import torch
from torch import nn

from TTS.encoder.models.base_encoder import BaseEncoder


class LSTMWithProjection(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, proj_size, bias=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        o, (_, _) = self.lstm(x)
        return self.linear(o)


class LSTMWithoutProjection(nn.Module):
    def __init__(self, input_dim, lstm_dim, proj_dim, num_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_dim, num_layers=num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(lstm_dim, proj_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.relu(self.linear(hidden[-1]))


class LSTMSpeakerEncoder(BaseEncoder):
    """
    这是一个LSTM音频编码器的类实现。它用于将原始音频信号或频谱帧转换成固定维度的音频特征表示。

    构造函数中的参数说明如下：
    - input_dim: 输入的维度，可以是原始音频信号的采样点数或者频谱帧的维度。
    - proj_dim: LSTM投影层的维度，默认为256。
    - lstm_dim: LSTM层的维度，默认为768。
    - num_lstm_layers: LSTM层数，默认为3。
    - use_lstm_with_projection: 是否在LSTM层之间使用投影层，默认为True。
    - use_torch_spec: 是否使用torch_spec来计算频谱，默认为False。
    - audio_config: 音频配置参数。

    该类的forward方法用于进行模型的前向传递。具体流程如下：
    1. 如果use_torch_spec为True，则将输入x从三维张量调整为二维张量，并通过torch_spec计算频谱。
    2. 对输入x进行Instance Normalization，并调整维度从(batch_size, input_dim, sequence_length)变为(batch_size, sequence_length, input_dim)。
    3. 将调整后的x输入到LSTM层中进行处理。
    4. 如果use_lstm_with_projection为True，则取最后一层LSTM的输出作为最终结果；否则，直接使用LSTMWithoutProjection的输出。
    5. 如果l2_norm为True，则对输出进行L2归一化处理。
    6. 返回最终的音频特征表示。
    """
    def __init__(
        self,
        input_dim,
        proj_dim=256,
        lstm_dim=768,
        num_lstm_layers=3,
        use_lstm_with_projection=True,
        use_torch_spec=False,
        audio_config=None,
    ):
        super().__init__()
        self.use_lstm_with_projection = use_lstm_with_projection
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config
        self.proj_dim = proj_dim

        layers = []
        # choise LSTM layer
        if use_lstm_with_projection:
            layers.append(LSTMWithProjection(input_dim, lstm_dim, proj_dim))
            for _ in range(num_lstm_layers - 1):
                layers.append(LSTMWithProjection(proj_dim, lstm_dim, proj_dim))
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = LSTMWithoutProjection(input_dim, lstm_dim, proj_dim, num_lstm_layers)

        self.instancenorm = nn.InstanceNorm1d(input_dim)

        if self.use_torch_spec:
            self.torch_spec = self.get_torch_mel_spectrogram_class(audio_config)
        else:
            self.torch_spec = None

        self._init_layers()

    def _init_layers(self):
        for name, param in self.layers.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x, l2_norm=True):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.use_torch_spec:
                    x.squeeze_(1)
                    x = self.torch_spec(x)
                x = self.instancenorm(x).transpose(1, 2)
        d = self.layers(x)
        if self.use_lstm_with_projection:
            d = d[:, -1]
        if l2_norm:
            d = torch.nn.functional.normalize(d, p=2, dim=1)
        return d
