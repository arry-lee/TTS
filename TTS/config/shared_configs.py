from dataclasses import asdict, dataclass
from typing import List

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig


@dataclass
class BaseAudioConfig(Coqpit):
    """用于定义音频处理参数的基本配置。用于初始化```TTS.utils.audio.AudioProcessor。```

    Args:
        fft_size (int):
            STFT频率级别数，也就是线性频谱图帧的大小。默认为1024。

        win_length (int):
            每个音频帧都被长度为```win_length```的窗口进行窗口处理，然后用零填充以匹配```fft_size```。默认为1024。

        hop_length (int):
            相邻的STFT列之间的音频样本数。默认为1024。

        frame_shift_ms (int):
            基于毫秒和采样率设置```hop_length```。

        frame_length_ms (int):
            基于毫秒和采样率设置```win_length```。

        stft_pad_mode (str):
            STFT中使用的填充方法。'reflect'或'center'。默认为'reflect'。

        sample_rate (int):
            音频采样率。默认为22050。

        resample (bool):
            启用/禁用将音频重采样为```sample_rate```。默认为```False```.

        preemphasis (float):
            预加重系数。默认为0.0。

        ref_level_db (int): 20
            用于重新调整音频信号并忽略低于该水平的参考Db级别。假定20Db为空气的声音。默认为20。

        do_sound_norm (bool):
            启用/禁用对音频样本之间的音量差异进行声音标准化。默认为False。

        log_func (str):
            用于幅度到DB转换的Numpy对数函数。默认为'np.log10'。

        do_trim_silence (bool):
            启用/禁用裁剪音频片段开头和结尾的静音。默认为```True```.

        do_amp_to_db_linear (bool, optional):
            启用/禁用线性频谱图的幅度到dB转换。默认为True。

        do_amp_to_db_mel (bool, optional):
            启用/禁用mel频谱图的幅度到dB转换。默认为True。

        pitch_fmax (float, optional):
            F0帧的最大频率。默认为```640```.

        pitch_fmin (float, optional):
            F0帧的最小频率。默认为```1```.

        trim_db (int):
            用于静音修剪的静音阈值。默认为45。

        do_rms_norm (bool, optional):
            加载音频文件时启用/禁用RMS音量标准化。默认为False。

        db_level (int, optional):
            用于rms标准化的dB级别。范围为-99至0。默认为None。

        power (float):
            用于扩展声谱图级别的指数，在运行Griffin Lim之前有助于减少合成声音中的伪影。默认为1.5。

        griffin_lim_iters (int):
            Griffin Lim迭代次数。默认为60。

        num_mels (int):
            定义每个mel频谱图帧的帧长度的mel基础帧数量。默认为80。

        mel_fmin (float): 用于mel基础滤波器的最小频率级别。男性声音约为50，女性声音约为95。需要根据数据集进行调整。默认为0。

        mel_fmax (float):
            用于mel基础滤波器的最大频率级别。需要根据数据集进行调整。

        spec_gain (int):
            在将幅度转换为DB时应用的增益。默认为20。

        signal_norm (bool):
            启用/禁用信号标准化。默认为True。

        min_level_db (int):
            计算的melspectrograms的最小db阈值。默认为-100。

        symmetric_norm (bool):
            启用/禁用对称标准化。如果设置为True，则在范围[-k, k]内执行标准化，否则在范围[0, k]内。默认为True。

        max_norm (float):
            定义规范化范围的```k```。默认为4.0。

        clip_norm (bool):
            启用/禁用对标准化音频信号中超出范围的值进行裁剪。默认为True。

        stats_path (str):
            计算的统计文件的路径。默认为None。
    """
    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # f0 params
    pitch_fmax: float = 640.0
    pitch_fmin: float = 1.0
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)


@dataclass
class BaseDatasetConfig(Coqpit):
    """TTS数据集的基本配置。

    Args:
        formatter (str):
            定义在```TTS.tts.datasets.formatter```中使用的格式化器名称。默认为`""`。

        dataset_name (str):
            数据集的唯一名称。默认为`""`。

        path (str):
            数据集文件的根目录。默认为`""`。

        meta_file_train (str):
            数据集元文件的名称。或多说话人数据集的要在训练中忽略的说话人列表。默认为`""`。

        ignored_speakers (List):
            不在训练中使用的说话人ID列表。默认为None。

        language (str):
            数据集的语言代码。如果定义，则会覆盖`phoneme_language`。默认为`""`。

        phonemizer (str):
            用于该数据集语言的音素转换器。默认情况下，它使用`DEF_LANG_TO_PHONEMIZER`。默认为`""`。

        meta_file_val (str):
            定义用于验证的实例的数据集元文件的名称。

        meta_file_attn_mask (str):
            用于训练持续时间预测器需要注意力掩码的模型所使用的文件的路径。
    """
    formatter: str = ""
    dataset_name: str = ""
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: List[str] = None
    language: str = ""
    phonemizer: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("formatter", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)


@dataclass
class BaseTrainingConfig(TrainerConfig):
    """Base config to define the basic 🐸TTS training parameters that are shared
    among all the models. It is based on ```Trainer.TrainingConfig```.

    Args:
        model (str):
            Name of the model that is used in the training.

        num_loader_workers (int):
            Number of workers for training time dataloader.

        num_eval_loader_workers (int):
            Number of workers for evaluation time dataloader.
    """

    model: str = None
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False
