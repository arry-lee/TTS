### Speaker Encoder

This is an implementation of https://arxiv.org/abs/1710.10467. This model can be used for voice and speaker embedding.

With the code here you can generate d-vectors for both multi-speaker and single-speaker TTS datasets, then visualise and explore them along with the associated audio files in an interactive chart.

Below is an example showing embedding results of various speakers. You can generate the same plot with the provided notebook as demonstrated in [this video](https://youtu.be/KW3oO7JVa7Q).

![](umap.png)

Download a pretrained model from [Released Models](https://github.com/mozilla/TTS/wiki/Released-Models) page.

To run the code, you need to follow the same flow as in TTS.

- Define 'config.json' for your needs. Note that, audio parameters should match your TTS model.
- Example training call ```python speaker_encoder/train.py --config_path speaker_encoder/config.json --data_path ~/Data/Libri-TTS/train-clean-360```
- Generate embedding vectors ```python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth model/config/path/config.json dataset/path/ output_path``` . This code parses all .wav files at the given dataset path and generates the same folder structure under the output path with the generated embedding files.
- Watch training on Tensorboard as in TTS

## 说话人编码器
这是一个实现了Speaker Encoder论文的模型，可以用于语音和说话人嵌入。

使用这里的代码，您可以为多说话人和单一说话人的TTS数据集生成d-向量，然后在交互式图表中可视化和探索这些向量以及相关的音频文件。

下面是一个展示不同说话人嵌入结果的示例。您可以使用提供的notebook生成相同的图表，具体操作可参考[这个视频](https://youtu.be/KW3oO7JVa7Q)。

![](umap.png)

从[Released Models](https://github.com/mozilla/TTS/wiki/Released-Models)页面下载预训练模型。

要运行代码，您需要按照TTS的流程进行操作。

- 为您的需求定义 'config.json'。请注意，音频参数应与您的TTS模型相匹配。
- 示例训练调用：```python speaker_encoder/train.py --config_path speaker_encoder/config.json --data_path ~/Data/Libri-TTS/train-clean-360```
- 生成嵌入向量：```python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth model/config/path/config.json dataset/path/ output_path```。此代码会解析给定数据集路径下的所有.wav文件，并在输出路径下生成相同的文件夹结构，其中包含生成的嵌入文件。
- 跟随TTS中的步骤，在Tensorboard上查看训练进度。
