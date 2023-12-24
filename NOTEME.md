## 目录结构

└─TTS
    ├─bin  脚本
    ├─config  模型配置
    │  └─__pycache__
    ├─demos  示例程序
    │  └─xtts_ft_demo
    │      └─utils
    ├─encoder  编码器 for Seq2Seq
    │  ├─configs  编码器配置
    │  ├─models  编码器模型
    │  └─utils 
    ├─server  服务器
    │  ├─static
    │  └─templates
    ├─tts  声学模型或端到端
    │  ├─configs
    │  │  └─__pycache__
    │  ├─datasets  数据集
    │  ├─layers  模块
    │  │  ├─align_tts
    │  │  ├─bark
    │  │  │  └─hubert
    │  │  ├─delightful_tts
    │  │  ├─feed_forward
    │  │  ├─generic
    │  │  ├─glow_tts
    │  │  ├─overflow
    │  │  ├─tacotron
    │  │  ├─tortoise
    │  │  ├─vits
    │  │  └─xtts
    │  │      └─trainer
    │  ├─models  模型
    │  ├─utils  
    │  │  ├─assets
    │  │  │  └─tortoise
    │  │  ├─monotonic_align  对齐
    │  │  ├─text  文本前端
    │  │  │  ├─bangla
    │  │  │  │  └─__pycache__
    │  │  │  ├─belarusian
    │  │  │  │  └─__pycache__
    │  │  │  ├─chinese_mandarin
    │  │  │  │  └─__pycache__
    │  │  │  ├─english
    │  │  │  │  └─__pycache__
    │  │  │  ├─french
    │  │  │  │  └─__pycache__
    │  │  │  ├─japanese
    │  │  │  │  └─__pycache__
    │  │  │  ├─korean
    │  │  │  │  └─__pycache__
    │  │  │  ├─phonemizers  音素化
    │  │  │  │  └─__pycache__
    │  │  │  └─__pycache__
    │  │  └─__pycache__
    │  └─__pycache__
    ├─utils  音频工具
    │  ├─audio
    │  └─__pycache__
    ├─vc  声音转换
    │  ├─configs
    │  ├─models
    │  └─modules
    │      └─freevc
    │          ├─speaker_encoder
    │          └─wavlm
    ├─vocoder  声码器模型
    │  ├─configs
    │  ├─datasets
    │  ├─layers
    │  ├─models
    │  └─utils
    └─__pycache__

## 关系
encoder 包含 lstm.LSTMSpeakerEncoder 和 resnet.ResNetSpeakerEncoder;
setup_encoder_model 包含两者，被EmbeddingManager用于实例化encoder;
EmbeddingManager 被 speakers 调用，SpeakerManager 继承
SpeakerManager 通过 get_speaker_manager 实例化
最终是由config实例化，use_d_vector_file， d_vector_file，use_speaker_embedding

SpeakerManager 和 get_speaker_balancer_weights 被 BaseTTS 调用
speaker_manager 是初始化的参数之一，被用于 init_multispeaker

BaseTTS 被所有TTS模型继承
get_data_loader

## 3rd-party

fsspec 是一个 Python 库，它提供了一种统一的接口来访问不同文件系统的数据。它的目标是为不同的存储后端（如本地文件系统、Hadoop 分布式文件系统、Amazon S3 等）提供一个一致的 API，使得在不同的文件系统之间进行读取和写入操作更加简单和灵活。

通过 fsspec，你可以使用相同的操作方式来处理各种文件系统，而不需要关心底层实现的细节。它的设计理念类似于 Python 标准库中的 open() 函数，但更加通用和可扩展。

fsspec 提供了许多功能，包括文件和目录的读写、文件的追加、文件的复制和移动、文件的元数据操作等。它还支持并行操作和流式读取，方便处理大规模数据集。


## function
- config.loadconfig 从配置文件生成配置示例

## ignore
encoder.configs.emotion_encoder_config




## 术语
F0帧（Fundamental Frequency Frame）：F0帧是指音频信号中的基频帧，也称为音调。它表示声音振动的周期性特征，即人耳所感知到的音高。F0帧的最大频率和最小频率分别是pitch_fmax和pitch_fmin。

RMS（Root Mean Square）：RMS是指音频信号的均方根值。它代表了音频信号的能量强度或音量大小。通过计算音频信号的平方值的平均值，并取其平方根得到RMS值。在音频处理中，可以使用RMS进行音频音量的标准化、静音检测和裁剪等操作。

STFT 短时傅里叶变换

## 运行
tts --model_path=E:\models\tts_models--zh-CN--baker--tacotron2-DDC-GST\model_file.pth --config_path=E:\models\tts_models--zh-CN--baker--tacotron2-DDC-GST\config.json --text=我爱我的祖国
