# WeNet API

We refer [vosk](https://github.com/alphacep/vosk-api/blob/master/src/vosk_api.h)
for the interface design.


We are going to implement the following interfaces:

- [x] non-streaming recognition 非流式语音识别
- [] streaming recognition      流式语音识别
- [] nbest                      即语境偏置词，是指在语音识别（ASR）系统中，为了提高在相关领域词汇（俗称“热词”）上的识别准确率，而特别引入或强调的词汇。这些词汇通常与用户的特定需求或上下文环境紧密相关，例如领域专有名词、用户通讯录中的人名等。
- [] contextual biasing word    在生成文本时，alignment指的是语言模型在输出中对齐用户输入的方式，确保生成的文本与用户输入相关联。
- [] alignment                  语言支持工具是指那些旨在帮助人们克服语言障碍的资源和技术。在计算机编程和软件开发中，后置处理器是一种机制，用于在构建
或生成过程中执行额外的操作。例如，在Spring框架中，后置处理器（PostProcessor）用于在Bean实例化、依赖注入和初始化完成后执行自定义操作。此外，在JMeter等测试工具中，后置处理器用于处理测试结果和数据。
- [] language support(post processor)   对音频内容进行标签检查的过程，提高识别精准度wenet的推理流程，哪块代码负责什么，编码，解码
- [] label check                
