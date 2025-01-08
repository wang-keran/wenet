from os import PathLike
from typing import Dict, List, Union
from wenet.text.base_tokenizer import BaseTokenizer, T as Type


# 这段代码定义了一个名为 HuggingFaceTokenizer 的类，继承自 BaseTokenizer，用于与 Hugging Face 的 Transformers 库的分词器进行交互。
class HuggingFaceTokenizer(BaseTokenizer):

    # 初始化方法
    def __init__(self, model: Union[str, PathLike], *args, **kwargs) -> None:
        # NOTE(Mddct): don't build here, pickle issues
        self.model = model
        self.tokenizer = None

        self.args = args
        self.kwargs = kwargs

    # 在序列化对象时调用，控制如何保存对象的状态。
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    # 在反序列化对象时调用，控制如何恢复对象的状态。
    def __setstate__(self, state):
        self.__dict__.update(state)
        recovery = {'tokenizer': None}
        self.__dict__.update(recovery)

    # 构建 Hugging Face 的分词器。
    def _build_hugging_face(self):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, **self.kwargs)
            self.t2i = self.tokenizer.get_vocab()

    # 将输入文本 line 转换为 tokens 列表。
    def text2tokens(self, line: str) -> List[Type]:
        self._build_hugging_face()
        return self.tokenizer.tokenize(line)

    # 将 tokens 列表转换为字符串文本。
    def tokens2text(self, tokens: List[Type]) -> str:
        self._build_hugging_face()
        ids = self.tokens2ids(tokens)
        return self.tokenizer.decode(ids)

    # 将 tokens 列表转换为其对应的 id 列表。
    def tokens2ids(self, tokens: List[Type]) -> List[int]:
        self._build_hugging_face()
        return self.tokenizer.convert_tokens_to_ids(tokens)

    # 将 id 列表转换为对应的 tokens 列表。
    def ids2tokens(self, ids: List[int]) -> List[Type]:
        self._build_hugging_face()
        return self.tokenizer.convert_ids_to_tokens(ids)

    # 获取词汇表的大小。
    def vocab_size(self) -> int:
        self._build_hugging_face()
        # TODO: we need special tokenize size in future
        return len(self.tokenizer)

    # 获取符号表。
    @property
    def symbol_table(self) -> Dict[Type, int]:
        self._build_hugging_face()
        return self.t2i

# 总结：HuggingFaceTokenizer 类用于与 Hugging Face Transformers 库的分词器交互，提供了从文本到 tokens 的转换、tokens 到文本的转换、以及 tokens 和 ids 之间的相互转换功能。
# 使得自然语言处理任务中的文本预处理变得更加方便。