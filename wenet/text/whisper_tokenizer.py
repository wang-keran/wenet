from os import PathLike
from typing import Dict, List, Optional, Tuple, Union
from wenet.text.base_tokenizer import BaseTokenizer

from wenet.utils.file_utils import read_non_lang_symbols


# 定义了一个名为 WhisperTokenizer 的类，继承自 BaseTokenizer，用于处理文本的分词和反向操作。
class WhisperTokenizer(BaseTokenizer):

    # 传进来multilingual和num_languages，返回
    def __init__(
        self,
        multilingual: bool,
        num_languages: int = 99,
        language: Optional[str] = None,
        task: Optional[str] = None,
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        *args,
        **kwargs,
    ) -> None:
        # NOTE(Mddct): don't build here, pickle issues
        self.tokenizer = None
        # TODO: we don't need this in future
        self.multilingual = multilingual
        self.num_languages = num_languages
        self.language = language
        self.task = task

        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        # TODO(Mddct): add special tokens, like non_lang_syms
        del self.non_lang_syms

    # 实现自定义序列化和反序列化方法，确保在序列化时不保存 tokenizer 属性，避免潜在的 pickle 问题。
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        recovery = {'tokenizer': None}
        self.__dict__.update(recovery)

    # 构建分词器并初始化词表（token-to-id 和 id-to-token 映射）。
    def _build_tiktoken(self):
        if self.tokenizer is None:
            # 这里是whisper模型专用的分词器
            from whisper.tokenizer import get_tokenizer
            # 初始化自带的分词器
            self.tokenizer = get_tokenizer(multilingual=self.multilingual,
                                           num_languages=self.num_languages,
                                           language=self.language,
                                           task=self.task)
            self.t2i = {}
            self.i2t = {}
            for i in range(self.tokenizer.encoding.n_vocab):
                unit = str(
                    self.tokenizer.encoding.decode_single_token_bytes(i))
                if len(unit) == 0:
                    unit = str(i)
                unit = unit.replace(" ", "<space>")
                # unit = bytes(unit, 'utf-8')
                self.t2i[unit] = i
                self.i2t[i] = unit
            assert len(self.t2i) == len(self.i2t)

    # 将输入文本行进行分词。
    def tokenize(self, line: str) -> Tuple[List[str], List[int]]:
        self._build_tiktoken()
        ids = self.tokenizer.encoding.encode(line)
        text = [self.i2t[d] for d in ids]
        return text, ids

    # 将 ID 列表反向转换为文本。
    def detokenize(self, ids: List[int]) -> Tuple[str, List[str]]:
        self._build_tiktoken()
        tokens = [self.i2t[d] for d in ids]
        text = self.tokenizer.encoding.decode(ids)
        return text, tokens

    # 将输入文本转换为 tokens。
    def text2tokens(self, line: str) -> List[str]:
        self._build_tiktoken()
        return self.tokenize(line)[0]

    # 将 tokens 转换为文本。
    def tokens2text(self, tokens: List[str]) -> str:
        self._build_tiktoken()
        ids = [self.t2i[t] for t in tokens]
        return self.detokenize(ids)[0]

    # 将 tokens 转换为 ID。
    def tokens2ids(self, tokens: List[str]) -> List[int]:
        self._build_tiktoken()
        ids = [self.t2i[t] for t in tokens]
        return ids

    # 将 ID 转换为 tokens。
    def ids2tokens(self, ids: List[int]) -> List[str]:
        self._build_tiktoken()
        return [self.tokenizer.encoding.decode([id]) for id in ids]

    # 返回词汇表的大小。
    def vocab_size(self) -> int:
        self._build_tiktoken()
        return len(self.t2i)

    # 返回符号表（token 到 ID 的映射）。
    @property
    def symbol_table(self) -> Dict[str, int]:
        self._build_tiktoken()
        return self.t2i

# 总结：WhisperTokenizer 类为处理多语言文本提供了强大的分词和反向操作功能。
# 它通过对 Whisper 模型的封装，实现了灵活的文本编码和解码，并提供了 token 与 ID 之间的映射。