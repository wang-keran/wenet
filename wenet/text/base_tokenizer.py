from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List, Tuple, Union

# 定义一个类型别名 T，它可以是 str（字符串）或 bytes（字节串）。这使得在整个类中处理字符串和字节串时更加灵活。
T = Union[str, bytes]


class BaseTokenizer(ABC):

    # 将输入字符串 line 分词，并返回分词结果和相应的 ID 列表。
    def tokenize(self, line: str) -> Tuple[List[T], List[int]]:
        tokens = self.text2tokens(line)
        ids = self.tokens2ids(tokens)
        return tokens, ids

    # 根据 ID 列表重建原始文本。
    def detokenize(self, ids: List[int]) -> Tuple[str, List[T]]:
        tokens = self.ids2tokens(ids)
        text = self.tokens2text(tokens)
        return text, tokens

    # 将输入字符串转换为 tokens，必须在子类中实现。
    @abstractmethod
    def text2tokens(self, line: str) -> List[T]:
        raise NotImplementedError("abstract method")

    # 将 tokens 转换为字符串文本，必须在子类中实现。
    @abstractmethod
    def tokens2text(self, tokens: List[T]) -> str:
        raise NotImplementedError("abstract method")

    # 将 tokens 转换为对应的 ID 列表，必须在子类中实现。
    @abstractmethod
    def tokens2ids(self, tokens: List[T]) -> List[int]:
        raise NotImplementedError("abstract method")

    # 将 ID 列表转换为 tokens，必须在子类中实现。
    @abstractmethod
    def ids2tokens(self, ids: List[int]) -> List[T]:
        raise NotImplementedError("abstract method")

    # 返回词汇表的大小，必须在子类中实现。
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError("abstract method")

    # 返回一个字典，将符号（tokens）映射到其对应的 ID，必须在子类中实现。
    @abstractproperty
    def symbol_table(self) -> Dict[T, int]:
        raise NotImplementedError("abstract method")

# BaseTokenizer 类提供了文本分词和反分词的基本框架，并定义了一系列必须由子类实现的抽象方法。
# 这种设计使得子类可以根据具体的需求和算法实现自己的分词逻辑，而基类则提供了一些通用的操作（如分词和反分词），有助于实现代码的重用和统一接口。
# 通过这样的设计，任何具体的分词器都可以在此基础上进行扩展，比如实现不同的文本预处理策略或不同的编码方式。