import re

from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.text.base_tokenizer import BaseTokenizer


# 定义了一个名为 CharTokenizer 的类，继承自 BaseTokenizer，实现了基于字符的分词器。
class CharTokenizer(BaseTokenizer):

    # 初始化方法
    def __init__(
        self,
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        self.non_lang_syms_pattern = None
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(
                r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        if not isinstance(symbol_table, Dict):
            # 读取符号表文件，并返回一个符号到 ID 的映射字典。
            self._symbol_table = read_symbol_table(symbol_table)
        else:
            # symbol_table = {"我": 1, "是": 2, "{NOISE}": 3}
            self._symbol_table = symbol_table
        if not isinstance(non_lang_syms, List):
            # 从指定文件中读取非语言符号，验证符号格式。
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        self.char_dict = {v: k for k, v in self._symbol_table.items()}
        self.split_with_space = split_with_space
        self.connect_symbol = connect_symbol
        self.unk = unk

    # 将输入字符串 line 转换为 tokens 列表。
    def text2tokens(self, line: str) -> List[str]:
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.split_with_space:
                    part = part.split(" ")
                for ch in part:
                    if ch == ' ':
                        ch = "▁"
                    tokens.append(ch)
        return tokens

    # 将 tokens 列表转换为字符串文本。
    def tokens2text(self, tokens: List[str]) -> str:
        return self.connect_symbol.join(tokens)

    # 将 tokens 列表转换为其对应的 id 列表。
    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for ch in tokens:
            if ch in self._symbol_table:
                ids.append(self._symbol_table[ch])
            elif self.unk in self._symbol_table:
                ids.append(self._symbol_table[self.unk])
        return ids

    # 将 id 列表转换为对应的 tokens 列表。
    def ids2tokens(self, ids: List[int]) -> List[str]:
        content = [self.char_dict[w] for w in ids]
        return content

    # 获取词汇表的大小。
    def vocab_size(self) -> int:
        return len(self.char_dict)

    # 获取符号表。
    @property
    def symbol_table(self) -> Dict[str, int]:
        return self._symbol_table

# 总结：CharTokenizer 类实现了基于字符的分词器，能够将字符串文本转换为字符级的 tokens，并能够将这些 tokens 转换为其对应的 id。