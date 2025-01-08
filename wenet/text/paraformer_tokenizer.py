from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.paraformer.search import paraformer_beautify_result
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_seg_dict


# 读取分词字典文件并返回一个字典。
def read_seg_dict(path):
    seg_table = {}
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            assert len(arr) == 2
            seg_table[arr[0]] = arr[1]
    return seg_table


# 主要功能是根据给定的符号表和分词字典将文本转换为 tokens，以及将 tokens 还原为文本。
class ParaformerTokenizer(CharTokenizer):

    # 初始化
    def __init__(self,
                 symbol_table: Union[str, PathLike, Dict],
                 seg_dict: Optional[Union[str, PathLike, Dict]] = None,
                 split_with_space: bool = False,
                 connect_symbol: str = '',
                 unk='<unk>') -> None:
        super().__init__(symbol_table, None, split_with_space, connect_symbol,
                         unk)
        self.seg_dict = seg_dict
        if seg_dict is not None and not isinstance(seg_dict, Dict):
            self.seg_dict = read_seg_dict(seg_dict)

    # 将输入文本 line 转换为 tokens。
    def text2tokens(self, line: str) -> List[str]:
        assert self.seg_dict is not None

        # TODO(Mddct): duplicated here, refine later
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line)
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(tokenize_by_seg_dict(self.seg_dict, part))
        return tokens

    # 将 tokens 列表还原为文本。
    def tokens2text(self, tokens: List[str]) -> str:
        return paraformer_beautify_result(tokens)

# 总结：ParaformerTokenizer 类通过继承 CharTokenizer 实现了一个分词器，能够读取分词字典并使用该字典对文本进行分词，同时也能将分词后的 tokens 转换回文本。