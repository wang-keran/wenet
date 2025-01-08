from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_bpe_model


# 定义了一个名为 BpeTokenizer 的类，继承自 CharTokenizer，实现了基于字节对编码（BPE，Byte Pair Encoding）的分词器。
class BpeTokenizer(CharTokenizer):

    # 初始化方法 
    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        super().__init__(symbol_table, non_lang_syms, split_with_space,
                         connect_symbol, unk)
        self._model = bpe_model
        # NOTE(Mddct): multiprocessing.Process() issues
        #              don't build sp here
        self.bpe_model = None

    # 构建并加载 BPE 模型。
    def _build_sp(self):
        if self.bpe_model is None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(self._model)

    # 将输入字符串 line 转换为 tokens 列表。
    def text2tokens(self, line: str) -> List[str]:
        self._build_sp()
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
                tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
        return tokens

    # 将 tokens 列表转换为字符串文本。
    def tokens2text(self, tokens: List[str]) -> str:
        self._build_sp()
        text = super().tokens2text(tokens)
        return text.replace("▁", ' ').strip()

# 总结：BpeTokenizer 类实现了基于字节对编码（BPE）的分词器，利用 sentencepiece 库来加载和使用 BPE 模型。
# 通过定义的方法，该类能够将字符串文本转换为 tokens，并能够将 tokens 重建为原始文本。