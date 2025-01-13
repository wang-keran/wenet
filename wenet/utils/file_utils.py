# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


# 读取指定文件中的每一行，并返回一个字符串列表。
def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


# 从指定文件中读取非语言符号，验证符号格式。
def read_non_lang_symbols(non_lang_sym_path):
    """read non-linguistic symbol from file.

    The file format is like below:

    {NOISE}\n
    {BRK}\n
    ...


    Args:
        non_lang_sym_path: non-linguistic symbol file path, None means no any
        syms.

    """
    if non_lang_sym_path is None:
        return []
    else:
        syms = read_lists(non_lang_sym_path)
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        for sym in syms:
            if non_lang_syms_pattern.fullmatch(sym) is None:

                class BadSymbolFormat(Exception):
                    pass

                raise BadSymbolFormat(
                    "Non-linguistic symbols should be "
                    "formatted in {xxx}/<xxx>/[xxx], consider"
                    " modify '%s' to meet the requirment. "
                    "More details can be found in discussions here : "
                    "https://github.com/wenet-e2e/wenet/pull/819" % (sym))
        return syms


# 读取符号表文件，并返回一个符号到 ID 的映射字典。
def read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table

# 总结：这段代码包含三个功能模块，分别用于读取列表、验证非语言符号格式以及读取符号表。
# read_non_lang_symbols 函数使用正则表达式确保符号格式的正确性，增强了代码的鲁棒性。