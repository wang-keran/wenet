# Copyright (c) 2021 NVIDIA CORPORATION
#               2023 58.com(Wuba) Inc AI Lab.
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
# 这里用到了words.txt文件

import multiprocessing
import numpy as np
import os
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
from swig_decoders import ctc_beam_search_decoder_batch, \
    Scorer, HotWordsScorer, map_batch
import yaml


class WenetModel(object):

    def __init__(self, model_config, device):
        # 解析模型配置中的参数，并将这些参数存储在 params 变量中。
        params = self.parse_model_parameters(model_config['parameters'])

        # 设置模型使用的设备
        self.device = device
        print("Using device", device)
        print("Successfully load model !")

        # load vocabulary，加载词汇表，并将结果存储在 self.id2vocab, self.vocab, space_id, blank_id, sos_eos 等变量中。
        # 如果词汇表中没有特定的 ID（如 space_id 或 blank_id），则使用默认值（如 -1 或 0）。
        # self.eos 和 self.sos 默认设置为词汇表的最后一个 ID，但如果 sos_eos 参数存在，则使用该值。
        # print("Successfully load vocabulary !") 输出成功加载词汇表的信息。
        ret = self.load_vocab(params["vocab_path"])
        self.id2vocab, self.vocab, space_id, blank_id, sos_eos = ret
        self.space_id = space_id if space_id else -1
        self.blank_id = blank_id if blank_id else 0
        self.eos = self.sos = sos_eos if sos_eos else len(self.vocab) - 1
        print("Successfully load vocabulary !")
        self.params = params

        # beam search setting
        self.beam_size = params.get("beam_size")
        self.cutoff_prob = params.get("cutoff_prob")

        # language model
        # 从 params 中获取语言模型路径 lm_path，并根据路径加载语言模型。
        # 如果路径存在，则创建 Scorer 对象，并将其存储在 self.scorer 中。
        # 语言模型用于在生成序列时提供额外的评分，帮助选择更合适的词。
        lm_path = params.get("lm_path", None)
        alpha, beta = params.get('alpha'), params.get('beta')
        self.scorer = None
        if os.path.exists(lm_path):
            self.scorer = Scorer(alpha, beta, lm_path, self.vocab)

        # load hotwords，设置自带的热词记录为空
        self.hotwords_scorer = None
        # 获取热词文件的路径
        hotwords_path = params.get("hotwords_path", None)
        # 检测热词路径是否存在
        if os.path.exists(hotwords_path):
            # 存在就加载热词
            self.hotwords = self.load_hotwords(hotwords_path)
            # 并计算这些热词的最大长度 max_order。
            max_order = 4
            # 如果self.hotwords存在，则创建一个 HotWordsScorer 对象，并将其赋值给 self.hotwords_scorer。
            if self.hotwords is not None:
                for w in self.hotwords:
                    max_order = max(max_order, len(w))
                self.hotwords_scorer = HotWordsScorer(self.hotwords,
                                                      self.vocab,
                                                      window_length=max_order,
                                                      SPACE_ID=-2,
                                                      is_character_based=True)
                # 最后，打印出成功加载热词的信息，并输出热词的最大长度。
                print(
                    f"Successfully load hotwords! Hotwords orders = {max_order}"
                )

        # 从传入的参数字典 params 中获取名为 'bidecoder' 的参数值，并将其赋值给实例变量 self.bidecoder。
        self.bidecoder = params.get('bidecoder')
        # rescore setting
        # 从 params 字典中获取键为 "rescoring" 的值，如果该键不存在，则返回默认值 0。
        self.rescoring = params.get("rescoring", 0)
        # 打印出是否使用了重评分（rescoring）
        print("Using rescoring:", bool(self.rescoring))
        # 通过 print("Successfully load all parameters!") 打印出成功加载所有参数的信息。
        print("Successfully load all parameters!")

        # 从 model_config 通过log_probs获取的配置
        log_probs_config = pb_utils.get_input_config_by_name(
            model_config, "log_probs")
        # Convert Triton types to numpy types，将log_prob_config的datatype字段转换为 NumPy 数据类型
        log_probs_dtype = pb_utils.triton_string_to_numpy(
            log_probs_config['data_type'])

        # 如果是上面获取的数据类型是np.float32，则将 self.dtype 设置为 torch.float32
        if log_probs_dtype == np.float32:
            self.dtype = torch.float32
        # 否则设置为 torch.float16。
        else:
            self.dtype = torch.float16

    # 初始化时创建一个缓存对象，但目前并没有实现具体的缓存逻辑。
    def generate_init_cache(self):
        encoder_out = None
        return encoder_out

    # 打开words.txt文件转换为字典列表形式
    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        # 初始化内部字典变量
        id2vocab = {}
        # 初始化其他变量
        space_id, blank_id, sos_eos = None, None, None
        # 打开文件
        with open(vocab_file, "r", encoding="utf-8") as f:
            # 循环读取文件内容
            for line in f:
                # 去除首尾空白字符
                line = line.strip()
                # 读取字符和行号id
                char, id = line.split()
                # 将id和行内容对应起来
                id2vocab[int(id)] = char
                # 存储空格字符
                if char == " ":
                    space_id = int(id)
                # 存储<blank>字符
                elif char == "<blank>":
                    blank_id = int(id)
                # 存储<sos/eos>字符
                elif char == "<sos/eos>":
                    sos_eos = int(id)
        # 创建一个长度为 id2vocab 字典大小的列表，初始值全部为0。测试上传远程仓库
        vocab = [0] * len(id2vocab)
        # 向里面写入读取出来的数据（不包含blank，sos/eos，空格这些特殊字符
        for id, char in id2vocab.items():
            vocab[id] = char
        # 方法返回一个元组，包含：id2vocab：字符到ID的映射字典；vocab：按ID索引的字符列表；space_id：空格字符的ID；
        # blank_id：<blank> 标识符的ID。；sos_eos：<sos/eos> 标识符的ID。
        return (id2vocab, vocab, space_id, blank_id, sos_eos)

    # 将热词文件加载到configs里
    def load_hotwords(self, hotwords_file):
        """
        load hotwords.yaml
        """
        # 读取hotwords.yaml热词文件
        with open(hotwords_file, 'r', encoding="utf-8") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        return configs

    # 解析传入的 model_parameters 字典，并将其值赋给预定义的 model_p 字典中的相应键。
    # 这里的数据来自config.pbtxt的parameter部分
    def parse_model_parameters(self, model_parameters):
        # 初始化字典model_p,每个键一个默认值
        model_p = {
            "beam_size": 10,
            "cutoff_prob": 0.999,
            "vocab_path": None,
            "lm_path": None,
            "hotwords_path": None,
            "alpha": 2.0,
            "beta": 1.0,
            "rescoring": 0,
            "bidecoder": 1
        }
        # get parameter configurations
        # 遍历 model_parameters 字典中的每一项，键存在于 model_p 中，则根据 model_p 中对应键的类型进行类型转换，
        # 并将转换后的值赋给 model_p 中的相应键。
        for li in model_parameters.items():
            # 获取键值对给key和value
            key, value = li
            # 获取value中的string_value对应的键值
            true_value = value["string_value"]
            # 不是model_p中的键值对元素继续查找
            if key not in model_p:
                continue
            # 获取model_p字典中键key对应的值的类型
            key_type = type(model_p[key])
            # 键值类型是空值，则将true_value赋值给model_p[key]
            if key_type == type(None):
                model_p[key] = true_value
            # 否则，将true_value转换为与key_type相同的类型后再赋值给model_p[key]
            else:
                model_p[key] = key_type(true_value)
        # 确保 model_p 中的 "vocab_path" 键对应的值不为 None，否则会抛出 AssertionError。
        assert model_p["vocab_path"] is not None
        # 返回赋值后的model_p字典
        return model_p

    # 推测函数
    # self：指代类的实例，使得这个函数能够访问类的其他方法和属性。
    # batch_log_probs：包含多个序列的对数概率矩阵，每个序列在矩阵中的一行代表一个时间步的概率分布。
    # batch_log_probs_idx：通常与batch_log_probs配合使用，记录了对数概率的有效索引，有时用于处理稀疏矩阵或特定格式的数据。
    # seq_lens：一个列表或数组，记录了每个输入序列的实际长度（不考虑填充）。
    # rescore_index：指定了哪些序列在初步解码后需要进行额外的打分（rescoring）。这通常用于对特定条件（如包含特定关键词）的序列进行优化。
    # batch_states：一个列表，包含了解码所需的多种状态信息，如trie树、起始位置、编码器历史记录和当前编码器输出。
    def infer(self, batch_log_probs, batch_log_probs_idx, seq_lens,
              rescore_index, batch_states):
        """
        batch_states = [trieVector, batch_start,
                       batch_encoder_hist, cur_encoder_out]
        """
        # 从batch_states中分解出解码所需的各种状态
        trie_vector, batch_start, batch_encoder_hist, cur_encoder_out = batch_states
        # 基于系统CPU核心数和输入批次大小来确定并行处理的数量，以提高解码效率，选最小的来干
        num_processes = min(multiprocessing.cpu_count(), len(batch_log_probs))

        # 前缀束搜索解码,给后面重打分使用
        score_hyps = self.batch_ctc_prefix_beam_search_cpu(
            batch_log_probs,
            batch_log_probs_idx,
            seq_lens,
            trie_vector,
            batch_start,
            self.beam_size,
            self.blank_id,
            self.space_id,
            self.cutoff_prob,
            num_processes,
            self.scorer,
            self.hotwords_scorer,
        )

        # 重新评分
        if self.rescoring and len(rescore_index) != 0:
            # find the end of sequence
            rescore_encoder_hist = []
            rescore_encoder_lens = []
            rescore_hyps = []
            res_idx = list(rescore_index.keys())
            max_length = -1
            for idx in res_idx:
                hist_enc = batch_encoder_hist[idx]
                if hist_enc is None:
                    cur_enc = cur_encoder_out[idx]
                    cur_mask_len = int(0 + seq_lens[idx])
                else:
                    cur_enc = torch.cat([hist_enc, cur_encoder_out[idx]],
                                        axis=0)
                    cur_mask_len = int(len(hist_enc) + seq_lens[idx])
                rescore_encoder_hist.append(cur_enc)
                rescore_encoder_lens.append(cur_mask_len)
                rescore_hyps.append(score_hyps[idx])
                if cur_enc.shape[0] > max_length:
                    max_length = cur_enc.shape[0]
            best_index = self.batch_rescoring(rescore_hyps,
                                              rescore_encoder_hist,
                                              rescore_encoder_lens, max_length)

        best_sent = []
        j = 0
        for idx, li in enumerate(score_hyps):
            if idx in rescore_index and self.rescoring:
                best_sent.append(li[best_index[j]][1])
                j += 1
            else:
                best_sent.append(li[0][1])

        # 使用词表示将每个句子转换为字符串，并返回最终结果。
        final_result = map_batch(best_sent, self.vocab, num_processes)

        return final_result, cur_encoder_out

    # 实现了批量CTC前缀束搜索解码器，调用了swig_decoders中的方法。
    def batch_ctc_prefix_beam_search_cpu(self, batch_log_probs_seq,
                                         batch_log_probs_idx, batch_len,
                                         batch_root, batch_start, beam_size,
                                         blank_id, space_id, cutoff_prob,
                                         num_processes, scorer,
                                         hotwords_scorer):
        """
        Return: Batch x Beam_size elements, each element is a tuple
                (score, list of ids),
        """

        # 将批次长度列表赋值给新变量。
        batch_len_list = batch_len
        # 将每个批次的概率序列和索引列表分别存储在两个新的列表中。
        batch_log_probs_seq_list = []
        batch_log_probs_idx_list = []
        # 遍历批次长度列表，将每个批次的概率序列和索引列表添加到对应的列表中。
        for i in range(len(batch_len_list)):
            cur_len = int(batch_len_list[i])
            batch_log_probs_seq_list.append(
                batch_log_probs_seq[i][0:cur_len].tolist())
            batch_log_probs_idx_list.append(
                batch_log_probs_idx[i][0:cur_len].tolist())
        # 获取结果序列
        score_hyps = ctc_beam_search_decoder_batch(
            batch_log_probs_seq_list, batch_log_probs_idx_list, batch_root,
            batch_start, beam_size, num_processes, blank_id, space_id,
            cutoff_prob, scorer, hotwords_scorer)
        return score_hyps

    # 批量重打分
    def batch_rescoring(self, score_hyps, hist_enc, hist_mask_len, max_len):
        """
        score_hyps: [((ctc_score, (id1, id2, id3, ....)), (), ...), ....]包含每个批次中候选序列及其CTC分数的列表。
        hist_enc: [len1xF, len2xF, .....]历史编码器输出的张量列表。
        hist_mask: [1x1xlen1, 1x1xlen2] 每个历史编码器输出的实际长度。
        return bzx1  best_index 最佳路径对应的索引
        """
        # batch_size批次大小
        bz = len(hist_enc)
        # 编码器输出的特征维度（最后一个维度）
        f = hist_enc[0].shape[-1]
        # 波束搜索的波束宽度，即保存多少条最佳前缀路径
        beam_size = self.beam_size
        # 存储每个批次中编码器输出的实际长度。
        encoder_lens = np.zeros((bz, 1), dtype=np.int32)
        # 存储所有批次的编码器输出，初始化为零。
        encoder_out = torch.zeros((bz, max_len, f), dtype=self.dtype)
        # 存储所有候选序列的ID列表。
        hyps = []
        # 存储每个候选序列的CTC分数。
        ctc_score = torch.zeros((bz, beam_size), dtype=self.dtype)
        # 记录最长候选序列的长度。
        max_seq_len = 0
        # 处理每个批次的历史编码器输出
        for i in range(bz):
            # 将每个批次的历史编码器输出复制到 encoder_out 中。
            cur_len = hist_enc[i].shape[0]
            encoder_out[i, 0:cur_len] = hist_enc[i]
            # 更新 encoder_lens，记录每个批次的实际长度。
            encoder_lens[i, 0] = hist_mask_len[i]

            # process candidate处理候选序列
            # 如果某个批次的候选序列数量少于 beam_size，则补充无效候选序列（分数设为 -10000）。
            if len(score_hyps[i]) < beam_size:
                to_append = (beam_size - len(score_hyps[i])) * [(-10000, ())]
                score_hyps[i] = list(score_hyps[i]) + to_append
            # 遍历每个候选序列，提取其 CTC 分数和 ID 列表，并更新 ctc_score 和 hyps。
            for idx, c in enumerate(score_hyps[i]):
                score, idlist = c
                if score < -10000:
                    score = -10000
                ctc_score[i][idx] = score
                hyps.append(list(idlist))
                # 更新 max_seq_len，记录最长候选序列的长度。
                if len(hyps[-1]) > max_seq_len:
                    max_seq_len = len(hyps[-1])

        # 初始化填充后的候选序列张量。添加 <sos> 和 <eos> 标记，因此最大长度加2。
        max_seq_len += 2
        # 初始化填充后的候选序列张量，初始值为 <eos>。
        hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len),
                                   dtype=np.int64)
        hyps_pad_sos_eos = hyps_pad_sos_eos * self.eos  # fill eos
        # 如果使用双向解码器，则初始化反向填充后的候选序列张量。
        if self.bidecoder:
            r_hyps_pad_sos_eos = np.ones((bz, beam_size, max_seq_len),
                                         dtype=np.int64)
            r_hyps_pad_sos_eos = r_hyps_pad_sos_eos * self.eos

        # 初始化候选序列长度张量。
        hyps_lens_sos = np.ones((bz, beam_size), dtype=np.int32)
        # 填充候选序列
        bz_id = 0
        # 遍历所有候选序列
        for idx, cand in enumerate(hyps):
            # 计算当前批次的索引 bz_id 和偏移量 bz_offset。
            bz_id = idx // beam_size
            length = len(cand) + 2
            bz_offset = idx % beam_size
            # 构建带 <sos> 和 <eos> 的填充候选序列 pad_cand。
            pad_cand = [self.sos] + cand + [self.eos]
            # 将 pad_cand 填充到 hyps_pad_sos_eos 中。
            hyps_pad_sos_eos[bz_id][bz_offset][0:length] = pad_cand
            # 如果使用双向解码器，构建反向填充候选序列 r_pad_cand 并填充到 r_hyps_pad_sos_eos 中。
            if self.bidecoder:
                r_pad_cand = [self.sos] + cand[::-1] + [self.eos]
                r_hyps_pad_sos_eos[bz_id][bz_offset][0:length] = r_pad_cand
            # 更新 hyps_lens_sos，记录每个候选序列的实际长度。
            hyps_lens_sos[bz_id][idx % beam_size] = len(cand) + 1
        # 这里直接调用decoder.onnx返回最佳索引，通过最佳索引获得最佳结果
        in0 = pb_utils.Tensor.from_dlpack("encoder_out",
                                          to_dlpack(encoder_out))
        in1 = pb_utils.Tensor("encoder_out_lens", encoder_lens)
        in2 = pb_utils.Tensor("hyps_pad_sos_eos", hyps_pad_sos_eos)
        in3 = pb_utils.Tensor("hyps_lens_sos", hyps_lens_sos)
        input_tensors = [in0, in1, in2, in3]
        if self.bidecoder:
            in4 = pb_utils.Tensor("r_hyps_pad_sos_eos", r_hyps_pad_sos_eos)
            input_tensors.append(in4)
        in5 = pb_utils.Tensor.from_dlpack("ctc_score", to_dlpack(ctc_score))
        input_tensors.append(in5)
        # 发送数据
        request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['best_index'],
            inputs=input_tensors)
        # 发送推理请求启动推理
        response = request.exec()
        # 从响应中获取名为 best_index 的输出张量。
        best_index = pb_utils.get_output_tensor_by_name(response, 'best_index')
        # 使用 from_dlpack 将张量转换为 PyTorch 张量。
        best_index = from_dlpack(best_index.to_dlpack()).clone()
        # 将张量移动到 CPU 并转换为 NumPy 数组。
        best_index = best_index.cpu().numpy()[:, 0]
        # 返回最佳索引数组。
        return best_index

    def __del__(self):
        print("remove wenet model")
