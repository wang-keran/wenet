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

import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing
from torch.utils.dlpack import from_dlpack
from swig_decoders import ctc_beam_search_decoder_batch, \
    Scorer, HotWordsScorer, PathTrie, TrieVector, map_batch
import json
import os
import yaml


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # 获取模型配置参数，包括模型的最大批处理大小，模型名称，模型版本等。
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration获取输出层配置信息
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types将字符串转换成numpy类型
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration，获取batch_log_probs的输入层配置信息
        batch_log_probs = pb_utils.get_input_config_by_name(
            model_config, "batch_log_probs")
        # 从配置中提取 dims 字段的最后一维大小，并赋值给 self.beam_size。
        # 这通常表示 beam search 中的 beam 宽度。
        self.beam_size = batch_log_probs['dims'][-1]

        # 获取encoder_out的配置信息,从配置中提取data_type字段，获取输出类型
        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        # 从 encoder_config 的 dims 字段中提取最后一维大小，赋值给 self.feature_size，
        # 这通常表示编码器输出的特征维度。
        self.feature_size = encoder_config['dims'][-1]

        # 将 self.lm 和 self.hotwords_scorer 初始化为 None，表明当前没有加载语言模型或热词评分器。
        self.lm = None
        self.hotwords_scorer = None
        # 使用配置文件中的parameters初始化 CTC重打分相关的资源
        self.init_ctc_rescore(self.model_config['parameters'])
        print('Initialized Rescoring!')

    # 
    def init_ctc_rescore(self, parameters):
        # 使用python自带方法获取CPU核心数
        num_processes = multiprocessing.cpu_count()
        # 截断概率，默认为0.9999
        cutoff_prob = 0.9999
        # 语言模型权重参数
        alpha = 2.0
        beta = 1.0
        # 双向解码器默认为0
        bidecoder = 0
        # 语言模型路径和词汇表路径，初始为None。
        lm_path, vocab_path = None, None
        # 根据输入有什么更新参数
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "num_processes":
                num_processes = int(value)
            elif key == "blank_id":
                blank_id = int(value)
            elif key == "cutoff_prob":
                cutoff_prob = float(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "hotwords_path":
                hotwords_path = value
            elif key == "alpha":
                alpha = float(value)
            elif key == "beta":
                beta = float(value)
            elif key == "vocabulary":
                vocab_path = value
            elif key == "bidecoder":
                bidecoder = int(value)

        # 设置多进程数量，截断概率，加载词汇表，设置空格ID=-1默认，空白ID=0默认，起始符号ID，结束符号ID默认是最后一个元素的ID
        self.num_processes = num_processes
        self.cutoff_prob = cutoff_prob
        ret = self.load_vocab(vocab_path)
        id2vocab, vocab, space_id, blank_id, sos_eos = ret
        self.space_id = space_id if space_id else -1
        self.blank_id = blank_id if blank_id else 0
        self.eos = self.sos = sos_eos if sos_eos else len(vocab) - 1

        # 如果语言模型路径存在，加载语言模型
        if lm_path and os.path.exists(lm_path):
            self.lm = Scorer(alpha, beta, lm_path, vocab)
            print("Successfully load language model!")
        # 热词存在就加载热词
        if hotwords_path and os.path.exists(hotwords_path):
            self.hotwords = self.load_hotwords(hotwords_path)
            # 初始化 max_order 为 4，然后遍历热词列表，更新 max_order 为热词中最长的单词长度。
            # 这用于后续设置评分窗口的长度。
            max_order = 4
            if self.hotwords is not None:
                for w in self.hotwords:
                    max_order = max(max_order, len(w))
                # 如果成功加载了热词，则初始化 HotWordsScorer 热词评分器对象。热词评分器在swig_decoder中实现。
                self.hotwords_scorer = HotWordsScorer(self.hotwords,
                                                      vocab,
                                                      window_length=max_order,
                                                      SPACE_ID=-2,
                                                      is_character_based=True)
                print(
                    f"Successfully load hotwords! Hotwords orders = {max_order}"
                )
        # 给词表和双向解码器是否启用赋值，用于后续的解码操作。
        self.vocabulary = vocab
        self.bidecoder = bidecoder

    # 加载词汇表，加载后分割数据，构建映射，将特殊字符存储到对应的变量中。
    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        space_id, blank_id, sos_eos = None, None, None
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
                if char == " ":
                    space_id = int(id)
                elif char == "<blank>":
                    blank_id = int(id)
                elif char == "<sos/eos>":
                    sos_eos = int(id)
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return (id2vocab, vocab, space_id, blank_id, sos_eos)

    # 加载热词，加载后返回热词列表。
    def load_hotwords(self, hotwords_file):
        """
        load hotwords.yaml
        """
        with open(hotwords_file, 'r', encoding="utf-8") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        return configs

    # 实现了代码的重打分操作，包括加载语言模型和热词评分器。
    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_encoder_out, batch_encoder_lens = [], []
        batch_log_probs, batch_log_probs_idx = [], []
        batch_count = []
        batch_root = TrieVector()
        batch_start = []
        root_dict = {}

        encoder_max_len = 0
        hyps_max_len = 0
        total = 0
        # 提取数据
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request,
                                                     "encoder_out_lens")
            in_2 = pb_utils.get_input_tensor_by_name(request,
                                                     "batch_log_probs")
            in_3 = pb_utils.get_input_tensor_by_name(request,
                                                     "batch_log_probs_idx")
            in_4 = pb_utils.get_input_tensor_by_name(request, "ctc_log_probs")
            
            print("in_0 shape is :",in_0.as_numpy().shape)
            print("encoder_out data:\n", in_0.as_numpy())
            print("encoder_out_lens data:\n", in_1.as_numpy())
            print("batch_log_probs data:\n", in_2.as_numpy())
            print("batch_log_probs_idx data:\n", in_3.as_numpy())
            #print("ctc_log_probs shape:\n", in_4.as_numpy().shape)
            #print("ctc_log_probs data:\n", in_4.as_numpy())
            
            batch_encoder_out.append(in_0.as_numpy())   # batch,feature_size，批次大小和特征维度
            encoder_max_len = max(encoder_max_len,
                                  batch_encoder_out[-1].shape[1])

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            cur_b_log_probs = in_2.as_numpy()
            cur_b_log_probs_idx = in_3.as_numpy()
            for i in range(cur_batch):
                cur_len = cur_b_lens[i]
                cur_probs = cur_b_log_probs[i][
                    0:cur_len, :].tolist()  # T X Beam
                cur_idx = cur_b_log_probs_idx[i][
                    0:cur_len, :].tolist()  # T x Beam
                batch_log_probs.append(cur_probs)
                batch_log_probs_idx.append(cur_idx)
                root_dict[total] = PathTrie()
                batch_root.append(root_dict[total])
                batch_start.append(True)
                total += 1

        # swig_decoder实现了CTC前缀束搜索解码，这里获取假设
        score_hyps = ctc_beam_search_decoder_batch(
            batch_log_probs,
            batch_log_probs_idx,
            batch_root,
            batch_start,
            self.beam_size,
            min(total, self.num_processes),
            blank_id=self.blank_id,
            space_id=-2,
            cutoff_prob=self.cutoff_prob,
            ext_scorer=self.lm,
            hotwords_scorer=self.hotwords_scorer)
        all_hyps = []
        all_ctc_score = []
        max_seq_len = 0
        # 遍历候选词序列：对于每个候选词序列列表seq_cand，检查其长度是否等于束大小self.beam_size。
        # 补充不足的候选词序列：如果候选词序列的数量少于束大小，则用填充项（分数为负无穷，
        # 假设词序列为(0,)）补齐到束大小。这样可以确保后续处理时每个批次都有相同数量的候选词序列。
        # 收集候选词序列和分数：将每个候选词序列及其对应的分数添加到全局列表all_hyps和all_ctc_score中，
        # 并更新最大序列长度max_seq_len。
        for seq_cand in score_hyps:
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"),
                                                                 (0, ))]

            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        # 初始化和准备用于后续处理的张量，特别是为每个候选词序列构建填充了起始符（SOS）和结束符（EOS）的张量。
        beam_size = self.beam_size
        feature_size = self.feature_size
        hyps_max_len = max_seq_len + 2
        # in_ctc_score：初始化一个形状为(total, beam_size)的零矩阵，用于存储每个候选词序列的CTC分数。
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        # in_hyps_pad_sos_eos：初始化一个形状为(total, beam_size, hyps_max_len)的全1矩阵，并用结束符（EOS）填充。
        # 这个矩阵将用于存储每个候选词序列，包括起始符（SOS）和结束符（EOS）。
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos
        print(in_hyps_pad_sos_eos.shape)
        # 如果使用双向解码器（self.bidecoder），还会初始化一个类似的张量in_r_hyps_pad_sos_eos，
        # 用于存储反转后的候选词序列。
        if self.bidecoder:
            in_r_hyps_pad_sos_eos = np.ones(
                (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        # 初始化假设的输出张量
        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        # 初始化编码器输出张量in_encoder_out和编码器输出长度张量in_encoder_out_lens，
        in_encoder_out = np.zeros((total, encoder_max_len, feature_size),
                                  dtype=self.data_type)
        in_encoder_out_lens = np.zeros(total, dtype=np.int32)
        st = 0
        # 这段代码的主要功能是将处理后的候选词序列及其相关信息填充到之前初始化的张量中，以便后续传递给解码器模型进行进一步处理。
        for b in batch_count:
            t = batch_encoder_out.pop(0)
            in_encoder_out[st:st + b, 0:t.shape[1]] = t
            in_encoder_out_lens[st:st + b] = batch_encoder_lens.pop(0)
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_len = len(cur_hyp) + 2
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    if self.bidecoder:
                        r_in_hyp = [self.sos] + cur_hyp[::-1] + [self.eos]
                        in_r_hyps_pad_sos_eos[st + i][j][0:cur_len] = r_in_hyp
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b
        print("hyps_pad_sos_eos shape is :",in_hyps_pad_sos_eos.shape)
        print("hyps_lens_sos shape is :",in_hyps_lens_sos.shape)
        # 将CTC的分数和候选词序列还有编码器输出填充到张量中后，将这些张量作为输入张量传递给解码器模型。
        in_encoder_out_lens = np.expand_dims(in_encoder_out_lens, axis=1)
        in_tensor_0 = pb_utils.Tensor("encoder_out", in_encoder_out)
        in_tensor_1 = pb_utils.Tensor("encoder_out_lens", in_encoder_out_lens)
        in_tensor_2 = pb_utils.Tensor("hyps_pad_sos_eos", in_hyps_pad_sos_eos)
        in_tensor_3 = pb_utils.Tensor("hyps_lens_sos", in_hyps_lens_sos)
        
        # 打印张量的维度大小和存储的信息
        print("in_tensor_0 shape is:", in_tensor_0.as_numpy().shape)
        print("in_tensor_0 data:\n", in_tensor_0.as_numpy())
        print("in_tensor_1 shape is:", in_tensor_1.as_numpy().shape)
        print("in_tensor_1 data:\n", in_tensor_1.as_numpy())
        print("in_tensor_2 shape is:", in_tensor_2.as_numpy().shape)
        print("in_tensor_2 data:\n", in_tensor_2.as_numpy())
        print("in_tensor_3 shape is:", in_tensor_3.as_numpy().shape)
        print("in_tensor_3 data:\n", in_tensor_3.as_numpy())
        
        input_tensors = [in_tensor_0, in_tensor_1, in_tensor_2, in_tensor_3]
        if self.bidecoder:
            in_tensor_4 = pb_utils.Tensor("r_hyps_pad_sos_eos",
                                          in_r_hyps_pad_sos_eos)
            print("r_hyps_pad_sos_eos shape is :",in_r_hyps_pad_sos_eos.shape)
            print("r_hyps_pad_sos_eos data:\n", in_r_hyps_pad_sos_eos)
            input_tensors.append(in_tensor_4)
        in_tensor_5 = pb_utils.Tensor("ctc_score", in_ctc_score)
        print("ctc_score shape is :", in_ctc_score.shape)
        print("ctc_score data:\n", in_ctc_score)
        input_tensors.append(in_tensor_5)

        # 这里给了解码器进行解码,并将解码结果填充到输出张量中。
        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['best_index'],
            inputs=input_tensors)

        # 这里执行解码器模型的推理请求，并将推理结果填充到输出张量中。来自triton_python_backend_utils的exec()方法。
        inference_response = inference_request.exec()
        
        # 判断推理是否有错误,有就抛出异常停止运行
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        # 没有就提取输出张量
        else:
            # Extract the output tensors from the inference response.
            best_index = pb_utils.get_output_tensor_by_name(
                inference_response, 'best_index')
            if best_index.is_cpu():
                best_index = best_index.as_numpy()
                print("best_index  is :",best_index)
            else:
                best_index = from_dlpack(best_index.to_dlpack())
                best_index = best_index.cpu().numpy()
                print("best_index  is :",best_index)

            hyps = []
            idx = 0
            # 获取最佳候选词序列
            for cands, cand_lens in zip(in_hyps_pad_sos_eos, in_hyps_lens_sos):
                best_idx = best_index[idx][0]
                best_cand_len = cand_lens[best_idx] - 1  # remove sos
                best_cand = cands[best_idx][1:1 + best_cand_len].tolist()
                hyps.append(best_cand)
                idx += 1
            
            # 使用map_batch函数将候选词序列映射到词汇表，生成最终的文本输出。
            hyps = map_batch(
                hyps, self.vocabulary,
                min(multiprocessing.cpu_count(), len(in_ctc_score)))
            st = 0
            # 构建响应对象,用于返回给客户端
            for b in batch_count:
                sents = np.array(hyps[st:st + b])
                out0 = pb_utils.Tensor("OUTPUT0",
                                       sents.astype(self.out0_dtype))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[out0])
                responses.append(inference_response)
                st += b
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
