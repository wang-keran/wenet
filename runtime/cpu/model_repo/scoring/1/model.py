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
import torch

from typing import List, Tuple
import math

# 这个模型文件的输入输出是不是也得改？模型本身是不是也得改？因为decoder.onnx和encoder.onnx输入输出已经不一样了，目前代码是GPU直接粘贴过来的，和pbtxt匹配


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
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        if 'data_type' not in output0_config:
            raise ValueError("Output 'OUTPUT0' does not specify 'data_type',OUTPUT0没有加载进来")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        # print(f"初始化的数据类型为：Initialized OUTPUT0 data_type: {self.out0_dtype}")

        # Get INPUT configuration
        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "output")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])
        self.feature_size = encoder_config['dims'][-1]

        probs_config = pb_utils.get_input_config_by_name(   # 这里相当于gpu版本的ctc_log_probs，可以转换为beam_log_probs和beam_log_probs_idx
            model_config, "probs")
        self.beam_size = probs_config['dims'][-1]

        self.lm = None
        self.hotwords_scorer = None
        self.init_ctc_rescore(self.model_config['parameters'])
        print('Initialized Rescoring!')

    def init_ctc_rescore(self, parameters):
        num_processes = multiprocessing.cpu_count()
        cutoff_prob = 0.9999
        alpha = 2.0
        beta = 1.0
        bidecoder = 0
        lm_path, vocab_path = None, None
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

        self.num_processes = num_processes
        self.cutoff_prob = cutoff_prob
        ret = self.load_vocab(vocab_path)
        id2vocab, vocab, space_id, blank_id, sos_eos = ret
        self.space_id = space_id if space_id else -1
        self.blank_id = blank_id if blank_id else 0
        self.eos = self.sos = sos_eos if sos_eos else len(vocab) - 1

        if lm_path and os.path.exists(lm_path):
            self.lm = Scorer(alpha, beta, lm_path, vocab)
            print("Successfully load language model!")
        if hotwords_path and os.path.exists(hotwords_path):
            self.hotwords = self.load_hotwords(hotwords_path)
            max_order = 4
            if self.hotwords is not None:
                for w in self.hotwords:
                    max_order = max(max_order, len(w))
                self.hotwords_scorer = HotWordsScorer(self.hotwords,
                                                      vocab,
                                                      window_length=max_order,
                                                      SPACE_ID=-2,
                                                      is_character_based=True)
                print(
                    f"Successfully load hotwords! Hotwords orders = {max_order}"
                )
        self.vocabulary = vocab
        self.bidecoder = bidecoder

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

    def load_hotwords(self, hotwords_file):
        """
        load hotwords.yaml
        """
        with open(hotwords_file, 'r', encoding="utf-8") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        return configs


    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> self.pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])
        """
        max_len = max([len(item) for item in xs])
        batchs = len(xs)
        ndim = xs[0].ndim
        if ndim == 1:
            pad_res = torch.zeros(batchs,
                                max_len,
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 2:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 3:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                xs[0].shape[2],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")
        pad_res.fill_(pad_value)
        for i in range(batchs):
            pad_res[i, :len(xs[i])] = xs[i]
        return pad_res

    def add_sos_eos(self, ys_pad: torch.Tensor, sos: int, eos: int,
                    ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add <sos> and <eos> labels.

        Args:
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
            sos (int): index of <sos>
            eos (int): index of <eeos>
            ignore_id (int): index of padding

        Returns:
            ys_in (torch.Tensor) : (B, Lmax + 1)
            ys_out (torch.Tensor) : (B, Lmax + 1)

        Examples:
            >>> sos_id = 10
            >>> eos_id = 11
            >>> ignore_id = -1
            >>> ys_pad
            tensor([[ 1,  2,  3,  4,  5],
                    [ 4,  5,  6, -1, -1],
                    [ 7,  8,  9, -1, -1]], dtype=torch.int32)
            >>> ys_in,ys_out=self.add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
            >>> ys_in
            tensor([[10,  1,  2,  3,  4,  5],
                    [10,  4,  5,  6, 11, 11],
                    [10,  7,  8,  9, 11, 11]])
            >>> ys_out
            tensor([[ 1,  2,  3,  4,  5, 11],
                    [ 4,  5,  6, 11, -1, -1],
                    [ 7,  8,  9, 11, -1, -1]])
        """
        _sos = torch.tensor([sos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        return self.pad_list(ys_in, eos), self.pad_list(ys_out, ignore_id)

    
    # decoder.onnx函数只负责计算正向和反向的得分，具体的计算要这两个乘对应的概率加上ctc权重乘以概率才是最终结果
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

        print("********************************start scoring********************************")
        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_encoder_out, batch_encoder_lens = [], []
        batch_log_probs = []
        batch_log_probs_idx = []
        batch_count = []
        batch_root = TrieVector()
        batch_start = []
        root_dict = {}

        encoder_max_len = 0
        hyps_max_len = 0
        total = 0
        
        reverse_weight = 0.3
        ctc_weight = 0.3
        prefix_len = 1
        
        print("********************************start get requests********************************")
        # 批次是1的情况下只循环一次，只有一条音频的非流式语音识别就是这种情况
        for request in requests:
            # Perform inference on the request and append it to responses list...
            print("********************************start get input from encoder and ctc********************************")
            in_0 = pb_utils.get_input_tensor_by_name(request, "output") # 多了时间维度T，中间那个变量
            in_1 = pb_utils.get_input_tensor_by_name(request,
                                                     "probs")
            print("********************************finish get input from encoder and ctc********************************")
            
            print(f"********************************type(in_1): {type(in_1)}*****************************")  # 查看 in_1 的类型
            in_1 = torch.tensor(in_1.as_numpy())  # Triton Tensor → NumPy → PyTorch
            print(f"***************88**after change type(in_1): {type(in_1)}*****************************")
            print(f"********************************type(batch_log_probs): {type(batch_log_probs_origin)}*****************************")  # 查看 batch_log_probs 的类型
            batch_log_probs_origin,batch_log_probs_idx_origin = torch.topk(in_1,self.beam_size,dim=2) #修改一下中间的10,不一定是正确的beam_size
            print(f"********************************type(batch_log_probs): {type(batch_log_probs_origin)}*****************************")  # 查看 batch_log_probs 的类型
            print("batch_log_probs_origin get beam_size is:",batch_log_probs_origin.shape)

            encoder_out_lens=0
            print(f"before get shape type(in_0): {type(in_0)}")  # 查看 in_0 的类型
            print(f"before get shape dir(in_0): {dir(in_0)}")  # 查看 in_0 具有什么属性
            output_batch_size,output_time_step,output_feature_size=in_0.shape()
            print("***************************************after get shape**********************************")
            # 如果批次大小或时间步数为 0，直接返回全零张量
            if output_batch_size==0 or output_time_step==0:
                encoder_out_lens = 0
                raise pb_utils.TritonModelException("The input tensor is empty.")
            # 对特征维度 F 求和，得到 (B, T)
            in_0_copy =in_0
            in_0_copy=torch.tensor(in_0_copy.as_numpy())
            encoder_out_sum = in_0_copy.sum(dim=-1)       # dim=-1 表示最后一个维度
            print("***************************************after get sum**********************************")
            # 检查每个时间步的和是否大于 0，得到 (B, T) 的布尔张量
            encoder_out_mask = encoder_out_sum > 0
            # 对时间维度 T 求和，得到 (B,)
            encoder_out_lens = encoder_out_mask.sum(dim=-1)

            # 新的decoder返回score和r_score，需要回来计算最终的得分,reverse_weight=0.3，这个与模型相关，从train.yaml中找到
            # 还差ctc_scores[i]和ctc_weight这两个变量,ctc_weight=0.3，这个从预训练模型的参数中找到
        
            
            batch_encoder_out.append(in_0.as_numpy())
            # 这里相对于之前有修改，因为中间多加了时间步为第二维度，但是我们需要第三维度特征维度最大值，
            # 所以把shape[1]改成shape[2],batch_encoder_out[-1].shape[2]，代表取出最新存入数据的第三维度
            encoder_max_len = max(encoder_max_len,
                                  batch_encoder_out[-1].shape[2])

            # 这里有问题，因为这个不是encoder_out_lens，这里是ctc_log_probs,差root和start，差在encoder_out_lens没有
            print("***************************Type of encoder_out_lens:", type(encoder_out_lens),"**********************************************")
            cur_b_lens = encoder_out_lens.cpu().numpy()
            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            print("***************************Type of batch_log_probs:", type(batch_log_probs),"**********************************************")
            cur_b_log_probs = batch_log_probs_origin.cpu().numpy()
            cur_b_log_probs_idx = batch_log_probs_idx_origin.cpu().numpy()
            print(cur_b_log_probs)
            print(cur_b_log_probs_idx)
            print("*********************8cur_batch:",cur_batch,"cur_batch range:",range(cur_batch),"********************************")
            for i in range(cur_batch):
                cur_len = cur_b_lens[i]
                cur_probs = cur_b_log_probs[i][
                    0:cur_len, :].tolist()  # T X Beam
                print("cur_probs is:",cur_probs)
                print("***************************Type of cur_probs:", type(cur_probs),"**********************************************")
                cur_idx = cur_b_log_probs_idx[i][
                    0:cur_len, :].tolist()  # T x Beam
                print("we are in i in range(cur_batch):",i)
                batch_log_probs.append(cur_probs)
                batch_log_probs_idx.append(cur_idx)
                root_dict[total] = PathTrie()
                batch_root.append(root_dict[total])
                batch_start.append(True)
                total += 1

        print(batch_log_probs_idx)
        print("********************************start ctc_beam_search_decoder_batch********************************")
        # 返回路径的总概率（总得分）和路径的解码结果，按总概率从高到底排序，无需再排序了，只返回了前beam_size个路径
        # 返回的结果数量和批次大小batch_size相同
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
        # 处理前缀束搜索的结果
        print("********************************start calculate ctc_beam_search_decoder_batch********************************")
        for seq_cand in score_hyps:
            # if candidates less than beam size
            # 现在还差hyps,hyps_lens的运算，准备给decoder输入
            # 如果路径不够的话就填充负无穷
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"),
                                                                 (0, ))]
            # 提取候选路径和得分
            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        beam_size = self.beam_size
        feature_size = self.feature_size
        # 长度加上作用两侧的sos和eos
        hyps_max_len = max_seq_len + 2
        # 初始化用于存储CTC得分的数组
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        # 给序列添加eos结束标识
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos
        print("*************",in_hyps_pad_sos_eos.shape,"**********************")
        if self.bidecoder:
            in_r_hyps_pad_sos_eos = np.ones(
                (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        in_encoder_out = np.zeros((total, encoder_max_len, feature_size),
                                  dtype=self.data_type)
        in_encoder_out_lens = np.zeros(total, dtype=np.int32)
        st = 0
        for b in batch_count:
            t = batch_encoder_out.pop(0)
            in_encoder_out[st:st + b, 0:t.shape[1]] = t
            in_encoder_out_lens[st:st + b] = batch_encoder_lens.pop(0)
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_len = len(cur_hyp) + 2
                    # 为了每个假设序列添加sos和eos，因此需要将长度加1
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    # 这里的长度是不包括eos的，因此需要减1
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    if self.bidecoder:
                        r_in_hyp = [self.sos] + cur_hyp[::-1] + [self.eos]
                        in_r_hyps_pad_sos_eos[st + i][j][0:cur_len] = r_in_hyp
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b
        # 减去第一个total，没有用的维度
        in_hyps_pad_sos_eos_origin = in_hyps_pad_sos_eos 
        in_hyps_lens_sos_origin = in_hyps_lens_sos
        _ = in_hyps_pad_sos_eos[0]
        _ = in_hyps_lens_sos[0]
        in_hyps_lens_sos=in_hyps_lens_sos.reshape(-1)
        print(in_hyps_pad_sos_eos.shape)
        print(f"*****************************************the type of in_hyps_pad_sos_eos:",type(in_hyps_pad_sos_eos),"*******************************************************")
        print("*******************************************************start decoder*******************************************************")
        print("in_hyps_pad_sos_eos",in_hyps_pad_sos_eos)    #这里的hyps数字过大且相同，都是5234,5234是<start>和<end>的值，输入可能不需要这两个数据，参考export_onnx_gpu中的写法就行，输出已经修改好了
        print("in_hyps_lens_sos",in_hyps_lens_sos)          #这里除了第一个是1其他的都是2
        print("encoder_out",in_0)
        # hyps [0,0],hyps_lens [0],encoder_out [1,0,256]，现在的in_hyps_pad_sos_eos就是缺少total维度的了
        in_tensor_0 = pb_utils.Tensor("hyps", in_hyps_pad_sos_eos)
        in_tensor_1 = pb_utils.Tensor("hyps_lens", in_hyps_lens_sos.astype(np.int64))
        in_tensor_2 = pb_utils.Tensor("encoder_out", in_0.as_numpy())
        
        input_tensors = [in_tensor_0, in_tensor_1, in_tensor_2]

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['score','r_score'],
            inputs=input_tensors)

        # 新的decoder返回score和r_score，需要回来计算最终的得分,reverse_weight=0.3，这个与模型相关，从train.yaml中找到
        # 还差ctc_scores[i]和ctc_weight这两个变量,ctc_weight=0.3，这个从预训练模型的参数中找到
        
        inference_response = inference_request.exec()
        print("******** start tuili *******")
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.获取正向反向打分结果
            # score(num_hyps, max_hyps_len, vocab_size),r_score (num_hyps表示有 10 个假设序列, max_hyps_len表示每个假设序列的最大长度为 20, vocab_size示每个时间步上有 5000 个词汇的概率分布)。
            score = pb_utils.get_output_tensor_by_name(
                inference_response, 'score')
            r_score = pb_utils.get_output_tensor_by_name(
                inference_response, 'r_score')
            if score.is_cpu():
                score = score.as_numpy()
                r_score = r_score.as_numpy()
            else:
                score = from_dlpack(score.to_dlpack())
                score = score.cpu().numpy()
                r_score = from_dlpack(r_score.to_dlpack())
                r_score = r_score.cpu().numpy()

        # 开始使用循环计算最终得分score和选择路线，这里应该有一个根据ctc_scores[i].nbest的循环
            best_score = -float('inf')
            best_index = 0
            confidences = []
            tokens_confidences = []
            for i,hyp in enumerate(all_hyps):
                # 用于累加当前假设的总得分。
                result_score=0.0
                # tc 是各个 token 的置信度列表，通过解码概率的指数表示。
                tc = []  # tokens confidences
                # 这个循环遍历当前假设中的每个标记 w，j 是标记的索引。
                for j, w in enumerate(hyp):
                    # 从解码器的输出 decoder_out 中提取当前标记的得分 s。
                    s = score[i][j + (prefix_len - 1)][w]
                    # 将当前标记的得分 s 累加到总得分 score 中。
                    result_score += s
                    # 通过取 s 的指数值，计算出当前标记的置信度并添加到列表 tc 中。
                    # 这个指数运算使得得分转化为一种概率形式，表示模型对这个标记的信心程度。
                    tc.append(math.exp(s))
                if r_score.dim() > 0:
                    reverse_score = 0.0
                    # 这个循环遍历当前假设中的每个标记，以计算反向得分。
                    for j, w in enumerate(hyp):
                        # 从反向解码输出 r_decoder_out 中提取当前标记的反向得分 s。
                        s = r_score[i][len(hyp) - j - 1 +
                                            (prefix_len - 1)][w]
                        # 累加
                        reverse_score += s
                        # 更新当前标记的置信度 tc[j]，将正向和反向的置信度平均化，以综合考虑两种得分。
                        tc[j] = (tc[j] + math.exp(s)) / 2
                    # 将反向解码中结束标记的得分也加入到 r_score 中。
                    reverse_score += r_score[i][len(hyp) + (prefix_len - 1)][self.eos]
                    result_score = result_score * (1 - reverse_weight) + reverse_score * reverse_weight
                confidences.append(math.exp(score / (len(hyp) + 1)))
                if score > best_score:
                    best_score = score.item()
                    best_index = i
                # 将当前假设中各个标记的置信度列表 tc 添加到 tokens_confidences 中，以便后续分析和使用。
                tokens_confidences.append(tc)


            for cands, cand_lens in zip(in_hyps_pad_sos_eos_origin, in_hyps_lens_sos_origin):
                best_idx = best_index
                best_cand_len = cand_lens[best_idx] - 1  # remove sos
                best_cand = cands[best_idx][1:1 + best_cand_len].tolist()
                hyps.append(best_cand)
                idx += 1

            hyps = map_batch(
                hyps, self.vocabulary,
                min(multiprocessing.cpu_count(), len(in_ctc_score)))
            st = 0
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
