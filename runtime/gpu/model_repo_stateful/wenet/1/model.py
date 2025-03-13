# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import json
import torch
from swig_decoders import PathTrie, TrieVector

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from wenet_onnx_model import WenetModel

from torch.utils.dlpack import from_dlpack


# Python模型使用CPU推理的原因：
# 1. 模型部署到GPU上，GPU的显存有限，需要减少模型参数的数量，因此需要使用模型压缩工具进行模型压缩。
# 2. GPU数量有限，这样可以减少GPU的成本
# 3. python模型的输入输出相较于onnx来说过多，而GPU适合处理大量重复计算，所以使用CPU进行推理
# 4. python模型比onnx模型更轻量化，适合使用CPU进行推理
# 5. 延迟控制：CPU推理的延迟比GPU推理的延迟要低，适合对延迟要求较高的场景
# triton会根据config.pbtxt中模型部署的位置自动把数据传输到CPU或者GPU上
class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
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

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # get device# get device根据args字典中的model_instance_kind值来确定模型运行的设备类型。如果model_instance_kind为"GPU"，则将设备设置为'cuda'，否则设置为'cpu'。这一逻辑确保了模型能够在适当的硬件上运行。
        if args["model_instance_kind"] == "GPU":
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # get parameter configurations使用解析后的model_config和确定的设备类型来初始化WenetModel对象，并将其存储在self.model属性中。
        self.model = WenetModel(self.model_config, self.device)

        # Get OUTPUT0 configuration获取模型配置中的输出配置（如OUTPUT0），并将其数据类型从Triton类型转换为NumPy类型。
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Convert Triton types to numpy types将triton格式转为numpy格式
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # use to record every sequence state
        self.seq_states = {}
        print("Finish Init")

    # 处理传入的请求列表requests，并生成相应的响应列表responses，
    # 一次初始化，一次finalize释放，多次调用这个实现函数，每次接收到客户端请求时才会进行调用
    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        batch_log_probs, batch_log_probs_idx, batch_len, batch_states = [], [], [], []
        cur_encoder_out = []

        batch_encoder_hist = []
        batch_start = []

        trieVector = TrieVector()

        rescore_index = {}
        batch_idx2_corrid = {}

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        # 这里是获取config.pbtxt中输入数据的部分,分别来自ctc和encoder模块
        batch_idx = 0
        # 按照请求进行循环，每次都是只有一个request，由客户端输入决定
        for request in requests:
            # Get INPUT0,为什么后面都要跟着()[0]这是去掉了第一个维度batch,后面两个是time_steps和beam_size
            # 这些数据(长度,ctc结果)都要修改成numpy格式,为了更方便使用其中存储的数据,不需要进行推理
            # 所以要转化为numpy格式
            in_0 = pb_utils.get_input_tensor_by_name(request, "log_probs")
            batch_log_probs.append(in_0.as_numpy()[0])
            # print("in_0.numpy() is :",in_0.as_numpy())
            # print("in_0.numpy[0] is :",in_0.as_numpy()[0])

            # ()[0]也是去除了批次大小(去除第一个维度)，
            # 使用triton类型的数据方便在triton后端传递和CPU模型GPU模型间传递。
            in_1 = pb_utils.get_input_tensor_by_name(request, "log_probs_idx")
            batch_log_probs_idx.append(in_1.as_numpy()[0])
            if self.model.rescoring:
                in_2 = pb_utils.get_input_tensor_by_name(request, "chunk_out")
                # important to use clone or this tensor
                # the tensor will be released after one inference
                # 先将triton数据转成dlpack数据,再将dlpack张量转成pytorch张量并进行克隆,
                # 因为原始张量再一次推理后被释放,克隆可以确保数据在后续使用时仍然有效
                # 这里不转化为numpy数组,因为后续还需要使用pytorch张量进行操作,直接使用pytorch进行推理
                # PyTorch 张量支持动态计算图和自动求导等功能，更适合复杂的模型操作。后面重打分推理需要使用这个序列
                # 所以不能动,必须保持数据是pytorch格式,方便后续操作
                # 这里的数据最后发送给了decoder.onnx，而decoder.onnx使用GPU推理，所以需要转化为pytorch张量
                # 这是GPU上的数据，直接在GPU上克隆一份，
                # 你要是使用numpy提取就是有三个副本，CPU一个GPU两个，dlpack就是两个副本
                in_2 = from_dlpack(in_2.to_dlpack()).clone()
                # cur_encoder_out:用于存储当前批次中所有请求的 chunk_out 数据。
                # 假设 in_2 是一个包含单个元素的张量列表，去掉第一个维度并添加到 cur_encoder_out 列表中。
                # 因为第一维度是batch,所有推理都是在batch内部来的,不需要使用这个维度,减少计算量和复杂度
                cur_encoder_out.append(in_2[0])
            in_3 = pb_utils.get_input_tensor_by_name(request, "chunk_out_lens")
            # 转成numpy数组进行存储,这里只有一个数据维度表示长度,不需要去掉批次.
            # 保持numpy数组存储为了方便提取数据,方便在CPU上进行运算，所以使用numpy数组
            # 如果in_3是在GPU上,就直接复制到CPU上并转换成numpy数组
            # 检查 in_3 是否在 GPU 上,直接triton后端给送到CPU上进行操作
            dlpack_tensor = in_3.to_dlpack()
            torch_tensor = from_dlpack(dlpack_tensor)
            if torch_tensor.is_cuda:
                print("in_3 is on GPU")
            else:
                print("in_3 is on CPU")
            batch_len.append(in_3.as_numpy())

            # 这是config.pbtxt的sequence_batching中的参数,作用是初始化配置适应不同的模型,这两个[0的作用是什么]
            # 去掉前两个维度吧，START维度是[batch_size]，READY，CORRID，END维度一样
            # [0,1]分别表示false和true，in_1.as_numpy()[0][0]是取得第一批次的第一个时间步，
            # in_1.as_numpy()[0,0]是去掉前两个维度
            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            print("start is :",start)

            # 表示当前序列是否是新的序列(批次)
            if start:
                batch_start.append(True)
            else:
                batch_start.append(False)

            # 获取ready数输入,将张量转换为 NumPy 数组并提取第一个元素的值，得到 ready 布尔值,表示当前序列是否准备好,状态如何更新?
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            print("ready is :",ready)

            # 获取corrid输入,将张量转换为 NumPy 数组并提取第一个元素的值，得到 corrid 整数值,表示当前序列的 ID,用于管理序列状态
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            print("corrid is :",corrid)

            # 获取end输入,将张量转换为 NumPy 数组并提取第一个元素的值，得到 end 布尔值,表示当前序列是否结束
            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]
            print("end is :",end)
            
            # 每个batch中每个帧的表示格式
            # 序号   START   READY   CORRID   END
            #  0    1         1      1001      0
            #  1    0         1      1001      0
            #  2    0         1      1001      0
            #  3    0         1      1001      1
            #下一块：
            #  0    1         1      1002      0
            

            # 开始的时候初始化状态,初始化缓存和前缀树等数据
            # 如果初始化失败，就丢弃这个batch的状态
            if start and ready:
                # intialize states,返回0或者None
                encoder_out = self.model.generate_init_cache()
                root = PathTrie()
                # register this sequence
                self.seq_states[corrid] = [root, encoder_out]

            # 结束的时候准备rescore
            if end and ready:
                rescore_index[batch_idx] = 1

            # 如果准备好了,则将当前状态加入到trieVector中,这是一个请求批次的内部
            if ready:
                root, encoder_out = self.seq_states[corrid]
                trieVector.append(root)
                batch_idx2_corrid[batch_idx] = corrid
                batch_encoder_hist.append(encoder_out)

            # 批次数量加1
            batch_idx += 1

        # 设置批次的状态,上面把切块的数据组合成一个批次进行重打分
        batch_states = [
            trieVector, batch_start, batch_encoder_hist, cur_encoder_out
        ]
        # 进行推理
        res_sents, new_states = self.model.infer(batch_log_probs,
                                                 batch_log_probs_idx,
                                                 batch_len, rescore_index,
                                                 batch_states)
        # 获取当前编码器的输出
        cur_encoder_out = new_states
        for i in range(len(res_sents)):
            sent = np.array(res_sents[i])
            # 获取输出张量的数据类型,创建输出张量
            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           sent.astype(self.output0_dtype))
            # 创建一个响应,包含输出张量
            response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            # 将响应对象放到响应列表中
            responses.append(response)
            # 获取当前序列的 ID
            corr = batch_idx2_corrid[i]
            # 检查当前批次索引 i 是否在 rescore_index 中。如果在，则表示该序列已经结束。这是上面的END输入的作用。
            if i in rescore_index:
                # this response ends, remove it通过删除 self.seq_states[corr]，清理已完成的序列状态，释放内存。
                del self.seq_states[corr]
            # 没有结束
            else:
                # 判断是否使用了重打分功能，如果使用了重打分功能，则更新当前序列的状态。
                if self.model.rescoring:
                    # 如果当前序列的历史编码器输出为空，则直接将其设置为当前批次的编码器输出 cur_encoder_out[i]。
                    if self.seq_states[corr][1] is None:
                        self.seq_states[corr][1] = cur_encoder_out[i]
                    # 如果已有历史编码器输出，则将当前批次的编码器输出 cur_encoder_out[i] 拼接到历史输出后面，沿轴 0（时间维度）进行拼接。
                    else:
                        new_hist = torch.cat(
                            [self.seq_states[corr][1], cur_encoder_out[i]],
                            axis=0)
                        # 更新序列的历史编码器输出为新的拼接结果。
                        self.seq_states[corr][1] = new_hist

        # 确保输入的请求列表 requests 和生成的响应列表 responses 长度一致。防止因为逻辑错误导致请求响应不匹配
        assert len(requests) == len(responses)
        # 返回结果列表给triton服务器
        return responses

    # 结束收尾
    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        del self.model


# 为什么除了重打分的代码在CPU上运行而其他模型在GPU上运行？
# 1.model.py在CPU上运行的原因：model.py模型输入输出类型过多，不适合使用GPU进行推理，GPU适合大量重复计算而model.py全是
# 复杂少量的计算，而CPU最适合这种复杂少量轻量化的计算，所以放在CPU上运行。
# 这也导致了in_2数据使用pytorch类型而其他数据使用numpy数据，因为in_2数据要用到decoder.onnx推理中，
# decoder.onnx模型使用GPU加速推理，pytorch也能被GPU加速，numpy数据主要使用CPU推理，numpy底层由C语言实现，最适合在CPU上运行。
# 而且将部分模型放在CPU上运行，可以减少GPU的显存占用，提高GPU的效率，节省成本。
# 2.decoder.onnx和encoder.onnx模型原本都是pytorch模型导出的模型，可以被GPU加速，所以使用GPU进行推理。
# 3.特征提取模型使用GPU进行推理因为Kaldi的模型使用GPU进行推理加速，所以使用GPU进行推理。这样既减少了GPU的显存占用，
# 也可以提高GPU的效率，节省成本。

# START,READY,CORRID,END这些参数的作用可以少传或者不传吗？
# 在这个模型中不行，这些参数是必须的，不能少传或者不传。最多少传END参数，等待5秒后自动结束，但是这样会影响效率，所以最好都要传递进来
# 但是在其他模型中，可以根据需要来设置参数，也可以不设置这些参数，比如非流式的GPU模型，当然非流式的也根本没办法使用这些参数因为不符合语法规则，但是流式模型确实可以选择其中的部分参数进行使用。
# 在同一个sequence_id下的序列中，triton的所有模型都可以获取数据，是triton统一管理的，不是链式传递的，所以所有模型都可以访问。
# https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#stateful-models
# 