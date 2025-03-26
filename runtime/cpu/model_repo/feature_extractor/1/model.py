import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
import torch
import numpy as np
import kaldifeat
import _kaldifeat
from typing import List
import json

# 这个模型文件的输入输出是不是也得改？模型本身是不是也得改？因为decoder.onnx和encoder.onnx输入输出已经不一样了，目前代码是GPU直接粘贴过来的，和pbtxt匹配
# 初始化滤波器
class Fbank(torch.nn.Module):

    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)


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
        # 获取模型配置和最大批量大小
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # 判断使用什么设备，GPU，CPU都能跑
        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # 获取输出的张量的类型（speech）
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "chunk")
        # Convert Triton types to numpy types，将Triton类型转换为numpy类型
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        
        # 获取输出的张量的类型（att_cache）
        output2_config = pb_utils.get_output_config_by_name(
            model_config, "att_cache")
        self.output2_dtype = pb_utils.triton_string_to_numpy(
            output2_config['data_type'])

        # 获取输出的张量的类型（cnn_cache）
        output3_config = pb_utils.get_output_config_by_name(
            model_config, "cnn_cache")
        self.output3_dtype = pb_utils.triton_string_to_numpy(
            output3_config['data_type'])


        # 获取帧移长度等配置信息
        params = self.model_config['parameters']
        # 初始化特征提取器
        opts = kaldifeat.FbankOptions()
        # 禁用随机噪声，因为输入已经归一化了
        opts.frame_opts.dither = 0

        for li in params.items():
            # 解包键值对
            key, value = li
            value = value["string_value"]
            if key == "num_mel_bins":
                opts.mel_opts.num_bins = int(value)
            elif key == "frame_shift_in_ms":
                opts.frame_opts.frame_shift_ms = float(value)
            elif key == "frame_length_in_ms":
                opts.frame_opts.frame_length_ms = float(value)
            elif key == "sample_rate":
                opts.frame_opts.samp_freq = int(value)
        # 获取设备
        opts.device = torch.device(self.device)
        self.opts = opts
        # 获取特征提取器
        self.feature_extractor = Fbank(self.opts)
        # 获取特征大小
        self.feature_size = opts.mel_opts.num_bins  #feature_size

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
        print("********************************start feature extract********************************")
        batch_count = []
        total_waves = []
        batch_len = []
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")

            cur_b_wav = input0.as_numpy()   # 如果是动态批次
            cur_b_wav = cur_b_wav * (1 << 15)  # b x -1
            cur_b_wav_lens = input1.as_numpy()  # b x 1
            cur_batch = cur_b_wav.shape[0]
            cur_len = cur_b_wav.shape[1]
            batch_count.append(cur_batch)
            batch_len.append(cur_len)
            for wav, wav_len in zip(cur_b_wav, cur_b_wav_lens):
                print("************************开始获取特征长度**********************")
                wav_len = wav_len[0]
                print("*********************************特征长度获取完成********************************")
                wav = torch.tensor(wav[0:wav_len],
                                   dtype=torch.float32,
                                   device=self.device)
                total_waves.append(wav)

        features = self.feature_extractor(total_waves)
        idx = 0
        for b, l in zip(batch_count, batch_len):
            expect_feat_len = _kaldifeat.num_frames(l, self.opts.frame_opts)
            speech = torch.zeros((b, expect_feat_len, self.feature_size),
                                 dtype=self.output0_dtype,
                                 device=torch.device(self.device))
            speech_lengths = torch.zeros((b, 1),
                                         dtype=torch.int32,
                                         device=torch.device(self.device))
            
            for i in range(b):
                f = features[idx]
                f_l = f.shape[0]
                #speech[i, 0:f_l, :] = f.to(self.output0_dtype)
                speech[i, 0:f_l, :] = f.to(torch.float32)
                speech_lengths[i][0] = f_l
                idx += 1
            # put speech feature on device will cause empty output
            # we will follow this issue and now temporarily put it on cpu
            speech = speech.cpu()
            speech_lengths = speech_lengths.cpu()
            
            batch = 1
            decoding_window = speech.shape[1]  # 动态获取 decoding_window
            feature_size = self.feature_size
            offset = 0
            num_blocks = 12
            head = 4
            required_cache_size = decoding_window  # 根据实际情况调整
            d_k = 128
            output_size = 256
            cnn_module_kernel = 7
           
            # inference_response = pb_utils.InferenceResponse(
            #     output_tensors=[out0, out1])
            
            # 创建张量,这里有问题，找不到output0_dtype
            chunk_output = speech
            print("chunk_output shape:", chunk_output.shape)
            print("chunk_output is :",chunk_output)
            # 这里两个cache都是空的，因为都是非流式语音识别，所以没有用到。
            att_cache_out = torch.zeros((num_blocks, head, required_cache_size, d_k), dtype=torch.float32)
            print("att_cache_out shape:", att_cache_out.shape)
            print("att_cache_out data:", att_cache_out)
            # **去掉 batch 维度**
            cnn_cache_out = torch.zeros((num_blocks, batch, output_size, cnn_module_kernel), dtype=torch.float32)
            print("cnn_cache_out shape:", cnn_cache_out.shape)
            print("cnn_cache_out data:", cnn_cache_out)
        
            # 将张量转换为发送格式
            chunk_send=pb_utils.Tensor.from_dlpack("chunk", to_dlpack(chunk_output))
            att_cache_send=pb_utils.Tensor.from_dlpack("att_cache", to_dlpack(att_cache_out))
            cnn_cache_send=pb_utils.Tensor.from_dlpack("cnn_cache", to_dlpack(cnn_cache_out))
            
            # 3. 传给 Triton            
            # 创建推理响应返回的 Tensor 对象
            #inference_response = pb_utils.InferenceResponse(
            #    output_tensors=[chunk_tensor, offset_tensor, att_cache_tensor, cnn_cache_tensor])
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[chunk_send, att_cache_send, cnn_cache_send])
            
            responses.append(inference_response)
            print("********************************end feature extract********************************")
        return responses