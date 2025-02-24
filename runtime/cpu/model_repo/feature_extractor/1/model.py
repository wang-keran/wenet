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
        output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # 获取输出的张量的类型（speech）
        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "offset")
        # Convert Triton types to numpy types，将Triton类型转换为numpy类型
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        
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
        # # (args.chunk_size - 1) * model.encoder.embed.subsampling_rate + model.encoder.embed.right_context + 1
        # decoding_window=(args.chunk_size - 1) * \
        # model.encoder.embed.subsampling_rate + \
        # model.encoder.embed.right_context + 1 if args.chunk_size > 0 else 67
        # offset = args['chunk_size'] * args['left_chunks']
        # # attention_cache:
        # num_blocks=configs['encoder_conf']['num_blocks']    #train.yaml中的数据
        # head=configs['encoder_conf']['attention_heads'] #在train.yaml中
        # required_cache_size=args['chunk_size'] * args['left_chunks']
        # args['output_size'] // args['head'] * 2
        # arguments['output_size'] = configs['encoder_conf']['output_size']   #在train.yaml中

        # # cnn_cache
        # num_blocks=configs['encoder_conf']['num_blocks']    #train.yaml中的数据
        # args['batch']=1
        # arguments['output_size'] = configs['encoder_conf']['output_size']   #在train.yaml中
        # args['cnn_module_kernel'] - 1
        # arguments['cnn_module_kernel'] = configs['encoder_conf'].get(
        # 'cnn_module_kernel', 1)

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
        batch_count = []
        total_waves = []
        batch_len = []
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")

            cur_b_wav = input0.as_numpy()
            cur_b_wav = cur_b_wav * (1 << 15)  # b x -1
            cur_b_wav_lens = input1.as_numpy()  # b x 1
            cur_batch = cur_b_wav.shape[0]
            cur_len = cur_b_wav.shape[1]
            batch_count.append(cur_batch)
            batch_len.append(cur_len)
            for wav, wav_len in zip(cur_b_wav, cur_b_wav_lens):
                wav_len = wav_len[0]
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
                                 device=self.device)
            speech_lengths = torch.zeros((b, 1),
                                         dtype=torch.int32,
                                         device=self.device)
            
            for i in range(b):
                f = features[idx]
                f_l = f.shape[0]
                speech[i, 0:f_l, :] = f.to(self.output0_dtype)
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
           
            out0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            out1 = pb_utils.Tensor.from_dlpack("speech_lengths",
                                               to_dlpack(speech_lengths))
            # inference_response = pb_utils.InferenceResponse(
            #     output_tensors=[out0, out1])
            
            # 创建张量
            chunk_output = torch.zeros(batch, decoding_window, feature_size, dtype=self.output0_dtype)
            offset_output = torch.zeros(offset, dtype=self.output1_dtype)
            att_cache_out = torch.zeros(batch, num_blocks, head, required_cache_size, d_k, dtype=self.output2_dtype)
            cnn_cache = torch.zeros(num_blocks, batch, output_size, cnn_module_kernel, dtype=self.output3_dtype)
        
            # 将张量转换为DLpack格式
            chunk_dlpack = to_dlpack(chunk_output)
            offset_dlpack = to_dlpack(offset_output)
            att_cache_dlpack = to_dlpack(att_cache_out)
            cnn_cache_dlpack = to_dlpack(cnn_cache)
        
            # 封装为 pb_utils.Tensor 对象
            chunk_tensor = pb_utils.Tensor("chunk", chunk_dlpack)
            offset_tensor = pb_utils.Tensor("offset", offset_dlpack)
            att_cache_tensor = pb_utils.Tensor("att_cache", att_cache_dlpack)
            cnn_cache_tensor = pb_utils.Tensor("cnn_cache", cnn_cache_dlpack)
            
            # 创建推理响应返回的 Tensor 对象
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[chunk_tensor, offset_tensor, att_cache_tensor, cnn_cache_tensor])
            
            responses.append(inference_response)
        return responses
        #  # 提取特征
    #         feature = self.feature_extractor([cur_wav])[0]
    #         feature = feature.to(self.output0_dtype)

    #         # 计算特征长度
    #         feature_len = feature.shape[0]

    #         # 生成输出张量
    #         chunk_output = feature.unsqueeze(0)  # [1, decoding_window, feature_size]
    #         offset_output = torch.tensor([0], dtype=self.output1_dtype)  # [1]
    #         att_cache_output = torch.zeros(self.num_blocks, self.head, feature_len, self.d_k, dtype=self.output2_dtype)
    #         cnn_cache_output = torch.zeros(self.num_blocks, 1, self.output_size, self.cnn_module_kernel, dtype=self.output3_dtype)

    #         # 将张量转换为DLpack格式
    #         chunk_dlpack = to_dlpack(chunk_output)
    #         offset_dlpack = to_dlpack(offset_output)
    #         att_cache_dlpack = to_dlpack(att_cache_output)
    #         cnn_cache_dlpack = to_dlpack(cnn_cache_output)

    #         # 封装为 pb_utils.Tensor 对象
    #         chunk_tensor = pb_utils.Tensor("chunk", chunk_dlpack)
    #         offset_tensor = pb_utils.Tensor("offset", offset_dlpack)
    #         att_cache_tensor = pb_utils.Tensor("att_cache", att_cache_dlpack)
    #         cnn_cache_tensor = pb_utils.Tensor("cnn_cache", cnn_cache_dlpack)

    #         # 创建推理响应返回的 Tensor 对象
    #         inference_response = pb_utils.InferenceResponse(
    #             output_tensors=[chunk_tensor, offset_tensor, att_cache_tensor, cnn_cache_tensor])

    #         responses.append(inference_response)

    # return responses
