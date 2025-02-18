# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint as ckpt

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.class_utils import (
    WENET_EMB_CLASSES,
    WENET_MLP_CLASSES,
    WENET_NORM_CLASSES,
    WENET_SUBSAMPLE_CLASSES,
    WENET_ATTENTION_CLASSES,
    WENET_ACTIVATION_CLASSES,
)
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.utils.common import mask_to_bias


# 主要是用于构建编码器（Encoder）模块，具体是Transformer编码器和Conformer编码器。
# BaseEncoder类是所有编码器的基础，负责初始化和定义基本的前向传播逻辑。
class BaseEncoder(torch.nn.Module):

    # 初始化输入层、位置编码、归一化层等。
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            query_bias: whether use bias in attention.linear_q
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            value_bias: whether use bias in attention.linear_v
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
            use_sdpa: whether to use SDPA, currently only support transformer for now
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        pos_emb_class = WENET_EMB_CLASSES[pos_enc_layer_type]
        # NOTE(Mddct): head_dim == output_size // attention_heads for most of
        #    speech tasks,  but for other task (LLM),
        #    head_dim == hidden_size * attention_heads. refactor later
        self.embed = WENET_SUBSAMPLE_CLASSES[input_layer](
            input_size, output_size, dropout_rate,
            pos_emb_class(output_size, positional_dropout_rate)
            if pos_enc_layer_type != 'rope_pos' else pos_emb_class(
                output_size, output_size //
                attention_heads, positional_dropout_rate))

        assert layer_norm_type in ['layer_norm', 'rms_norm']
        self.normalize_before = normalize_before
        self.after_norm = WENET_NORM_CLASSES[layer_norm_type](output_size,
                                                              eps=norm_eps)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.use_sdpa = use_sdpa

    # 返回输出大小
    def output_size(self) -> int:
        return self._output_size

    # 前向函数，定义了如何处理输入数据 xs，并返回编码器的输出以及子采样的掩码。
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        # 获取时间步长
        T = xs.size(1)
        # 使用 make_pad_mask 函数根据输入长度 xs_lens 创建一个掩码，并通过按位取反 ~ 转换为有效位置的掩码，是中间那个1位。
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        # 如果归一化不为空就对输入数据进行归一化
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # 将输入特征 xs 通过嵌入层进行处理,转为稠密向量，同时获取位置编码 pos_emb 和更新后的掩码 masks。
        xs, pos_emb, masks = self.embed(xs, masks)
        # 将有效位置的掩码保存到 mask_pad 变量中，以便在后续的处理过程中使用，确保模型在计算时只关注有效的输入数据。
        mask_pad = masks  # (B, 1, T/subsample_rate)
        # 根据不同的设置（如动态块大小）生成相应的块掩码。
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
            # Since we allow up to 1s(100 frames) delay, the maximum
            # chunk_size is 100 / 4 = 25.
            max_chunk_size=int(100.0 / self.embed.subsampling_rate))
        # （可选）如果使用 SDPA，则将块掩码转换为偏置形式。
        # SDPA 不支持动态块，因此需要将块掩码转换为偏置形式。SDPA 代表 Sparse Dynamic Positioning Attention，是一种用于优化注意力机制的方法，特别是在处理序列数据时。它主要通过稀疏化注意力计算来提高效率，适应长序列输入。
        # 稀疏化注意力：传统的自注意力机制计算复杂度为 O(n2)O(n2)，其中 nn 是输入序列的长度。SDPA 通过只计算部分重要位置之间的注意力关系，从而降低计算复杂度，提高处理速度。
        if self.use_sdpa:
            chunk_masks = mask_to_bias(chunk_masks, xs.dtype)
        # 根据是否启用梯度检查点机制，选择使用 forward_layers_checkpointed 或 forward_layers 进行前向传播编码。
        # 在前向传播期间，它会保存每一层的输出结果，以便在反向传播时使用。这意味着所有中间计算结果都被存储在内存中。
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb,
                                                  mask_pad)
        # 使用了梯度检查点（gradient checkpointing）技术。在前向传播期间，它不会保存所有中间输出，而是选择在反向传播时重新计算这些输出。这可以显著降低内存消耗，尤其是在处理较大的模型或长序列时。
        #     forward_layers_checkpointed 适合处理大模型或长序列的情况，因为它可以显著降低内存使用，但可能会增加计算时间，因为它需要在反向传播时重新计算中间结果。
        #选择使用哪种方法通常取决于模型的大小和可用的内存资源。
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        # 如果启用了归一化，则对编码器的输出进行归一化，便于节省模型推理时间和突出特征
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    # 该方法遍历 self.encoders 中的所有编码器层，并依次将输入 xs、chunk_masks、pos_emb 和 mask_pad 传递给每个编码器层进行处理。
    # 每一层的输出会更新 xs 和 chunk_masks，而返回的其他值被忽略（用下划线 _ 表示）。
    # 最后，返回最终的输出张量 xs
    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    # 该方法的主要目的是实现梯度检查点（gradient checkpointing），以降低内存使用。
    # 在每一层的前向传播中，使用 ckpt.checkpoint 方法替代直接调用 layer.__call__，这是为了在反向传播时重新计算这一层的输出，从而节省内存。
    # 通过这样做，可以处理更大的模型或者更长的序列，因为不需要在前向传播中保存所有中间计算结果。
    # 返回最终输出张量 xs，与 forward_layers 方法相同。
    @torch.jit.unused
    def forward_layers_checkpointed(self, xs: torch.Tensor,
                                    chunk_masks: torch.Tensor,
                                    pos_emb: torch.Tensor,
                                    mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = ckpt.checkpoint(layer.__call__,
                                                    xs,
                                                    chunk_masks,
                                                    pos_emb,
                                                    mask_pad,
                                                    use_reentrant=False)
        return xs

    # 分块处理：处理一个输入块（chunk），并返回当前块的输出及新的注意力缓存和 CNN 缓存。
    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        # 这里确保输入的批次大小为1，即只处理一条音频数据。
        assert xs.size(0) == 1
        # tmp_masks：创建一个与输入长度相同的掩码，所有值为1，表示所有位置都有效，用于表示没有任何时间步需要被掩盖或忽略。
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        # self.global_cmvn(xs)：如果设置了全局均值方差归一化，应用该方法对输入数据进行归一化处理。
        # 中心化（减去均值）：将输入的每一帧的特征值减去全局均值，确保数据的平均值接近 0。
        # 缩放（除以标准差）：将结果除以全局标准差，保证特征的标准差为 1，从而得到尺度一致的数据。
        # 这样可以加速收敛，使音频特征更明显，加速收敛，减少噪声干扰，通过torch来实现，但在这里面是none没有使用
        # 给掩码扩张维度，位置在原来张量的第1个维度位置上，(1, T)变成(1,1, T)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        # 将输入特征 xs 通过嵌入层进行处理，得到经过嵌入的特征 xs 和位置编码 pos_emb，将离散的音素向量转换为更高维度的稠密的嵌入向量，并且获得新的嵌入向量的位置编码，利于下一步的推理。
        # 为以前输入的向量提供位置信息，以帮助模型理解输入数据的位置关系。
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        # 获取当前注意力缓存的层数和时间维度。
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        # 获取音频块大小
        chunk_size = xs.size(1)
        print("forward_chunk中块大小为：",chunk_size)
        # attention_key_size 计算注意力键的大小，是之前的计算结果缓存大小加一块。
        attention_key_size = cache_t1 + chunk_size
        # 先去utils文件夹里找到对应的类，再去subsampling里面找，然后通过torch到embedding里找到需要的方法实现
        # offset:已经处理过输入帧的数量，cache_t1是已经处理过时间步的数量的缓存
        # 为当前输入的嵌入向量提供位置信息
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1,
                                               size=attention_key_size)
        # 计算下一个缓存起始位置：根据 required_cache_size 的值计算下一次缓存的起始位置。
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if elayers == 0:
                kv_cache = (att_cache, att_cache)
            else:
                i_kv_cache = att_cache[i:i + 1]
                size = att_cache.size(-1) // 2
                kv_cache = (i_kv_cache[:, :, :, :size], i_kv_cache[:, :, :,
                                                                   size:])
            xs, _, new_kv_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=kv_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            new_att_cache = torch.cat(new_kv_cache, dim=-1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        #print("归一化前的音频：")
        #print(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)

        #print("归一化后的音频：")
        #print(xs)
        print("使用了forward_chunk()方法444444444444444")
         # 打印 hidden-dim
        print(f"hidden-dim: {self._output_size}")

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        # 拼接注意力缓存，CNN缓存，返回结果
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    # 分块逐步前向传播：实现逐块输入的前向传播，适合流式输入场景。该方法会逐步处理输入数据，每次处理一个块并更新缓存。
    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        # 判断解码块数是否大于0,小于等于0不解码
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk确定必须使用动态块或者大于0的静态块
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        # 获取下采样率，选择不进行子采样的情况，下采样率是1,作用是压缩输入的步长，减少计算量
        subsampling = self.embed.subsampling_rate
        # 当前帧和右侧的上下文长度（最远可以看到多远），0+1,保证最远只能看到当前帧，右侧上下文全都看不见，加1是为了能看到第一个块的第一帧
        context = self.embed.right_context + 1  # Add current frame
        # 如果不加当前帧，模型在处理每个 chunk 时，需要依赖当前帧的信息来做出准确的预测。如果 context 不包含当前帧，模型将无法正确处理当前帧的数据，从而影响解码的准确性。
        # decoding_window 的计算不再包括当前帧，导致窗口大小与实际需求不符。例如，假设 decoding_chunk_size = 4，subsampling_rate = 4，right_context = 2，
        # 则： [ \text{decoding_window} = (4 - 1) \times 4 + 2 = 3 \times 4 + 2 = 14 ] 这意味着窗口大小为 14 帧，但实际上是少了当前帧的信息。
        # 每次处理的步长，每次处理块大小*块数（要躲开下采样的帧），确保在每次前向传播时，新的输入块与之前的计算结果保持合理的重叠，从而充分利用缓存（如注意力缓存和卷积缓存），提高效率。
        stride = subsampling * decoding_chunk_size
        # -1减去context加上的当前帧，+context考虑右侧上下文，但是没有右侧上下文，只有一个当前帧，每4帧用1帧，第一个chunk的第一帧已经包含在context中了，不减去1的话
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
#         虽然代码中没有显式的 left_context 变量，但左侧上下文信息是通过以下方式隐式处理的：

# 动态块掩码 (chunk_masks)：控制哪些帧是可以被当前帧看到的，包括左侧上下文。
# 注意力机制中的缓存 (att_cache)：存储之前块的注意力键值对，作为左侧上下文信息使用。
# 子采样和嵌入层 (embed)：通过重叠输入确保上下文信息的传递。
# 逐块前向传播 (forward_chunk_by_chunk)：逐步处理每个块，并更新缓存，确保每个块的处理都考虑到之前的上下文信息。
        # 获取帧的数量（时间步长）
        num_frames = xs.size(1)
        # 初始化缓存存储中间结果，因为是中间结果，所以肯定是编码过的块
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        # 存储解码块输出信息
        outputs = []
        # 在逐块处理输入时，offset 表示已经处理过的输入帧的数量。每次处理完一个块后，offset 会增加，以指向下一个要处理的块的起始位置。
        # 这样可以确保每个输入块都在正确的位置进行解码，从而保持数据的连续性。
        offset = 0
        # 缓存左侧的编码过的 encoder 块，负数表示全都参考，0表示全都不参考，正数表示参考的大小。
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        # 从第零帧开始到第num_frames - context + 1帧，步长为stride
        for cur in range(0, num_frames - context + 1, stride):
            #print(111)  # 调试用
            # 求当前处理块的长度，即当前帧cur到cur+decoding_window之间的帧数，不能大于num_frames一块的帧数
            end = min(cur + decoding_window, num_frames)
            # chunk_xs：从输入张量 xs 中提取当前块的数据。: 表示在其他维度上选择所有数据，cur:end 则选择当前块的帧范围。
            chunk_xs = xs[:, cur:end, :]
            
            # 打印 chunk_xs 的维度
            print(f"chunk_xs 的维度: {chunk_xs.shape}")
            
            # 确认xs和y的维度是不是一样
            (y, att_cache,
             cnn_cache) = self.forward_chunk(chunk_xs, offset,
                                             required_cache_size, att_cache,
                                             cnn_cache)
             
            # 打印 y 的维度
            print(f"y 的维度: {y.shape}")
            
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)),
                           device=ys.device,
                           dtype=torch.bool)
        #print(ys)       # 调试用
        print("2222222222222222222222222")
        return ys, masks


# 实现一个 Transformer 编码器模块。
class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    # 初始化
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        selfattention_layer_type: str = "selfattn",
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing,
                         use_sdpa, layer_norm_type, norm_eps)

        assert selfattention_layer_type in ['selfattn', 'rope_abs_selfattn']
        # 创建编码器层
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    attention_heads, output_size, attention_dropout_rate,
                    query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                    head_dim),
                mlp_class(output_size,
                          linear_units,
                          dropout_rate,
                          activation,
                          mlp_bias,
                          n_expert=n_expert,
                          n_expert_activated=n_expert_activated),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])


# 实现了一个 Conformer 编码器，通过结合卷积和自注意力机制来提取和转换特征。
class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    # 初始化
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        conv_bias: bool = True,
        gradient_checkpointing: bool = False,
        use_sdpa: bool = False,
        layer_norm_type: str = 'layer_norm',
        norm_eps: float = 1e-5,
        n_kv_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        mlp_type: str = 'position_wise_feed_forward',
        mlp_bias: bool = True,
        n_expert: int = 8,
        n_expert_activated: int = 2,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        # 先去BaseEncoder进行初始化，然后去WENET_EMB_CLASSES初始化，EMB在Embedding.py中,使用RelPositionEncoding类，再走到PositionEncoding类，PositionEncoding类是基类，接着去BaseEncoder中的Conv2dSubsampling4(BaseSubsampling)类进行初始化在subsampling.py中
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, gradient_checkpointing,
                         use_sdpa, layer_norm_type, norm_eps)
        # 根据提供的 activation_type 选择合适的激活函数，实例化。
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # 自注意力模块定义
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            query_bias,
            key_bias,
            value_bias,
            use_sdpa,
            n_kv_head,
            head_dim,
        )
        # 定义前馈网络（MLP）所需的参数。
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            mlp_bias,
            n_expert,
            n_expert_activated,
        )
        # 定义卷积模块所需的参数。
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, conv_bias)

        print("111111111111111")
        # 创建编码器层
        # 根据 mlp_type 获取对应的 MLP 类。
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        # 编码器层列表：使用 torch.nn.ModuleList 存储多个 ConformerEncoderLayer 实例。
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                # 先到attention.py里找到RelPositionMultiHeadedAttention(MultiHeadedAttention)进行初始化，再找到MultiHeadedAttention最底层初始化，最底层里全是utorch.nn
                # RelPositionMultiHeadedAttention也全是torch.nn
                # PositionFeedForward里面也全是torch.nn
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args) if macaron_style else None,
                # 这里也是拿torch.nn实现的
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])

# 总结：实现了transformer和conformer编码器