# Copyright (c) 2021 Mobvoi Inc (Chao Yang)
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

import sys
import random
import math

import torchaudio
import torch


# 将分贝（dB）值转换为幅度（amp）值。
def db2amp(db):
    # pow(10, db / 20)表示将db除以20的结果作为指数，计算10的该次方。
    return pow(10, db / 20)


# 将幅度转为分贝
def amp2db(amp):
    # 先求出来10的几次方再乘以20获得分贝
    return 20 * math.log10(amp)


# 生成了一个在分贝域（db-domain）上的多项式失真函数。
# f(x)=a⋅x的m次方⋅(1−x)的n次方+x
def make_poly_distortion(conf):
    """Generate a db-domain ploynomial distortion function

        f(x) = a * x^m * (1-x)^n + x

    Args:
        conf: a dict {'a': #int, 'm': #int, 'n': #int}

    Returns:
        The ploynomial function, which could be applied on
        a float amplitude value
    """
    a = conf['a']
    m = conf['m']
    n = conf['n']

    def poly_distortion(x):
        abs_x = abs(x)
        if abs_x < 0.000001:
            x = x
        else:
            db_norm = amp2db(abs_x) / 100 + 1
            if db_norm < 0:
                db_norm = 0
            db_norm = a * pow(db_norm, m) * pow((1 - db_norm), n) + db_norm
            if db_norm > 1:
                db_norm = 1
            db = (db_norm - 1) * 100
            amp = db2amp(db)
            if amp >= 0.9997:
                amp = 0.9997
            if x > 0:
                x = amp
            else:
                x = -amp
        return x

    return poly_distortion



# 多项式失真函数
def make_quad_distortion():
    return make_poly_distortion({'a': 1, 'm': 1, 'n': 1})


# the amplitude are set to max for all non-zero point
# 生成一个最大失真函数
def make_max_distortion(conf):
    """Generate a max distortion function

    Args:
        conf: a dict {'max_db': float }
            'max_db': the maxium value.

    Returns:
        The max function, which could be applied on
        a float amplitude value
    """
    max_db = conf['max_db']
    if max_db:
        max_amp = db2amp(max_db)  # < 0.997
    else:
        max_amp = 0.997

    def max_distortion(x):
        if x > 0:
            x = max_amp
        elif x < 0:
            x = -max_amp
        else:
            x = 0.0
        return x

    return max_distortion


# 目的是将分贝（dB）域的掩码转换为振幅（amplitude）域的掩码。
def make_amp_mask(db_mask=None):
    """Get a amplitude domain mask from db domain mask

    Args:
        db_mask: Optional. A list of tuple. if None, using default value.

    Returns:
        A list of tuple. The amplitude domain mask
    """
     # 检查是否为空，空的话使用默认值
    if db_mask is None:
        db_mask = [(-110, -95), (-90, -80), (-65, -60), (-50, -30), (-15, 0)]
    # 将分贝掩码转换为振幅掩码
    # 使用列表推导式遍历db_mask中的每个元组
    # 调用db2amp函数将其转换为振幅值
    # 转换后的振幅值组成新的元组
    amp_mask = [(db2amp(db[0]), db2amp(db[1])) for db in db_mask]
    # 返回结果
    return amp_mask


# 将分贝（dB）域的掩码转换为振幅（amplitude）域的掩码。
default_mask = make_amp_mask()


# 生成一个振幅域掩码，该掩码在 [-100db, 0db] 范围内随机生成
def generate_amp_mask(mask_num):
    """Generate amplitude domain mask randomly in [-100db, 0db]

    Args:
        mask_num: the slot number of the mask

    Returns:
        A list of tuple. each tuple defines a slot.
        e.g. [(-100, -80), (-65, -60), (-50, -30), (-15, 0)]
        for #mask_num = 4
    """
    a = [0] * 2 * mask_num
    a[0] = 0
    m = []
    for i in range(1, 2 * mask_num):
        # 随机数生成
        a[i] = a[i - 1] + random.uniform(0.5, 1)
    # 掩码生成
    max_val = a[2 * mask_num - 1]
    # 掩码范围
    for i in range(0, mask_num):
        l = ((a[2 * i] - max_val) / max_val) * 100
        r = ((a[2 * i + 1] - max_val) / max_val) * 100
        m.append((l, r))
    return make_amp_mask(m)


# 该函数根据提供的配置（掩码数量和最大值）将输入振幅值映射到新的值。
# 如果输入值位于正或负振幅掩码槽位内，则将其映射到最大值；否则，将其映射到0。
def make_fence_distortion(conf):
    """Generate a fence distortion function

    In this fence-like shape function, the values in mask slots are
    set to maxium, while the values not in mask slots are set to 0.
    Use seperated masks for Positive and negetive amplitude.

    Args:
        conf: a dict {'mask_number': int,'max_db': float }
            'mask_number': the slot number in mask.
            'max_db': the maxium value.

    Returns:
        The fence function, which could be applied on
        a float amplitude value
    """
    mask_number = conf['mask_number']
    max_db = conf['max_db']
    max_amp = db2amp(max_db)  # 0.997
    if mask_number <= 0:
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    def fence_distortion(x):
        is_in_mask = False
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        return x

    return fence_distortion



# 生成一个锯齿状（jag-like）的失真函数。这个函数基于一个配置字典 conf 来决定哪些值（振幅）会被保留，哪些值会被设置为0。
# 它使用了两个独立的掩码（mask）来分别处理正振幅和负振幅的情况。
def make_jag_distortion(conf):
    """Generate a jag distortion function

    In this jag-like shape function, the values in mask slots are
    not changed, while the values not in mask slots are set to 0.
    Use seperated masks for Positive and negetive amplitude.

    Args:
        conf: a dict {'mask_number': #int}
            'mask_number': the slot number in mask.

    Returns:
        The jag function,which could be applied on
        a float amplitude value
    """
    # 掩码生成
    mask_number = conf['mask_number']
    # 如果 mask_number 小于或等于0，则使用默认的掩码 default_mask作为正振幅的掩码，而负振幅的掩码则通过 make_amp_mask([(-50, 0)]) 生成
    if mask_number <= 0:
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    # 如果 mask_number 大于0，则使用 generate_amp_mask(mask_number)生成正振幅和负振幅的掩码。
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    # 锯齿失真函数 
    def jag_distortion(x):
        # 对于输入的 x（振幅值），首先检查它是否属于正振幅或负振幅的掩码范围内。
        is_in_mask = False
        # x 是正数，则遍历正振幅掩码 positive_mask，检查 x 是否在某个掩码范围内。如果是，则返回 x；否则，如果 x 不在任何掩码范围内，则返回0.0。
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        # 如果 x 是负数，则先取其绝对值 abs_x，然后遍历负振幅掩码 negative_mask，检查 abs_x 是否在某个掩码范围内。
        # 如果是，则返回原始的 x（注意这里返回的是原始的负数，而不是绝对值）；否则，如果 x 不在任何掩码范围内，则返回0.0。
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        # 如果 x 恰好为0，则直接返回 x
        return x

    return jag_distortion


# gaining 20db means amp = amp * 10
# gaining -20db means amp = amp / 10
# 生成一个基于分贝（dB）域的增益函数
def make_gain_db(conf):
    """Generate a db domain gain function

    Args:
        conf: a dict {'db': #float}
            'db': the gaining value

    Returns:
        The db gain function, which could be applied on
        a float amplitude value
    """
    # 从配置字典 conf 中获取分贝值 db。
    db = conf['db']

    #  并返回调整后的振幅值
    def gain_db(x):
        # 返回0.997或者计算增益调整后的振幅值，哪个小返回哪个，保证振幅不溢出
        return min(0.997, x * pow(10, db / 20))

    # 返回调整后的振幅值
    return gain_db


# 用于对波形数据进行失真处理。
# 具体来说，该函数会根据指定的概率 rate 对波形数据中的每个样本点进行失真操作。
# 失真操作由传入的函数 func 完成。
def distort(x, func, rate=0.8):
    """Distort a waveform in sample point level

    Args:
        x: the origin wavefrom
        func: the distort function
        rate: sample point-level distort probability

    Returns:
        the distorted waveform
    """
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            x[0][i] = func(float(x[0][i]))
    return x


# 对输入的矩阵 x 进行随机变换
# 它会遍历矩阵的每一列，并以一定的概率（由 rate 参数决定）对每一列的元素应用一个随机选择的函数 func。
# 这个过程是通过 random.uniform(0, 1) 生成一个在 [0, 1) 范围内的随机数 a，然后判断 a 是否小于 rate 来决定是否应用变换。
def distort_chain(x, funcs, rate=0.8):
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            for func in funcs:
                x[0][i] = func(float(x[0][i]))
    return x


# x is numpy
# 对音频数据 x 进行不同的失真处理
# 函数根据 distort_type 参数的不同值，调用相应的失真处理函数，并将结果赋值给 x。
# 如果 distort_type 不是支持的类型，则打印 "unsupport type" 并返回原始音频数据 x。
def distort_wav_conf(x, distort_type, distort_conf, rate=0.1):
    if distort_type == 'gain_db':
        gain_db = make_gain_db(distort_conf)
        x = distort(x, gain_db)
    elif distort_type == 'max_distortion':
        max_distortion = make_max_distortion(distort_conf)
        x = distort(x, max_distortion, rate=rate)
    elif distort_type == 'fence_distortion':
        fence_distortion = make_fence_distortion(distort_conf)
        x = distort(x, fence_distortion, rate=rate)
    elif distort_type == 'jag_distortion':
        jag_distortion = make_jag_distortion(distort_conf)
        x = distort(x, jag_distortion, rate=rate)
    elif distort_type == 'poly_distortion':
        poly_distortion = make_poly_distortion(distort_conf)
        x = distort(x, poly_distortion, rate=rate)
    elif distort_type == 'quad_distortion':
        quad_distortion = make_quad_distortion()
        x = distort(x, quad_distortion, rate=rate)
    elif distort_type == 'none_distortion':
        pass
    else:
        print('unsupport type')
    return x


# 对音频失真并保存
def distort_wav_conf_and_save(distort_type, distort_conf, rate, wav_in,
                              wav_out):
    x, sr = torchaudio.load(wav_in)
    x = x.detach().numpy()
    out = distort_wav_conf(x, distort_type, distort_conf, rate)
    torchaudio.save(wav_out, torch.from_numpy(out), sr)


if __name__ == "__main__":
    # 失真类型
    distort_type = sys.argv[1]
    #输入输出路径
    wav_in = sys.argv[2]
    wav_out = sys.argv[3]
    # 初始化配置变量为None
    conf = None
    # 设置失真率的默认值为0.1。
    rate = 0.1
    # 按照失真类型不同来运用不同的失真函数
    if distort_type == 'new_jag_distortion':
        conf = {'mask_number': 4}
    elif distort_type == 'new_fence_distortion':
        conf = {'mask_number': 1, 'max_db': -30}
    elif distort_type == 'poly_distortion':
        conf = {'a': 4, 'm': 2, "n": 2}
    distort_wav_conf_and_save(distort_type, conf, rate, wav_in, wav_out)

# 总结：主要用于检测和分析音频文件中的失真情况该文件可能包含用于自动分析 .wav 文件并检测可能由于过载导致的失真区域的代码。
# 失真可能会降低录音的质量，因此该工具在处理大量录音时特别有用，例如来自户外音乐会的用户生成的录音片段。
# 音频文件失真可以看作是波形失真的一个子集，特别是在数字音频处理和传输的上下文中