import numpy as np


# 编辑距离是一种用于衡量两个字符串之间差异的度量标准。它定义为将一个字符串转换成另一个字符串所需的最少单字符编辑次数（插入、删除或替换）
def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case,字符串相等或其中一个等于0时的处理方法
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space,距离数组是一个全为0的2行n+1列的数组，其中对象格式都是int32
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix，初始化距离数组
    for j in range(n + 1):
        distance[0][j] = j

    # calculate levenshtein distance。计算levenstein距离
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


# 包含真实文本和预测文本的列表
def cal_cer(references, predictions):
    # 累计所有预测文本与真实文本之间的编辑距离。
    errors = 0
    # 用于累计所有真实文本的长度。
    lengths = 0
    # 将文本中的元素一一对应起来并打包
    for ref, pred in zip(references, predictions):
        # 将打包好的元素转换为字符列表并计算编辑长度，然后相加求和
        cur_ref = list(ref)
        cur_hyp = list(pred)
        cur_error = _levenshtein_distance(cur_ref, cur_hyp)
        errors += cur_error
        lengths += len(cur_ref)
    # 返回字符错误率
    return float(errors) / lengths
