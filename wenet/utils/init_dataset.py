import copy
from typing import Optional
from wenet.dataset.dataset import Dataset

from wenet.text.base_tokenizer import BaseTokenizer


# 初始化 ASR （自动语音识别）数据集函数
def init_asr_dataset(data_type,
                     data_list_file,
                     tokenizer: Optional[BaseTokenizer] = None,
                     conf=None,
                     partition=True):
    return Dataset(data_type, data_list_file, tokenizer, conf, partition)


# 初始化数据集函数
def init_dataset(dataset_type,
                 data_type,
                 data_list_file,
                 tokenizer: Optional[BaseTokenizer] = None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['asr', 'ssl']

    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        conf = cv_conf

    if dataset_type == 'asr':
        return init_asr_dataset(data_type, data_list_file, tokenizer, conf,
                                partition)
    else:
        from wenet.ssl.init_dataset import init_dataset as init_ssl_dataset
        return init_ssl_dataset(data_type, data_list_file, conf, partition)

# 该文件定义了两个函数 init_asr_dataset 和 init_dataset，用于初始化 ASR 和 SSL 数据集。
# init_dataset 函数根据数据集类型和分割类型选择合适的初始化函数，并对非训练集的配置进行调整。