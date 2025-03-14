#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

onnx_model_dir=/ws/onnx_model
model_repo=/ws/model_repo

# Convert config.pbtxt in model_repo and move models。将 config.pbtxt 文件转换并移动到 model_repo 中的模型目录中。
python3 scripts/convert.py --config=$onnx_model_dir/train.yaml --vocab=$onnx_model_dir/words.txt \
        --model_repo=$model_repo --onnx_model_dir=$onnx_model_dir

#--vocab=$onnx_model_dir/words.txt 
# Start server，启动服务端，服务端需要cuda
tritonserver --model-repository=${model_repo} --pinned-memory-pool-byte-size=1024000000 --cuda-memory-pool-byte-size=0:1024000000
