# Copyright (c) 2021 Shaoshang Qi
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

import copy


# 这段代码的功能是更新配置字典，允许通过一系列指定的覆盖项来修改原有的配置。
# 定义一个名为 override_config 的函数，接受两个参数：
#   configs：原始配置字典。
#   override_list：包含覆盖项的列表，每个覆盖项以特定格式指定。
def override_config(configs, override_list):
    # 功能：使用 deepcopy 创建原始配置的深拷贝，以便对其进行修改而不影响原始配置。
    new_configs = copy.deepcopy(configs)
    # 功能：遍历传入的覆盖项列表 override_list。
    for item in override_list:
        # 分割和验证格式，功能：将每个覆盖项按空格分割为两个部分，如果不等于两个部分，打印错误信息并跳过该项。
        arr = item.split()
        if len(arr) != 2:
            print(f"the overrive {item} format not correct, skip it")
            continue
        # 处理键路径：将第一个部分（键路径）按照点（.）分割，以便逐层访问配置字典。s_configs 变量用于在遍历时逐层深入配置字典。
        keys = arr[0].split('.')
        s_configs = new_configs
        # 遍历键路径：遍历 keys 列表，获取每个键和其索引。
        for i, key in enumerate(keys):
            # 键存在性检查：检查当前键是否存在于当前配置字典中。如果不存在，打印错误信息。
            if key not in s_configs:
                print(f"the overrive {item} format not correct, skip it")
            # 修改配置项：如果当前键是路径的最后一个部分，就获取当前配置项的类型 param_type，如果该配置项的类型不是布尔型，尝试将覆盖值（arr[1]）转换为该类型并赋值；如果是布尔型，则根据 arr[1] 的值判断并赋值为 True 或 False；打印成功覆盖的消息。
            if i == len(keys) - 1:
                param_type = type(s_configs[key])
                if param_type != bool:
                    s_configs[key] = param_type(arr[1])
                # 逐层深入： 如果当前键不是路径的最后一部分，更新 s_configs，使其指向下一级配置。
                else:
                    s_configs[key] = arr[1] in ['true', 'True']
                print(f"override {arr[0]} with {arr[1]}")
            else:
                s_configs = s_configs[key]
    # 返回新的配置：返回更新后的配置字典 new_configs。
    return new_configs
# 总结：这个函数的目的是通过提供的覆盖项列表来灵活地修改原始配置字典。
# 它支持通过点（.）语法来访问和更新嵌套的配置项，同时确保对配置项类型的正确处理。
# 错误和异常情况会被打印出来，以帮助用户调试和理解覆盖过程。
