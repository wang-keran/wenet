// Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/utils.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

// 总结：用于测试wenet::TopK函数。该函数的目的是从一个浮点数数组（或向量）中找出最大的K个值以及它们对应的索引。
TEST(UtilsTest, TopKTest) {
  using ::testing::ElementsAre;
  using ::testing::FloatNear;
  using ::testing::Pointwise;
  std::vector<float> data = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
  std::vector<float> values;
  std::vector<int32_t> indices;
  wenet::TopK(data, 3, &values, &indices);
  EXPECT_THAT(values, Pointwise(FloatNear(1e-8), {10, 9, 8}));
  ASSERT_THAT(indices, ElementsAre(9, 4, 8));
}
