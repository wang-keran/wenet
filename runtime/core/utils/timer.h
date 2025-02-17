// Copyright (c) 2021 Mobvoi Inc (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_TIMER_H_
#define UTILS_TIMER_H_

#include <chrono>

namespace wenet {

// 获取当前时间
class Timer {
 public:
  Timer() : time_start_(std::chrono::steady_clock::now()) {}
  // 重置计时
  void Reset() { time_start_ = std::chrono::steady_clock::now(); }
  // return int in milliseconds，返回经过的时间（毫秒）
  int Elapsed() const {
    auto time_now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_now -
                                                                 time_start_)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> time_start_;
};
}  // namespace wenet

#endif  // UTILS_TIMER_H_
