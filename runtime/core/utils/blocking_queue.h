// Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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

// 总结：实现了一个阻塞队列，用于多线程间的数据传递。BlockingQueue类提供了Push和Pop方法，用于向队列中添加和取出数据。
#ifndef UTILS_BLOCKING_QUEUE_H_
#define UTILS_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <limits>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

#include "utils/utils.h"

namespace wenet {

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity = std::numeric_limits<int>::max())
      : capacity_(capacity) {}

  void Push(const T& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(value);
    }
    not_empty_condition_.notify_one();
  }

  void Push(T&& value) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      while (queue_.size() >= capacity_) {
        not_full_condition_.wait(lock);
      }
      queue_.push(std::move(value));
    }
    not_empty_condition_.notify_one();
  }

  void Push(const std::vector<T>& values) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      for (auto& value : values) {
        while (queue_.size() >= capacity_) {
          not_empty_condition_.notify_one();
          not_full_condition_.wait(lock);
        }
        queue_.push(value);
      }
    }
    not_empty_condition_.notify_one();
  }

  void Push(std::vector<T>&& values) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& value : values) {
      while (queue_.size() >= capacity_) {
        not_empty_condition_.notify_one();
        not_full_condition_.wait(lock);
      }
      queue_.push(std::move(value));
    }
    not_empty_condition_.notify_one();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty()) {
      not_empty_condition_.wait(lock);
    }
    T t(std::move(queue_.front()));
    queue_.pop();
    not_full_condition_.notify_one();
    return t;
  }

  // num can be greater than capacity,but it needs to be used with care
  std::vector<T> Pop(size_t num) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<T> block_data;
    while (block_data.size() < num) {
      while (queue_.empty()) {
        not_full_condition_.notify_one();
        not_empty_condition_.wait(lock);
      }
      block_data.push_back(std::move(queue_.front()));
      queue_.pop();
    }
    not_full_condition_.notify_one();
    return block_data;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void Clear() {
    while (!Empty()) {
      Pop();
    }
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable not_full_condition_;
  std::condition_variable not_empty_condition_;
  std::queue<T> queue_;

 public:
  WENET_DISALLOW_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace wenet

#endif  // UTILS_BLOCKING_QUEUE_H_
