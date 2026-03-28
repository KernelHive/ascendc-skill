/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file registry.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CUBE_UTIL_REGISTRY_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_UTIL_REGISTRY_H_

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <pthread.h>
#include <lock.h>

namespace optiling {
namespace cachetiling {
template <typename ObjectTypePtr, typename... Args>
class Registry {
 public:
  using Creator = std::function<ObjectTypePtr(Args...)>;
  Registry(){};
  ~Registry() = default;

  void Register(const OpType &type, const Creator &creator) {
    if (Exist(type)) {
      return;
    }
    registry_map_[type] = creator;
  }

  ObjectTypePtr Create(const OpType &type, Args... args) {
    std::lock_guard<std::mutex> _(mtx_);

    if (!Exist(type)) {
      return nullptr;
    }

    return registry_map_[type](args...);
  }

  inline bool Exist(const OpType &type) const { return registry_map_.count(type) != 0; }

 private:
  Registry(const Registry &) = delete;
  Registry &operator=(const Registry &) = delete;
  std::unordered_map<OpType, Creator> registry_map_;
  std::mutex mtx_;
};

template <typename ObjectTypePtr>
class FactoryInst {
 public:
  inline void Add(const OpType &type, const ObjectTypePtr &ptr) {
    auto thread_id = pthread_self();
    lock_.wrlock();
    auto it = inst_.find(thread_id);
    if (it != inst_.end()) {
      it->second[type] = ptr;
    } else {
      auto ret = inst_.emplace(std::make_pair(thread_id, ObjectTypePtrArry({nullptr})));
      ret.first->second[type] = ptr;
    }
    lock_.unlock();
  }

  inline ObjectTypePtr Get(const OpType &type) {
    if (type >= kOpTypeNum) {
      return nullptr;
    }

    ObjectTypePtr ptr = nullptr;
    lock_.rdlock();
    auto it = inst_.find(pthread_self());
    if (it != inst_.end()) {
      ptr = it->second[type];
    }

    lock_.unlock();
    return ptr;
  }

  inline void Clear() {
    // only for UT
    lock_.wrlock();
    inst_.clear();
    lock_.unlock();
  }

 private:
  using ObjectTypePtrArry = std::array<ObjectTypePtr, static_cast<size_t>(kOpTypeNum)>;
  std::map<pthread_t, ObjectTypePtrArry> inst_;
  RWLock lock_;
};

template <typename ObjectTypePtr, typename... Args>
class Register {
 public:
  explicit Register(const OpType &type, Registry<ObjectTypePtr, Args...> &registry,
                    const typename Registry<ObjectTypePtr, Args...>::Creator &creator) {
    registry.Register(type, creator);
  }
  ~Register() = default;

  template <typename DerivedObjectType>
  static ObjectTypePtr DefaultCreator(Args... args) {
    return ObjectTypePtr(new (std::nothrow) DerivedObjectType(args...));
  }
};
}  // namespace cachetiling
}  // namespace optiling

#define DECLARE_REGISTRY_TYPE(registry_name, object_type, ...)                                            \
  optiling::cachetiling::Registry<std::shared_ptr<object_type>, ##__VA_ARGS__> &registry_name() noexcept; \
  typedef optiling::cachetiling::Register<std::shared_ptr<object_type>, ##__VA_ARGS__> Register##registry_name

#define DEFINE_REGISTRY_TYPE(registry_name, object_type, ...)                                              \
  optiling::cachetiling::Registry<std::shared_ptr<object_type>, ##__VA_ARGS__> &registry_name() noexcept { \
    static optiling::cachetiling::Registry<std::shared_ptr<object_type>, ##__VA_ARGS__> inst;              \
    return inst;                                                                                           \
  }

#define REGISTER_TYPE_CLASS(registry_name, type, derived_clazz)           \
  static Register##registry_name g_##registry_name##type(type, registry_name(), \
                                                   Register##registry_name::DefaultCreator<derived_clazz>)
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_UTIL_REGISTRY_H_