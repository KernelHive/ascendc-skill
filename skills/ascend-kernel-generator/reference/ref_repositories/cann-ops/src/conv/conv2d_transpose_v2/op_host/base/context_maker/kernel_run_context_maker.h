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
 * \file kernel_run_context_maker.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_BASE_CONTEXT_MAKER_KERNEL_RUN_CONTEXT_MAKER_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_BASE_CONTEXT_MAKER_KERNEL_RUN_CONTEXT_MAKER_H_

#include <memory>
#include <vector>
#include <cstring>
#include <cstddef>
#include "runtime/kernel_run_context.h"
#include "runtime/context_extend.h"
#include "runtime/storage_shape.h"
#include "runtime/tiling_context.h"
#include "buffer_pool.h"
#include "any_value.h"
#include "node.h"

namespace optiling {
struct KernelRunContextHolder {
  template <typename T>
  T* GetContext() {
    return reinterpret_cast<T*>(context);
  }
  gert::ComputeNodeInfo* MutableComputeNodeInfo() {
    return reinterpret_cast<gert::ComputeNodeInfo*>(compute_node_extend_holder.get());
  }
  size_t kernel_input_num;
  size_t kernel_output_num;
  std::unique_ptr<uint8_t[]> context_holder;
  std::vector<AsyncAnyValue> value_holder;
  std::unique_ptr<uint8_t[]> compute_node_extend_holder;
  gert::bg::BufferPool buffer_pool;
  KernelRunContext* context;
};
KernelRunContextHolder BuildKernelRunContext(size_t input_num, size_t output_num);

class KernelRunContextMaker {
 public:
  KernelRunContextMaker() = default;
  KernelRunContextMaker& SetOpType(const std::string op_type);
  KernelRunContextMaker& KernelIONum(size_t input_num, size_t output_num);
  KernelRunContextMaker& NodeIoNum(size_t input_num, size_t output_num);
  KernelRunContextMaker& IrInputNum(size_t input_num);
  KernelRunContextMaker& IrInstanceNum(std::vector<uint32_t> instance_num);
  KernelRunContextMaker& NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                     ge::Format storage_format);
  KernelRunContextMaker& NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                      ge::Format storage_format);
  KernelRunContextMaker& NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value);
  KernelRunContextMaker& Inputs(std::vector<void*> inputs);
  KernelRunContextMaker& Outputs(std::vector<void*> outputs);

  KernelRunContextHolder Build();

 private:
  ge::NodePtr FakeNode();

 private:
  size_t kernel_input_num_;
  size_t kernel_output_num_;
  size_t node_input_num_;
  size_t node_output_num_;
  std::string optype_ = "node";
  std::vector<uint32_t> ir_instance_num_;
  std::vector<gert::CompileTimeTensorDesc> node_input_tds_;
  std::vector<gert::CompileTimeTensorDesc> node_output_tds_;
  std::vector<void*> inputs_;
  std::vector<void*> outputs_;
  std::vector<std::pair<std::string, ge::AnyValue>> attrs_;
  ge::ComputeGraphPtr graph_;
};

class TilingContextMaker {
 public:
  TilingContextMaker& SetOpType(const std::string op_type) {
    base_maker_.SetOpType(op_type);
    return *this;
  }
  TilingContextMaker& NodeIoNum(size_t input_num, size_t output_num);
  TilingContextMaker& IrInputNum(size_t input_num) {
    base_maker_.IrInputNum(input_num);
    return *this;
  }
  TilingContextMaker& IrInstanceNum(std::vector<uint32_t> instance_num) {
    base_maker_.IrInstanceNum(std::move(instance_num));
    return *this;
  }
  TilingContextMaker& NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format, ge::Format storage_format) {
    base_maker_.NodeInputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextMaker& NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                   ge::Format storage_format) {
    base_maker_.NodeOutputTd(index, dt, origin_format, storage_format);
    return *this;
  }
  TilingContextMaker& NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
    base_maker_.NodeAttrs(std::move(keys_to_value));
    return *this;
  }
  TilingContextMaker& InputShapes(std::vector<void*> input_shapes);
  TilingContextMaker& OutputShapes(std::vector<void*> output_shapes);
  TilingContextMaker& CompileInfo(void* compile_info);
  TilingContextMaker& PlatformInfo(void* platform_info);
  TilingContextMaker& TilingData(void* tiling_data);
  TilingContextMaker& Workspace(gert::ContinuousVector* workspace);
  TilingContextMaker& ConstInput(std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors);

  KernelRunContextHolder Build();

 private:
  void UpdateInputs();

 private:
  enum InputsAppend { kInputsCompileInfo, kInputsPlatformInfo, kInputsTilingFunc, kInputsAppendEnd };

  KernelRunContextMaker base_maker_;
  std::vector<void*> input_shapes_;
  std::vector<void*> output_shapes_;
  std::vector<void*> outputs_{gert::TilingContext::kOutputNum};

  void* platform_info_{nullptr};
  void* compile_info_{nullptr};
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_BASE_CONTEXT_MAKER_KERNEL_RUN_CONTEXT_MAKER_H_
