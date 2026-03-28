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
 * \file kernel_run_context_maker.cc
 * \brief
 */
#include <utils/op_desc_utils.h>
#include <utils/graph_utils.h>
#include "compute_graph.h"
#include "lowering/bg_kernel_context_extend.h"
#include "runtime/tiling_context.h"
#include "kernel_run_context_maker.h"

using namespace optiling;

KernelRunContextHolder BuildKernelRunContext(size_t input_num, size_t output_num) {
  return KernelRunContextMaker().KernelIONum(input_num, output_num).Build();
}

KernelRunContextMaker& KernelRunContextMaker::SetOpType(const std::string op_type) {
  optype_ = op_type;
  return *this;
}

KernelRunContextMaker& KernelRunContextMaker::KernelIONum(size_t input_num, size_t output_num) {
  kernel_input_num_ = input_num;
  kernel_output_num_ = output_num;
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::NodeIoNum(size_t input_num, size_t output_num) {
  node_input_num_ = input_num;
  node_output_num_ = output_num;
  node_input_tds_.resize(input_num);
  node_output_tds_.resize(output_num);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::IrInputNum(size_t input_num) {
  ir_instance_num_ = std::vector<uint32_t>(input_num, 1);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::IrInstanceNum(std::vector<uint32_t> instance_num) {
  ir_instance_num_ = std::move(instance_num);
  return *this;
}
ge::NodePtr KernelRunContextMaker::FakeNode() {
  auto op_desc = std::make_shared<ge::OpDesc>(optype_.c_str(), optype_.c_str());
  size_t input_index = 0;
  for (size_t ir_index = 0; ir_index < ir_instance_num_.size(); ++ir_index) {
    auto ir_ins_num = ir_instance_num_[ir_index];
    auto prefix = "x_" + std::to_string(ir_index) + "_";
    op_desc->AppendIrInput(prefix, ge::kIrInputDynamic);
    for (size_t i = 0; i < ir_ins_num; ++i, ++input_index) {
      auto td = ge::GeTensorDesc();
      if (node_input_tds_.size() > input_index) {
        td.SetOriginFormat(node_input_tds_[input_index].GetOriginFormat());
        td.SetFormat(node_input_tds_[input_index].GetStorageFormat());
        td.SetDataType(node_input_tds_[input_index].GetDataType());
        td.SetOriginDataType(node_input_tds_[input_index].GetDataType());
      }
      op_desc->AddInputDesc(prefix + std::to_string(i), td);
    }
  }
  for (size_t i = 0; i < node_output_num_; ++i) {
    auto td = ge::GeTensorDesc();
    if (node_output_tds_.size() > i) {
      td.SetOriginFormat(node_output_tds_[i].GetOriginFormat());
      td.SetFormat(node_output_tds_[i].GetStorageFormat());
      td.SetDataType(node_output_tds_[i].GetDataType());
      td.SetOriginDataType(node_output_tds_[i].GetDataType());
    }
    op_desc->AddOutputDesc("y" + std::to_string(i), td);
  }
  for (const auto& attr : attrs_) {
    op_desc->AppendIrAttrName(attr.first);
    op_desc->SetAttr(attr.first, attr.second);
  }
  graph_ = std::make_shared<ge::ComputeGraph>("tmp");
  auto fake_node = graph_->AddNode(op_desc);
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    auto op_data = ge::OpDescBuilder(std::to_string(i), "Data").AddInput("x").AddOutput("y").Build();
    auto data_node = graph_->AddNode(op_data);
    ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), fake_node->GetInDataAnchor(i));
  }
  return fake_node;
}
KernelRunContextHolder KernelRunContextMaker::Build() {
  KernelRunContextHolder holder;
  holder.kernel_input_num = kernel_input_num_;
  holder.kernel_output_num = kernel_output_num_;
  size_t size = sizeof(KernelRunContext) + sizeof(AsyncAnyValue*) * (kernel_input_num_ + kernel_output_num_);
  holder.context_holder = std::unique_ptr<uint8_t[]>(new uint8_t[size]);
  errno_t ret = memset_s(holder.context_holder.get(), size, 0xff, size);
  if (ret != 0) {
    throw std::runtime_error("Failed to initialize memory with memset_s");
  }
  holder.value_holder.resize(kernel_input_num_ + kernel_output_num_);
  holder.compute_node_extend_holder = gert::bg::CreateComputeNodeInfo(FakeNode(), holder.buffer_pool);
  if (holder.compute_node_extend_holder) {
    auto compute_node_info = reinterpret_cast<gert::ComputeNodeInfo*>(holder.compute_node_extend_holder.get());
    compute_node_info->SetNodeName(
        holder.buffer_pool.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeName())));
    compute_node_info->SetNodeType(
        holder.buffer_pool.GetBufById(reinterpret_cast<size_t>(compute_node_info->GetNodeType())));
  }

  holder.context = reinterpret_cast<KernelRunContext*>(holder.context_holder.get());
  holder.context->input_size = kernel_input_num_;
  holder.context->output_size = kernel_output_num_;
  holder.context->compute_node_info =
      holder.compute_node_extend_holder ? holder.compute_node_extend_holder.get() : nullptr;
  holder.context->output_start = &(holder.context->values[holder.context->input_size]);

  for (size_t i = 0; i < kernel_input_num_ + kernel_output_num_; ++i) {
    holder.context->values[i] = &holder.value_holder[i];
  }

  if (inputs_.size() == kernel_input_num_) {
    for (size_t i = 0; i < inputs_.size(); ++i) {
      holder.value_holder[i].data.pointer = inputs_[i];
    }
  }
  if (outputs_.size() == kernel_output_num_) {
    for (size_t i = 0; i < outputs_.size(); ++i) {
      holder.value_holder[i + kernel_input_num_].data.pointer = outputs_[i];
    }
  }
  return holder;
}
KernelRunContextMaker& KernelRunContextMaker::NodeInputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                          ge::Format storage_format) {
  node_input_tds_[index].SetDataType(dt);
  node_input_tds_[index].SetOriginFormat(origin_format);
  node_input_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::NodeOutputTd(int32_t index, ge::DataType dt, ge::Format origin_format,
                                                           ge::Format storage_format) {
  node_output_tds_[index].SetDataType(dt);
  node_output_tds_[index].SetOriginFormat(origin_format);
  node_output_tds_[index].SetStorageFormat(storage_format);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::Inputs(std::vector<void*> inputs) {
  inputs_ = std::move(inputs);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::Outputs(std::vector<void*> outputs) {
  outputs_ = std::move(outputs);
  return *this;
}
KernelRunContextMaker& KernelRunContextMaker::NodeAttrs(
    std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
  attrs_ = std::move(keys_to_value);
  return *this;
}

TilingContextMaker& TilingContextMaker::NodeIoNum(size_t input_num, size_t output_num) {
  base_maker_.KernelIONum(input_num + output_num + kInputsAppendEnd, gert::TilingContext::kOutputNum);
  base_maker_.NodeIoNum(input_num, output_num);
  return *this;
}
TilingContextMaker& TilingContextMaker::InputShapes(std::vector<void*> input_shapes) {
  input_shapes_ = std::move(input_shapes);
  UpdateInputs();
  return *this;
}
TilingContextMaker& TilingContextMaker::OutputShapes(std::vector<void*> output_shapes) {
  output_shapes_ = std::move(output_shapes);
  UpdateInputs();
  return *this;
}
TilingContextMaker& TilingContextMaker::CompileInfo(void* compile_info) {
  compile_info_ = compile_info;
  UpdateInputs();
  return *this;
}
TilingContextMaker& TilingContextMaker::PlatformInfo(void* platform_info) {
  platform_info_ = platform_info;
  UpdateInputs();
  return *this;
}
TilingContextMaker& TilingContextMaker::TilingData(void* tiling_data) {
  outputs_[gert::TilingContext::kOutputTilingData] = tiling_data;
  base_maker_.Outputs(outputs_);
  return *this;
}
TilingContextMaker& TilingContextMaker::Workspace(gert::ContinuousVector* workspace) {
  outputs_[gert::TilingContext::kOutputWorkspace] = workspace;
  base_maker_.Outputs(outputs_);
  return *this;
}
TilingContextMaker& TilingContextMaker::ConstInput(
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors) {
  std::vector<void*> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  for (size_t i = 0; i < const_tensors.size(); i++) {
    inputs[const_tensors[i].first] = const_tensors[i].second.get();
  }
  for (const auto output_shape : output_shapes_) {
    inputs.push_back(output_shape);
  }
  inputs.push_back(compile_info_);  // kInputsCompileInfo
  inputs.push_back(platform_info_);
  inputs.push_back(nullptr);  // kInputsTilingFunc
  base_maker_.Inputs(std::move(inputs));
  return *this;
}
KernelRunContextHolder TilingContextMaker::Build() {
  return base_maker_.Build();
}
void TilingContextMaker::UpdateInputs() {
  std::vector<void*> inputs;
  for (const auto input_shape : input_shapes_) {
    inputs.push_back(input_shape);
  }
  for (const auto output_shape : output_shapes_) {
    inputs.push_back(output_shape);
  }
  inputs.push_back(compile_info_);  // kInputsCompileInfo
  inputs.push_back(platform_info_);
  inputs.push_back(nullptr);  // kInputsTilingFunc
  base_maker_.Inputs(std::move(inputs));
}
