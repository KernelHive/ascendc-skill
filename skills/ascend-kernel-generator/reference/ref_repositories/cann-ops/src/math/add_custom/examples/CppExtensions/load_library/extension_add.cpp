/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file extension_add.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// register forward implementation for NPU device
at::Tensor my_op_impl_npu(const at::Tensor &self, const at::Tensor &other)
{
    // alloc output memory
    at::Tensor result = at::Tensor(self);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
    return result;
}

// register backward implementation for NPU device
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_npu(const at::Tensor &self)
{
    at::Tensor result = at::Tensor(self); // Create output memory
    return {result, result};
}

// register forward implementation for Meta device
at::Tensor my_op_impl_meta(const at::Tensor &self, const at::Tensor &other)
{
    return empty_like(self);
}

// register backward implementation for Meta device
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_meta(const at::Tensor &self)
{
    auto result = empty_like(self);
    return std::make_tuple(result, result);
}

// implement forward and backward binding by inheriting the torch::autograd::Function class
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
public:
    static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        return my_op_impl_npu(self, other);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto result = my_op_backward_impl_npu(grad_output);
        return {std::get<0>(result), std::get<1>(result)};
    }
};

// call apply() method when using it
at::Tensor my_op_impl_autograd(const at::Tensor &self, const at::Tensor &other)
{
    return MyAddFunction::apply(self, other);
}

// register the schemas for my_op and my_op_backward in the myops namespace
TORCH_LIBRARY(myops, m)
{
    m.def("my_op(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward(Tensor self) -> (Tensor, Tensor)");
}

// register forward and backward implementations for the NPU device
// the device name used by the NPU device in PyTorch 2.1 and above is PrivateUse1. 
// in versions below 2.1, XLA is used. If the version is below 2.1, PrivateUse1 needs to be changed to XLA.
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("my_op", &my_op_impl_npu);
    m.impl("my_op_backward", &my_op_backward_impl_npu);
}

// bind the NPU's autograd implementation to the operation
// if the version is below PyTorch 2.1, AutogradPrivateUse1 needs to be changed to AutogradXLA.
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m)
{
    m.impl("my_op", &my_op_impl_autograd);
}

// register forward and backward implementations for the Meta device
TORCH_LIBRARY_IMPL(myops, Meta, m)
{
    m.impl("my_op", &my_op_impl_meta);
    m.impl("my_op_backward", &my_op_backward_impl_meta);
}
