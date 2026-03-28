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
 * @file extension_add1.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

// register forward implementation for NPU device
at::Tensor my_op_impl_npu1(const at::Tensor &self, const at::Tensor &other)
{
    // alloc output memory
    at::Tensor result = at::Tensor(self);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnAddCustom, self, other, result);
    return result;
}

// register backward implementation for NPU device
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_npu1(const at::Tensor &self)
{
    at::Tensor result = at::Tensor(self); // Create output memory

    return {result, result};
}

// register forward implementation for Meta device
at::Tensor my_op_impl_meta1(const at::Tensor &self, const at::Tensor &other)
{
    return empty_like(self);
}

// register backward implementation for Meta device
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl_meta1(const at::Tensor &self)
{
    auto result = empty_like(self);
    return std::make_tuple(result, result);
}

// look up the implementation registered for different devices for this operation
at::Tensor my_op_impl1(const at::Tensor &self, const at::Tensor &other)
{
    static auto op =
        torch::Dispatcher::singleton().findSchemaOrThrow("myops::my_op1", "").typed<decltype(my_op_impl1)>();
    return op.call(self, other);
}

// look up the implementation registered for different devices for this operation
std::tuple<at::Tensor, at::Tensor> my_op_backward_impl1(const at::Tensor &self)
{
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("myops::my_op_backward1", "")
                         .typed<decltype(my_op_backward_impl1)>();
    return op.call(self);
}

// implement forward and backward binding by inheriting the torch::autograd::Function class
class MyAddFunction1 : public torch::autograd::Function<MyAddFunction1> {
public:
    static at::Tensor forward(AutogradContext *ctx, at::Tensor self, at::Tensor other)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        return my_op_impl1(self, other);
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto result = my_op_backward_impl1(grad_output);
        return {std::get<0>(result), std::get<1>(result)};
    }
};

// call apply() method when using it
at::Tensor my_op_impl_autograd1(const at::Tensor &self, const at::Tensor &other)
{
    return MyAddFunction1::apply(self, other);
}

// register forward and backward implementations for the NPU device
// the device name used by the NPU device in PyTorch 2.1 and above is PrivateUse1. 
// in versions below 2.1, XLA is used. If the version is below 2.1, PrivateUse1 needs to be changed to XLA.
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m)
{
    m.impl("my_op1", &my_op_impl_npu1);
    m.impl("my_op_backward1", &my_op_backward_impl_npu1);
}

// bind the NPU's autograd implementation to the operation
// if the version is below PyTorch 2.1, AutogradPrivateUse1 needs to be changed to AutogradXLA.
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m)
{
    m.impl("my_op1", &my_op_impl_autograd1);
}

// register forward and backward implementations for the Meta device
TORCH_LIBRARY_IMPL(myops, Meta, m)
{
    m.impl("my_op1", &my_op_impl_meta1);
    m.impl("my_op_backward1", &my_op_backward_impl_meta1);
}
