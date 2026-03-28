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
 * @file register.cpp
 */
#include <torch/extension.h>
#include <torch/library.h>

#include "function.h"

// Register two schema: my_op and my_op_backward in the myops namespace
TORCH_LIBRARY(myops, m)
{
    m.def("my_op(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward(Tensor self) -> (Tensor, Tensor)");
    m.def("my_op1(Tensor self, Tensor other) -> Tensor");
    m.def("my_op_backward1(Tensor self) -> (Tensor, Tensor)");
}

// bind c++ interface to python interface by pybind
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_custom", &my_op_impl_autograd, "x + y");
    m.def("add_custom1", &my_op_impl_autograd1, "x + y");
}
