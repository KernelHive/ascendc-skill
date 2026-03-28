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
 * @file function.h
 */
#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <ATen/ATen.h>

at::Tensor my_op_impl_autograd(const at::Tensor &self, const at::Tensor &other);
at::Tensor my_op_impl_autograd1(const at::Tensor &self, const at::Tensor &other);

#endif //  FUNCTION_H_
