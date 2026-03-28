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
 * \file upsample_linear1d_l0.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_LINEAR1D_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_LINEAR1D_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *UpsampleLinear1dNcdhw(const aclTensor *x, const aclIntArray *outputSize, const bool alignCorners,
    const aclTensor *y, const double scale, aclOpExecutor *executor);
}

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_UPSAMPLE_LINEAR1D_H_
