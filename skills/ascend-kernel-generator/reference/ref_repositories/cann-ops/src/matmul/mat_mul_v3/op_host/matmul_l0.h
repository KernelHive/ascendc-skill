/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *MatMulNd(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                          const bool transposeX1, const bool transposeX2, const bool offsetX,
                          const int64_t opImplModeEnum, aclOpExecutor *executor);

const aclTensor *MatMulNdFp162Fp32(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *offsetW, const bool transposeX1, const bool transposeX2,
                                   const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor *executor);

const aclTensor *MatMulNz(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                          const bool transposeX1, const bool transposeX2, const bool offsetX,
                          const int64_t opImplModeEnum, aclOpExecutor *executor);
/*
包括ND输入NZ输出和NZ输入NZ输出两种切K模式
*/
const aclTensor *MatMulNzFp162Fp32(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *offsetW, const bool transposeX1, const bool transposeX2,
                                   const bool offsetX, const int64_t opImplModeEnum, aclOpExecutor *executor);

// 输入self=ND，输入mat2=NZ
const aclTensor *MatMulNdNz(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias, const aclTensor *offsetW,
                            const bool transposeX1, const bool transposeX2, const bool offsetX,
                            const int64_t opImplModeEnum, aclOpExecutor *executor);

// 输入self=NZ，输入mat2=NZ, 输出ND
const aclTensor* MatMulNzNzNd(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* offsetW,
                              const bool transposeX1, const bool transposeX2, const bool offsetX,
                              const int64_t opImplModeEnum, aclOpExecutor* executor);

const aclTensor *MatMulV3Nd(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                            const bool transposeX1, const bool transposeX2, const bool offsetX, const bool enableHf32,
                            aclOpExecutor* executor);

}  // namespace l0op

#endif  // PTA_NPU_OP_API_INC_LEVEL0_OP_MATMUL_OP_H_

