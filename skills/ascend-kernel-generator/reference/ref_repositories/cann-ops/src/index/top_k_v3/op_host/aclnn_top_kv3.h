/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TOP_K_V3_H_
#define OP_API_INC_TOP_K_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTopk的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：完成计算输入的k个极值及下标。
 *
 * @param [in] self: npu
 * npu device侧的aclTensor，数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE。
 * 支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] k:
 * int64_t类型整数。表示计算维度上输出的极值个数。取值范围为[0, self.size(dim)]。
 * @param [in] dim:
 * int64_t类型整数。表示计算维度。取值范围为[-self.dim(), self.dim())。
 * @param [in] largest:
 * bool类型数据。True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出。
 * @param [in] sorted:
 * bool类型数据。True表示输出结果需要排序（若largest为True则结果从大到小排序，若largest为False则结果从小到大排序），
 * False表示输出结果不排序，按输入时的数据顺序输出。
 * @param [in] valuesOut:
 * dnpu device侧的aclTensor，数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE，
 * 且数据类型与self保持一致，支持连续和非连续的Tensor，数据格式支持ND。
 * @param [in] indicesOut:
 * npu device侧的aclTensor，数据类型支持INT64。支持连续和非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor* self, int64_t k, int64_t dim, bool largest,
                                                bool sorted, aclTensor* valuesOut, aclTensor* indicesOut,
                                                uint64_t* workspaceSize, aclOpExecutor** executor);
/**
 * @brief aclnnTopk的第二段接口，用于执行计算。
 *
 * 算子功能：完成计算输入的k个极值及下标。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTopkGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTopk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TOP_K_V3_H_
