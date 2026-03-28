/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_
#define OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIndexSelect的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素向下取整的结果
 * 计算公式：
 * 以输入为三维张量为例：
 *   x=$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$
 *   idx=[1, 0],
 * dim为0, index_select(0, idx)：   I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$
 * dim为1, index_select(1, idx)：   J=index[j];  &nbsp;&nbsp;&nbsp;    y$[l][j][n]$ = x$[l][J][n]$
 * dim为2, index_select(2, idx)：   K=index[k]; &nbsp;  y$[l][m][k]$ = x$[l][m][K]$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(Self)] -->B([l0op::Contiguous])
 *   B --> In_0([l0op::cast]) --> Op([GatherV2])
 *   In_1[(index)] --> con([l0op::Contiguous])--> Op
 *   In_2(dim) --> a(dimVec) --> Op
 *   Op --> C([l0op::cast]) --> D([l0op::ViewCopy])  --> Out[(out)]
 * ```

 *
 * @param [in] self: 原始张量。npu device侧的aclTensor，
 *
 数据类型支持FLOAT、FLOAT16、BFLOAT16、INT64、INT32、INT16、INT8、UINT8、BOOL、DOUBLE、COMPLEX64、COMPLEX128，数据格式支持ND，支持非连续的Tensor。
 * @param [in] dim: host侧的int64。
 * @param [in] index: indecies，切片用，npu device侧的aclTensor。
 * 数据类型支持INT64、INT32，数据格式支持ND，维度是0D或1D。 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor* self, int64_t dim, const aclTensor* index,
                                                       aclTensor* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnIndexSelect的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnIndexSelectGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIndexSelect(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_INDEX_SELECT_H_
