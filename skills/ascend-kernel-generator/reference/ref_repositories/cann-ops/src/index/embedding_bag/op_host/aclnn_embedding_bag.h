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
 * \file aclnn_embedding_bag.h
 * \brief
 */
#ifndef OP_API_INC_EMBEDDING_BAG_H_
#define OP_API_INC_EMBEDDING_BAG_H_

#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnEmbeddingBag第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：根据indices从weight中获取一组被聚合的数，然后根据offsets的便宜和mode指定的聚合模式进行max、sum、mean等聚合。
 *
 * @param [in] weight: npu device侧的aclTensor，数据类型支持float32 float16 bfloat16，必须是2D。
 * @param [in] indices: npu device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64。必须是1D或2D tensor。
 * @param [in] offsets: npu
 * device侧的aclTensor，数据类型支持UINT8、INT8、INT16、INT32、INT64，但和indices必须有一个是INT32或INT64,
 *                      在indices是1D时，offsets必须是1D tensor.
 * @param [in] scaleGradByFreq: bool数据类型，用于控制缩放梯度。
 * @param [in] mode: int64类型，用于控制聚合模式。
 * @param [in] sparse: bool数据类型，用于控制稀疏模式。
 * @param [in] perSampleWeights: npu device侧的aclTensor，指定样本权重。必须是1D
 * tensor，数据类型与weight一致，仅在sum可以不是null。
 * @param [in] includeLastOffset: bool数据类型，控制是否包含最后的偏移。
 * @param [in] paddingIdx: int64类型，取值范围是[-n,n-1]，其中n是weigit第一维元素个数。
 * @param [out] workspaceSize: 返回用户需要在npu侧申请的workspace大小。
 * @param [out] executor: 返回op执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor* weight, const aclTensor* indices,
                                                        const aclTensor* offsets, bool scaleGradByFreq, int64_t mode,
                                                        bool sparse, const aclTensor* perSampleWeights,
                                                        bool includeLastOffset, int64_t paddingIdx, aclTensor* output,
                                                        aclTensor* offset2bag, aclTensor* bagSize,
                                                        aclTensor* maxIndices, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnEmbeddingBag的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceIndexCopyGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_INDEX_COPY_H_
