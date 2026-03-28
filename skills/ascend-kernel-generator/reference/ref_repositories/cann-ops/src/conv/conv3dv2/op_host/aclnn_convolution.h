/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CONVOLUTION_H_
#define OP_API_INC_CONVOLUTION_H_

#include "aclnn/aclnn_base.h"
#define ACLNN_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief convolution接口，计算并获取workspace大小
* @domain aclnn_ops_infer
*
* @param [in] input: npu，feature map
* device侧的aclTensor，数据类型浮点类型FLOAT16，FLOAT32，FLOAT64
* 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
* @param [in] weight: npu, kernels
* device侧的aclTensor，数据类型与input一致
* 支持非连续的Tensor，数据格式与input一致
* @param [in] bias: npu，偏置
* device侧的aclTensor，数据类型与input一致
* 支持非连续的Tensor，数据格式与input一致
* @param [in] stride: 步长
* int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：3D卷积的步长数组的有效长度是3位
* @param [in] padding: 补边
* int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：3D卷积的padding数组的有效长度是3位
* @param [in] dilation: kernel中元素的间隔，>1代表空洞卷积
* int64的数组，数组长度需等于input的维度-2（也等于kernel size -1），例：3D卷积的dilation数组的有效长度是3位
* @param [in] transposed: 是否转置
* bool，暂不支持，False即可
* @param [in] outputPadding：转置卷积时生效，对输出的补边
* int64的数组，数组长度需等于input的维度-2，值必须分别小于stride或者dilation的最大值，例：3D转置卷积的dilation数组的有效长度是3位
* @param [in] groups：分组数，表示从输入通道到输出通道的块链接个数
* int64，大于0且能整除input和output的通道数， input通道数 = weight通道数*groups
* @param [out] output: npu
* device侧的aclTensor，数据类型与input一致
* broadcast之后的shape，数据格式与input一致
* @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含算子计算流程。
* @return aclnnStatus: 返回状态码。
*/
ACLNN_API aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor* input, const aclTensor* weight,
                                                    const aclTensor* bias, const aclIntArray* stride,
                                                    const aclIntArray* padding, const aclIntArray* dilation,
                                                    bool transposed, const aclIntArray* outputPadding,
                                                    const int64_t groups, aclTensor* output, int8_t cubeMathType,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
* @brief convolution接口，进行kernellaunch
*
* @param [in] workspace: 在npu device侧申请的workspace内存起址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由aclnnConvolutionGetWorkspaceSize获取。
* @param [in] stream: acl stream流。
* @param [in] executor: op执行器，包含了算子计算流程。调用该接口后，executor不再可用
* @return aclnnStatus: 返回状态码。
*/
ACLNN_API aclnnStatus aclnnConvolution(void* workspace, const uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_H_