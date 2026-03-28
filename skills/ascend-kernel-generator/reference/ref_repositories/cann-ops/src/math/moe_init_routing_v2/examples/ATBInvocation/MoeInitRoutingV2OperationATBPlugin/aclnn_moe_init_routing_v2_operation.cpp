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
 * @file aclnn_add_operation.cpp
 */
#include "aclnn_moe_init_routing_v2_operation.h"
#include "aclnn_moe_init_routing_v2.h"
using namespace common;
MoeInitRoutingV2Operation::MoeInitRoutingV2Operation(const std::string &name, MoeInitRoutingV2AttrParam param){
    attrParam = param;
    opName_ = name;
}   
    
atb::SVector<int64_t> MoeInitRoutingV2Operation::GetCopyTensorStride(atb::Dims &tensorDims)
{
    atb::SVector<int64_t> tmpStrides(tensorDims.dimNum, 1);
    if (tensorDims.dimNum > 8) {  // 8: tensor最大维度数量
        printf("tensor's dimNum is larger than 8, GetCopyTensorStride failed.");
        return tmpStrides;
    }
    for (int64_t i = static_cast<int64_t>(tensorDims.dimNum) - 2; i >= 0; i--) {
        tmpStrides[i] = (tensorDims.dims[i + 1] * tmpStrides[i + 1]);
    }
    return tmpStrides;
}

std::shared_ptr<AclnnTensor> MoeInitRoutingV2Operation::CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx)
{
    auto aclnnTensor = std::make_shared<AclnnTensor>();
    aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    // 创建Aclnn tensor
    aclnnTensor->tensor = aclCreateTensor(atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.desc.dtype,
        aclnnTensor->strides.data(),
        0,
        atbTensor.desc.format,
        atbTensor.desc.shape.dims,
        atbTensor.desc.shape.dimNum,
        atbTensor.deviceData);
    return aclnnTensor;
}

atb::Status MoeInitRoutingV2Operation::UpdateAclnnVariantPack(const atb::VariantPack &variantPack)
{
    // 更新inTensor的device地址
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        int ret = -1;
        if (!aclInTensors_[i]->needUpdateTensorDataPtr) {
            continue;
        }
        aclInTensors_[i]->atbTensor = variantPack.inTensors.at(i);
        ret = aclSetInputTensorAddr(aclExecutor_,
            aclInTensors_[i]->tensorIdx,
            aclInTensors_[i]->tensor,
            aclInTensors_[i]->atbTensor.deviceData);
        if (ret != 0) {
            printf("set input fail");
            return atb::ERROR_CANN_ERROR;
        }
    }
    // 更新outTensor的device地址
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        int ret = -1;
        if (!aclOutTensors_[i]->needUpdateTensorDataPtr) {
            continue;
        }
        aclOutTensors_[i]->atbTensor = variantPack.outTensors.at(i);
        ret = aclSetOutputTensorAddr(aclExecutor_,
            aclOutTensors_[i]->tensorIdx,
            aclOutTensors_[i]->tensor,
            aclOutTensors_[i]->atbTensor.deviceData);
        if (ret != 0) {
            printf("set output fail");
            return atb::ERROR_CANN_ERROR;
        }
    }
    return atb::NO_ERROR;
}

atb::Status MoeInitRoutingV2Operation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) 
{
    aclInTensors_.resize(GetInputNum());
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        auto aclnnTensor = CreateAclnnTensor(variantPack.inTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            printf("creat input tensor %ld fail", i);
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclInTensors_[i] = aclnnTensor;
    }
    aclOutTensors_.resize(GetOutputNum());
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        auto aclnnTensor = CreateAclnnTensor(variantPack.outTensors.at(i), i);
        if (aclnnTensor->tensor == nullptr) {
            printf("creat output tensor %ld fail", i);
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclOutTensors_[i] = aclnnTensor;
    }
    if(attrParam.drop_pad_mode == 0 && attrParam.expert_tokens_count_or_cumsum_flag > 0 ){
        auto ret = aclnnMoeInitRoutingV2GetWorkspaceSize(aclInTensors_.at(0)->tensor,
            aclInTensors_.at(1)->tensor,
            attrParam.active_num,
            attrParam.expert_capacity,
            attrParam.expert_num,
            attrParam.drop_pad_mode,
            attrParam.expert_tokens_count_or_cumsum_flag,
            attrParam.expert_tokens_before_capacity_flag,
            attrParam.start_expertId,
            attrParam.end_expertId,
            attrParam.device_id,
            aclOutTensors_.at(0)->tensor,
            aclOutTensors_.at(1)->tensor,
            aclOutTensors_.at(2)->tensor,
            nullptr,
            &workspaceSize_,
            &aclExecutor_);
            workspaceSize = workspaceSize_;
        return ret;
    }
    else if(attrParam.drop_pad_mode == 1 && attrParam.expert_tokens_before_capacity_flag == true){
        auto ret = aclnnMoeInitRoutingV2GetWorkspaceSize(aclInTensors_.at(0)->tensor,
            aclInTensors_.at(1)->tensor,
            attrParam.active_num,
            attrParam.expert_capacity,
            attrParam.expert_num,
            attrParam.drop_pad_mode,
            attrParam.expert_tokens_count_or_cumsum_flag,
            attrParam.expert_tokens_before_capacity_flag,
            attrParam.start_expertId,
            attrParam.end_expertId,
            attrParam.device_id,
            aclOutTensors_.at(0)->tensor,
            aclOutTensors_.at(1)->tensor,
            nullptr,
            aclOutTensors_.at(2)->tensor,
            &workspaceSize_,
            &aclExecutor_);
            workspaceSize = workspaceSize_;
        return ret;
    }
    else{
        auto ret = aclnnMoeInitRoutingV2GetWorkspaceSize(aclInTensors_.at(0)->tensor,
            aclInTensors_.at(1)->tensor,
            attrParam.active_num,
            attrParam.expert_capacity,
            attrParam.expert_num,
            attrParam.drop_pad_mode,
            attrParam.expert_tokens_count_or_cumsum_flag,
            attrParam.expert_tokens_before_capacity_flag,
            attrParam.start_expertId,
            attrParam.end_expertId,
            attrParam.device_id,
            aclOutTensors_.at(0)->tensor,
            aclOutTensors_.at(1)->tensor,
            nullptr,
            nullptr,
            &workspaceSize_,
            &aclExecutor_);
            workspaceSize = workspaceSize_;
        return ret;
    }
}

atb::Status MoeInitRoutingV2Operation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context) {
    aclrtStream stream = context->GetExecuteStream();
    if (!stream) {
        printf("get stream fail");
        return atb::ERROR_INVALID_PARAM;
    }
    // 更新数据传入的地址
    int ret = UpdateAclnnVariantPack(variantPack);
    if (ret != 0) {
        printf("UpdateAclnnVariantPack fail");
        return atb::ERROR_CANN_ERROR;
    }
    ret = aclnnMoeInitRoutingV2(workspace, workspaceSize_, aclExecutor_, stream);
    return ret;
}

atb::Status MoeInitRoutingV2Operation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    int64_t n = inTensorDesc.at(0).shape.dims[0];
    int64_t cols = inTensorDesc.at(0).shape.dims[1];
    int64_t k = inTensorDesc.at(1).shape.dims[1];
    int64_t expandedRowIdxNum = n * k;
    int64_t outActiveNum = attrParam.active_num > 0 ? std::min(n * k, attrParam.active_num) : n * k;
    outTensorDesc.at(0).dtype = inTensorDesc.at(0).dtype;
    outTensorDesc.at(0).format = inTensorDesc.at(0).format;
    outTensorDesc.at(1).dtype = ACL_INT32; 
    outTensorDesc.at(1).format = inTensorDesc.at(0).format;
    const int INPUT_DIMS_FIRST = 3;    
    const int INPUT_INDEX_THIRD = 2;
    const int OUTPUT_INDEX_THIRD = 2;
    if (attrParam.drop_pad_mode > 0) {
        outTensorDesc.at(0).shape.dimNum = INPUT_DIMS_FIRST; // 第一个输出是三维tensor
        outTensorDesc.at(0).shape.dims[0] = attrParam.expert_num;
        outTensorDesc.at(0).shape.dims[1] = attrParam.expert_capacity;
        outTensorDesc.at(0).shape.dims[INPUT_INDEX_THIRD] = cols; // 第一个输出的第三维等于第一个输入的第二维
    }
    else{
        outTensorDesc.at(0).shape.dimNum = 2; // 输出第一个tensor是2维
        outTensorDesc.at(0).shape.dims[0] = outActiveNum;
        outTensorDesc.at(0).shape.dims[1] = cols;
    }
    outTensorDesc.at(1).shape.dimNum = 1;
    outTensorDesc.at(1).shape.dims[0] = expandedRowIdxNum;
    if(attrParam.drop_pad_mode == 0 && attrParam.expert_tokens_count_or_cumsum_flag > 0){
        outTensorDesc.at(OUTPUT_INDEX_THIRD).dtype = ACL_INT32; // 第二个输出的类型为int32
        outTensorDesc.at(OUTPUT_INDEX_THIRD).format = inTensorDesc.at(0).format; // 第二个输出的格式和第一个输出相等
        outTensorDesc.at(OUTPUT_INDEX_THIRD).shape.dimNum = 1; // 第二个输出是一维tensor
        outTensorDesc.at(OUTPUT_INDEX_THIRD).shape.dims[0] = attrParam.expert_num; // 第二个输出的第一维等于expert_num
    }
    else if(attrParam.drop_pad_mode == 0 && attrParam.expert_tokens_before_capacity_flag){
        outTensorDesc.at(OUTPUT_INDEX_THIRD).dtype = ACL_INT32;  // 第二个输出的类型为int32
        outTensorDesc.at(OUTPUT_INDEX_THIRD).format = inTensorDesc.at(0).format; // 第二个输出的格式和第一个输出相等
        outTensorDesc.at(OUTPUT_INDEX_THIRD).shape.dimNum = 1; // 第二个输出是一维tensor
        outTensorDesc.at(OUTPUT_INDEX_THIRD).shape.dims[0] = attrParam.expert_num;// 第二个输出的第一维等于expert_num
    }

    return atb::NO_ERROR;
}
