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
#include "aclnn_add_operation.h"
#include "aclnn_add_custom.h"
using namespace common;
AddOperation::AddOperation(const std::string &name, AddAttrParam param){
    attrParam = param;
    opName_ = name;
}   
    
atb::SVector<int64_t> AddOperation::GetCopyTensorStride(atb::Dims &tensorDims)
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

std::shared_ptr<AclnnTensor> AddOperation::CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx)
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

atb::Status AddOperation::UpdateAclnnVariantPack(const atb::VariantPack &variantPack)
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

atb::Status AddOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) 
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
    auto ret = aclnnAddCustomGetWorkspaceSize(aclInTensors_.at(0)->tensor,
        aclInTensors_.at(1)->tensor,
        aclOutTensors_.at(0)->tensor,
        &workspaceSize_,
        &aclExecutor_);
    workspaceSize = workspaceSize_;
    return ret;
}

atb::Status AddOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context) {
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
    ret = aclnnAddCustom(workspace, workspaceSize_, aclExecutor_, stream);
    return ret;
}

atb::Status AddOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    outTensorDesc.at(0) = inTensorDesc.at(0);
    return atb::NO_ERROR;
}
