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
#include "aclnn_addcdiv_operation.h"
#include "aclnn_addcdiv.h"

using namespace common;

AddCdivOperation::AddCdivOperation(const std::string &name, AddCdivAttrParam param){
    attrParam = param;
    opName_ = name;
}   
    
atb::SVector<int64_t> AddCdivOperation::GetCopyTensorStride(atb::Dims &tensorDims)
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

std::shared_ptr<AclnnTensor> AddCdivOperation::CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx)
{
    auto aclnnTensor = std::make_shared<AclnnTensor>();
    aclnnTensor->tensorIdx = static_cast<int>(tensorIdx);
    aclnnTensor->needUpdateTensorDataPtr = true;
    aclnnTensor->atbTensor = atbTensor;
    aclnnTensor->strides = GetCopyTensorStride(atbTensor.desc.shape);
    
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

atb::Status AddCdivOperation::UpdateAclnnVariantPack(const atb::VariantPack &variantPack)
{
    // 更新inTensor的device地址
    // 无需更新aclScalar的device地址，因为最后一个输入张量可能是 aclScalar，而 aclScalar 不需要像普通张量那样更新设备地址
    for (size_t i = 0; i < aclInTensors_.size() - 1; ++i) {
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

atb::Status AddCdivOperation::Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) 
{
    aclInTensors_.resize(GetInputNum());
    size_t scalarIndex = 3;

    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        // 不需要使用 CreateAclnnTensor 函数来创建，而是使用 aclCreateScalar 函数单独创建
        // 因为索引为 scalarIndex 的输入是 aclScalar，它的创建方式与普通张量不同
        if (i == scalarIndex)
        {
            continue;
        }

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
    printf("creat output tensor success\n"); 
    aclScalar *aclInScalar = aclCreateScalar(variantPack.inTensors.at(3).deviceData, variantPack.inTensors.at(3).desc.dtype);
    if (aclInScalar == nullptr) 
    {
        printf("创建Scalar失败\n");
        return atb::ERROR_INTERNAL_ERROR;
    }

    if (aclInTensors_.at(1)->tensor == nullptr || aclInTensors_.at(2)->tensor == nullptr || aclOutTensors_.at(0)->tensor == nullptr) 
    {
        printf("检测到空指针\n");
        return atb::ERROR_INTERNAL_ERROR;
    }

    auto ret = aclnnAddcdivGetWorkspaceSize(aclInTensors_.at(0)->tensor,
        aclInTensors_.at(1)->tensor,
        aclInTensors_.at(2)->tensor,
        aclInScalar,
        aclOutTensors_.at(0)->tensor,
        &workspaceSize_,
        &aclExecutor_);

    workspaceSize = workspaceSize_;
    return ret;
}

atb::Status AddCdivOperation::Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context) {
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
    ret = aclnnAddcdiv(workspace, workspaceSize_, aclExecutor_, stream);
    return ret;
}

atb::Status AddCdivOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const
{
    outTensorDesc.at(0) = inTensorDesc.at(0);
    return atb::NO_ERROR;
}
