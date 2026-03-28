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
 * @file aclnn_eye_operation.h
 */
#ifndef ACLNN_EYE_OPERATION_H
#define ACLNN_EYE_OPERATION_H
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <vector>

#include "atb/infer_op_params.h"
namespace common{
    struct EyeAttrParam
    {
        uint64_t num_rows;
        uint64_t num_columns = 0;
        std::vector<int64_t> batchShape = {1};
        aclIntArray* batch_shape = aclCreateIntArray(batchShape.data(),batchShape.size());
        uint64_t dtype = 0;
    };

    struct AclnnTensor
    {
    public:
        atb::Tensor atbTensor; //
        aclTensor *tensor = nullptr;
        int tensorIdx = -1; // aclTensor在aclExecutor中的index
        bool needUpdateTensorDataPtr = false;
        atb::SVector<int64_t> strides = {};
    };
    class EyeOperation: public atb::Operation{
    public:
        EyeOperation(const std::string &name, EyeAttrParam param);
        atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
        atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, 
                                                uint64_t workspaceSize, atb::Context *context) override;
        atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const;
        atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);
        std::shared_ptr<AclnnTensor> CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx);
        atb::Status UpdateAclnnVariantPack(const atb::VariantPack &variantPack);
        static constexpr int inputNum = 1;   // 算子入参个数
        static constexpr int outputNum = 1;  // 算子出参个数
        uint32_t GetInputNum() const
        {
            return inputNum; 
        }

        uint32_t GetOutputNum() const
        {
            return outputNum; 
        }
        std::string GetName() const
        {
            return opName_;
        }

        aclOpExecutor *aclExecutor_ = nullptr;
        EyeAttrParam attrParam;
        std::string opName_;
        uint64_t workspaceSize_;

        atb::SVector<std::shared_ptr<AclnnTensor>> aclInTensors_;
        atb::SVector<std::shared_ptr<AclnnTensor>> aclOutTensors_;
    };
}
#endif