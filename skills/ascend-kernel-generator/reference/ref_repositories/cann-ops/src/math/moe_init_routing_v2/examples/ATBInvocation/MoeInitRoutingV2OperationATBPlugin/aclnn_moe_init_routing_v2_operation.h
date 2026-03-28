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
 * @file aclnn_add_operation.h
 */
#ifndef ACLNN_MOE_INIT_ROUTING_V2_OPERATION_H
#define ACLNN_MOE_INIT_ROUTING_V2_OPERATION_H
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include <iostream>

#include "atb/infer_op_params.h"

namespace common{
    struct MoeInitRoutingV2AttrParam
    {
        int64_t active_num = 0;
        int64_t expert_capacity = 0;
        int64_t expert_num = 0;
        int64_t drop_pad_mode = 0;
        int64_t expert_tokens_count_or_cumsum_flag = 0;
        bool expert_tokens_before_capacity_flag = false;
        int64_t start_expertId = 0;
        int64_t end_expertId = 0;
        int64_t device_id = 0;
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

    class MoeInitRoutingV2Operation: public atb::Operation{
    public:
        MoeInitRoutingV2Operation(const std::string &name, MoeInitRoutingV2AttrParam param);
        atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
        atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, 
                                                uint64_t workspaceSize, atb::Context *context) override;
        atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const;
        atb::SVector<int64_t> GetCopyTensorStride(atb::Dims &tensorDims);
        std::shared_ptr<AclnnTensor> CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx);
        atb::Status UpdateAclnnVariantPack(const atb::VariantPack &variantPack);
        static constexpr uint32_t inputNum = 2;   // 算子入参个数
        static constexpr uint32_t outputNum = 2;  // 算子出参个数
        MoeInitRoutingV2AttrParam attrParam;
        uint32_t GetInputNum() const
        {
            return inputNum; 
        }
        uint32_t GetOutputNum() const
        {
            uint32_t output = outputNum;
            if(attrParam.drop_pad_mode == 0 && attrParam.expert_tokens_count_or_cumsum_flag > 0){
                output++;
            }
            else if(attrParam.drop_pad_mode == 1 && attrParam.expert_tokens_before_capacity_flag == true){
                output++;
            }
            return output; 
        }
        std::string GetName() const
        {
            return opName_;
        }
        aclOpExecutor *aclExecutor_ = nullptr;
        
        std::string opName_;
        uint64_t workspaceSize_;

        atb::SVector<std::shared_ptr<AclnnTensor>> aclInTensors_;
        atb::SVector<std::shared_ptr<AclnnTensor>> aclOutTensors_;
    };
}
#endif