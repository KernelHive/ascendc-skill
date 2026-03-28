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
 * \file group_norm_swish_grad_def.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class GroupNormSwishGrad : public OpDef {
public:
    explicit GroupNormSwishGrad(const char* name) : OpDef(name)
    {
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dgamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dbeta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16})
            .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("num_groups")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NCHW");
        this->Attr("swish_scale")
            .AttrType(OPTIONAL)
            .Float(1.0);
        this->Attr("dgamma_is_require")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->Attr("dbeta_is_require")
            .AttrType(OPTIONAL)
            .Bool(true);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(GroupNormSwishGrad);
}
