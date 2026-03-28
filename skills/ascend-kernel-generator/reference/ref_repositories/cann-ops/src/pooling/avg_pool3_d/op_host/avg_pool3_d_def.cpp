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
 * \file avg_pool3_d_def.cpp
 * \brief op_host of avg_pool3d
 */

 #include "register/op_def_registry.h"

namespace ops {

class AvgPool3D : public OpDef {
public:
    explicit AvgPool3D(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("ksize")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("strides")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("pads")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("ceil_mode")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("count_include_pad")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->Attr("divisor_override")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NDHWC");

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", config);
    }
};

OP_ADD(AvgPool3D);
}  // namespace ops