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
 * \file upsample_bicubic2d_grad_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
    class UpsampleBicubic2dGrad : public OpDef {
    public:
        explicit UpsampleBicubic2dGrad(const char* name) : OpDef(name) {
            this->Input("grad_output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("align_corners")
                .AttrType(OPTIONAL)
                .Bool(false);
            this->Attr("scales_h")
                .AttrType(OPTIONAL)
                .Float(0.0);
            this->Attr("scales_w")
                .AttrType(OPTIONAL)
                .Float(0.0);
            this->Output("grad_input")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });

            this->AICore().AddConfig("ascend910b");
        }
    };
    OP_ADD(UpsampleBicubic2dGrad);
}