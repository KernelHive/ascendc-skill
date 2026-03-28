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
 * \file grid_sampler2_d_grad_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class GridSampler2DGrad : public OpDef {
public:
    explicit GridSampler2DGrad(const char *name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .UnknownShapeFormat({ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC})
            .InitValue({{ScalarType::FLOAT32, 0}, {ScalarType::FLOAT16, 0}, {ScalarType::FLOAT16, 0}});
        this->Output("dgrid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("interpolation_mode").AttrType(REQUIRED).String("bilinear");
        this->Attr("padding_mode").AttrType(REQUIRED).String("zeros");
        this->Attr("align_corners").AttrType(REQUIRED).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(GridSampler2DGrad);
}  // namespace ops