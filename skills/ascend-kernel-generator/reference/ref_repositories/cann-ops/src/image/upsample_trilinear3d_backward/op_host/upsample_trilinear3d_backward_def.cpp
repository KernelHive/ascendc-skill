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
 * \file upsample_trilinear3d_backward_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class UpsampleTrilinear3dBackward : public OpDef {
public:
    explicit UpsampleTrilinear3dBackward(const char *name) : OpDef(name)
    {
        this->Input("grad_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("output_size").AttrType(REQUIRED).ListInt();
        this->Attr("input_size").AttrType(REQUIRED).ListInt();
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scale_d").AttrType(OPTIONAL).Float();
        this->Attr("scale_h").AttrType(OPTIONAL).Float();
        this->Attr("scale_w").AttrType(OPTIONAL).Float();

        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UpsampleTrilinear3dBackward);
}  // namespace ops
