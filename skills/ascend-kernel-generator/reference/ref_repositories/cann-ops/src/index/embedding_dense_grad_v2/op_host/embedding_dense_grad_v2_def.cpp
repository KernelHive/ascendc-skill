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
 * @file embedding_dense_grad_v2_def.cpp
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
    class EmbeddingDenseGradV2 : public OpDef {
    public:
        explicit EmbeddingDenseGradV2(const char* name) : OpDef(name)
        {
            this->Input("grad")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("sort_indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("pos_idx")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND})
                    .InitValue(0);
            this->Attr("num_weights")
                    .AttrType(REQUIRED)
                    .Int();
            this->Attr("padding_idx")
                    .AttrType(OPTIONAL)
                    .Int(-1);
            this->Attr("scale_grad_by_freq")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            OpAICoreConfig aicore_config;
            aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(EmbeddingDenseGradV2);
} // namespace ops
