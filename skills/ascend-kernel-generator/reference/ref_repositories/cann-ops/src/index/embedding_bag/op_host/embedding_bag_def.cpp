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
 * \file embedding_bag.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
    class EmbeddingBag : public OpDef {
    public:
        explicit EmbeddingBag(const char* name) : OpDef(name)
        {
            this->Input("weight")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("offsets")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("per_sample_weight")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("offset2bag")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("bag_size")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("max_indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("mode")
                    .AttrType(OPTIONAL)
                    .String("mean");
            this->Attr("scale_grad_by_freq")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            this->Attr("sparse")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            this->Attr("include_last_offset")
                    .AttrType(OPTIONAL)
                    .Bool(false);
            this->Attr("padding_idx")
                    .AttrType(OPTIONAL)
                    .Int(-1);
            OpAICoreConfig aicore_config;
            aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "embedding_bag_apt");
            this->AICore().AddConfig("ascend910_93", aicore_config);
            this->AICore().AddConfig("ascend910b", aicore_config);

            OpAICoreConfig config_n;
            config_n.Input("weight")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Input("indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Input("offsets")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Input("per_sample_weight")
                    .ParamType(OPTIONAL)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Output("y")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Output("offset2bag")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Output("bag_size")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.Output("max_indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32, ge::DT_INT32})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            config_n.DynamicCompileStaticFlag(true)
                    .DynamicRankSupportFlag(true)
                    .DynamicShapeSupportFlag(true)
                    .NeedCheckSupportFlag(false)
                    .ExtendCfgInfo("opFile.value", "embedding_bag_apt");

            this->AICore().AddConfig("ascend310p", config_n);
            this->AICore().AddConfig("ascend910", config_n);
        }
    };

    OP_ADD(EmbeddingBag);
}