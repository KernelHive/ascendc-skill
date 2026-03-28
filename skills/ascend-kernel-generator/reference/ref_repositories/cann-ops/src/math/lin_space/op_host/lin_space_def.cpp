/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class LinSpace : public OpDef {
 public:
  explicit LinSpace(const char* name) : OpDef(name) {
    this->Input("start")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Input("stop")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->Input("num")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910");
    this->AICore().AddConfig("ascend310p");
    
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false);
    config.Input("start")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    config.Input("stop")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    config.Input("num")
        .ParamType(REQUIRED)
        .ValueDepend(OPTIONAL)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND});
    config.Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND});
    this->AICore().AddConfig("ascend910b", config);
    this->AICore().AddConfig("ascend910_93", config);
  }
};

OP_ADD(LinSpace);
}  // namespace ops