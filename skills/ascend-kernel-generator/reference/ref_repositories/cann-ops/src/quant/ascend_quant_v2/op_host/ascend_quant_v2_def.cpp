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
 * \file ascend_quant_v2_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class AscendQuantV2 : public OpDef {
public:
  explicit AscendQuantV2(const char* name) : OpDef(name)
  {
    this->Input("x")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("scale")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("offset")
      .ParamType(OPTIONAL)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                 ge::DT_INT4, ge::DT_INT4, ge::DT_INT4})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("sqrt_mode").AttrType(OPTIONAL).Bool(false);
    this->Attr("round_mode").AttrType(OPTIONAL).String("round");
    this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);
    this->Attr("axis").AttrType(OPTIONAL).Int(-1);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
          .DynamicRankSupportFlag(true)
          .DynamicShapeSupportFlag(true)
          .NeedCheckSupportFlag(false);
    this->AICore().AddConfig("ascend910b", config);

    OpAICoreConfig config_310p;
    config_310p.Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    config_310p.Input("scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    config_310p.Input("offset")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    config_310p.Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    config_310p.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false);

    this->AICore().AddConfig("ascend310p", config_310p); 
  }
};

OP_ADD(AscendQuantV2);
}  // namespace ops
