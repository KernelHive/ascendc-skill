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
 * \file stft.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static ge::graphStatus StftCheckSupport(const ge::Operator& op, ge::AscendString& result) {
  std::string resultJsonStr = R"({"ret_code": "0", "reason":"Stft don't support graph mode"})";
  result = ge::AscendString(resultJsonStr.c_str());
  return ge::GRAPH_FAILED;
}

class STFT : public OpDef {
 public:
  explicit STFT(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("plan")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("window")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_COMPLEX64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("hop_length").AttrType(OPTIONAL).Int();
    this->Attr("win_length").AttrType(OPTIONAL).Int();
    this->Attr("normalized").AttrType(OPTIONAL).Bool(false);
    this->Attr("onesided").AttrType(OPTIONAL).Bool(true);
    this->Attr("return_complex").AttrType(OPTIONAL).Bool(true);
    this->Attr("n_fft").AttrType(REQUIRED).Int();
    this->AICore().SetCheckSupport(StftCheckSupport);
    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(true)
        .PrecisionReduceFlag(true);
    this->AICore().AddConfig("ascend910b", aicConfig);
    this->AICore().AddConfig("ascend910_93", aicConfig);
  }
};
OP_ADD(STFT);
}  // namespace ops
