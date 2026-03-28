/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {

static const int64_t AXIS_DEFAULT = -2;

static const std::vector<ge::DataType> varDataType910b = {
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT
};

static const std::vector<ge::DataType> indiceDataType910b = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::DataType> maskDataType910b = {
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8
};

static const std::vector<ge::Format> format910b = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

static const std::vector<ge::DataType> varDataType310p = {
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_FLOAT,
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32,
    ge::DT_FLOAT16, ge::DT_FLOAT
};

static const std::vector<ge::DataType> indiceDataType310p = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::DataType> maskDataType310p = {
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8,
    ge::DT_UINT8, ge::DT_UINT8
};

static const std::vector<ge::Format> format310p = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND
};

class ScatterList : public OpDef {
public:
    explicit ScatterList(const char* name) : OpDef(name) {
        this->Input("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("indice")
            .ParamType(REQUIRED)
            .DataType(indiceDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType(maskDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Output("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType910b)
            .Format(format910b)
            .UnknownShapeFormat(format910b);
        this->Attr("reduce")
            .AttrType(OPTIONAL)
            .String("update");
        this->Attr("axis")
            .AttrType(OPTIONAL)
            .Int(AXIS_DEFAULT);

        OpAICoreConfig config910b;
        config910b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config910b);
        this->AICore().AddConfig("ascend910_93", config910b);

        OpAICoreConfig config310p;
        config310p.Input("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType310p)
            .Format(format310p)
            .UnknownShapeFormat(format310p);
        config310p.Input("indice")
            .ParamType(REQUIRED)
            .DataType(indiceDataType310p)
            .Format(format310p)
            .UnknownShapeFormat(format310p);
        config310p.Input("updates")
            .ParamType(REQUIRED)
            .DataType(varDataType310p)
            .Format(format310p)
            .UnknownShapeFormat(format310p);
        config310p.Input("mask")
            .ParamType(OPTIONAL)
            .DataType(maskDataType310p)
            .Format(format310p)
            .UnknownShapeFormat(format310p);
        config310p.Output("var")
            .ParamType(DYNAMIC)
            .DataType(varDataType310p)
            .Format(format310p)
            .UnknownShapeFormat(format310p);
        config310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend310p", config310p);
    }
};

OP_ADD(ScatterList);
}  // namespace ops
