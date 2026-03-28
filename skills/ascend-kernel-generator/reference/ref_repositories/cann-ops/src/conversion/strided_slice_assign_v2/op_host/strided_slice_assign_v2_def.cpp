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
 * \file strided_slice_assign_v2.cpp
 * \brief
 */
#include <vector>
#include <cstdint>
#include "register/op_def_registry.h"

static const std::vector<ge::DataType> inputDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT64, ge::DT_DOUBLE, ge::DT_INT8};

static const std::vector<ge::DataType> idxDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

namespace ops {
class StridedSliceAssignV2 : public OpDef {
public:
    explicit StridedSliceAssignV2(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("input_value")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("begin")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("end")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("strides")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Input("axes")
            .ParamType(OPTIONAL)
            .ValueDepend(REQUIRED)
            .DataType(idxDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(inputDataType)
            .Format(format)
            .UnknownShapeFormat(format)
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(StridedSliceAssignV2);
}  // namespace ops