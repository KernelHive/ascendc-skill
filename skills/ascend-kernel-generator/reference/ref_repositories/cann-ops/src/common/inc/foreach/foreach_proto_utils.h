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
 * \file foreach_proto_utils.h
 * \brief
 */

#ifndef FOREACH_PROTO_UITLS_H_
#define FOREACH_PROTO_UITLS_H_

#include <algorithm>
#include "register/tilingdata_base.h"

inline ge::DataType DtypeTensor2Scalar(ge::DataType dtype) {
    switch(dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
        case ge::DT_BF16:
            return ge::DT_FLOAT;
        case ge::DT_INT32:
            return ge::DT_INT64;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

inline ge::DataType DtypeScalarToTensor2(ge::DataType dtype) {
    switch(dtype) {
        case ge::DT_FLOAT16:
            return ge::DT_FLOAT16;
        case ge::DT_FLOAT:
            return ge::DT_FLOAT;
        case ge::DT_BF16:
            return ge::DT_FLOAT;
        case ge::DT_INT32:
            return ge::DT_INT32;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

#define FOREACH_OPDEF_BEGIN(NAME)                                   \
    class Foreach##NAME: public OpDef {                             \
        public:                                                     \
            explicit Foreach##NAME(const char* name) : OpDef(name) {

#define FOREACH_OPDEF_END_HOST_CONFIG(NAME)                           \
            this->AICore().AddConfig("ascend910b");                 \
        }                                                           \
    };

#define FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(...)                \
    std::vector<ge::DataType> tensor_dtype_list = {__VA_ARGS__};    \
    std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);

#define FOREACH_SCALAR_DTYPE_PREPARE                    \
    std::vector<ge::DataType> scalar_dtype_list;        \
    std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_dtype_list](ge::DataType dtype){scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype));});

#define FOREACH_SCALAR_TENSOR_DTYPE_PREPARE                    \
    std::vector<ge::DataType> scalar_tensor_dtype_list;        \
    std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_tensor_dtype_list](ge::DataType dtype){scalar_tensor_dtype_list.push_back(DtypeScalarToTensor2(dtype));});

#define FOREACH_OPDEF_PARAM_TENSOR(PARAM_TYPE, NAME)    \
    this->PARAM_TYPE(#NAME)                             \
    .ParamType(REQUIRED)                                \
    .DataType(tensor_dtype_list)                        \
    .Format(format_list)                                \
    .UnknownShapeFormat(format_list)                    \
    .AutoContiguous();

#define FOREACH_OPDEF_PARAM_SCALAR_TENSOR(PARAM_TYPE, NAME)    \
    this->PARAM_TYPE(#NAME)                                    \
    .ParamType(REQUIRED)                                       \
    .DataType(scalar_tensor_dtype_list)                        \
    .Format(format_list)                                       \
    .UnknownShapeFormat(format_list);

#define FOREACH_OPDEF_PARAM_TENSORLIST(PARAM_TYPE, NAME)\
    this->PARAM_TYPE(#NAME)                             \
    .ParamType(DYNAMIC)                                 \
    .DataType(tensor_dtype_list)                        \
    .Format(format_list)                                \
    .UnknownShapeFormat(format_list)                    \
    .AutoContiguous();

#define FOREACH_OPDEF_PARAM_SCALAR(PARAM_TYPE, NAME)    \
    this->PARAM_TYPE(#NAME)                             \
    .Scalar()                                           \
    .ParamType(REQUIRED)                                \
    .DataType(scalar_dtype_list)                        \
    .Format(format_list)                                \
    .UnknownShapeFormat(format_list)                    \
    .AutoContiguous();

#define FOREACH_OPDEF_PARAM_SCALARLIST(PARAM_TYPE, NAME)\
    this->PARAM_TYPE(#NAME)                             \
    .ScalarList()                                       \
    .ParamType(REQUIRED)                                \
    .DataType(scalar_dtype_list)                        \
    .Format(format_list)                                \
    .UnknownShapeFormat(format_list)                    \
    .AutoContiguous();

#define FOREACH_UNARY_PARAM(...)                        \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)            \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_LIST_PARAM(...)                  \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_LIST_ALPHA_PARAM(...)            \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_SCALAR(Input, alpha)            \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_SCALAR_PARAM(...)                \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)            \
    FOREACH_OPDEF_PARAM_SCALAR(Input, scalar)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_SCALARLIST_PARAM(...)            \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)            \
    FOREACH_OPDEF_PARAM_SCALARLIST(Input, scalars)       \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_POINTWISE_LIST_PARAM(...)               \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x3)           \
    FOREACH_OPDEF_PARAM_TENSOR(Input, scalars)          \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_POINTWISE_SCALAR_PARAM(...)             \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x3)           \
    FOREACH_OPDEF_PARAM_SCALAR(Input, scalar)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_POINTWISE_SCALAR_TENSOR_PARAM(...)             \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_TENSOR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x3)           \
    FOREACH_OPDEF_PARAM_SCALAR_TENSOR(Input, scalar)          \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_POINTWISE_SCALARLIST_PARAM(...)         \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x3)           \
    FOREACH_OPDEF_PARAM_SCALARLIST(Input, scalars)      \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_SCALAR_TENSOR_PARAM(...)                \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_TENSOR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)            \
    FOREACH_OPDEF_PARAM_SCALAR_TENSOR(Input, scalar)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_BINARY_LIST_ALPHA_TENSOR_PARAM(...)            \
    FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(__VA_ARGS__)\
    FOREACH_SCALAR_TENSOR_DTYPE_PREPARE                        \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)           \
    FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)           \
    FOREACH_OPDEF_PARAM_SCALAR_TENSOR(Input, alpha)            \
    FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)

#define FOREACH_OPDEF(CORE_VERSION, FOREACH_TYPE, NAME, ...)    \
    FOREACH_OPDEF_BEGIN(NAME)                                   \
    FOREACH_##FOREACH_TYPE##_PARAM(__VA_ARGS__)                 \
    FOREACH_OPDEF_END_##CORE_VERSION(NAME)                      \
    OP_ADD(Foreach##NAME);

#endif
