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
 * \file diag_flat.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
    class DiagFlat : public OpDef {
    public:
        explicit DiagFlat(const char* name) : OpDef(name)
        {
            this->Input("x")
                    .ParamType(REQUIRED)
                    .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,
                ge::DT_INT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64}
                        )
                    .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}
                        );
            this->Output("y")
                    .ParamType(REQUIRED)
                    .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_DOUBLE, ge::DT_INT16, ge::DT_INT32,
                ge::DT_INT64, ge::DT_INT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8, ge::DT_COMPLEX64}
                        )
                    .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}
                        );
            this->Attr("diagonal")
                    .AttrType(OPTIONAL)
                    .Int(0);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(DiagFlat);
}