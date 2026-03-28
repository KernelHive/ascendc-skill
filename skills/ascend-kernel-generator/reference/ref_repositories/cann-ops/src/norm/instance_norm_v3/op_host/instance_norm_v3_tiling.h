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
 * \file instance_norm_v3_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_INSTANCE_NORM_V3_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_INSTANCE_NORM_V3_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InstanceNormV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, C);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, W);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNums);
TILING_DATA_FIELD_DEF(uint32_t, nAxisPerCore);
TILING_DATA_FIELD_DEF(uint32_t, nAxisPerCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, reduceNums);
TILING_DATA_FIELD_DEF(uint32_t, ubFactor);
TILING_DATA_FIELD_DEF(uint32_t, cAxisFactor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InstanceNormV3, InstanceNormV3TilingData)

struct InstanceNormV3CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

template <typename T1, typename T2>
inline T1 FloorAlign(T1 a, T2 b)
{
    return b == 0 ? a : (a) / b * b;
}

class InstanceNormV3TilingHelper {
public:
    explicit InstanceNormV3TilingHelper(gert::TilingContext *_context, const InstanceNormV3CompileInfo *_compileInfo)
        : context(_context), compileInfo(_compileInfo)
    {}

    ~InstanceNormV3TilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(InstanceNormV3TilingData *tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();

    gert::TilingContext *context;
    const InstanceNormV3CompileInfo *compileInfo;

    ge::DataType xDtype{ge::DataType::DT_FLOAT16};
    uint32_t dtSize{2};

    uint32_t N{1};
    uint32_t C{1};
    uint32_t H{1};
    uint32_t W{1};
    uint32_t reduceNums{1};
    uint32_t useCoreNums{1};
    uint32_t nAxisPerCore{1};
    uint32_t nAxisPerCoreTail{1};
    uint32_t ubFactor{1};
    uint32_t cAxisFactor{1};

    uint32_t coreNums{1};
    uint32_t ubSize{1};

    float eps{1e-6};
    float avgFactor{0.0};
    const char *dataFormat{"NCHW"};

    uint32_t ubTilingStrategy{0};
};

}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_INSTANCE_NORM_V3_TILING_H
