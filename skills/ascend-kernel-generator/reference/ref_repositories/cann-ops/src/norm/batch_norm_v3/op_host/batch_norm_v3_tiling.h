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
 * \file batch_norm_v3_tiling.h
 * \brief
 */

#ifndef BATCH_NORM_V3_TILING_H
#define BATCH_NORM_V3_TILING_H

#include <register/tilingdata_base.h>
#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"

using namespace ge;

namespace optiling {

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret) \
    if ((ptr) == nullptr) {                                \
        std::printf("nullptr error!");                     \
        return ret;                                        \
    }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg) \
    do {                                                      \
        std::printf("op[%s], %s", op_name, err_msg);          \
    } while (0)

BEGIN_TILING_DATA_DEF(BatchNormV3BaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3, BatchNormV3BaseTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3WelfordTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, r0UbFactor);
TILING_DATA_FIELD_DEF(int64_t, r0UbLoop);
TILING_DATA_FIELD_DEF(int64_t, r0UbTail);
TILING_DATA_FIELD_DEF(int64_t, procNR0);
TILING_DATA_FIELD_DEF(int64_t, nR0Loop);
TILING_DATA_FIELD_DEF(int64_t, lastLoopNR0);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_1000, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1001, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1002, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1003, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1012, BatchNormV3WelfordTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_1013, BatchNormV3WelfordTilingData)

BEGIN_TILING_DATA_DEF(BatchNormV3FullReduceTilingData)
TILING_DATA_FIELD_DEF(int64_t, patternR1);
TILING_DATA_FIELD_DEF(int64_t, patternR0);
TILING_DATA_FIELD_DEF(int64_t, patternA);
TILING_DATA_FIELD_DEF(int64_t, patternR0Align);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbFactor);
TILING_DATA_FIELD_DEF(int64_t, aUbLoop);
TILING_DATA_FIELD_DEF(int64_t, aUbTail);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbLoop);
TILING_DATA_FIELD_DEF(int64_t, tailCoreAUbTail);
TILING_DATA_FIELD_DEF(int64_t, aUbSize);
TILING_DATA_FIELD_DEF(int64_t, rUbSize);
TILING_DATA_FIELD_DEF(int64_t, dichotomizeAddDiffSize);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, coefficient0);
TILING_DATA_FIELD_DEF(float, coefficient1);
TILING_DATA_FIELD_DEF(float, momentum);
TILING_DATA_FIELD_DEF(float, momentumReverse);
TILING_DATA_FIELD_DEF(float, batchVarScale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchNormV3_2000, BatchNormV3FullReduceTilingData)
REGISTER_TILING_DATA_CLASS(BatchNormV3_2001, BatchNormV3FullReduceTilingData)

struct ParamsBatchNormV3 {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
    int64_t patternR1;
    int64_t patternR0;
    int64_t patternR0Align;
    int64_t patternA;
    float epsilon;
    float momentum;
    float momentumReverse;
    std::string nodeName;
    ge::DataType xDtype;
};

struct BatchNormV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
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
inline T1 FloorDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a) / (b);
}

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2>
inline T1 FloorAlign(T1 a, T2 b)
{
    return b == 0 ? a : (a) / b * b;
}

class BatchNormV3TilingBase : public TilingBaseClass {
public:
    explicit BatchNormV3TilingBase(gert::TilingContext *context) : TilingBaseClass(context)
    {}
    ~BatchNormV3TilingBase() override
    {}
    ParamsBatchNormV3 commonParams;

protected:
    bool IsCapable() override
    {
        return true;
    };
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    uint64_t GetTilingKey() const override
    {
        return 0;
    };
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    bool CheckInputDtype();
    bool CheckInputShape();
};

class BatchNormV3WelfordTiling : public BatchNormV3TilingBase {
public:
    explicit BatchNormV3WelfordTiling(gert::TilingContext *context) : BatchNormV3TilingBase(context)
    {}
    ~BatchNormV3WelfordTiling() override
    {}
    BatchNormV3WelfordTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t usedCoreNum;
    uint64_t welfordTilingkey;
    void DoUbTiling(int64_t &aUbFactor, int64_t &r0UbFactor);
    uint32_t FindDichotomizeAddDiffSize(uint32_t parallelN);
};

class BatchNormV3FullReduceTiling : public BatchNormV3TilingBase {
public:
    explicit BatchNormV3FullReduceTiling(gert::TilingContext *context_) : BatchNormV3TilingBase(context_)
    {}
    ~BatchNormV3FullReduceTiling() override
    {}
    BatchNormV3FullReduceTilingData td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t usedCoreNum;
    uint64_t fullReduceTilingkey;
    int64_t DoUbTiling(const int64_t blockFactor, int64_t &aUbSize, int64_t &rUbSize);
};

}  // namespace optiling
#endif  // BATCH_NORM_V3_TILING_H
