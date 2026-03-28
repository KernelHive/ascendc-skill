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
 * \file layer_norm_v4_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_V4_TILING_H
#define LAYER_NORM_V4_TILING_H

#include "tiling/tiling_base.h"
#include "tiling/tiling_type.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_templates_registry.h"

namespace optiling {

#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_LOG_I(OPS_DESC, fmt, ...)                            \
    std::printf("[%s]" fmt, __func__, ##__VA_ARGS__);            \
    std::printf("\n")

#define OPS_LOG_E(OPS_DESC, fmt, ...)                            \
    std::printf("[%s]" fmt, __func__, ##__VA_ARGS__);            \
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

#define OP_CHECK(cond, log_func, return_expr) \
    if (cond) {                               \
        log_func;                             \
        return_expr;                          \
    }

BEGIN_TILING_DATA_DEF(LayerNormV4TilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4, LayerNormV4TilingData)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataSingleRead)
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, rowAlign);
TILING_DATA_FIELD_DEF(uint32_t, nRow);
TILING_DATA_FIELD_DEF(uint32_t, tailNRow);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, tailLoop);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
TILING_DATA_FIELD_DEF(uint32_t, blockLength);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_100, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_110, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_111, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_120, LayerNormV4TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormV4_122, LayerNormV4TilingDataSingleRead)

BEGIN_TILING_DATA_DEF(LayerNormV4TilingDataTranspose)
TILING_DATA_FIELD_DEF(uint64_t, col);                     // 输入tensor的行
TILING_DATA_FIELD_DEF(uint64_t, row);                     // 输入tensor的列，即reduce的轴
TILING_DATA_FIELD_DEF(uint64_t, blockDim);                // 实际使用的core数量
TILING_DATA_FIELD_DEF(uint64_t, blockFormer);             // 整核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, blockTail);               // 尾核处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubFormer);                // ub整循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfFormerBlock);     // 整核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubLoopOfTailBlock);       // 尾核处理的ub循环次数
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfFormerBlock);     // 整核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, ubTailOfTailBlock);       // 尾核ub尾循环处理的row大小
TILING_DATA_FIELD_DEF(uint64_t, bFormer);                 // ubFormer借轴大小，ubFormer->16*bFormer
TILING_DATA_FIELD_DEF(uint64_t, dichotomizeAddDiffSize);  // row与小于row的最近二次幂的差值
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, coefficient);
TILING_DATA_FIELD_DEF(uint32_t, nullptrGamma);
TILING_DATA_FIELD_DEF(uint32_t, nullptrBeta);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormV4_200, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_210, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_211, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_220, LayerNormV4TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormV4_222, LayerNormV4TilingDataTranspose)

struct ParamsLayerNomrV4 {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    float eps = 0;
    float coefficient = 0;
    uint64_t rowAlign;
    uint64_t gammaNullPtr;
    uint64_t betaNullPtr;
    uint64_t meanAndRstdNullPtr = 0;
    ge::DataType tensorDtype;
    ge::DataType paramDtype;
    bool isAscend310P;
};

enum LayerNormV4TilingKey : int64_t {
    // FLOAT32/FLOAT16/BFLOAT16 -- 0/1/2
    // Single Read
    LAYER_NORM_SINGLE_READ_FLOAT32_FLOAT32 = 100,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT32 = 110,
    LAYER_NORM_SINGLE_READ_FLOAT16_FLOAT16 = 111,
    LAYER_NORM_SINGLE_READ_BFLOAT16_FLOAT32 = 120,
    LAYER_NORM_SINGLE_READ_BFLOAT16_BFLOAT16 = 122,
    // Transpose
    LAYER_NORM_TRANSPOSE_FLOAT32_FLOAT32 = 200,
    LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT32 = 210,
    LAYER_NORM_TRANSPOSE_FLOAT16_FLOAT16 = 211,
    LAYER_NORM_TRANSPOSE_BFLOAT16_FLOAT32 = 220,
    LAYER_NORM_TRANSPOSE_BFLOAT16_BFLOAT16 = 222,
};

struct LayerNormV4CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    bool isAscend310P = false;
};

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

class LayerNormV4TilingBase : public TilingBaseClass {
public:
    explicit LayerNormV4TilingBase(gert::TilingContext *context_) : TilingBaseClass(context_)
    {}
    ~LayerNormV4TilingBase() override
    {}
    ParamsLayerNomrV4 commonParams;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
};

class LayerNormV4SingleReadTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4SingleReadTiling(gert::TilingContext *context_) : LayerNormV4TilingBase(context_)
    {}
    ~LayerNormV4SingleReadTiling() override
    {}
    LayerNormV4TilingDataSingleRead td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
};

class LayerNormV4TransposeTiling : public LayerNormV4TilingBase {
public:
    explicit LayerNormV4TransposeTiling(gert::TilingContext *context_) : LayerNormV4TilingBase(context_)
    {}
    ~LayerNormV4TransposeTiling() override
    {}
    LayerNormV4TilingDataTranspose td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    struct BlockTilingData {
        uint64_t blockDim;
        uint64_t blockFormer;
        uint64_t blockTail;
    };

    struct UbTilingData {
        uint64_t ubFormer;
        uint64_t bFormer;
        uint64_t ubLoopOfFormerBlock;
        uint64_t ubLoopOfTailBlock;
        uint64_t ubTailOfFormerBlock;
        uint64_t ubTailOfTailBlock;
    };
    void DoBlockTiling(BlockTilingData &blockTilingParams);
    void DoUbTiling(const BlockTilingData &blockTilingParams, UbTilingData &ubTilingParams);
    uint64_t CalcBorrowFactor(uint64_t oriFactor);
    uint32_t FindDichotomizeAddDiffSize();
};

}  // namespace optiling
#endif  // LAYER_NORM_V4_TILING_H
