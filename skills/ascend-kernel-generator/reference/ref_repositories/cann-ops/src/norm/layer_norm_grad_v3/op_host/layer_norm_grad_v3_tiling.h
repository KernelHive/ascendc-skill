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
 * \file layer_norm_grad_v3_tiling.h
 * \brief
 */

#ifndef LAYER_NORM_GRAD_V3_TILING_H
#define LAYER_NORM_GRAD_V3_TILING_H

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

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

template <typename T>
inline T *GetCompileInfoPtr(gert::TilingParseContext *context)
{
    return context->GetCompiledInfo<T>();
}

constexpr uint64_t LNG_TEMPLATE_KEY_WEIGHT = 100;
constexpr uint64_t LNG_DETERMINISTIC_KEY_WEIGHT = 10;
constexpr uint64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t FLOAT_SIZE = 4;
constexpr uint64_t HALF_SIZE = 2;

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, colSize);
TILING_DATA_FIELD_DEF(uint32_t, rowSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3, LayerNormGradV3TilingData)

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataTranspose)
TILING_DATA_FIELD_DEF(uint64_t, row);                     // 输入tensor的行
TILING_DATA_FIELD_DEF(uint64_t, col);                     // 输入tensor的列，即reduce的轴
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
TILING_DATA_FIELD_DEF(uint64_t, deterministicComputeWspSize);  // 确定性计算需要的pdGamma或pdBeta workspace size大小
TILING_DATA_FIELD_DEF(float, coefficient);                     // 1/col
TILING_DATA_FIELD_DEF(float, placeHolder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_301, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_302, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_303, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_304, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_305, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_311, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_312, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_313, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_314, LayerNormGradV3TilingDataTranspose)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_315, LayerNormGradV3TilingDataTranspose)

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataWorkspace)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubLoop);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, colAlignM);
TILING_DATA_FIELD_DEF(int64_t, colAlignV);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_201, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_202, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_203, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_204, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_205, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_211, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_212, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_213, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_214, LayerNormGradV3TilingDataWorkspace)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_215, LayerNormGradV3TilingDataWorkspace)

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataSingleRead)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, colAlignM);
TILING_DATA_FIELD_DEF(int64_t, colAlignV);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, bufferElemNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_101, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_102, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_103, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_104, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_105, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_111, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_112, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_113, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_114, LayerNormGradV3TilingDataSingleRead)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_115, LayerNormGradV3TilingDataSingleRead)

BEGIN_TILING_DATA_DEF(LayerNormGradV3TilingDataCommon)
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, colAlignM);
TILING_DATA_FIELD_DEF(int64_t, colAlignV);
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, lastRBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, nlastRBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, lastBrcbBufferBytes);
TILING_DATA_FIELD_DEF(int64_t, wholeBufferElemNums);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayerNormGradV3_401, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_402, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_403, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_404, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_405, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_411, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_412, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_413, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_414, LayerNormGradV3TilingDataCommon)
REGISTER_TILING_DATA_CLASS(LayerNormGradV3_415, LayerNormGradV3TilingDataCommon)

// TilingKey生成方式：LNGTemplateKey * 100 + isDeterministicKey * 10 + dtypeKey
enum class LNGDtypeKey {
    FLOAT_FLOAT = 1,
    FLOAT16_FLOAT16 = 2,
    FLOAT16_FLOAT = 3,
    BFLOAT16_BFLOAT16 = 4,
    BFLOAT16_FLOAT = 5
};

enum class LNGTemplateKey { SINGEL_READ = 1, WORKSPACE = 2, TRANSPOSE = 3, COMMON = 4 };

struct ParamsLayerNormGradV3 {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint64_t colSize = 1;
    uint64_t rowSize = 1;
    uint64_t colAlign = 1;
    ge::DataType dyDtype;
    ge::DataType gammaDtype;
    uint64_t isDeterministicKey;
    LNGDtypeKey dtypeKey;
};

struct LayerNormGradV3CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

class LayerNormGradV3TilingBase : public TilingBaseClass {
public:
    explicit LayerNormGradV3TilingBase(gert::TilingContext *context_) : TilingBaseClass(context_)
    {}
    ~LayerNormGradV3TilingBase() override
    {}
    ParamsLayerNormGradV3 commonParams;

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

class LayerNormGradV3WorkspaceTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3WorkspaceTiling(gert::TilingContext *context_) : LayerNormGradV3TilingBase(context_)
    {}
    ~LayerNormGradV3WorkspaceTiling() override
    {}
    LayerNormGradV3TilingDataWorkspace td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
};

class LayerNormGradV3SingleReadTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3SingleReadTiling(gert::TilingContext *context_) : LayerNormGradV3TilingBase(context_)
    {}
    ~LayerNormGradV3SingleReadTiling() override
    {}
    LayerNormGradV3TilingDataSingleRead td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    // for vector
    int64_t colAlignV;
    // for mte
    // colAlignM >= colAlignV
    int64_t colAlignM;
    bool dbFlag;
};

class LayerNormGradV3TransposeTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3TransposeTiling(gert::TilingContext *context_) : LayerNormGradV3TilingBase(context_)
    {}
    ~LayerNormGradV3TransposeTiling() override
    {}
    LayerNormGradV3TilingDataTranspose td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    uint64_t CalcBorrowFactor(uint64_t oriFactor);
    uint32_t FindDichotomizeAddDiffSize();
};

class LayerNormGradV3CommonTiling : public LayerNormGradV3TilingBase {
public:
    explicit LayerNormGradV3CommonTiling(gert::TilingContext *context_) : LayerNormGradV3TilingBase(context_)
    {}
    ~LayerNormGradV3CommonTiling() override
    {}
    LayerNormGradV3TilingDataCommon td_;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    int64_t CalculateUbFormer();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    int64_t row_{-1};
    int64_t col_{-1};
    // for vector
    int64_t colAlignV_{-1};
    // for mte
    // colAlignM >= colAlignV
    int64_t colAlignM_{-1};
};

}  // namespace optiling
#endif  // LAYER_NORM_GRAD_V3_TILING_H
