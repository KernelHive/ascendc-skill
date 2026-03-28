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
 * \file deep_norm_grad_tiling.cpp
 * \brief
 */
#include <iostream>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "deep_norm_grad_tiling.h"

namespace optiling {
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")

#define OP_LOGE(op_name, ...)            \
    std::printf(op_name, ##__VA_ARGS__); \
    std::printf("\n")

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

}  // namespace optiling

namespace optiling {

static constexpr uint32_t BLOCK_ALIGN_SIZE = 32;
static constexpr uint32_t REDUCE_BUF_ELEM = 64;

static constexpr uint32_t DTYPE_BYTES_FP32 = 4;
static constexpr uint32_t DTYPE_BYTES_FP16 = 2;
static constexpr uint32_t DTYPE_BYTES_BF16 = 2;

// dtypeKey: 0:fp32、1:fp16、2:bf16
static constexpr uint32_t DTYPE_KEY_FP32 = 0;
static constexpr uint32_t DTYPE_KEY_FP16 = 1;
static constexpr uint32_t DTYPE_KEY_BF16 = 2;

// cutDKey:  0: merge N; 1：cut D; 2: large N small D
static constexpr uint32_t KEY_MERGE_N = 0;
static constexpr uint32_t KEY_CUT_D = 1;
static constexpr uint32_t KEY_LARGE_N_SMALL_D = 2;

static constexpr uint32_t SMALL_D_STAGE = 500;
constexpr int32_t INPUT_DY_INDEX = 0;
constexpr int32_t INPUT_X_INDEX = 1;
constexpr int32_t INPUT_GX_INDEX = 2;
constexpr int32_t INPUT_GAMMA_INDEX = 3;
constexpr int32_t INPUT_MEAN_INDEX = 4;
constexpr int32_t INPUT_RSTD_INDEX = 5;
constexpr int32_t OUTPUT_DX_INDEX = 0;
constexpr int32_t OUTPUT_DGX_INDEX = 1;
constexpr int32_t OUTPUT_DBETA_INDEX = 2;
constexpr int32_t OUTPUT_DGAMMA_INDEX = 3;
constexpr size_t MAX_DIM_X = 8;
constexpr size_t MIN_DIM_X = 2;
constexpr size_t MAX_DIM_GAMMA = 7;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t NUM_5 = 5;
constexpr uint32_t NUM_8 = 8;
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_17 = 17;
constexpr uint32_t NUM_21 = 21;

inline void SetBaseConfig(gert::TilingContext *context, DeepNormGradTilingData &tiling, uint32_t &dDimNum)
{
    const gert::StorageShape *dyShape = context->GetInputShape(0);
    const gert::StorageShape *gammaShape = context->GetInputShape(3);
    uint32_t dyDims = dyShape->GetStorageShape().GetDimNum();
    uint32_t gammaDims = gammaShape->GetStorageShape().GetDimNum();

    uint32_t nDimNum = 1;
    for (uint32_t i = 0; i < dyDims - gammaDims; i++) {
        nDimNum *= dyShape->GetStorageShape().GetDim(i);
    }
    dDimNum = gammaShape->GetStorageShape().GetShapeSize();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNumAiv();

    uint32_t useCoreNum = CeilDiv(nDimNum, CeilDiv(nDimNum, maxCoreNum));
    uint32_t nDealPerCore = CeilDiv(nDimNum, useCoreNum);
    uint32_t nDealLastCore = nDimNum - nDealPerCore * (useCoreNum - 1);
    uint32_t fixedOutputFlag = context->GetDeterministic() == 1 ? 1 : 0;
    OP_LOGI(context->GetNodeName(), "[DeepNormGrad] GetDeterministic state: %u", context->GetDeterministic());

    tiling.set_useCoreNum(useCoreNum);
    tiling.set_nDimNum(nDimNum);
    tiling.set_dDimNum(dDimNum);
    tiling.set_nDealPerCore(nDealPerCore);
    tiling.set_nDealLastCore(nDealLastCore);
    tiling.set_fixedOutputFlag(fixedOutputFlag);
    context->SetBlockDim(useCoreNum);
}

inline void CalCutStageValue(uint32_t &cutStage, uint32_t &ubElemNum, uint32_t &tensorJustNNum,
    uint32_t &elemWithoutDInUB, uint32_t tensorWithDNum, uint32_t &blockElem)
{
    cutStage = (ubElemNum - tensorJustNNum * elemWithoutDInUB) / tensorWithDNum;
    cutStage = cutStage / blockElem * blockElem;
}

inline void CalMergeCountValue(uint32_t &mergeCount, uint32_t &ubElemNum, uint32_t tensorJustDNum,
    uint32_t &elemWithDInUB, uint32_t &tensorNDNum, uint32_t &tensorJustNNum, uint32_t &elemWithoutDInUB)
{
    mergeCount = ((ubElemNum - tensorJustDNum * elemWithDInUB) /
                  (tensorNDNum * elemWithDInUB + tensorJustNNum * elemWithoutDInUB));
}

inline void SetFp32Config(DeepNormGradTilingData &tiling, uint32_t &ubElemNum, uint32_t &elemWithDInUB,
    uint32_t &elemWithoutDInUB, uint32_t &blockElem, uint32_t &dDimNum, uint32_t &cutDKey)
{
    uint32_t cutStageMergeN;
    uint32_t mergeCountMergeN;

    uint32_t tensorJustNNum;
    uint32_t tensorNDNum;
    uint32_t tensorJustDNum;
    uint32_t otherTensorJustDNum;

    if (dDimNum < SMALL_D_STAGE) {
        tensorJustNNum = NUM_2;        // mean\rstd
        tensorNDNum = NUM_8;           // dy\x\gx\dx\dgx\tmp_ND\brcb_tmp_ND1\brcb_tmp_ND2
        tensorJustDNum = NUM_3;        // gamma\dbeta\dgamma
        otherTensorJustDNum = NUM_16;  // brcb need ailgn brcb_tmp_ND1\brcb_tmp_ND2

        CalCutStageValue(cutStageMergeN,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);
        CalMergeCountValue(mergeCountMergeN,
            ubElemNum,
            tensorJustDNum + otherTensorJustDNum,
            elemWithDInUB,
            tensorNDNum,
            tensorJustNNum,
            elemWithoutDInUB);
    } else {
        tensorJustNNum = NUM_2;       // mean\rstd
        tensorNDNum = NUM_5;          // dy\x\gx\dx\dgx
        tensorJustDNum = NUM_3;       // gamma\dbeta\dgamma
        otherTensorJustDNum = 0;  // not use brcb

        CalCutStageValue(cutStageMergeN,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);
        CalMergeCountValue(mergeCountMergeN,
            ubElemNum,
            tensorJustDNum + otherTensorJustDNum,
            elemWithDInUB,
            tensorNDNum,
            tensorJustNNum,
            elemWithoutDInUB);
    }

    // == merge N ==
    uint32_t mergeNCount;
    if (dDimNum < SMALL_D_STAGE) {
        cutDKey = KEY_LARGE_N_SMALL_D;
        mergeNCount = mergeCountMergeN;
    } else if (dDimNum <= cutStageMergeN) {
        cutDKey = KEY_MERGE_N;
        mergeNCount = mergeCountMergeN;
    } else {
        cutDKey = KEY_CUT_D;
        mergeNCount = 1;
    }

    // == cut D ==
    uint32_t cutStageCutD;
    uint32_t cutDTime;
    uint32_t cutDPerTime;
    uint32_t cutDLastTime;
    if (cutDKey == KEY_CUT_D) {
        tensorJustNNum = NUM_4;       // mean\rstd\tmp_mean_pd_buf\tmp_var_pd_buf
        tensorNDNum = NUM_5;          // dy\x\gx\dx\dgx
        tensorJustDNum = NUM_3;       // gamma\dgamma\dbeta
        otherTensorJustDNum = 0;  // not use brcb

        CalCutStageValue(cutStageCutD,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);

        cutDTime = CeilDiv(dDimNum, cutStageCutD);
        cutDPerTime = cutStageCutD;
        cutDLastTime = dDimNum - cutStageCutD * (cutDTime - 1);
    } else {
        cutDTime = 1;
        cutDPerTime = dDimNum;
        cutDLastTime = dDimNum;
    }

    tiling.set_mergeNCount(mergeNCount);
    tiling.set_cutDTime(cutDTime);
    tiling.set_cutDPerTime(cutDPerTime);
    tiling.set_cutDLastTime(cutDLastTime);
}

inline void SetFp16Bf16Config(DeepNormGradTilingData &tiling, uint32_t &ubElemNum, uint32_t &elemWithDInUB,
    uint32_t &elemWithoutDInUB, uint32_t &blockElem, uint32_t &dDimNum, uint32_t &cutDKey)
{
    uint32_t cutStageMergeN;
    uint32_t mergeCountMergeN;

    uint32_t tensorJustNNum;
    uint32_t tensorNDNum;
    uint32_t tensorJustDNum;
    uint32_t otherTensorJustDNum;

    if (dDimNum < SMALL_D_STAGE) {
        tensorJustNNum = NUM_4;        // mean(fp32)\rstd(fp32)
        tensorNDNum = NUM_21;          // dy\x\gx\dx\dgx\dy_t(fp32)\x_t(fp32)\gx_t(fp32)\dx_t(fp32)\dgx_t(fp32)
                                   // tmp_ND(fp32)\brcb_tmp_ND1(fp32)\brcb_tmp_ND2(fp32)
        tensorJustDNum = NUM_5;        // gamma\dbeta(fp32)\dgamma(fp32)
        otherTensorJustDNum = 32;  // brcb need ailgn brcb_tmp_ND1\brcb_tmp_ND2

        CalCutStageValue(cutStageMergeN,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);
        CalMergeCountValue(mergeCountMergeN,
            ubElemNum,
            tensorJustDNum + otherTensorJustDNum,
            elemWithDInUB,
            tensorNDNum,
            tensorJustNNum,
            elemWithoutDInUB);
    } else {
        tensorJustNNum = NUM_4;       // mean\rstd
        tensorNDNum = NUM_5;          // dy\x\gx\dx\dgx
        tensorJustDNum = NUM_17;      // gamma\dbeta(fp32)\dgamma(fp32)\dy_t(fp32)\x_t(fp32)\gx_t(fp32)
                                  // gamma_t(fp32)\dx_t(fp32)\dgx_t(fp32)
        otherTensorJustDNum = 0;  // not use brcb

        CalCutStageValue(cutStageMergeN,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);
        CalMergeCountValue(mergeCountMergeN,
            ubElemNum,
            tensorJustDNum + otherTensorJustDNum,
            elemWithDInUB,
            tensorNDNum,
            tensorJustNNum,
            elemWithoutDInUB);
    }

    // == merge N ==
    uint32_t mergeNCount;
    if (dDimNum < SMALL_D_STAGE) {
        cutDKey = KEY_LARGE_N_SMALL_D;
        mergeNCount = mergeCountMergeN;
    } else if (dDimNum <= cutStageMergeN) {
        cutDKey = KEY_MERGE_N;
        mergeNCount = mergeCountMergeN;
    } else {
        cutDKey = KEY_CUT_D;
        mergeNCount = 1;
    }

    // == cut D ==
    uint32_t cutStageCutD;
    uint32_t cutDTime;
    uint32_t cutDPerTime;
    uint32_t cutDLastTime;
    if (cutDKey == 1) {
        tensorJustNNum = NUM_8;       // mean(fp32)\rstd(fp32)\tmp_mean_pd_buf(fp32)\tmp_var_pd_buf(fp32)
        tensorNDNum = NUM_17;         // dy\x\gx\dx\dgx\dy_t(fp32)\x_t(fp32)\gx_t(fp32)
                                  // gamma_t(fp32)\dx_t(fp32)\dgx_t(fp32)
        tensorJustDNum = NUM_5;       // gamma\dgamma(fp32)\dbeta(fp32)
        otherTensorJustDNum = 0;  // not use brcb

        CalCutStageValue(cutStageCutD,
            ubElemNum,
            tensorJustNNum,
            elemWithoutDInUB,
            tensorNDNum + tensorJustDNum + otherTensorJustDNum,
            blockElem);

        cutDTime = CeilDiv(dDimNum, cutStageCutD);
        cutDPerTime = cutStageCutD;
        cutDLastTime = dDimNum - cutStageCutD * (cutDTime - 1);
    } else {
        cutDTime = 1;
        cutDPerTime = dDimNum;
        cutDLastTime = dDimNum;
    }

    tiling.set_mergeNCount(mergeNCount);
    tiling.set_cutDTime(cutDTime);
    tiling.set_cutDPerTime(cutDPerTime);
    tiling.set_cutDLastTime(cutDLastTime);
}

static ge::graphStatus CheckInputOutputShapeNull(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *meanShape = context->GetInputShape(INPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetInputShape(INPUT_RSTD_INDEX);
    const gert::StorageShape *dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    const gert::StorageShape *dgxShape = context->GetOutputShape(OUTPUT_DGX_INDEX);
    const gert::StorageShape *dbetaShape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    const gert::StorageShape *dgammaShape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gxShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rstdShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgxShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dbetaShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dgammaShape);
    return ge::GRAPH_SUCCESS;
}

static bool CheckInputOutputShapeDim(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *meanShape = context->GetInputShape(INPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetInputShape(INPUT_RSTD_INDEX);
    const gert::StorageShape *dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    const gert::StorageShape *dgxShape = context->GetOutputShape(OUTPUT_DGX_INDEX);
    const gert::StorageShape *dbetaShape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    const gert::StorageShape *dgammaShape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);

    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gxDimNum = gxShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t dxDimNum = dxShape->GetStorageShape().GetDimNum();
    size_t dgxDimNum = dgxShape->GetStorageShape().GetDimNum();
    size_t dbetaDimNum = dbetaShape->GetStorageShape().GetDimNum();
    size_t dgammaDimNum = dgammaShape->GetStorageShape().GetDimNum();

    // Check shape dim range
    OP_TILING_CHECK((dyDimNum > MAX_DIM_X) || (dyDimNum < MIN_DIM_X),
        OP_LOGE(
            context->GetNodeName(), "Input dy shape invaild, dim num should in range[%lu, %lu].", MIN_DIM_X, MAX_DIM_X),
        return false);
    OP_TILING_CHECK((gammaDimNum > MAX_DIM_GAMMA) || (gammaDimNum < MIN_DIM_GAMMA),
        OP_LOGE(context->GetNodeName(),
            "Input gamma shape invaild, dim num should in range[%lu, %lu].",
            MIN_DIM_GAMMA,
            MAX_DIM_GAMMA),
        return false);

    // Check shape dim relationship
    OP_TILING_CHECK(
        (xDimNum != dyDimNum) || (gxDimNum != dyDimNum) || (dxDimNum != dyDimNum) || (dgxDimNum != dyDimNum),
        OP_LOGE(context->GetNodeName(), "Input gx/x/dx/dgx shape invaild, dim num is not equal dy dim."),
        return false);
    OP_TILING_CHECK((rstdDimNum != dyDimNum) || (meanDimNum != dyDimNum),
        OP_LOGE(context->GetNodeName(), "Input mean/rstd shape invaild, dim num is not equal dy dim num."),
        return false);
    OP_TILING_CHECK((dgammaDimNum != gammaDimNum) || (dbetaDimNum != gammaDimNum),
        OP_LOGE(context->GetNodeName(), "Output dgamma/dbeta shape invaild, dim num is not equal input gamma dim num."),
        return false);
    OP_TILING_CHECK(dyDimNum <= gammaDimNum,
        OP_LOGE(context->GetNodeName(), "dy dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckXYShapeValue(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *gxShape = context->GetInputShape(INPUT_GX_INDEX);
    const gert::StorageShape *dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    const gert::StorageShape *dgxShape = context->GetOutputShape(OUTPUT_DGX_INDEX);

    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < dyDimNum; i++) {
        OP_TILING_CHECK(dyShape->GetStorageShape().GetDim(i) == 0,
            OP_LOGE(context->GetNodeName(), "Input dy shape can not be 0."),
            return false);
        OP_TILING_CHECK((xShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(i)) ||
                            (gxShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Input x/gx shape invaild, shape is not equal dy shape."),
            return false);
        OP_TILING_CHECK((dxShape->GetStorageShape().GetDim(i) != xShape->GetStorageShape().GetDim(i)) ||
                            (dgxShape->GetStorageShape().GetDim(i) != gxShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Output dx/dgx shape invaild, shape is not equal x/gx shape."),
            return false);
    }
    return true;
}

static bool CheckMeanRstdShapeValue(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *meanShape = context->GetInputShape(INPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetInputShape(INPUT_RSTD_INDEX);

    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < dyDimNum - gammaDimNum; i++) {
        OP_TILING_CHECK((rstdShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(i)) ||
                            (meanShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Input rstd/mean shape invaild, shape is not equal dy first few dim."),
            return false);
    }
    return true;
}

static bool CheckBetaGammaShapeValue(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *dbetaShape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    const gert::StorageShape *dgammaShape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);

    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            (gammaShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i)),
            OP_LOGE(context->GetNodeName(), "Input gamma shape invaild, gamma shape is not equal dy last few dim."),
            return false);
        OP_TILING_CHECK((dgammaShape->GetStorageShape().GetDim(i) != gammaShape->GetStorageShape().GetDim(i)),
            OP_LOGE(context->GetNodeName(), "Output dgamma shape invaild, shape is not equal gamma shape."),
            return false);
        OP_TILING_CHECK(
            (dbetaShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i)),
            OP_LOGE(context->GetNodeName(), "Output dbeta shape invaild, dbeta shape is not equal dy last few dim."),
            return false);
    }
    return true;
}

static bool CheckInputOutputShapeValue(const gert::TilingContext *context)
{
    OP_TILING_CHECK(!CheckXYShapeValue(context), , return false);
    OP_TILING_CHECK(!CheckMeanRstdShapeValue(context), , return false);
    OP_TILING_CHECK(!CheckBetaGammaShapeValue(context), , return false);
    return true;
}

ge::graphStatus SetWorkspace(gert::TilingContext *context)
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkSpaceSize = 20 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkSpaceSize;

    OP_LOGD(context->GetNodeName(), "[DeepNormGrad] TilingFunc end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4DeepNormGradCompileInfo(gert::TilingContext *context)
{
    OP_LOGD(context->GetNodeName(), "[DeepNormGrad] TilingFunc begin");
    DeepNormGradTilingData tiling;
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != CheckInputOutputShapeNull(context),
        OP_LOGE(context->GetNodeName(), "Input shape dim invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputOutputShapeDim(context),
        OP_LOGE(context->GetNodeName(), "Input shape dim invalid."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckInputOutputShapeValue(context),
        OP_LOGE(context->GetNodeName(), "Input shape value invalid."),
        return ge::GRAPH_FAILED);

    uint32_t dDimNum;
    SetBaseConfig(context, tiling, dDimNum);

    // == data type ==
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeKey = DTYPE_KEY_FP32;
    uint32_t dtypeBytes = DTYPE_BYTES_FP32;
    if (dataType == ge::DT_FLOAT) {
        dtypeKey = DTYPE_KEY_FP32;
        dtypeBytes = DTYPE_BYTES_FP32;
    } else if (dataType == ge::DT_FLOAT16) {
        dtypeKey = DTYPE_KEY_FP16;
        dtypeBytes = DTYPE_BYTES_FP16;
    } else if (dataType == ge::DT_BF16) {
        dtypeKey = DTYPE_KEY_BF16;
        dtypeBytes = DTYPE_BYTES_BF16;
    } else {
        OP_LOGE(context->GetNodeName(), "[DeepNormGrad] input dtype not support!");
        return ge::GRAPH_FAILED;
    }
    uint32_t blockElem = BLOCK_ALIGN_SIZE / dtypeBytes;

    // == data size ==
    // cutDKey:  0: merge N; 1：cut D; 2: large N small D
    uint64_t maxUBSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUBSize);

    // == cal cut stage ==
    uint32_t workspaceSizeInUB = BLOCK_ALIGN_SIZE;
    uint32_t syncSpaceInUB = 4 * BLOCK_ALIGN_SIZE;
    uint32_t otherSpaceInUB = 512;
    uint32_t maxUBSizeAligned = (maxUBSize - workspaceSizeInUB - syncSpaceInUB - otherSpaceInUB) / BLOCK_ALIGN_SIZE *
                                BLOCK_ALIGN_SIZE;        // 195680
    uint32_t ubElemNum = maxUBSizeAligned / dtypeBytes;  // fp32: 48920 fp16: 97840

    uint32_t elemWithDInUB = CeilAlign(dDimNum, blockElem);
    uint32_t elemWithoutDInUB = CeilAlign(1, blockElem);

    uint32_t cutDKey;
    if (dataType == ge::DT_FLOAT) {
        SetFp32Config(tiling, ubElemNum, elemWithDInUB, elemWithoutDInUB, blockElem, dDimNum, cutDKey);
    } else {
        SetFp16Bf16Config(tiling, ubElemNum, elemWithDInUB, elemWithoutDInUB, blockElem, dDimNum, cutDKey);
    }

    // attr config
    float alpha = *context->GetAttrs()->GetFloat(0);
    tiling.set_alpha(*reinterpret_cast<uint32_t *>(&alpha));

    uint32_t tilingCode = dtypeKey * 10 + cutDKey * 1;
    context->SetTilingKey(tilingCode);

    // other config
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace
    return SetWorkspace(context);
}

static ge::graphStatus TilingPrepare4DeepNormGrad(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}
struct DeepNormGradCompileInfo {};
IMPL_OP_OPTILING(DeepNormGrad)
    .Tiling(Tiling4DeepNormGradCompileInfo)
    .TilingParse<DeepNormGradCompileInfo>(TilingPrepare4DeepNormGrad);

}  // namespace optiling
