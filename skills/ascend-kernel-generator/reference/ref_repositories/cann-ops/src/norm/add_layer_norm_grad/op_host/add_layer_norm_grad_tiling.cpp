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
 * \file add_layer_norm_grad_tiling.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_layer_norm_grad_tiling.h"

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

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
}  // namespace optiling

namespace optiling {
static constexpr uint32_t FLOAT_DTYPE_KEY = 1;
static constexpr uint32_t FLOAT16_DTYPE_KEY = 2;
static constexpr uint32_t BFLOAT16_DTYPE_KEY = 3;
static constexpr uint32_t FLOAT_CUT_N_AXIS_TILING_KEY = 10;
static constexpr uint32_t FLOAT16_CUT_N_AXIS_TILING_KEY = 20;
static constexpr uint32_t BFLOAT16_CUT_N_AXIS_TILING_KEY = 30;
static constexpr uint32_t FLOAT_CUT_REDUCE_AXIS_TILING_KEY = 40;
static constexpr uint32_t FLOAT16_CUT_REDUCE_AXIS_TILING_KEY = 50;
static constexpr uint32_t BFLOAT16_CUT_REDUCE_AXIS_TILING_KEY = 60;
static constexpr uint32_t TILING_ADDITIONAL_OUTPUT = 1;
static constexpr uint32_t REDUCE_AXIS_IN_UB_MAX = 4096;
static constexpr uint32_t SIZE_OF_BF16_FP16 = 2;
static constexpr uint32_t BLOCK_ALIGN = 32;
static constexpr uint32_t BLOCK_NUMBER = BLOCK_ALIGN / sizeof(float);
static constexpr uint32_t BLOCK_NUMBER_FP16 = BLOCK_ALIGN / 2;
static constexpr uint32_t WORKSPACE_SIZE = 512;
static constexpr uint32_t SYNC_SAPCE = 512;
static constexpr uint32_t REDUCE_BUF = 64 * 4;
static constexpr uint32_t COEXIST_TENSOR_INPUT32_ONLY_D = 4;       // gamma_que/d_gamma_que/d_beta_que/
static constexpr uint32_t COEXIST_TENSOR_INPUT32_WITH_N = 6;       // dy_que/x_1_que/x_2_que/output_d_x_que/dx_input_que
static constexpr uint32_t COEXIST_TENSOR_INPUT32_D_1 = 2;          // rstd_que/mean_que/
static constexpr uint32_t COEXIST_FP32_TENSOR_INPUT16_ONLY_D = 3;  // d_gamma_que/d_beta_que/gamma_fp32_buf/
static constexpr uint32_t COEXIST_FP32_TENSOR_INPUT16_WITH_N = 5;
static constexpr uint32_t COEXIST_FP32_TENSOR_INPUT16_D_1 = 2;     // rstd_que/mean_que/
static constexpr uint32_t COEXIST_FP16_TENSOR_INPUT16_ONLY_D = 1;  // gamma_que/
static constexpr uint32_t COEXIST_FP16_TENSOR_INPUT16_WITH_N = 5;  // dy_que/x_1_que/x_2_que/output_d_x_que/dx_input_que
static constexpr uint32_t COEXIST_FP16_TENSOR_INPUT16_D_1 = 0;
constexpr int32_t INPUT_DY_INDEX = 0;
constexpr int32_t INPUT_X1_INDEX = 1;
constexpr int32_t INPUT_X2_INDEX = 2;
constexpr int32_t INPUT_MEAN_INDEX = 3;
constexpr int32_t INPUT_RSTD_INDEX = 4;
constexpr int32_t INPUT_GAMMA_INDEX = 5;
constexpr int32_t OUTPUT_DX_INDEX = 0;
constexpr int32_t OUTPUT_DGAMMA_INDEX = 1;
constexpr int32_t OUTPUT_DBETA_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;

static uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

static uint32_t ROUND_UP(uint32_t x, uint32_t factor)
{
    if (factor > 0) {
        return (x + factor - 1) / factor * factor;
    }
    return 0;
}

static uint32_t ROUND_DOWN(uint32_t x, uint32_t factor)
{
    if (factor > 0) {
        return (x - factor + 1) / factor * factor;
    }
    return 0;
}

static inline void ComputeCoexistFactor(const bool hasDxInput, uint32_t &coexistTensorInput32WithN,
    uint32_t &coexistFp32TensorInput16WithN, uint32_t &coexistFp16TensorInput16WithN)
{
    if (!hasDxInput) {
        coexistTensorInput32WithN = COEXIST_TENSOR_INPUT32_WITH_N - 1;
        coexistFp32TensorInput16WithN = COEXIST_FP32_TENSOR_INPUT16_WITH_N - 1;
        coexistFp16TensorInput16WithN = COEXIST_FP16_TENSOR_INPUT16_WITH_N - 1;
    } else {
        coexistTensorInput32WithN = COEXIST_TENSOR_INPUT32_WITH_N;
        coexistFp32TensorInput16WithN = COEXIST_FP32_TENSOR_INPUT16_WITH_N;
        coexistFp16TensorInput16WithN = COEXIST_FP16_TENSOR_INPUT16_WITH_N;
    }
}

void AddLayerNormGradTilingImpl(const uint32_t dtypeKey, const uint32_t actualAvailUb, int32_t &ndAvailUb,
    bool &cutDPath, TilingStruct &tilingStruct, const bool hasDxInput)
{
    uint32_t coexistTensorInput32WithN;
    uint32_t coexistFp32TensorInput16WithN;
    uint32_t coexistFp16TensorInput16WithN;
    ComputeCoexistFactor(
        hasDxInput, coexistTensorInput32WithN, coexistFp32TensorInput16WithN, coexistFp16TensorInput16WithN);
    if (dtypeKey == FLOAT_DTYPE_KEY) {
        tilingStruct.dInnerLength = tilingStruct.numLastDim;
        ndAvailUb = actualAvailUb / sizeof(float) -
                    ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER) * COEXIST_TENSOR_INPUT32_ONLY_D;
        tilingStruct.nAvailInUb =
            ndAvailUb / (ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER) * coexistTensorInput32WithN +
                            ROUND_UP(1, BLOCK_NUMBER) * COEXIST_TENSOR_INPUT32_D_1);
        if (tilingStruct.nAvailInUb < 1 || ndAvailUb < 0) {
            cutDPath = true;
            tilingStruct.nAvailInUb = 1;
            ndAvailUb = actualAvailUb / sizeof(float) - ROUND_UP(1, BLOCK_NUMBER) * COEXIST_TENSOR_INPUT32_D_1;
            tilingStruct.dInnerLength = ndAvailUb / (COEXIST_TENSOR_INPUT32_ONLY_D + coexistTensorInput32WithN);
            tilingStruct.dInnerLength = ROUND_DOWN(tilingStruct.dInnerLength, BLOCK_NUMBER);  // 6120
        }
    } else if (dtypeKey == FLOAT16_DTYPE_KEY || dtypeKey == BFLOAT16_DTYPE_KEY) {
        tilingStruct.dInnerLength = tilingStruct.numLastDim;
        ndAvailUb = actualAvailUb - (ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER) *
                                            COEXIST_FP32_TENSOR_INPUT16_ONLY_D * sizeof(float) +
                                        ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER_FP16) *
                                            COEXIST_FP16_TENSOR_INPUT16_ONLY_D * SIZE_OF_BF16_FP16);
        tilingStruct.nAvailInUb =
            ndAvailUb /
            (ROUND_UP(1, BLOCK_NUMBER) * COEXIST_FP32_TENSOR_INPUT16_D_1 * sizeof(float) +
                ROUND_UP(1, BLOCK_NUMBER_FP16) * COEXIST_FP16_TENSOR_INPUT16_D_1 * SIZE_OF_BF16_FP16 +
                ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER) * coexistFp32TensorInput16WithN * sizeof(float) +
                ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER_FP16) * coexistFp16TensorInput16WithN *
                    SIZE_OF_BF16_FP16);
        if (tilingStruct.nAvailInUb < 1 || ndAvailUb < 0) {
            cutDPath = true;
            tilingStruct.nAvailInUb = 1;
            ndAvailUb = actualAvailUb -
                        (ROUND_UP(1, BLOCK_NUMBER) * COEXIST_FP32_TENSOR_INPUT16_D_1 * sizeof(float) +
                            ROUND_UP(1, BLOCK_NUMBER_FP16) * COEXIST_FP16_TENSOR_INPUT16_D_1 * SIZE_OF_BF16_FP16);
            tilingStruct.dInnerLength =
                actualAvailUb /
                ((coexistFp16TensorInput16WithN + COEXIST_FP16_TENSOR_INPUT16_ONLY_D) * SIZE_OF_BF16_FP16 +
                    (coexistFp32TensorInput16WithN + COEXIST_FP32_TENSOR_INPUT16_ONLY_D) * sizeof(float));
            tilingStruct.dInnerLength = ROUND_DOWN(tilingStruct.dInnerLength, BLOCK_NUMBER_FP16);  // 4656
        }
    } else {
        cutDPath = true;
        tilingStruct.nAvailInUb = 1;
        tilingStruct.dInnerLength = REDUCE_AXIS_IN_UB_MAX;
    }
}

void AddLayerNormGradGetTilingKey(const uint32_t dtypeKey, const bool cutDPath, uint32_t &tilingKey,
    uint32_t &roundUpNumLastDimDtype, const TilingStruct &tilingStruct, const bool hasDxInput)
{
    if (dtypeKey == FLOAT_DTYPE_KEY && !cutDPath) {
        tilingKey = FLOAT_CUT_N_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * sizeof(float);
    } else if (dtypeKey == FLOAT16_DTYPE_KEY && !cutDPath) {
        tilingKey = FLOAT16_CUT_N_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * SIZE_OF_BF16_FP16;
    } else if (dtypeKey == BFLOAT16_DTYPE_KEY && !cutDPath) {
        tilingKey = BFLOAT16_CUT_N_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * SIZE_OF_BF16_FP16;
    } else if (dtypeKey == FLOAT_DTYPE_KEY && cutDPath) {
        tilingKey = FLOAT_CUT_REDUCE_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * sizeof(float);
    } else if (dtypeKey == FLOAT16_DTYPE_KEY && cutDPath) {
        tilingKey = FLOAT16_CUT_REDUCE_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * SIZE_OF_BF16_FP16;
    } else if (dtypeKey == BFLOAT16_DTYPE_KEY && cutDPath) {
        tilingKey = BFLOAT16_CUT_REDUCE_AXIS_TILING_KEY;
        roundUpNumLastDimDtype = tilingStruct.roundUpNumLastDim * SIZE_OF_BF16_FP16;
    }
    if (hasDxInput) {
        tilingKey += TILING_ADDITIONAL_OUTPUT;
    }
}

static bool CheckDimNumLimit(size_t x1DimNum, size_t gammaDimNum)
{
    OP_TILING_CHECK(x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE("CheckDimNumLimit", "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_TILING_CHECK(gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE("CheckDimNumLimit", "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    return true;
}

template <typename T>
static bool CheckAllNotNull(std::initializer_list<T> ptrList)
{
    for (auto &ptr : ptrList) {
        if (nullptr == ptr) {
            return false;
        }
    }
    return true;
}

template <typename T>
static inline bool CheckEqualsAll(std::initializer_list<T> eleList)
{
    bool ret = true;
    if (eleList.size() > 0) {
        const T *fontPtr = eleList.begin();
        for (const T *curtPtr = eleList.begin(); curtPtr != eleList.end(); curtPtr++) {
            ret = ret && (*(curtPtr) == *(fontPtr));
            fontPtr = curtPtr;
        }
    }
    return ret;
}

static inline bool HasNoZero(const gert::StorageShape *shapePtr, size_t shapeDim)
{
    for (uint32_t i = 0; i < shapeDim; i++) {
        OP_TILING_CHECK(shapePtr->GetStorageShape().GetDim(i) == 0,
            OP_LOGE("HasNoZero", "Invaild shape which have zero dim."),
            return false);
    }
    return true;
}

static inline bool CheckDyGammaMean(const gert::StorageShape *dyShape, const gert::StorageShape *gammaShape,
    const gert::StorageShape *meanShape, size_t dyDimNum, size_t gammaDimNum)
{
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_TILING_CHECK(
            gammaShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i),
            OP_LOGE("CheckAddLn", "Gamma shape invaild, gamma shape is not equal dy last few dim."),
            return false);
    }
    for (uint32_t i = 0; i < dyDimNum; i++) {
        if (i < dyDimNum - gammaDimNum) {
            OP_TILING_CHECK(meanShape->GetStorageShape().GetDim(i) != dyShape->GetStorageShape().GetDim(i),
                OP_LOGE("CheckAddLn", "Output mean/rstd reduce dim is not equal dy first few dim."),
                return false);
        } else {
            OP_TILING_CHECK(meanShape->GetStorageShape().GetDim(i) != 1,
                OP_LOGE("CheckAddLn", "Output mean/rstd reduce dim is not equal 1."),
                return false);
        }
    }
    return true;
}

static bool CheckInputOutputShape(const gert::TilingContext *context)
{
    const gert::StorageShape *dyShape = context->GetInputShape(INPUT_DY_INDEX);
    const gert::StorageShape *x1Shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape *meanShape = context->GetInputShape(INPUT_MEAN_INDEX);
    const gert::StorageShape *rstdShape = context->GetInputShape(INPUT_RSTD_INDEX);
    const gert::StorageShape *gammaShape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape *dxShape = context->GetOutputShape(OUTPUT_DX_INDEX);
    const gert::StorageShape *dgammaShape = context->GetOutputShape(OUTPUT_DGAMMA_INDEX);
    const gert::StorageShape *dbetaShape = context->GetOutputShape(OUTPUT_DBETA_INDEX);
    bool noNullShape =
        CheckAllNotNull({dyShape, x1Shape, x2Shape, meanShape, rstdShape, gammaShape, dgammaShape, dbetaShape});
    OP_TILING_CHECK(!noNullShape, OP_LOGE("CheckShape", "Tensor's shape have nullptr"), return false);
    size_t dyDimNum = dyShape->GetStorageShape().GetDimNum();
    size_t x1DimNum = dyShape->GetStorageShape().GetDimNum();
    size_t meanDimNum = meanShape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstdShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t dgammaDimNum = dgammaShape->GetStorageShape().GetDimNum();
    size_t dbetaDimNum = dbetaShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(!CheckDimNumLimit(x1DimNum, gammaDimNum), OP_LOGE("CheckShape", "Bad x1/gamma dim"), return false);
    OP_TILING_CHECK(!CheckEqualsAll<size_t>({dyDimNum, rstdDimNum, meanDimNum}),
        OP_LOGE("CheckShape", "Dim num of dy/mean/rstd not same."),
        return false);
    OP_TILING_CHECK(!CheckEqualsAll<size_t>({dgammaDimNum, gammaDimNum, dbetaDimNum}),
        OP_LOGE("CheckShape", "Dim num of gamma/dgamma/dbeta not same."),
        return false);
    OP_TILING_CHECK(dyDimNum < gammaDimNum,
        OP_LOGE("CheckShape", "Input gamma shape invaild, dim num bigger than dy dim num."),
        return false);
    bool x1NoZeroDim = HasNoZero(x1Shape, x1DimNum);
    bool meanNoZeroDim = HasNoZero(meanShape, meanDimNum);
    OP_TILING_CHECK(
        !(x1NoZeroDim && meanNoZeroDim), OP_LOGE("CheckShape", "Bad x1/mean which have 0 dim."), return false);
    bool x1x2dydxEqs = CheckEqualsAll<const gert::StorageShape>({*(dyShape), *(x1Shape), *(x2Shape), *(dxShape)});
    bool meanRstdEqs = CheckEqualsAll<const gert::StorageShape>({*(meanShape), *(rstdShape)});
    OP_TILING_CHECK(!(x1x2dydxEqs && meanRstdEqs),
        OP_LOGE("CheckShape", "Tensor x1/x2/dy/dx OR mean/rstd should have same shape."),
        return false);
    bool checkDyGammaMean = CheckDyGammaMean(dyShape, gammaShape, meanShape, dyDimNum, gammaDimNum);
    OP_TILING_CHECK(!checkDyGammaMean, OP_LOGE("CheckShape", "Tensor dy/gamma/mean have bad rels."), return false);
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        int64_t gammaDimValue = gammaShape->GetStorageShape().GetDim(i);
        int64_t dgammaDimValue = dgammaShape->GetStorageShape().GetDim(i);
        int64_t dyDimValue = dyShape->GetStorageShape().GetDim(dyDimNum - gammaDimNum + i);
        int64_t dbetaDimValue = dbetaShape->GetStorageShape().GetDim(i);
        OP_TILING_CHECK(!CheckEqualsAll<int64_t>({gammaDimValue, dgammaDimValue, dyDimValue, dbetaDimValue}),
            OP_LOGE("CheckShape", "Shape of gamma/dgamma/dbeta should equal to dy last few dim."),
            return false);
    }
    return true;
}

ge::graphStatus BasicCheck(const gert::TilingContext *context, TilingStruct &tilingStruct, bool &hasDxInput)
{
    const gert::StorageShape *dyShape = context->GetInputShape(0);
    const gert::StorageShape *gammaShape = context->GetInputShape(5);
    if (dyShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "dyShape is empty");
        return ge::GRAPH_FAILED;
    }
    auto dyTensor = context->GetInputDesc(0);
    if (dyTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "dyTensor is empty");
        return ge::GRAPH_FAILED;
    }
    uint32_t gammaDims = gammaShape->GetStorageShape().GetDimNum();
    uint32_t numLastDim = 1;
    for (uint32_t i = 0; i < gammaDims; i++) {
        numLastDim *= gammaShape->GetStorageShape().GetDim(i);
    }

    uint32_t inputYDim = 1;
    for (uint32_t i = 0; i < dyShape->GetStorageShape().GetDimNum(); i++) {
        inputYDim *= dyShape->GetStorageShape().GetDim(i);
    }

    tilingStruct.numLastDim = numLastDim;
    if (tilingStruct.numLastDim == 0) {
        OP_LOGE(context->GetNodeName(), "numLastDim is 0");
        return ge::GRAPH_FAILED;
    }

    auto dxInputShape = context->GetOptionalInputShape(6);
    hasDxInput = (dxInputShape != nullptr);
    if (hasDxInput) {
        OP_LOGI(context->GetNodeName(), "hasDxInput");
        int64_t DxInputSize = 1;
        for (size_t i = 0; i < dxInputShape->GetStorageShape().GetDimNum(); i++) {
            DxInputSize *= dxInputShape->GetStorageShape().GetDim(i);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static void CalNInUb(const uint32_t blockNumberTdtype, TilingStruct &tilingStruct)
{
    // == norm N in one core ==
    tilingStruct.nInOneCoreNorm = tilingStruct.nInOneCoreLength;
    tilingStruct.gmOneCoreElemXYNorm = tilingStruct.nInOneCoreNorm * tilingStruct.numLastDim;
    tilingStruct.nAvailInUbNorm =
        (tilingStruct.nInOneCoreNorm < tilingStruct.nAvailInUb) ? tilingStruct.nInOneCoreNorm : tilingStruct.nAvailInUb;
    tilingStruct.nMiddleCountNorm = CEIL_DIV(tilingStruct.nInOneCoreNorm, tilingStruct.nAvailInUbNorm);
    tilingStruct.nInUbTotalNormTail =
        tilingStruct.nInOneCoreNorm - (tilingStruct.nAvailInUbNorm * (tilingStruct.nMiddleCountNorm - 1));

    // == tail N in one core ==
    tilingStruct.nInOneCoreTail = tilingStruct.nInOneCoreLengthTail;
    tilingStruct.gmOneCoreElemXYTail = tilingStruct.nInOneCoreTail * tilingStruct.numLastDim;
    tilingStruct.nAvailInUbTail =
        (tilingStruct.nInOneCoreTail < tilingStruct.nAvailInUb) ? tilingStruct.nInOneCoreTail : tilingStruct.nAvailInUb;
    tilingStruct.nMiddleCountTail = CEIL_DIV(tilingStruct.nInOneCoreTail, tilingStruct.nAvailInUbTail);
    tilingStruct.nInUbTotalTailTail =
        tilingStruct.nInOneCoreTail - (tilingStruct.nAvailInUbNorm * (tilingStruct.nMiddleCountTail - 1));

    uint32_t dRstdInUb = 1;
    tilingStruct.dyPadRight = ROUND_UP(tilingStruct.numLastDim, blockNumberTdtype) - tilingStruct.numLastDim;
    tilingStruct.rstdPadRight = ROUND_UP(dRstdInUb, BLOCK_NUMBER) - dRstdInUb;

    tilingStruct.dOuterLength = CEIL_DIV(tilingStruct.numLastDim, tilingStruct.dInnerLength);
    tilingStruct.dInnerLengthTail =
        tilingStruct.numLastDim - tilingStruct.dInnerLength * (tilingStruct.dOuterLength - 1);

    tilingStruct.roundUpNumLastDim = ROUND_UP(tilingStruct.numLastDim, blockNumberTdtype);
    tilingStruct.roundUp1Dtype = ROUND_UP(1, BLOCK_NUMBER);
    tilingStruct.roundUpNumLastDimFloat = ROUND_UP(tilingStruct.numLastDim, BLOCK_NUMBER) * sizeof(float);
}

void SetTiling(TilingStruct &tilingStruct, AddLayerNormGradTilingData &tiling, const uint32_t &numCore,
    const uint32_t &numFirstDim, const uint32_t &roundUpNumLastDimDtype)
{
    tiling.set_numCore(numCore);
    tiling.set_numLastDim(tilingStruct.numLastDim);
    tiling.set_numFirstDim(numFirstDim);
    tiling.set_nInOneCoreLength(tilingStruct.nInOneCoreLength);
    tiling.set_nInOneCoreLengthTail(tilingStruct.nInOneCoreLengthTail);
    tiling.set_ndInOneCoreLength(tilingStruct.ndInOneCoreLength);
    tiling.set_nAvailInUb(tilingStruct.nAvailInUb);
    tiling.set_dInnerLength(tilingStruct.dInnerLength);
    tiling.set_dInnerLengthTail(tilingStruct.dInnerLengthTail);
    tiling.set_dOuterLength(tilingStruct.dOuterLength);
    tiling.set_nInOneCoreNorm(tilingStruct.nInOneCoreNorm);
    tiling.set_gmOneCoreElemXYNorm(tilingStruct.gmOneCoreElemXYNorm);
    tiling.set_nAvailInUbNorm(tilingStruct.nAvailInUbNorm);
    tiling.set_nMiddleCountNorm(tilingStruct.nMiddleCountNorm);
    tiling.set_ndRoundUpDtypeNorm(tilingStruct.ndRoundUpDtypeNorm);
    tiling.set_n1RoundUpFloatNorm(tilingStruct.n1RoundUpFloatNorm);
    tiling.set_nInUbTotalNormTail(tilingStruct.nInUbTotalNormTail);
    tiling.set_nInOneCoreTail(tilingStruct.nInOneCoreTail);
    tiling.set_gmOneCoreElemXYTail(tilingStruct.gmOneCoreElemXYTail);
    tiling.set_nAvailInUbTail(tilingStruct.nAvailInUbTail);
    tiling.set_nMiddleCountTail(tilingStruct.nMiddleCountTail);
    tiling.set_ndRoundUpDtypeTail(tilingStruct.ndRoundUpDtypeTail);
    tiling.set_n1RoundUpFloatTail(tilingStruct.n1RoundUpFloatTail);
    tiling.set_nInUbTotalTailTail(tilingStruct.nInUbTotalTailTail);
    tiling.set_dyPadRight(tilingStruct.dyPadRight);
    tiling.set_rstdPadRight(tilingStruct.rstdPadRight);
    tiling.set_roundUpNumLastDim(tilingStruct.roundUpNumLastDim);
    tiling.set_roundUpNumLastDimDtype(roundUpNumLastDimDtype);
    tiling.set_roundUp1Dtype(tilingStruct.roundUp1Dtype);
    tiling.set_roundUpNumLastDimFloat(tilingStruct.roundUpNumLastDimFloat);
}

static void TilingLog(AddLayerNormGradTilingData &tiling, const uint32_t &tilingKey)
{
    OP_LOGI("[AddLayerNormGrad]", "[tilingKey]: %u", tilingKey);
    OP_LOGI("[AddLayerNormGrad]", "[numCore]: %u", tiling.get_numCore());
    OP_LOGI("[AddLayerNormGrad]", "[numLastDim]: %u", tiling.get_numLastDim());
    OP_LOGI("[AddLayerNormGrad]", "[numFirstDim]: %u", tiling.get_numFirstDim());
    OP_LOGI("[AddLayerNormGrad]", "[nInOneCoreLength]: %u", tiling.get_nInOneCoreLength());
    OP_LOGI("[AddLayerNormGrad]", "[nInOneCoreLengthTail]: %u", tiling.get_nInOneCoreLengthTail());
    OP_LOGI("[AddLayerNormGrad]", "[ndInOneCoreLength]: %u", tiling.get_ndInOneCoreLength());
    OP_LOGI("[AddLayerNormGrad]", "[nAvailInUb]: %u", tiling.get_nAvailInUb());
    OP_LOGI("[AddLayerNormGrad]", "[dInnerLength]: %u", tiling.get_dInnerLength());
    OP_LOGI("[AddLayerNormGrad]", "[dInnerLengthTail]: %u", tiling.get_dInnerLengthTail());
    OP_LOGI("[AddLayerNormGrad]", "[dOuterLength]: %u", tiling.get_dOuterLength());
    OP_LOGI("[AddLayerNormGrad]", "[nInOneCoreNorm]: %u", tiling.get_nInOneCoreNorm());
    OP_LOGI("[AddLayerNormGrad]", "[gmOneCoreElemXYNorm]: %u", tiling.get_gmOneCoreElemXYNorm());
    OP_LOGI("[AddLayerNormGrad]", "[nAvailInUbNorm]: %u", tiling.get_nAvailInUbNorm());
    OP_LOGI("[AddLayerNormGrad]", "[nMiddleCountNorm]: %u", tiling.get_nMiddleCountNorm());
    OP_LOGI("[AddLayerNormGrad]", "[ndRoundUpDtypeNorm]: %u", tiling.get_ndRoundUpDtypeNorm());
    OP_LOGI("[AddLayerNormGrad]", "[n1RoundUpFloatNorm]: %u", tiling.get_n1RoundUpFloatNorm());
    OP_LOGI("[AddLayerNormGrad]", "[nInUbTotalNormTail]: %u", tiling.get_nInUbTotalNormTail());
    OP_LOGI("[AddLayerNormGrad]", "[nInOneCoreTail]: %u", tiling.get_nInOneCoreTail());
    OP_LOGI("[AddLayerNormGrad]", "[gmOneCoreElemXYTail]: %u", tiling.get_gmOneCoreElemXYTail());
    OP_LOGI("[AddLayerNormGrad]", "[nAvailInUbTail]: %u", tiling.get_nAvailInUbTail());
    OP_LOGI("[AddLayerNormGrad]", "[nMiddleCountTail]: %u", tiling.get_nMiddleCountTail());
    OP_LOGI("[AddLayerNormGrad]", "[ndRoundUpDtypeTail]: %u", tiling.get_ndRoundUpDtypeTail());
    OP_LOGI("[AddLayerNormGrad]", "[n1RoundUpFloatTail]: %u", tiling.get_n1RoundUpFloatTail());
    OP_LOGI("[AddLayerNormGrad]", "[nInUbTotalTailTail]: %u", tiling.get_nInUbTotalTailTail());
    OP_LOGI("[AddLayerNormGrad]", "[dyPadRight]: %u", tiling.get_dyPadRight());
    OP_LOGI("[AddLayerNormGrad]", "[rstdPadRight]: %u", tiling.get_rstdPadRight());
    OP_LOGI("[AddLayerNormGrad]", "[roundUpNumLastDim]: %u", tiling.get_roundUpNumLastDim());
    OP_LOGI("[AddLayerNormGrad]", "[roundUpNumLastDimDtype]: %u", tiling.get_roundUpNumLastDimDtype());
    OP_LOGI("[AddLayerNormGrad]", "[roundUp1Dtype]: %u", tiling.get_roundUp1Dtype());
    OP_LOGI("[AddLayerNormGrad]", "[roundUpNumLastDimFloat]: %u", tiling.get_roundUpNumLastDimFloat());
}

static inline bool CheckAndSetDtype(ge::DataType dyDataType, uint32_t &dtypeKey, uint32_t &blockNumberTdtype)
{
    if (ge::DT_FLOAT == dyDataType) {
        dtypeKey = FLOAT_DTYPE_KEY;
        blockNumberTdtype = BLOCK_ALIGN / sizeof(float);
    } else if (ge::DT_FLOAT16 == dyDataType) {
        dtypeKey = FLOAT16_DTYPE_KEY;
        blockNumberTdtype = BLOCK_ALIGN / SIZE_OF_BF16_FP16;
    } else if (ge::DT_BF16 == dyDataType) {
        dtypeKey = BFLOAT16_DTYPE_KEY;
        blockNumberTdtype = BLOCK_ALIGN / SIZE_OF_BF16_FP16;
    } else {
        OP_LOGE("CheckAndSetDtype", "dtype is not support");
        return false;
    }
    return true;
}

static inline void SetTilingDataPart1(TilingStruct *tilingStruct, uint32_t roundUpNumLastDimDtype)
{
    tilingStruct->ndRoundUpDtypeNorm = tilingStruct->nAvailInUbNorm * roundUpNumLastDimDtype;
    tilingStruct->ndRoundUpDtypeTail = tilingStruct->nAvailInUbTail * roundUpNumLastDimDtype;
    tilingStruct->n1RoundUpFloatNorm = tilingStruct->nAvailInUbNorm * tilingStruct->roundUp1Dtype * sizeof(float);
    tilingStruct->n1RoundUpFloatTail = tilingStruct->nAvailInUbTail * tilingStruct->roundUp1Dtype * sizeof(float);
}

static inline void SetSocInfo(gert::TilingContext *context, uint64_t &maxUbSize, uint64_t &l1Size, uint32_t &maxCoreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    maxCoreNum = ascendcPlatform.GetCoreNumAiv();
}

static ge::graphStatus Tiling4AddLayerNormGrad(gert::TilingContext *context)
{
    AddLayerNormGradTilingData tiling;
    TilingStruct tilingStruct;
    OP_LOGD(context->GetNodeName(), "Begin to do Tiling4AddLayerNormGrad");
    bool hasDxInput = false;
    BasicCheck(context, tilingStruct, hasDxInput);
    OP_TILING_CHECK(!CheckInputOutputShape(context),
        OP_LOGE(context->GetNodeName(), "Input shape is Invaild."),
        return ge::GRAPH_FAILED);
    const gert::StorageShape *dyShape = context->GetInputShape(0);
    const gert::StorageShape *gammaShape = context->GetInputShape(5);
    auto dyTensor = context->GetInputDesc(0);
    auto dyDataType = dyTensor->GetDataType();
    uint32_t dtypeKey = 0;
    uint32_t blockNumberTdtype = 0;
    OP_TILING_CHECK(!CheckAndSetDtype(dyDataType, dtypeKey, blockNumberTdtype),
        OP_LOGE(context->GetNodeName(), "Dtype is not support."),
        return ge::GRAPH_FAILED);
    uint32_t numFirstDim = 1;
    uint32_t gammaDims = gammaShape->GetStorageShape().GetDimNum();
    for (uint32_t i = 0; i < dyShape->GetStorageShape().GetDimNum() - gammaDims; i++) {
        numFirstDim *= dyShape->GetStorageShape().GetDim(i);
    }

    uint64_t maxUbSize;
    uint64_t l1Size;
    uint32_t maxCoreNum = 1;
    SetSocInfo(context, maxUbSize, l1Size, maxCoreNum);

    uint32_t numCore = CEIL_DIV(numFirstDim, CEIL_DIV(numFirstDim, maxCoreNum));
    tilingStruct.nInOneCoreLength = 0;
    if (numCore != 0) {
        tilingStruct.nInOneCoreLength = (numFirstDim + numCore - 1) / numCore;
    }
    tilingStruct.nInOneCoreLengthTail = numFirstDim - (tilingStruct.nInOneCoreLength * (numCore - 1));
    tilingStruct.ndInOneCoreLength = tilingStruct.nInOneCoreLength * tilingStruct.numLastDim;
    uint32_t actualAvailUb = maxUbSize - WORKSPACE_SIZE - SYNC_SAPCE - REDUCE_BUF;
    int32_t ndAvailUb;
    bool cutDPath = false;

    AddLayerNormGradTilingImpl(dtypeKey, actualAvailUb, ndAvailUb, cutDPath, tilingStruct, hasDxInput);
    CalNInUb(blockNumberTdtype, tilingStruct);

    uint32_t tilingKey = 0;
    uint32_t roundUpNumLastDimDtype = 1;

    AddLayerNormGradGetTilingKey(dtypeKey, cutDPath, tilingKey, roundUpNumLastDimDtype, tilingStruct, hasDxInput);
    SetTilingDataPart1(&tilingStruct, roundUpNumLastDimDtype);

    SetTiling(tilingStruct, tiling, numCore, numFirstDim, roundUpNumLastDimDtype);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(numCore);

    TilingLog(tiling, tilingKey);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // set Workspace
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t usrWorkSpaceSize = 20 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddLayerNormGrad(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<AddLayerNormGradCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddLayerNormGrad)
    .Tiling(Tiling4AddLayerNormGrad)
    .TilingParse<AddLayerNormGradCompileInfo>(TilingPrepare4AddLayerNormGrad);
}  // namespace optiling