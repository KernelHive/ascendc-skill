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
 * \file add_layer_norm_quant_tiling.cpp
 * \brief
 */
#include <vector>
#include <cmath>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "add_layer_norm_quant_tiling.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__); \
    std::printf("\n")
#define OP_LOGE(op_name, ...) std::printf(op_name, ##__VA_ARGS__)

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
}  // namespace optiling

namespace optiling {

static constexpr size_t X1_IDX = 0;
static constexpr size_t X2_IDX = 1;
static constexpr size_t GAMMA_IDX = 2;
static constexpr size_t BETA_IDX = 3;
static constexpr size_t BIAS_IDX = 4;
static constexpr size_t SCALE1_IDX = 5;
static constexpr size_t SCALE2_IDX = 6;
static constexpr size_t ZERO_POINT1_IDX = 7;
static constexpr size_t ZERO_POINT2_IDX = 8;

static constexpr size_t QUANT_MODE_IDX = 0;
static constexpr size_t EPS_IDX = 1;
static constexpr size_t X_OUT_IDX = 2;

static constexpr int64_t UB_RESERVED_BYTE = 256;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr int64_t FLOAT_BLOCK_ELEM = 8;
static constexpr int64_t RESERVED_WORKSPACE_SIZE_910B = 16 * 1024 * 1024;
static constexpr int64_t ROW_FACTOR = 16;

// Dtype Tiling: 1, 2, 3
static constexpr uint64_t TILING_DATATYPE_FP16 = 1000;  // 3
static constexpr uint64_t TILING_DATATYPE_FP32 = 2000;
static constexpr uint64_t TILING_DATATYPE_BF16 = 3000;

// AddLayerNormTiling
// BroadCast : 0, 1, 2
static constexpr uint64_t TILING_X3_ELEWISE = 1;
static constexpr uint64_t TILING_X3_BROADCAST = 2;
// UB Tiling for AddLayerNorm: 0, 2, 4, 5
static constexpr uint64_t TILING_SINGLE_ROW = 20;
static constexpr uint64_t TILING_SMALL_D = 40;
static constexpr uint64_t TILING_SLICE_EXT = 50;
// QuantizeTiling
// 0, 1
static constexpr uint64_t TILING_DYNAMIC_QUANT = 100;

static constexpr uint32_t MODE_DUAL_WITHOUT_OFFSETS = 200;
static constexpr uint32_t MODE_DUAL_WITH_OFFSETS = 211;
static constexpr uint32_t MODE_SOLE_WITHOUT_OFFSETS = 100;
static constexpr uint32_t MODE_SOLE_WITH_OFFSETS = 110;

static constexpr uint32_t MODE_CODE_INPUT_SCALES = 100;
static constexpr uint32_t MODE_CODE_INPUT_ZEROPOINTS1 = 10;
static constexpr uint32_t MODE_CODE_INPUT_ZEROPOINTS2 = 1;

const std::string OP_NAME = "AddLayerNormQuant";

inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

enum class TILING_TYPE { NORMAL, SINGLE_ROW, SLICE_EXT, SMALL_D };
enum class BIAS_TYPE { NO_BIAS, BROADCAST_BIAS, ELEWISE_BIAS };

template <typename T>
T GetOptionalAttr(const gert::RuntimeAttrs *attrs, const size_t idx, const T &defaultValue)
{
    const T *attrPtr = attrs->GetAttrPointer<T>(idx);
    if (attrPtr == nullptr) {
        OP_LOGD("GetOptionalAttr", "attr[%zu] get unsuccess, use default value", idx);
    }
    T outValue = (attrPtr == nullptr) ? defaultValue : (*attrPtr);
    return outValue;
}

/**
 * TILING TRY WITH OPTIONS BUFFER_NUMBER AND ENABLE_X_OUT BY FOLLOW SEQUENCE:
 * 1. NORMAL CASE
 * 2. SINGLE ROW
 * 3. SLICE EXT
 *
 * UB occupancy composition
 *  1. NORMAL CASE (COL=LAST_DIM, ROW=?)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + DT*COUNT(Gamma,Beta,(BIAS_BROADCAST?BIAS:NULL))*COL*BN +
 * DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN + DT_FLOAT*COUNT(x_buf,y_buf,z_buf)*ROW*COL + DT_FLOAT*64 + UB_RESERVED UB =
 * DT*(3+(ENABLE_X_OUT?1:0))*ROW*LAST_DIM*BN + DT*2*LAST_DIM*BN + 4*ROW*2*BN + 4*3*ROW*LAST_DIM + 4*64 + UB_RESERVED UB
 * = ROW*DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN + DT*2*LAST_DIM*BN + ROW*4*2*BN + ROW*4*3*LAST_DIM + 4*64 + UB_RESERVED
 *   UB = ROW*(DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN + 4*2*BN + 4*3*LAST_DIM) + DT*2*LAST_DIM*BN + 4*64 + UB_RESERVED
 *   ROW = (UB - (DT*2*LAST_DIM*BN + 4*64 + UB_RESERVED)) / (DT*(3+(ENABLE_X_OUT?1:0))*LAST_DIM*BN + 4*2*BN +
 * 4*3*LAST_DIM)
 *  2. SINGLE ROW (ROW=1, COL=LAST_DIM)
 *   UB = DT*COUNT(X1,Y)*ROW*COL*BN + DT*COUNT(Gamma,Beta)*COL*BN + DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN +
 * DT_FLOAT*COUNT(x_buf,y_buf)*COL + DT_FLOAT*64 + UB_RESERVED UB = DT*(2)*1*LAST_DIM*BN + DT*2*LAST_DIM*BN +
 * DT_FLOAT*1*2*BN + DT_FLOAT*2*LAST_DIM + DT_FLOAT*64 + UB_RESERVED UB >? DT*(2)*1*LAST_DIM*BN + DT*2*LAST_DIM*BN +
 * DT_FLOAT*1*2*BN + DT_FLOAT*2*LAST_DIM + DT_FLOAT*64 + UB_RESERVED
 *  3. SLICE EXT (ROW=1, COL=?)
 *   UB = DT*COUNT(X1,X2,Y,(ENABLE_X_OUT?X:NULL))*ROW*COL*BN + DT*COUNT(Gamma,Beta,(BIAS_BROADCAST?BIAS:NULL))*COL*BN +
 * DT_FLOAT*ROW*COUNT(Mean,Rstd)*BN + DT_FLOAT*COUNT(x_buf,y_buf,z_buf)*COL + DT_FLOAT*64 + UB_RESERVED UB =
 * DT*(3+(ENABLE_X_OUT?1:0))*1*COL*BN + DT*(2+BIAS_BROADCAST?1:0)*COL*BN + DT_FLOAT*1*2*BN + DT_FLOAT*3*COL +
 * DT_FLOAT*64 + UB_RESERVED UB = COL*DT*(3+(ENABLE_X_OUT?1:0))*1*BN + COL*DT*(2+BIAS_BROADCAST?1:0)*BN +
 * DT_FLOAT*1*2*BN + COL*DT_FLOAT*3 + DT_FLOAT*64 + UB_RESERVED UB = COL*(DT*(3+(ENABLE_X_OUT?1:0))*1*BN +
 * DT*(2+BIAS_BROADCAST?1:0)*BN + DT_FLOAT*3) + DT_FLOAT*1*2*BN + DT_FLOAT*64 + UB_RESERVED COL = (UB - (DT_FLOAT*1*2*BN
 * + DT_FLOAT*64 + UB_RESERVED)) / (DT*(3+(ENABLE_X_OUT?1:0))*1*BN + DT*(2+BIAS_BROADCAST?1:0)*BN + DT_FLOAT*3)
 *
 * @param maxUbSize
 * @param dataType
 * @param bufferNum
 * @param numCol
 * @param enableXOut
 * @param biasType
 * @param rowPerTime
 * @param colPerTime
 */
inline TILING_TYPE AddLayerNormQuantTilingImpl(uint64_t maxUbSize, ge::DataType dataType, int32_t bufferNum,
    int32_t numCol, bool enableXOut, enum BIAS_TYPE biasType, uint32_t &rowPerTime, uint32_t &colPerTime)
{
    int64_t dtSize = GetSizeByDataType(dataType);
    int64_t additionalOutputNum = enableXOut ? 1 : 0;
    int64_t biasNum = biasType == BIAS_TYPE::BROADCAST_BIAS ? 1 : 0;
    auto numColAligned = (numCol + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    // try Normal case:
    double avaUb = static_cast<double>(
        static_cast<long>(maxUbSize) -
        ((2 + biasNum) * numColAligned * dtSize + ROW_FACTOR * sizeof(float) * 4 + 32 + UB_RESERVED_BYTE));
    double liveTensorFactor =
        static_cast<double>(3 * numColAligned * bufferNum * dtSize + 2 * numColAligned * sizeof(float) + 2 * 4);
    double tmpRow = avaUb / liveTensorFactor;
    if (tmpRow > 1) {
        rowPerTime = floor(tmpRow);
        colPerTime = numCol;
        return TILING_TYPE::NORMAL;
    }
    // try SingleRow case
    uint64_t tmpUbSize = 3 * numColAligned * bufferNum * dtSize + 2 * numColAligned * sizeof(float) +
                         biasNum * numColAligned * dtSize + ROW_FACTOR * sizeof(float) * 4 + 32 + UB_RESERVED_BYTE;
    if (tmpUbSize < maxUbSize) {
        rowPerTime = 1;
        colPerTime = numCol;
        return TILING_TYPE::SINGLE_ROW;
    }
    // try Silce case
    int64_t numPerBlock = 1;
    if (dtSize > 0) {
        numPerBlock = BLOCK_SIZE / dtSize;
    }
    rowPerTime = 1;
    auto tmpCol = static_cast<double>(static_cast<long>(maxUbSize) - (4 * 2 * bufferNum + 4 * 64 + UB_RESERVED_BYTE)) /
                  static_cast<double>(
                      dtSize * (3 + additionalOutputNum) * bufferNum + dtSize * (2 + biasNum) * bufferNum + 4 * 3);
    colPerTime = floor(tmpCol);
    if (colPerTime % numPerBlock != 0) {  // colPerTime should be 32 align
        colPerTime = (colPerTime / numPerBlock) * numPerBlock;
    }
    return TILING_TYPE::SLICE_EXT;
}

inline bool CheckScaleOffset(bool isDynamicQuant, uint32_t mode)
{
    bool ret = true;
    if (isDynamicQuant) {
        ret = (mode % MODE_CODE_INPUT_SCALES) == 0;
    } else {
        ret = ((mode == MODE_DUAL_WITHOUT_OFFSETS) || (mode == MODE_DUAL_WITH_OFFSETS) ||
               (mode == MODE_SOLE_WITHOUT_OFFSETS) || (mode == MODE_SOLE_WITH_OFFSETS));
    }
    return ret;
}

static inline uint32_t ComputeOptCode(gert::TilingContext *context)
{
    uint32_t optionalScaleOffsetMode = 0;
    auto scale1Shape = context->GetOptionalInputShape(SCALE1_IDX);
    auto scale2Shape = context->GetOptionalInputShape(SCALE2_IDX);
    auto zeroPoint1Shape = context->GetOptionalInputShape(ZERO_POINT1_IDX);
    auto zeroPoint2Shape = context->GetOptionalInputShape(ZERO_POINT2_IDX);
    optionalScaleOffsetMode += (nullptr != scale1Shape) ? MODE_CODE_INPUT_SCALES : 0;
    optionalScaleOffsetMode += (nullptr != scale2Shape) ? MODE_CODE_INPUT_SCALES : 0;
    optionalScaleOffsetMode += (nullptr != zeroPoint1Shape) ? MODE_CODE_INPUT_ZEROPOINTS1 : 0;
    optionalScaleOffsetMode += (nullptr != zeroPoint2Shape) ? MODE_CODE_INPUT_ZEROPOINTS2 : 0;
    OP_LOGD(context->GetNodeName(), "optionalScaleOffsetMode=%d", optionalScaleOffsetMode);
    return optionalScaleOffsetMode;
}

static inline bool GetAttrs(gert::TilingContext *context, AddLayerNormQuantTilingData *tiling, bool &isDynamicQuant,
    bool &enableAdditionalOutput)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();

    const char *qms = attrs->GetAttrPointer<char>(QUANT_MODE_IDX);
    OP_TILING_CHECK(
        nullptr == qms, OP_LOGE("GetAttrs", "Get required attr quantMode failed, tiling failed"), return false);
    std::string quantModeStr = qms;
    isDynamicQuant = (quantModeStr == "dynamic");
    OP_LOGD("GetAttrs", "isDynamicQuant=%d", isDynamicQuant);

    float eps = GetOptionalAttr<float>(attrs, EPS_IDX, (float)1e-5);
    enableAdditionalOutput = GetOptionalAttr<bool>(attrs, X_OUT_IDX, false);
    OP_LOGD("GetAttrs", "Attrs: quantMode=%s, eps=%f, additionalOutput=%d", qms, eps, enableAdditionalOutput);
    tiling->set_eps(eps);
    if (enableAdditionalOutput) {
        tiling->set_isXOut(1);
    } else {
        tiling->set_isXOut(0);
    }
    return true;
}

static inline uint64_t ComputeTilingKey(
    ge::DataType dataType, bool isDynamicQuant, TILING_TYPE tilingType, BIAS_TYPE biasType)
{
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_DATATYPE_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_DATATYPE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_DATATYPE_BF16;
    }

    if (isDynamicQuant) {
        tilingKey += TILING_DYNAMIC_QUANT;
    }

    if (tilingType == TILING_TYPE::SINGLE_ROW) {
        tilingKey += TILING_SINGLE_ROW;
    } else if (tilingType == TILING_TYPE::SLICE_EXT) {
        tilingKey += TILING_SLICE_EXT;
    }

    if (biasType == BIAS_TYPE::BROADCAST_BIAS) {
        tilingKey += TILING_X3_BROADCAST;
    } else if (biasType == BIAS_TYPE::ELEWISE_BIAS) {
        tilingKey += TILING_X3_ELEWISE;
    }
    OP_LOGI("AddLayerNormQuant", "tilingKey = %lu", tilingKey);
    return tilingKey;
}

static inline BIAS_TYPE FindBiasType(gert::TilingContext *context, int64_t numRow, int64_t numCol)
{
    BIAS_TYPE biasType = BIAS_TYPE::NO_BIAS;
    auto biasShape = context->GetOptionalInputShape(BIAS_IDX);
    if (nullptr == biasShape) {
        OP_LOGW("AddLayerNormQuant", "Bias is null");
        biasType = BIAS_TYPE::NO_BIAS;
    } else {
        OP_LOGW("AddLayerNormQuant", "Bias is not null");
        int64_t x1Size = numRow * numCol;
        int64_t biasSize = 1;
        for (size_t i = 0; i < biasShape->GetStorageShape().GetDimNum(); i++) {
            biasSize *= biasShape->GetStorageShape().GetDim(i);
        }
        biasType = x1Size == biasSize ? (BIAS_TYPE::ELEWISE_BIAS) : (BIAS_TYPE::BROADCAST_BIAS);
    }
    OP_LOGI("AddLayerNormQuant", "Bias Type is %u", static_cast<uint32_t>(biasType));
    return biasType;
}

static inline void GetSocVersion(gert::TilingContext *context, uint64_t &maxUbSize, uint32_t &maxCoreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxUbSize);
    maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGI(context->GetNodeName(), "Get Platform Info: maxUbSize = %lu, maxCoreNum = %u", maxUbSize, maxCoreNum);
}

static inline void ComputeFusedAxis(
    gert::TilingContext *context, AddLayerNormQuantTilingData *tiling, int32_t &numRow, int32_t &numCol)
{
    numRow = 1;
    for (size_t i = 0; i < context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1; i++) {
        numRow *= context->GetInputShape(0)->GetStorageShape().GetDim(i);
    }
    numCol = context->GetInputShape(0)->GetStorageShape().GetDim(
        context->GetInputShape(0)->GetStorageShape().GetDimNum() - 1);
    float tempAve = (numCol == 0) ? 0 : float(1.0 / numCol);
    tiling->set_numFirstDim(numRow);
    tiling->set_numLastDim(numCol);
    tiling->set_aveFactor(tempAve);
    OP_LOGI("AddLayerNormQuant", "numRow = %d, numCol = %d", numRow, numCol);
}

static inline void DoBlockTiling(gert::TilingContext *context, AddLayerNormQuantTilingData *tiling, uint32_t maxCoreNum,
    int32_t numRow, uint32_t &firstdimPerCore, uint32_t &numCore, int32_t &nlFirstdimPerCoreNum)
{
    firstdimPerCore = CEIL_DIV(tiling->get_numFirstDim(), maxCoreNum);
    numCore = CEIL_DIV(tiling->get_numFirstDim(), firstdimPerCore);
    tiling->set_numCore(numCore);
    context->SetBlockDim(numCore);
    tiling->set_firstDimPerCore(CEIL_DIV(numRow, numCore));
    nlFirstdimPerCoreNum = tiling->get_firstDimPerCore();
    tiling->set_firstDimPerCoreTail(numRow - nlFirstdimPerCoreNum * (numCore - 1));
    OP_LOGI("AddLayerNormQuant", "numCore = %u", numCore);
}

static inline void SetWorkSapce4AddLayerNormQuant(gert::TilingContext *context, AddLayerNormQuantTilingData *tiling,
    TILING_TYPE tilingType, int32_t numRow, int32_t numCol)
{
    size_t workspaceSize = (tilingType == TILING_TYPE::SLICE_EXT) ? (numRow * numCol * sizeof(float)) : 1;
    tiling->set_workspaceSize(workspaceSize);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = RESERVED_WORKSPACE_SIZE_910B;
    currentWorkspace[0] = workspaceSize + sysWorkspaceSize;
    OP_LOGI("AddLayerNormQuant", "workspaceSize = %zu", workspaceSize);
}

static inline void PostUbTiling(AddLayerNormQuantTilingData *tiling, int32_t firstdimPerCore, int32_t numCol,
    uint32_t colPerTime, uint32_t &rowPerTime)
{
    rowPerTime =
        (rowPerTime > static_cast<uint32_t>(firstdimPerCore)) ? static_cast<uint32_t>(firstdimPerCore) : rowPerTime;
    tiling->set_firstDimPerTime(rowPerTime);

    int32_t colMoveCnt = CEIL_DIV(numCol, colPerTime);
    int32_t colTail = (numCol % colPerTime == 0) ? colPerTime : (numCol % colPerTime);
    tiling->set_lastDimPerTime(colPerTime);
    tiling->set_colMoveCnt(colMoveCnt);
    tiling->set_colTail(colTail);

    OP_LOGI("AddLayerNormQuant",
        "rowPerTime = %u, colPerTime = %u, colMoveCnt = %d, colTail = %d",
        rowPerTime,
        colPerTime,
        colMoveCnt,
        colTail);
}

static ge::graphStatus Tiling4AddLayerNormQuant(gert::TilingContext *context)
{
    AddLayerNormQuantTilingData tiling;
    OP_LOGI(context->GetNodeName(), "Begin to do Tiling4AddLayerNormQuant");
    bool isDynamicQuant = false;
    bool enableAdditionalOutput = false;
    OP_TILING_CHECK(!GetAttrs(context, &tiling, isDynamicQuant, enableAdditionalOutput),
        OP_LOGE(context->GetNodeName(), "GetAttrs failed"),
        return ge::GRAPH_FAILED);

    auto scale1Shape = context->GetOptionalInputShape(SCALE1_IDX);
    uint32_t optionalScaleOffsetMode = ComputeOptCode(context);

    OP_TILING_CHECK(!CheckScaleOffset(isDynamicQuant, optionalScaleOffsetMode),
        OP_LOGE(context->GetNodeName(),
            "Optional input Scales and Offsets are invalid, get mode = %u, tiling failed",
            optionalScaleOffsetMode),
        return ge::GRAPH_FAILED);

    tiling.set_scaleOffsetMode(optionalScaleOffsetMode);
    tiling.set_isPerTensor(0);
    if ((nullptr != scale1Shape)) {
        OP_LOGD(context->GetNodeName(), "scale1Shape size : %ld", scale1Shape->GetStorageShape().GetShapeSize());
        if (!isDynamicQuant && (scale1Shape->GetStorageShape().GetShapeSize() == 1)) {
            tiling.set_isPerTensor(1);
            OP_LOGD(context->GetNodeName(), "PerTensor Mode Open. ");
        }
    }

    auto dataType = context->GetInputDesc(0)->GetDataType();
    int64_t bufferNum = 1;

    uint64_t maxUbSize = 0;
    uint32_t maxCoreNum = 0;
    GetSocVersion(context, maxUbSize, maxCoreNum);

    int32_t numRow = 1;
    int32_t numCol = 1;
    uint32_t firstdimPerCore = 1;
    uint32_t numCore = 1;
    int32_t nlFirstdimPerCoreNum = 1;
    ComputeFusedAxis(context, &tiling, numRow, numCol);
    DoBlockTiling(context, &tiling, maxCoreNum, numRow, firstdimPerCore, numCore, nlFirstdimPerCoreNum);

    uint32_t rowPerTime;
    uint32_t colPerTime;
    BIAS_TYPE biasType = FindBiasType(context, numRow, numCol);

    // UB Tiling
    auto tilingType = AddLayerNormQuantTilingImpl(
        maxUbSize, dataType, bufferNum, numCol, enableAdditionalOutput, biasType, rowPerTime, colPerTime);
    PostUbTiling(&tiling, firstdimPerCore, numCol, colPerTime, rowPerTime);

    SetWorkSapce4AddLayerNormQuant(context, &tiling, tilingType, numRow, numCol);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint64_t tilingKey = ComputeTilingKey(dataType, isDynamicQuant, tilingType, biasType);
    context->SetTilingKey(tilingKey);

    OP_TILING_CHECK(tilingType == TILING_TYPE::SLICE_EXT,
        OP_LOGE(context->GetNodeName(), "D-axis is too big to do AddLayerNormQuant compute"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddLayerNormQuant(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

struct AddLayerNormQuantCompileInfo {};

IMPL_OP_OPTILING(AddLayerNormQuant)
    .Tiling(Tiling4AddLayerNormQuant)
    .TilingParse<AddLayerNormQuantCompileInfo>(TilingPrepare4AddLayerNormQuant);
}  // namespace optiling
