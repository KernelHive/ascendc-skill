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
 * \file upsample_trilinear3d_backward_tiling_func.cpp
 * \brief
 */
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "upsample_trilinear3d_backward_tiling_def.h"

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

constexpr int64_t SLIDE_SIZE = 16;
constexpr float MIN_SUPPORT_SCALE = 0.02f;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

static constexpr int32_t ALIGN_CORNERS_IDX = 2;
static constexpr int32_t SCALE_D_IDX = 3;
static constexpr int32_t SCALE_H_IDX = 4;
static constexpr int32_t SCALE_W_IDX = 5;

constexpr uint8_t RESERVED_LENGTH = 4;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr uint8_t BLOCK_SIZE = 32;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;

constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t D_INDEX = 0;
constexpr uint8_t H_INDEX = 1;
constexpr uint8_t W_INDEX = 2;

static constexpr int RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;

class UpsampleTrilinear3dBackwardTiling {
public:
    explicit UpsampleTrilinear3dBackwardTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    inline bool FloatEqual(const float a, const float b) const;
    inline float ComputeScales(const float scale, const int64_t inputSize, const int64_t outputSize) const;
    inline float AreaPixelComputeScale(const int64_t inputSize, const int64_t outputSize, const float scale) const;
    uint8_t GetDataTypeVal() const;
    int64_t GetSingleCoreK(const float scale, const int64_t outputSize) const;
    bool CheckShapes() const;
    bool CheckScales() const;

    void GetShapes();
    void GetScales();
    void GetWorkSpace(const int64_t needCoreNum);
    int64_t GetNeedCoreNum(const int64_t coreNumPlatform);
    int64_t GetNeedCoreNumByDirection(const int64_t coreNumPlatform, const uint8_t direction);
    void GetTCubeTilingW();
    void GetTCubeTilingH();
    void GetTCubeTilingD();
    void FillTilingData();

    template <typename T>
    inline auto GetOptionalAttr(const gert::RuntimeAttrs *attrs, const int32_t idx, const T &defaultValue) const -> T;

    template <typename T>
    inline auto Max(T a, T b, T c) const -> T;

    template <typename T>
    inline int64_t Ceil(T x) const;

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext *tilingContext = nullptr;
    gert::Shape outputShape;
    gert::Shape inputShape;

    bool alignCorners = false;
    float scaleD = 0.0f;
    float scaleH = 0.0f;
    float scaleW = 0.0f;

    bool needResizeW = true;
    bool needResizeH = true;
    bool needResizeD = true;

    int64_t batches = 0;
    int64_t inputShapes[3] = {0};
    int64_t outputShapes[3] = {0};
    float realScaleW = 0.0f;
    float realScaleH = 0.0f;
    float realScaleD = 0.0f;

    int64_t eachCoreSlideNums[3] = {0, 0, 0};
    int64_t remainders[3] = {0, 0, 0};
    int64_t tailStartSlideNums[3] = {0, 0, 0};
    int64_t groupCoreNums[3] = {0, 0, 0};
    int64_t inputRows[3] = {0, 0, 0};
    int64_t tailAvergingRows[3] = {0, 0, 0};
    int64_t needCoreNums[3] = {0, 0, 0};

    int64_t singleCoreKW = 0;
    int64_t singleCoreKH = 0;
    int64_t singleCoreKD = 0;

    UpsampleTrilinear3dBackwardTilingData tilingData;
};

ge::graphStatus UpsampleTrilinear3dBackwardTiling::Init()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleTrilinear3dBackwardTiling::RunBigKernelTiling()
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    alignCorners = GetOptionalAttr<bool>(attrs, ALIGN_CORNERS_IDX, false);
    scaleD = GetOptionalAttr<float>(attrs, SCALE_D_IDX, ZERO_FLOAT);
    scaleH = GetOptionalAttr<float>(attrs, SCALE_H_IDX, ZERO_FLOAT);
    scaleW = GetOptionalAttr<float>(attrs, SCALE_W_IDX, ZERO_FLOAT);

    outputShape = tilingContext->GetOutputShape(0)->GetStorageShape();
    inputShape = tilingContext->GetInputShape(0)->GetStorageShape();
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    GetShapes();
    if (!CheckShapes() || !CheckScales()) {
        return ge::GRAPH_FAILED;
    }
    GetScales();

    // 数据分核
    auto compileInfo = reinterpret_cast<const UpsampleTrilinearBackwardCompileInfo *>(tilingContext->GetCompileInfo());
    int64_t coreNumPlatform = compileInfo->coreNum;
    int64_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    GetWorkSpace(needCoreNum);
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(1);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

bool UpsampleTrilinear3dBackwardTiling::CheckShapes() const
{
    if (inputShapes[D_INDEX] <= 0 || inputShapes[H_INDEX] <= 0 || inputShapes[W_INDEX] <= 0) {
        return false;
    }
    if (outputShapes[D_INDEX] <= 0 || outputShapes[H_INDEX] <= 0 || outputShapes[W_INDEX] <= 0) {
        return false;
    }
    return true;
}

bool UpsampleTrilinear3dBackwardTiling::CheckScales() const
{
    float scalesD = ComputeScales(scaleD, inputShapes[D_INDEX], outputShapes[D_INDEX]);
    float scalesH = ComputeScales(scaleH, inputShapes[H_INDEX], outputShapes[H_INDEX]);
    float scalesW = ComputeScales(scaleW, inputShapes[W_INDEX], outputShapes[W_INDEX]);
    OP_TILING_CHECK(scalesD < MIN_SUPPORT_SCALE || scalesH < MIN_SUPPORT_SCALE || scalesW < MIN_SUPPORT_SCALE,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext->GetNodeName(),
            "scalesD, scalesH and scalesW are too small, scalesD [%f], scalesH [%f], scalesW [%f].",
            scalesD,
            scalesH,
            scalesW),
        return false);
    return true;
}

void UpsampleTrilinear3dBackwardTiling::GetShapes()
{
    batches = inputShape.GetDim(0) * inputShape.GetDim(1);
    for (int8_t i = 0; i < DIM; i++) {
        inputShapes[i] = outputShape.GetDim(i + BATCH_DIM);
        outputShapes[i] = inputShape.GetDim(i + BATCH_DIM);
    }

    tilingData.set_batches(batches);
    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);
}

void UpsampleTrilinear3dBackwardTiling::GetScales()
{
    realScaleD = AreaPixelComputeScale(inputShapes[D_INDEX], outputShapes[D_INDEX], scaleD);
    realScaleH = AreaPixelComputeScale(inputShapes[H_INDEX], outputShapes[H_INDEX], scaleH);
    realScaleW = AreaPixelComputeScale(inputShapes[W_INDEX], outputShapes[W_INDEX], scaleW);

    needResizeD = !FloatEqual(realScaleD, ONE_FLOAT);
    needResizeH = !FloatEqual(realScaleH, ONE_FLOAT);
    needResizeW = !FloatEqual(realScaleW, ONE_FLOAT);
    if (!needResizeD && !needResizeH && !needResizeW) {
        needResizeW = true;
    }

    tilingData.set_slideSize(SLIDE_SIZE);
    tilingData.set_scaleD(realScaleD);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_needResizeD(needResizeD);
    tilingData.set_needResizeH(needResizeH);
    tilingData.set_needResizeW(needResizeW);
}

inline float UpsampleTrilinear3dBackwardTiling::ComputeScales(
    const float scale, const int64_t inputSize, const int64_t outputSize) const
{
    if (scale > ZERO_FLOAT) {
        return scale;
    } else {
        return outputSize != 0 ? (static_cast<float>(inputSize) / outputSize) : ZERO_FLOAT;
    }
}

inline float UpsampleTrilinear3dBackwardTiling::AreaPixelComputeScale(
    const int64_t inputSize, const int64_t outputSize, const float scale) const
{
    if (outputSize == inputSize) {
        return ONE_FLOAT;
    }
    if (alignCorners) {
        return outputSize > 1 ? static_cast<float>(inputSize - 1) / (outputSize - 1) : ZERO_FLOAT;
    } else {
        return ComputeScales(scale, inputSize, outputSize);
    }
}

int64_t UpsampleTrilinear3dBackwardTiling::GetNeedCoreNum(const int64_t coreNumPlatform)
{
    int64_t needCoreNumW = 0;
    int64_t needCoreNumH = 0;
    int64_t needCoreNumD = 0;
    if (needResizeW) {
        singleCoreKW = GetSingleCoreK(realScaleW, outputShapes[W_INDEX]);
        needCoreNumW = GetNeedCoreNumByDirection(coreNumPlatform, W_INDEX);
        GetTCubeTilingW();
    }

    if (needResizeH) {
        singleCoreKH = GetSingleCoreK(realScaleH, outputShapes[H_INDEX]);
        needCoreNumH = GetNeedCoreNumByDirection(coreNumPlatform, H_INDEX);
        GetTCubeTilingH();
    }

    if (needResizeD) {
        singleCoreKD = GetSingleCoreK(realScaleD, outputShapes[D_INDEX]);
        needCoreNumD = GetNeedCoreNumByDirection(coreNumPlatform, D_INDEX);
        GetTCubeTilingD();
    }

    int64_t needCoreNum = Max(needCoreNumW, needCoreNumH, needCoreNumD);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

int64_t UpsampleTrilinear3dBackwardTiling::GetNeedCoreNumByDirection(
    const int64_t coreNumPlatform, const uint8_t direction)
{
    int64_t slideNum = CeilA2B(inputShapes[direction], SLIDE_SIZE);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    if (direction == W_INDEX) {
        inputRows[direction] = batches * outputShapes[D_INDEX] * outputShapes[H_INDEX];
    } else if (direction == H_INDEX) {
        inputRows[direction] = batches * outputShapes[D_INDEX];
    } else {
        inputRows[direction] = batches;
    }
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = SLIDE_SIZE;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[direction], groupCoreNum), SLIDE_SIZE);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[direction], tailAvergingRow));
    }

    int64_t needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNums[direction] = eachCoreSlideNum;
    remainders[direction] = remainder;
    tailStartSlideNums[direction] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[direction] = groupCoreNum;
    tailAvergingRows[direction] = tailAvergingRow;
    needCoreNums[direction] = needCoreNum;
    return needCoreNum;
}

void UpsampleTrilinear3dBackwardTiling::GetWorkSpace(const int64_t needCoreNum)
{
    uint8_t dataTypeSize = dataType == ge::DT_FLOAT ? BYTE_LEN_4 : BYTE_LEN_2;
    uint8_t size = BLOCK_SIZE / dataTypeSize;
    // 中间矩阵预留GM空间
    int64_t intermediateMatrixSizeW = 0;
    if (needResizeW && (needResizeH || needResizeD)) {
        intermediateMatrixSizeW = batches * outputShapes[D_INDEX] * outputShapes[H_INDEX] * inputShapes[W_INDEX];
        intermediateMatrixSizeW = CeilA2B(intermediateMatrixSizeW, size) * size;
    }
    int64_t intermediateMatrixSizeH = 0;
    if (needResizeH && needResizeD) {
        intermediateMatrixSizeH = batches * outputShapes[D_INDEX] * inputShapes[H_INDEX] * inputShapes[W_INDEX];
        intermediateMatrixSizeH = CeilA2B(intermediateMatrixSizeH, size) * size;
    }
    tilingData.set_intermediateMatrixSizeW(intermediateMatrixSizeW);
    tilingData.set_intermediateMatrixSizeH(intermediateMatrixSizeH);

    // 权重矩阵预留GM空间
    int64_t singleCoreK = Max(singleCoreKW, singleCoreKH, singleCoreKD);
    int64_t radioMatrixSize = SLIDE_SIZE * singleCoreK;
    tilingData.set_radioMatrixSize(radioMatrixSize);

    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = (intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * needCoreNum) * dataTypeSize +
                    RESERVED_WORKSPACE_SIZE;
}

void UpsampleTrilinear3dBackwardTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingW;
    mmTilingW.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingW.SetOrgShape(
        batches * outputShapes[D_INDEX] * outputShapes[H_INDEX], inputShapes[W_INDEX], outputShapes[W_INDEX]);
    mmTilingW.SetShape(batches * outputShapes[D_INDEX] * outputShapes[H_INDEX], SLIDE_SIZE, singleCoreKW);
    if (mmTilingW.GetTiling(tilingData.matmulTilingW) == -1) {
        return;
    }
}

void UpsampleTrilinear3dBackwardTiling::GetTCubeTilingH()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingH;
    mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingH.SetOrgShape(inputShapes[H_INDEX], inputShapes[W_INDEX], outputShapes[H_INDEX]);
    mmTilingH.SetShape(SLIDE_SIZE, inputShapes[W_INDEX], singleCoreKH);
    if (mmTilingH.GetTiling(tilingData.matmulTilingH) == -1) {
        return;
    }
}

void UpsampleTrilinear3dBackwardTiling::GetTCubeTilingD()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingD;
    mmTilingD.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingD.SetOrgShape(inputShapes[D_INDEX], inputShapes[H_INDEX] * inputShapes[W_INDEX], outputShapes[D_INDEX]);
    mmTilingD.SetShape(SLIDE_SIZE, inputShapes[H_INDEX] * inputShapes[W_INDEX], singleCoreKD);
    if (mmTilingD.GetTiling(tilingData.matmulTilingD) == -1) {
        return;
    }
}

void UpsampleTrilinear3dBackwardTiling::FillTilingData()
{
    tilingData.set_eachCoreSlideNums(eachCoreSlideNums);
    tilingData.set_remainders(remainders);
    tilingData.set_tailStartSlideNums(tailStartSlideNums);
    tilingData.set_groupCoreNums(groupCoreNums);
    tilingData.set_inputRows(inputRows);
    tilingData.set_tailAvergingRows(tailAvergingRows);
    tilingData.set_needCoreNums(needCoreNums);
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

int64_t UpsampleTrilinear3dBackwardTiling::GetSingleCoreK(const float scale, const int64_t outputSize) const
{
    int64_t singleCoreK = outputSize;
    if (!FloatEqual(scale, ZERO_FLOAT)) {
        singleCoreK = std::min(Ceil((SLIDE_SIZE + RESERVED_LENGTH) / scale) + RESERVED_LENGTH, outputSize);
    }
    return singleCoreK;
}

uint8_t UpsampleTrilinear3dBackwardTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return HALF_TYPE;
        case ge::DT_FLOAT:
            return FLOAT_TYPE;
        case ge::DT_BF16:
            return BFLOAT_TYPE;
        default:
            return 0;
    }
}

inline bool UpsampleTrilinear3dBackwardTiling::FloatEqual(const float a, const float b) const
{
    const float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

template <typename T>
inline auto UpsampleTrilinear3dBackwardTiling::GetOptionalAttr(
    const gert::RuntimeAttrs *attrs, const int32_t idx, const T &defaultValue) const -> T
{
    const T *attrPtr = attrs->GetAttrPointer<T>(idx);
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

template <typename T>
inline auto UpsampleTrilinear3dBackwardTiling::Max(T a, T b, T c) const -> T
{
    if (a > b) {
        return a > c ? a : c;
    }
    return b > c ? b : c;
}

template <typename T>
inline int64_t UpsampleTrilinear3dBackwardTiling::Ceil(T x) const
{
    int64_t floorX = int64_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

template <typename T1, typename T2>
inline auto UpsampleTrilinear3dBackwardTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

static ge::graphStatus Tiling4UpsampleTrilinear(gert::TilingContext *context)
{
    UpsampleTrilinear3dBackwardTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepare4UpsampleTrilinear(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleTrilinearBackwardCompileInfo>(context);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleTrilinear3dBackward)
    .Tiling(Tiling4UpsampleTrilinear)
    .TilingParse<UpsampleTrilinearBackwardCompileInfo>(TilingPrepare4UpsampleTrilinear);
}  // namespace optiling
