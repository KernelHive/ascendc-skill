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
 * \file upsample_nearest_exact3d_grad.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "upsample_nearest_exact3d_grad_tiling.h"

namespace optiling {
constexpr int64_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr int64_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr int64_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr int64_t BEST_PERFORMANCE_SIZE_4 = 64;

constexpr float BEST_PERFORMANCE_SCALE_1 = 50.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 24.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 10.0f;
constexpr float BEST_PERFORMANCE_SCALE_4 = 6.0f;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t RESERVED_LENGTH = 5;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;

constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t D_INDEX = 0;
constexpr uint8_t H_INDEX = 1;
constexpr uint8_t W_INDEX = 2;

constexpr int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

class UpsampleNearestExact3dGradTiling {
public:
    explicit UpsampleNearestExact3dGradTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    bool CheckScales() const;
    void SetScale();
    inline float ComputeScaleValue(int64_t inputSize, int64_t outputSize, const float scale) const;
    inline bool GetNeedResize(int64_t inputSize, int64_t outputSize, const float scale) const;
    void GetWorkSpace(int64_t needCoreNum);
    void GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeVal() const;
    uint8_t GetDataTypeSize() const;
    int64_t GetNeedCoreNum(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumW(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumH(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumD(int64_t coreNumPlatform);
    void GetTCubeTilingW();
    void GetTCubeTilingH();
    void GetTCubeTilingD();
    void FillTilingData();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int64_t Ceil(T1 x) const;

    template <typename T1>
    inline T1 Max(T1 a, T1 b, T1 c) const;

    template <typename T1>
    inline T1 Min(T1 a, T1 b, T1 c) const;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint8_t dataTypeSize = 0;

    gert::TilingContext *tilingContext = nullptr;
    gert::Shape gradOutputShape;
    const gert::ContinuousVector *outputSizeAttr = nullptr;
    const gert::ContinuousVector *inputSizeAttr = nullptr;
    const gert::ContinuousVector *scalesAttr = nullptr;
    float scaleD = 0.0f;
    float scaleH = 0.0f;
    float scaleW = 0.0f;

    float realScaleW = 0.0f;
    float realScaleH = 0.0f;
    float realScaleD = 0.0f;
    bool needResizeW = true;
    bool needResizeH = true;
    bool needResizeD = true;

    int64_t batches = 0;
    int64_t gradOutputShapes[3] = {0};
    int64_t gradInputShapes[3] = {0};

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

    int64_t tensorSizeW = 0;
    int64_t tensorSizeH = 0;
    int64_t tensorSizeD = 0;

    int64_t tensorSizeMappingW = 0;
    int64_t tensorSizeMappingH = 0;
    int64_t tensorSizeMappingD = 0;

    int64_t slideSize = 0;
    UpsampleNearestExact3dGradTilingData tilingData;
};

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleNearestExact3dGradTiling::Init()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearestExact3dGradTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        std::printf("ERROR:srcTensor == nullptr \n");
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        std::printf("ERROR:attrs == nullptr \n");
        return ge::GRAPH_FAILED;
    }

    size_t idx = 0;
    inputSizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    outputSizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    scalesAttr = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const float *scalesArray = reinterpret_cast<const float *>(scalesAttr->GetData());
    scaleD = scalesArray[D_INDEX];
    scaleH = scalesArray[H_INDEX];
    scaleW = scalesArray[W_INDEX];

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        std::printf("ERROR:temp == nullptr \n");
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    dataTypeSize = GetDataTypeSize();

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    gradOutputShape = srcShape->GetOriginShape();

    GetShapes();
    if (!CheckScales()) {
        std::printf("ERROR:CheckScales() == false \n");
        return ge::GRAPH_FAILED;
    }
    SetScale();
    GetSlideSize();

    // 数据分核
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dGradCompileInfo *>(tilingContext->GetCompileInfo());
    int64_t coreNumPlatform = compileInfo->coreNum;
    int64_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    GetWorkSpace(needCoreNum);
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(1);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestExact3dGradTiling::GetShapes()
{
    const int64_t *inputSizeArray = reinterpret_cast<const int64_t *>(inputSizeAttr->GetData());

    batches = inputSizeArray[0] * inputSizeArray[1];
    for (int8_t i = 0; i < DIM; i++) {
        gradInputShapes[i] = inputSizeArray[i + BATCH_DIM];
        gradOutputShapes[i] = gradOutputShape[i + BATCH_DIM];
    }
    tilingData.set_batches(batches);
    tilingData.set_gradInputShapes(gradInputShapes);
    tilingData.set_gradOutputShapes(gradOutputShapes);
}

bool UpsampleNearestExact3dGradTiling::CheckScales() const
{
    float checkScalesD = ComputeScaleValue(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    float checkScalesH = ComputeScaleValue(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    float checkScalesW = ComputeScaleValue(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    return true;
}

void UpsampleNearestExact3dGradTiling::SetScale()
{
    needResizeD = GetNeedResize(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    needResizeH = GetNeedResize(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    needResizeW = GetNeedResize(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    if (!needResizeD && !needResizeH && !needResizeW) {
        needResizeW = true;
    }
    tilingData.set_needResizeD(needResizeD);
    tilingData.set_needResizeH(needResizeH);
    tilingData.set_needResizeW(needResizeW);

    realScaleD = ComputeScaleValue(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    realScaleH = ComputeScaleValue(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    realScaleW = ComputeScaleValue(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    tilingData.set_scaleD(realScaleD);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
}

inline float UpsampleNearestExact3dGradTiling::ComputeScaleValue(
    int64_t inSize, int64_t outSize, const float scale) const
{
    if (scale > ZERO_FLOAT) {
        return scale;
    } else {
        return inSize != 0 ? (static_cast<float>(outSize) / inSize) : ZERO_FLOAT;
    }
}

inline bool UpsampleNearestExact3dGradTiling::GetNeedResize(int64_t inSize, int64_t outSize, const float scale) const
{
    if (scale > ZERO_FLOAT) {
        return !FloatEqual(scale, ONE_FLOAT);
    } else {
        return inSize != outSize;
    }
}

void UpsampleNearestExact3dGradTiling::GetSlideSize()
{
    auto maxScale = Max(realScaleW, realScaleH, realScaleD);
    if (maxScale <= BEST_PERFORMANCE_SCALE_4) {
        slideSize = BEST_PERFORMANCE_SIZE_4;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_3) {
        slideSize = BEST_PERFORMANCE_SIZE_3;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_2) {
        slideSize = BEST_PERFORMANCE_SIZE_2;
    } else {
        slideSize = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slideSize(slideSize);
}

int64_t UpsampleNearestExact3dGradTiling::GetNeedCoreNum(int64_t coreNumPlatform)
{
    int64_t needCoreNumW = 0;
    int64_t needCoreNumH = 0;
    int64_t needCoreNumD = 0;
    if (needResizeW) {
        singleCoreKW = Ceil(static_cast<float>(slideSize) * realScaleW) + RESERVED_LENGTH;
        if (singleCoreKW > gradOutputShapes[W_INDEX]) {
            singleCoreKW = gradOutputShapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatform);
        GetTCubeTilingW();
    }

    if (needResizeH) {
        singleCoreKH = Ceil(static_cast<float>(slideSize) * realScaleH) + RESERVED_LENGTH;
        if (singleCoreKH > gradOutputShapes[H_INDEX]) {
            singleCoreKH = gradOutputShapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatform);
        GetTCubeTilingH();
    }

    if (needResizeD) {
        singleCoreKD = Ceil(static_cast<float>(slideSize) * realScaleD) + RESERVED_LENGTH;
        if (singleCoreKD > gradOutputShapes[D_INDEX]) {
            singleCoreKD = gradOutputShapes[D_INDEX];
        }
        needCoreNumD = GetNeedCoreNumD(coreNumPlatform);
        GetTCubeTilingD();
    }

    int64_t tensorSize = Max(tensorSizeW, tensorSizeH, tensorSizeD);
    tilingData.set_tensorSize(tensorSize + slideSize);

    int64_t tensorSizeMapping = Max(tensorSizeMappingW, tensorSizeMappingH, tensorSizeMappingD);
    tilingData.set_tensorSizeMapping(tensorSizeMapping);

    int64_t needCoreNum = Max(needCoreNumW, needCoreNumH, needCoreNumD);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

int64_t UpsampleNearestExact3dGradTiling::GetNeedCoreNumW(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[W_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeW = std::max(eachCoreSlideNum, static_cast<int64_t>(1)) * slideSize;
    tensorSizeMappingW = Ceil(static_cast<float>(tensorSizeW) * realScaleW) + slideSize;

    inputRows[W_INDEX] = batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[W_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[W_INDEX], tailAvergingRow));
    }

    int64_t needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNums[W_INDEX] = eachCoreSlideNum;
    remainders[W_INDEX] = remainder;
    tailStartSlideNums[W_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[W_INDEX] = groupCoreNum;
    tailAvergingRows[W_INDEX] = tailAvergingRow;
    needCoreNums[W_INDEX] = needCoreNum;
    return needCoreNum;
}

int64_t UpsampleNearestExact3dGradTiling::GetNeedCoreNumH(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[H_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeH = std::max(eachCoreSlideNum, static_cast<int64_t>(1)) * slideSize;
    tensorSizeMappingH = Ceil(static_cast<float>(tensorSizeH) * realScaleH) + slideSize;

    if (batches * gradOutputShapes[D_INDEX] > gradInputShapes[W_INDEX]) {
        inputRows[H_INDEX] = batches * gradOutputShapes[D_INDEX];
    } else {
        inputRows[H_INDEX] = gradInputShapes[W_INDEX];
    }
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[H_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[H_INDEX], tailAvergingRow));
    }

    int64_t needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNums[H_INDEX] = eachCoreSlideNum;
    remainders[H_INDEX] = remainder;
    tailStartSlideNums[H_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[H_INDEX] = groupCoreNum;
    tailAvergingRows[H_INDEX] = tailAvergingRow;
    needCoreNums[H_INDEX] = needCoreNum;
    return needCoreNum;
}

int64_t UpsampleNearestExact3dGradTiling::GetNeedCoreNumD(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[D_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeD = std::max(eachCoreSlideNum, static_cast<int64_t>(1)) * slideSize;
    tensorSizeMappingD = Ceil(static_cast<float>(tensorSizeD) * realScaleD) + slideSize;

    if (batches > gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX]) {
        inputRows[D_INDEX] = batches;
    } else {
        inputRows[D_INDEX] = gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX];
    }
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[D_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[D_INDEX], tailAvergingRow));
    }

    int64_t needCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        needCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNums[D_INDEX] = eachCoreSlideNum;
    remainders[D_INDEX] = remainder;
    tailStartSlideNums[D_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[D_INDEX] = groupCoreNum;
    tailAvergingRows[D_INDEX] = tailAvergingRow;
    needCoreNums[D_INDEX] = needCoreNum;
    return needCoreNum;
}

void UpsampleNearestExact3dGradTiling::GetWorkSpace(int64_t needCoreNum)
{
    uint8_t size = 32 / dataTypeSize;
    // 中间矩阵预留GM空间
    int64_t intermediateMatrixSizeW = 0;
    if (needResizeW && (needResizeH || needResizeD)) {
        intermediateMatrixSizeW =
            batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX] * gradInputShapes[W_INDEX];
        intermediateMatrixSizeW = CeilA2B(intermediateMatrixSizeW, size) * size;
    }
    int64_t intermediateMatrixSizeH = 0;
    if (needResizeH && needResizeD) {
        intermediateMatrixSizeH =
            batches * gradOutputShapes[D_INDEX] * gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX];
        intermediateMatrixSizeH = CeilA2B(intermediateMatrixSizeH, size) * size;
    }
    tilingData.set_intermediateMatrixSizeW(intermediateMatrixSizeW);
    tilingData.set_intermediateMatrixSizeH(intermediateMatrixSizeH);

    // 权重矩阵预留GM空间
    int64_t singleCoreK = Max(singleCoreKW, singleCoreKH, singleCoreKD);
    int64_t radioMatrixSize = slideSize * singleCoreK;
    tilingData.set_radioMatrixSize(radioMatrixSize);

    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = (intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * needCoreNum) * dataTypeSize +
                    WORK_SPACE_SIZE;
}

void UpsampleNearestExact3dGradTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingW;
    mmTilingW.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingW.SetOrgShape(batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX],
        gradInputShapes[W_INDEX],
        gradOutputShapes[W_INDEX]);
    mmTilingW.SetShape(batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX], slideSize, singleCoreKW);
    if (mmTilingW.GetTiling(tilingData.matmulTilingW) == -1) {
        return;
    }
}

void UpsampleNearestExact3dGradTiling::GetTCubeTilingH()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingH;
    mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingH.SetOrgShape(gradInputShapes[H_INDEX], gradInputShapes[W_INDEX], gradOutputShapes[H_INDEX]);
    mmTilingH.SetShape(slideSize, gradInputShapes[W_INDEX], singleCoreKH);
    if (mmTilingH.GetTiling(tilingData.matmulTilingH) == -1) {
        return;
    }
}

void UpsampleNearestExact3dGradTiling::GetTCubeTilingD()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingD;
    mmTilingD.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingD.SetOrgShape(
        gradInputShapes[D_INDEX], gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX], gradOutputShapes[D_INDEX]);
    mmTilingD.SetShape(slideSize, gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX], singleCoreKD);
    if (mmTilingD.GetTiling(tilingData.matmulTilingD) == -1) {
        return;
    }
}

void UpsampleNearestExact3dGradTiling::FillTilingData()
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

uint8_t UpsampleNearestExact3dGradTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint8_t UpsampleNearestExact3dGradTiling::GetDataTypeVal() const
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

template <typename T1, typename T2>
inline T1 UpsampleNearestExact3dGradTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int64_t UpsampleNearestExact3dGradTiling::Ceil(T1 x) const
{
    int64_t floorX = int64_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

template <typename T1>
inline T1 UpsampleNearestExact3dGradTiling::Max(T1 a, T1 b, T1 c) const
{
    if (a > b) {
        return a > c ? a : c;
    }
    return b > c ? b : c;
}

template <typename T1>
inline T1 UpsampleNearestExact3dGradTiling::Min(T1 a, T1 b, T1 c) const
{
    if (a < b) {
        return a < c ? a : c;
    }
    return b < c ? b : c;
}

static ge::graphStatus Tiling4UpsampleNearestExact3dGradTiling(gert::TilingContext *context)
{
    UpsampleNearestExact3dGradTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleNearest3dGradCompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearestExact3dGrad)
    .Tiling(Tiling4UpsampleNearestExact3dGradTiling)
    .TilingParse<UpsampleNearest3dGradCompileInfo>(TilingPrepareTiling);

}  // namespace optiling
