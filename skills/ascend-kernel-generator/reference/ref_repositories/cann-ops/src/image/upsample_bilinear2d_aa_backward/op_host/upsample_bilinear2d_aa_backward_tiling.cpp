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
 * \file upsample_bilinear2d_aa_backward.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bilinear2d_aa_backward_tiling.h"

using namespace ge;
namespace optiling {
constexpr int64_t BEST_PERFORMANCE_SIZE = 16;
constexpr int64_t GOOD_PERFORMANCE_SIZE = 8;
constexpr int64_t RESERVED_LENGTH = 4;

constexpr float MIN_BEST_SCALE = 0.04f;
constexpr float MIN_SUPPORT_SCALE = 0.02f;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr uint32_t MATRIX_NUM = 2;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

class UpsampleBilinear2dAABackwardTiling {
public:
    explicit UpsampleBilinear2dAABackwardTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling();

private:
    float ComputeScales(int64_t inSize, int64_t outSize, const float *scale) const;
    bool CheckScales() const;
    void SetScale();
    inline float ComputeScaleValue(int64_t inSize, int64_t outSize, const float *scale) const;
    inline bool GetNeedResize(int64_t inSize, int64_t outSize, const float *scale) const;
    void GetWorkSpace(uint32_t needCoreNum);
    void GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeVal() const;
    uint8_t GetDataTypeSize() const;
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void GetTCubeTilingW();
    void GetTCubeTilingH();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slideSize = BEST_PERFORMANCE_SIZE;
    UpsampleBilinear2dAABackwardTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint8_t dataTypeSize = BYTE_LEN_4;
    gert::Shape inputShape;
    const bool *alignCorners = nullptr;
    const float *scaleH = nullptr;
    const float *scaleW = nullptr;
    float realScaleH = 0.0f;
    float realScaleW = 0.0f;
    const gert::ContinuousVector *inputSize = nullptr;
    const gert::ContinuousVector *outputSize = nullptr;
    int64_t slideStartListW[MAX_CORE_CONT] = {0};
    int64_t slideEndListW[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartListW[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndListW[MAX_CORE_CONT] = {0};
    int64_t tailRowStartListW[MAX_CORE_CONT] = {0};
    int64_t tailRowEndListW[MAX_CORE_CONT] = {0};

    int64_t slideStartListH[MAX_CORE_CONT] = {0};
    int64_t slideEndListH[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartListH[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndListH[MAX_CORE_CONT] = {0};
    int64_t tailRowStartListH[MAX_CORE_CONT] = {0};
    int64_t tailRowEndListH[MAX_CORE_CONT] = {0};

    int64_t outputShapes[4] = {0};
    int64_t inputShapes[4] = {0};

    int64_t singleCoreKW = 0;
    int64_t singleCoreKH = 0;

    bool needResizeW = true;
    bool needResizeH = true;
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

ge::graphStatus UpsampleBilinear2dAABackwardTiling::Init() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleBilinear2dAABackwardTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t idx = 0;
    outputSize = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    inputSize = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    alignCorners = attrs->GetAttrPointer<bool>(idx++);
    scaleH = attrs->GetAttrPointer<float>(idx++);
    scaleW = attrs->GetAttrPointer<float>(idx++);

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    dataTypeSize = GetDataTypeSize();

    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();

    if (!CheckScales()) {
        return ge::GRAPH_FAILED;
    }
    SetScale();
    GetSlideSize();
    GetShapes();

    auto compileInfo =
        reinterpret_cast<const UpsampleBilinear2dAABackwardCompileInfo *>(tilingContext->GetCompileInfo());
    uint32_t coreNumPlatform = 0;
    if (compileInfo != nullptr) {
        coreNumPlatform = compileInfo->coreNum;
    }
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    GetWorkSpace(needCoreNum);
    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(1);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

float UpsampleBilinear2dAABackwardTiling::ComputeScales(int64_t inSize, int64_t outSize, const float *scale) const
{
    if (*scale > ZERO_FLOAT) {
        return *scale;
    } else {
        return outSize != 0 ? (static_cast<float>(inSize) / outSize) : ZERO_FLOAT;
    }
}

bool UpsampleBilinear2dAABackwardTiling::CheckScales() const
{
    const int64_t *inputSizeArray = reinterpret_cast<const int64_t *>(inputSize->GetData());
    float scalesH = ComputeScales(inputSizeArray[H_INDEX], inputShape.GetDim(H_INDEX), scaleH);
    float scalesW = ComputeScales(inputSizeArray[W_INDEX], inputShape.GetDim(W_INDEX), scaleW);
    return true;
}

void UpsampleBilinear2dAABackwardTiling::GetSlideSize()
{
    slideSize = BEST_PERFORMANCE_SIZE;
    if (realScaleH > ZERO_FLOAT && realScaleH < MIN_BEST_SCALE) {
        slideSize = GOOD_PERFORMANCE_SIZE;
    }
    if (realScaleW > ZERO_FLOAT && realScaleW < MIN_BEST_SCALE) {
        slideSize = GOOD_PERFORMANCE_SIZE;
    }
    tilingData.set_slideSize(slideSize);
}

void UpsampleBilinear2dAABackwardTiling::SetScale()
{
    const int64_t *inputSizeArray = reinterpret_cast<const int64_t *>(inputSize->GetData());
    needResizeH = GetNeedResize(inputSizeArray[H_INDEX], inputShape.GetDim(H_INDEX), scaleH);
    needResizeW = GetNeedResize(inputSizeArray[W_INDEX], inputShape.GetDim(W_INDEX), scaleW);
    if (!needResizeH && !needResizeW) {
        needResizeH = true;
    }
    tilingData.set_needResizeH(needResizeH);
    tilingData.set_needResizeW(needResizeW);

    realScaleH = ComputeScaleValue(inputSizeArray[H_INDEX], inputShape.GetDim(H_INDEX), scaleH);
    realScaleW = ComputeScaleValue(inputSizeArray[W_INDEX], inputShape.GetDim(W_INDEX), scaleW);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);

    float supportH = (realScaleH >= ONE_FLOAT) ? realScaleH : ONE_FLOAT;
    float supportW = (realScaleW >= ONE_FLOAT) ? realScaleW : ONE_FLOAT;
    tilingData.set_supportH(supportH);
    tilingData.set_supportW(supportW);

    int16_t maxInterpSizeH = Ceil(supportH) * 2 + 1;
    int16_t maxInterpSizeW = Ceil(supportW) * 2 + 1;
    tilingData.set_maxInterpSizeH(maxInterpSizeH);
    tilingData.set_maxInterpSizeW(maxInterpSizeW);

    float invscaleH = (realScaleH >= ONE_FLOAT) ? ONE_FLOAT / realScaleH : ONE_FLOAT;
    float invscaleW = (realScaleW >= ONE_FLOAT) ? ONE_FLOAT / realScaleW : ONE_FLOAT;
    tilingData.set_invscaleH(invscaleH);
    tilingData.set_invscaleW(invscaleW);
}

inline float UpsampleBilinear2dAABackwardTiling::ComputeScaleValue(
    int64_t inSize, int64_t outSize, const float *scale) const
{
    if (*alignCorners) {
        if (outSize > 1) {
            return static_cast<float>(inSize - 1) / (outSize - 1);
        } else {
            return ZERO_FLOAT;
        }
    } else {
        return (scale != nullptr && *scale > ZERO_FLOAT) ? static_cast<float>(*scale)
                                                         : (static_cast<float>(inSize) / outSize);
    }
}

inline bool UpsampleBilinear2dAABackwardTiling::GetNeedResize(int64_t inSize, int64_t outSize, const float *scale) const
{
    if (*alignCorners) {
        return inSize != outSize;
    } else {
        return (scale != nullptr && *scale > ZERO_FLOAT) ? !FloatEqual(*scale, ONE_FLOAT) : inSize != outSize;
    }
}

void UpsampleBilinear2dAABackwardTiling::GetShapes()
{
    const int64_t *inputSizeArray = reinterpret_cast<const int64_t *>(inputSize->GetData());
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        inputShapes[i] = inputShape.GetDim(i);
        outputShapes[i] = inputSizeArray[i];
    }
    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);
}

uint32_t UpsampleBilinear2dAABackwardTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    int64_t outLength = slideSize + RESERVED_LENGTH;
    if (needResizeW) {
        singleCoreKW = inputShapes[W_INDEX];
        if (realScaleW > ZERO_FLOAT) {
            singleCoreKW = Ceil(outLength / realScaleW) + Ceil(tilingData.get_maxInterpSizeW()) + RESERVED_LENGTH;
            if (singleCoreKW > inputShapes[W_INDEX]) {
                singleCoreKW = inputShapes[W_INDEX];
            }
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatform);
        GetTCubeTilingW();
    }

    if (needResizeH) {
        singleCoreKH = inputShapes[H_INDEX];
        if (realScaleH > ZERO_FLOAT) {
            singleCoreKH = Ceil(outLength / realScaleH) + Ceil(tilingData.get_maxInterpSizeH()) + RESERVED_LENGTH;
            if (singleCoreKH > inputShapes[H_INDEX]) {
                singleCoreKH = inputShapes[H_INDEX];
            }
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatform);
        GetTCubeTilingH();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

uint32_t UpsampleBilinear2dAABackwardTiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    int64_t outputW = outputShapes[W_INDEX];
    int64_t slideNum = CeilA2B(outputW, slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // H维度总数
    int64_t inputH = inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShapes[H_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = BEST_PERFORMANCE_SIZE;

    if (remainder != 0) {
        int64_t minAvergingRows = BEST_PERFORMANCE_SIZE;
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(inputH, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputH, tailAvergingRows));
    }
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartListW[coreIndex] = coreIndex * eachCoreSlideNum * slideSize;
        slideEndListW[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slideSize;
        if (remainder != 0) {
            // 尾块处理
            int64_t groupIndex = groupCoreNum > 0 ? coreIndex / groupCoreNum : 0;
            if (groupIndex < remainder) {
                tailSlideStartListW[coreIndex] = (tailStartSlideNum + groupIndex) * slideSize;
                tailSlideEndListW[coreIndex] =
                    std::min(tailSlideStartListW[coreIndex] + slideSize, static_cast<int64_t>(outputW));
                int64_t coreIndexInGroup = groupCoreNum > 0 ? coreIndex % groupCoreNum : 0;
                tailRowStartListW[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndListW[coreIndex] =
                    std::min(tailRowStartListW[coreIndex] + tailAvergingRows, static_cast<int64_t>(inputH));
                needCoreNum++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    }
    tilingData.set_needCoreNumW(needCoreNum);
    return needCoreNum;
}

uint32_t UpsampleBilinear2dAABackwardTiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t outputH = outputShapes[H_INDEX];
    int64_t slideNum = CeilA2B(outputH, slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

    // W维度总数
    int64_t inputW = outputShapes[W_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = BEST_PERFORMANCE_SIZE;
    if (remainder != 0) {
        int64_t minAvergingRows = BEST_PERFORMANCE_SIZE;
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(inputW, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputW, tailAvergingRows));
    }

    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartListH[coreIndex] = coreIndex * eachCoreSlideNum * slideSize;
        slideEndListH[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slideSize;
        if (remainder != 0) {
            // 尾块处理
            int64_t groupIndex = groupCoreNum > 0 ? coreIndex / groupCoreNum : 0;
            if (groupIndex < remainder) {
                tailSlideStartListH[coreIndex] = (tailStartSlideNum + groupIndex) * slideSize;
                tailSlideEndListH[coreIndex] =
                    std::min(tailSlideStartListH[coreIndex] + slideSize, static_cast<int64_t>(outputH));
                int64_t coreIndexInGroup = groupCoreNum > 0 ? coreIndex % groupCoreNum : 0;
                tailRowStartListH[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndListH[coreIndex] =
                    std::min(tailRowStartListH[coreIndex] + tailAvergingRows, static_cast<int64_t>(inputW));
                needCoreNum++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    }
    tilingData.set_needCoreNumH(needCoreNum);
    return needCoreNum;
}

void UpsampleBilinear2dAABackwardTiling::GetWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    // 中间tensor
    uint64_t intermediateMatrixSize =
        outputShapes[N_INDEX] * outputShapes[C_INDEX] * inputShapes[H_INDEX] * outputShapes[W_INDEX];

    // 每个核的系数矩阵，每个核申请两个workspace空间，避免相互覆盖
    int64_t singleCoreK = singleCoreKW > singleCoreKH ? singleCoreKW : singleCoreKH;
    uint32_t radioMatrixWorkspaceSize = slideSize * singleCoreK * dataTypeSize;

    if (workspaces != nullptr) {
        workspaces[0] = intermediateMatrixSize * dataTypeSize + radioMatrixWorkspaceSize * needCoreNum * MATRIX_NUM +
                        WORK_SPACE_SIZE;
    }
    tilingData.set_radioMatrixSizeW(slideSize * singleCoreKW);
    tilingData.set_radioMatrixSizeH(slideSize * singleCoreKH);
    tilingData.set_intermediateMatrixSize(intermediateMatrixSize);
}

void UpsampleBilinear2dAABackwardTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingW;
    mmTilingW.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingW.SetOrgShape(
        inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShape[H_INDEX], outputShapes[W_INDEX], inputShapes[W_INDEX]);
    mmTilingW.SetShape(inputShapes[N_INDEX] * inputShapes[C_INDEX] * inputShape[H_INDEX], slideSize, singleCoreKW);
    if (mmTilingW.GetTiling(tilingData.matmulTilingW) == -1) {
        return;
    }
}

void UpsampleBilinear2dAABackwardTiling::GetTCubeTilingH()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingH;
    mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingH.SetOrgShape(outputShapes[H_INDEX], outputShapes[W_INDEX], inputShapes[H_INDEX]);
    mmTilingH.SetShape(slideSize, outputShapes[W_INDEX], singleCoreKH);
    if (mmTilingH.GetTiling(tilingData.matmulTilingH) == -1) {
        return;
    }
}

void UpsampleBilinear2dAABackwardTiling::FillTilingData()
{
    tilingData.set_slideStartListW(slideStartListW);
    tilingData.set_slideEndListW(slideEndListW);
    tilingData.set_tailSlideStartListW(tailSlideStartListW);
    tilingData.set_tailSlideEndListW(tailSlideEndListW);
    tilingData.set_tailRowStartListW(tailRowStartListW);
    tilingData.set_tailRowEndListW(tailRowEndListW);

    tilingData.set_slideStartListH(slideStartListH);
    tilingData.set_slideEndListH(slideEndListH);
    tilingData.set_tailSlideStartListH(tailSlideStartListH);
    tilingData.set_tailSlideEndListH(tailSlideEndListH);
    tilingData.set_tailRowStartListH(tailRowStartListH);
    tilingData.set_tailRowEndListH(tailRowEndListH);

    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

uint8_t UpsampleBilinear2dAABackwardTiling::GetDataTypeSize() const
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

uint8_t UpsampleBilinear2dAABackwardTiling::GetDataTypeVal() const
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
inline T1 UpsampleBilinear2dAABackwardTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBilinear2dAABackwardTiling::Ceil(T1 x) const
{
    int32_t floorX = int32_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

static ge::graphStatus Tiling4UpsampleBilinear2dAABackwardTiling(gert::TilingContext *context)
{
    UpsampleBilinear2dAABackwardTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleBilinear2dAABackwardCompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleBilinear2dAABackward)
    .Tiling(Tiling4UpsampleBilinear2dAABackwardTiling)
    .TilingParse<UpsampleBilinear2dAABackwardCompileInfo>(TilingPrepareTiling);

}  // namespace optiling
