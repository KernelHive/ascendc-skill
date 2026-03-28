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
 * \file upsample_nearest_exact2d_grad.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "upsample_nearest_exact2d_grad_tiling.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_4 = 64;

constexpr uint32_t BEST_PERFORMANCE_SCALE_1 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_2 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_3 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_4 = 5;

constexpr uint32_t BYTE = 8;
constexpr uint8_t TILING_KEY_HALF = 1;
constexpr uint8_t TILING_KEY_FLOAT = 2;
constexpr uint8_t TILING_KEY_BF16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;
constexpr int32_t RESERVED_VALUE = 4;

constexpr float ZERO_FLOAT = 0.0f;

class UpsampleNearestExact2dGradTiling {
public:
    explicit UpsampleNearestExact2dGradTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void setScale();
    inline float compute_scale_value(int64_t input_size, int64_t output_size, const float *scale);
    void getWorkSpace(uint32_t needCoreNum);
    void getOutputShape();
    void getSlideSize();
    uint8_t GetDataTypeSize();
    uint8_t GetDataTypeVal();
    uint8_t GetTilingKeyVal();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void getTCubeTiling_w(uint32_t needCoreNum);
    void getTCubeTiling_h(uint32_t needCoreNum);
    void setSingleCoreK();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b);

    template <typename T1>
    inline int32_t Ceil(T1 x);

private:
    int64_t slide_size = 0;
    UpsampleNearestExact2dGradTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize = 0;
    uint16_t funcType = 0;
    gert::Shape input_shape;
    uint8_t dim = 0;
    const float *scale_h = nullptr;
    const float *scale_w = nullptr;
    float realScale_h = 0.0f;
    float realScale_w = 0.0f;
    const gert::ContinuousVector *output_size = nullptr;
    const gert::ContinuousVector *input_size = nullptr;
    int64_t slideStartList_w[MAX_CORE_CONT] = {0};
    int64_t slideEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_w[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_w[MAX_CORE_CONT] = {0};

    int64_t slideStartList_h[MAX_CORE_CONT] = {0};
    int64_t slideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailSlideEndList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowStartList_h[MAX_CORE_CONT] = {0};
    int64_t tailRowEndList_h[MAX_CORE_CONT] = {0};
    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    int32_t tailBatchStartListH[MAX_CORE_CONT] = {0};
    int32_t tailBatchEndListH[MAX_CORE_CONT] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int32_t singleCoreK_w = 0;
    int32_t singleCoreK_h = 0;
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

void UpsampleNearestExact2dGradTiling::setScale()
{
    if (dim == H_INDEX) {
        const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
        realScale_h = compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[H_INDEX], scale_h);
        realScale_w = compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[W_INDEX], scale_w);

        tilingData.set_scale_h(realScale_h);
        tilingData.set_scale_w(realScale_w);
        float invscale_w = (realScale_w >= 1.0) ? 1.0 / realScale_w : 1.0;
        float invscale_h = (realScale_h >= 1.0) ? 1.0 / realScale_h : 1.0;

        tilingData.set_invscale_w(invscale_w);
        tilingData.set_invscale_h(invscale_h);
    }
}

inline float UpsampleNearestExact2dGradTiling::compute_scale_value(
    int64_t input_size, int64_t output_size, const float *scale)
{
    if (scale != nullptr && *scale > 0) {
        return static_cast<float>(*scale);
    } else {
        if (output_size > 0) {
            return static_cast<float>(input_size) / output_size;
        }
    }
    return ZERO_FLOAT;
}

ge::graphStatus UpsampleNearestExact2dGradTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    input_size = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    scale_h = attrs->GetAttrPointer<float>(H_INDEX);
    scale_w = attrs->GetAttrPointer<float>(W_INDEX);
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = temp->GetDataType();
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    auto src_shape = tilingContext->GetInputShape(0);
    // 固定是2
    dim = src_shape->GetStorageShape().GetDimNum() - 2;

    input_shape = src_shape->GetOriginShape();

    auto compileInfo = reinterpret_cast<const UpsampleNearestExact2dGradCompileInfo *>(tilingContext->GetCompileInfo());
    uint32_t coreNumPlatform = compileInfo->coreNum;
    tilingContext->SetTilingKey(GetTilingKeyVal());
    setScale();
    getSlideSize();
    getOutputShape();
    setSingleCoreK();
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    // 计算workspace，每个核的系数矩阵（是否要乘2，避免doubelBuffer矩阵相互影响？），中间矩阵大小
    getWorkSpace(needCoreNum);
    tilingContext->SetBlockDim(needCoreNum);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestExact2dGradTiling::setSingleCoreK()
{
    if (!FloatEqual(realScale_w, ZERO_FLOAT)) {
        singleCoreK_w = Ceil(slide_size * realScale_w) + RESERVED_VALUE;
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
    } else {
        singleCoreK_w = input_shapes[W_INDEX];
    }

    if (!FloatEqual(realScale_h, ZERO_FLOAT)) {
        singleCoreK_h = Ceil(slide_size * realScale_h) + RESERVED_VALUE;
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
    } else {
        singleCoreK_h = input_shapes[H_INDEX];
    }
}

uint32_t UpsampleNearestExact2dGradTiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(output_shapes[H_INDEX], slide_size);
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    if (coreNumPlatform != 0) {
        eachCoreSlideNum = slideNum / coreNumPlatform;
        remainder = slideNum % coreNumPlatform;
    }

    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = BEST_PERFORMANCE_SIZE;

    int64_t inputBatch = input_shapes[0] * input_shapes[1];
    int64_t tailAvergingBatch = 0;
    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = BEST_PERFORMANCE_SIZE;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(output_shapes[W_INDEX], groupCoreNum), minAvergingRows);
        tailAvergingBatch = CeilA2B(inputBatch, groupCoreNum);
        // 单个batch，每个分组的核数
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputBatch, tailAvergingBatch));
    }

    uint32_t needCoreNum_h = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_h[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_h[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        if (remainder != 0) {
            // 尾块处理, 核数不全都一样
            int groupIndex = 0;
            if (groupCoreNum != 0) {
                groupIndex = coreIndex / groupCoreNum;
            }
            if (groupIndex < remainder) {
                // 算出第几个分组
                tailSlideStartList_h[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
                tailSlideEndList_h[coreIndex] = std::min(
                    tailSlideStartList_h[coreIndex] + slide_size, static_cast<int64_t>(output_shapes[H_INDEX]));

                int64_t coreIndexInGroup = 0;
                coreIndexInGroup = (groupCoreNum == 0) ? 0 : coreIndex % groupCoreNum;
                tailRowStartList_h[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndList_h[coreIndex] = std::min(
                    tailRowStartList_h[coreIndex] + tailAvergingRows, static_cast<int64_t>(output_shapes[W_INDEX]));
                tailBatchStartListH[coreIndex] = coreIndexInGroup * tailAvergingBatch;
                tailBatchEndListH[coreIndex] =
                    std::min(tailBatchStartListH[coreIndex] + tailAvergingBatch, static_cast<int64_t>(inputBatch));
                needCoreNum_h++;
            }
        }
    }
    if (eachCoreSlideNum > 0) {
        needCoreNum_h = coreNumPlatform;
    }
    return needCoreNum_h;
}

void UpsampleNearestExact2dGradTiling::getTCubeTiling_h(uint32_t needCoreNum)
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size, output_shapes[W_INDEX], singleCoreK_h);
    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        return;
    }
}

void UpsampleNearestExact2dGradTiling::getTCubeTiling_w(uint32_t needCoreNum)
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shape.GetDim(N_INDEX) * input_shape.GetDim(C_INDEX) * input_shape.GetDim(H_INDEX),
        output_shapes[W_INDEX],
        input_shape.GetDim(W_INDEX));
    mmTiling_w.SetShape(input_shape.GetDim(N_INDEX) * input_shape.GetDim(C_INDEX) * input_shape.GetDim(H_INDEX),
        slide_size,
        singleCoreK_w);

    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

// 先只算w方向
void UpsampleNearestExact2dGradTiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    // 中间tensor
    uint64_t intermediate_matrix_size =
        output_shapes[0] * output_shapes[1] * input_shape.GetDim(2) * output_shapes[3] * dataTypeSize;

    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK_w;
    uint32_t radioMatrixWorkspaceSize_h = slide_size * singleCoreK_h;

    workspaces[0] = (intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum) * BYTE + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size(radioMatrixWorkspaceSize);
    tilingData.set_radio_matrix_size_h(radioMatrixWorkspaceSize_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
}

void UpsampleNearestExact2dGradTiling::getOutputShape()
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > 1) {
            output_shapes[i] = output_size_array[i];
        }
    }
    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleNearestExact2dGradTiling::getSlideSize()
{
    auto max_scale = realScale_h > realScale_w ? realScale_h : realScale_w;
    if (max_scale <= BEST_PERFORMANCE_SCALE_4) {
        slide_size = BEST_PERFORMANCE_SIZE_4;
    } else if (max_scale <= BEST_PERFORMANCE_SCALE_3) {
        slide_size = BEST_PERFORMANCE_SIZE_3;
    } else if (max_scale <= BEST_PERFORMANCE_SCALE_2) {
        slide_size = BEST_PERFORMANCE_SIZE_2;
    } else {
        slide_size = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline T1 UpsampleNearestExact2dGradTiling::CeilA2B(T1 a, T2 b)
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleNearestExact2dGradTiling::Ceil(T1 x)
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleNearestExact2dGradTiling::GetDataTypeSize()
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

uint8_t UpsampleNearestExact2dGradTiling::GetDataTypeVal()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return DATE_TYPE_FLOAT;
        case ge::DT_FLOAT16:
            return DATE_TYPE_FLOAT16;
        case ge::DT_BF16:
            return DATE_TYPE_HALF;
        default:
            return 0;
    }
}

uint8_t UpsampleNearestExact2dGradTiling::GetTilingKeyVal()
{
    return 1;
}
uint32_t UpsampleNearestExact2dGradTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScale_w, 1.0)) {
        needCoreNumW = GetNeedCoreNumW(coreNumPlatform);

        getTCubeTiling_w(needCoreNumW);
        tilingData.set_need_core_num_w(needCoreNumW);
    }

    if ((!FloatEqual(realScale_h, 1.0)) || (FloatEqual(realScale_w, 1.0))) {
        needCoreNumH = GetNeedCoreNumH(coreNumPlatform);
        tilingData.set_need_core_num_h(needCoreNumH);
        getTCubeTiling_h(needCoreNumH);
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}
uint32_t UpsampleNearestExact2dGradTiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    // 反向保持一致，还是按结果切
    //  先简单处理，直接均分给每个核；如果计算系数矩阵bound，可以优化，先按照批次分组分配，再分配给各个核，可以减少计算系数矩阵次数。

    int64_t outputSize = output_shapes[3];

    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = 0;
    if (coreNumPlatform != 0) {
        eachCoreSlideNum = slideNum / coreNumPlatform;
    }
    int64_t remainder = 0;
    if (coreNumPlatform != 0) {
        remainder = slideNum % coreNumPlatform;
    }
    // H维度总数
    int64_t input_h = input_shape.GetDim(0) * input_shape.GetDim(1) * input_shape.GetDim(2);
    // 获取最小分行数
    int64_t minAvergingRows = slide_size;
    // 按照剩余尾块数给核分组，然后每组核再均分行数
    int64_t groupCoreNum = remainder > 0 ? coreNumPlatform / remainder : 0;

    int64_t tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);

    groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_w[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;

        slideEndList_w[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;

        // 尾块处理
        int64_t groupIndex = 0;
        if (groupCoreNum != 0) {
            groupIndex = coreIndex / groupCoreNum;
        }
        if (groupIndex < remainder) {
            // 算出第几个分组
            tailSlideStartList_w[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
            tailSlideEndList_w[coreIndex] =
                std::min(tailSlideStartList_w[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
            int64_t coreIndexInGroup = 0;
            if (groupCoreNum != 0) {
                coreIndexInGroup = coreIndex % groupCoreNum;
            }
            tailRowStartList_w[coreIndex] = coreIndexInGroup * tailAvergingRows;
            tailRowEndList_w[coreIndex] =
                std::min(tailRowStartList_w[coreIndex] + tailAvergingRows, static_cast<int64_t>(input_h));

            needCoreNum++;
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    }
    return needCoreNum;
}

void UpsampleNearestExact2dGradTiling::FillTilingData()
{
    tilingData.set_slideStartList_w(slideStartList_w);
    tilingData.set_slideEndList_w(slideEndList_w);
    tilingData.set_tailSlideStartList_w(tailSlideStartList_w);
    tilingData.set_tailSlideEndList_w(tailSlideEndList_w);
    tilingData.set_tailRowStartList_w(tailRowStartList_w);
    tilingData.set_tailRowEndList_w(tailRowEndList_w);
    tilingData.set_slideStartList_h(slideStartList_h);
    tilingData.set_slideEndList_h(slideEndList_h);
    tilingData.set_tailSlideStartList_h(tailSlideStartList_h);
    tilingData.set_tailSlideEndList_h(tailSlideEndList_h);
    tilingData.set_tailRowStartList_h(tailRowStartList_h);
    tilingData.set_tailRowEndList_h(tailRowEndList_h);
    tilingData.set_tailBatchStartListH(tailBatchStartListH);
    tilingData.set_tailBatchEndListH(tailBatchEndListH);

    tilingData.set_dataType(GetDataTypeVal());

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleNearestExact2dGradTiling(gert::TilingContext *context)
{
    UpsampleNearestExact2dGradTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleNearestExact2dGradCompileInfo>(context);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearestExact2dGrad)
    .Tiling(tiling4UpsampleNearestExact2dGradTiling)
    .TilingParse<UpsampleNearestExact2dGradCompileInfo>(tilingPrepareTiling);

}  // namespace optiling
