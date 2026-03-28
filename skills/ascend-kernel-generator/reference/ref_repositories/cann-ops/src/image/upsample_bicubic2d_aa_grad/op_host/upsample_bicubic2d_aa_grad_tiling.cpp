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
 * \file upsample_bicubic2d_aa_grad_tiling.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "upsample_bicubic2d_aa_grad_tiling.h"

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE = 16;
constexpr uint32_t BYTE = 8;

constexpr uint32_t BYTE_REPEAT = 256;
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BF16 = 3;

constexpr uint64_t TILING_HALF_N_SCALAR = 14;
constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
constexpr uint64_t TILING_INT_N_SCALAR = 4;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr float MIN_SUPPORT_SCALE = 0.02;

constexpr uint32_t DIM_LEN = 4;
constexpr int64_t RESERVED_LENGTH = 2;
constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
constexpr uint32_t BINARY_SCALAR_UB_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_DIVIDER = 8;
constexpr uint32_t FOREACH_POW_SCALAR_DIVIDER = 4;
constexpr uint32_t FOREACH_COS_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_LIST_DIVIDER = 10;

constexpr uint16_t LOG2_BASIC_FOR_LOG2 = 1024;
constexpr uint32_t LOG2_HALF_FOR_LOG2 = 4;
constexpr uint32_t LOG2_FLOAT_FOR_LOG2 = 0;

constexpr uint8_t BYTE_PER_BLOCK = 32;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr int8_t INSIZE_INDEX = 0;
constexpr int8_t OUTSIZE_INDEX = 1;
constexpr int8_t ALIGN_INDEX = 2;
constexpr int8_t SCALEX_INDEX = 3;
constexpr int8_t SCALEY_INDEX = 4;
constexpr int8_t TEMP_INDEX = 0;
constexpr int8_t DOUBLE_SIZE = 2;

class UpsampleBicubic2dAAGradTiling {
public:
    explicit UpsampleBicubic2dAAGradTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void SetScale();
    inline float compute_scale_value(int64_t input_size, int64_t output_size, const float *scale);
    void getWorkSpace(uint32_t needCoreNum);
    void GetOutputShape();
    void GetSlideSize();
    void SetSingleCoreK();
    uint8_t GetDataTypeSize();
    uint64_t GetDataTypeVal();
    uint64_t GetTilingKeyVal();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void GetTCubeTilingW();
    void GetTCubeTilingH();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x);

private:
    int64_t slide_size = 0;
    UpsampleBicubicAAGradTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize = 0;
    gert::Shape input_shape;
    uint8_t dim = 0;
    const bool *alignCorners = nullptr;
    const float *scaleH = nullptr;
    const float *scaleW = nullptr;
    float realScaleH = 0;
    float realScaleW = 0;
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

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int32_t singleCoreK_w = 0;
    int32_t singleCoreK_h = 0;

    bool needExpandW = true;
    bool needExpandH = true;
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

ge::graphStatus UpsampleBicubic2dAAGradTiling::Init()
{
    return ge::GRAPH_SUCCESS;
}

void UpsampleBicubic2dAAGradTiling::SetScale()
{
    if (dim == H_INDEX) {
        const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());

        realScaleH = compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[H_INDEX], scaleH);
        realScaleW = compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[W_INDEX], scaleW);
        if (FloatEqual(realScaleW, 1.0)) {
            needExpandW = false;
        }
        if (FloatEqual(realScaleH, 1.0)) {
            needExpandH = false;
        }
        tilingData.set_scale_h(realScaleH);
        tilingData.set_scale_w(realScaleW);

        float supportW = (realScaleW >= 1.0) ? 2 * realScaleW : 2.0;
        float supportH = (realScaleH >= 1.0) ? 2 * realScaleH : 2.0;

        tilingData.set_support_w(supportW);
        tilingData.set_support_h(supportH);

        int16_t max_interp_size_w = Ceil(supportW) * 2 + 1;
        int16_t max_interp_size_h = Ceil(supportH) * 2 + 1;

        tilingData.set_max_interp_size_w(max_interp_size_w);
        tilingData.set_max_interp_size_h(max_interp_size_h);

        float invscaleW = (realScaleW >= 1.0) ? 1.0 / realScaleW : 1.0;

        float invscaleH = (realScaleH >= 1.0) ? 1.0 / realScaleH : 1.0;

        tilingData.set_invscale_w(invscaleW);
        tilingData.set_invscale_h(invscaleH);
    }
}

inline float UpsampleBicubic2dAAGradTiling::compute_scale_value(
    int64_t input_size, int64_t output_size, const float *scale)
{
    if (*alignCorners) {
        if (output_size == input_size) {
            return static_cast<float>(1);
        }
        if (input_size > 1) {
            return static_cast<float>(output_size - 1) / (input_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale != nullptr && *scale > 0.0) ? static_cast<float>(*scale)
                                                  : (static_cast<float>(output_size) / input_size);
    }
}

ge::graphStatus UpsampleBicubic2dAAGradTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    input_size = attrs->GetAttrPointer<gert::ContinuousVector>(INSIZE_INDEX);
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(OUTSIZE_INDEX);
    alignCorners = attrs->GetAttrPointer<bool>(ALIGN_INDEX);
    scaleH = attrs->GetAttrPointer<float>(SCALEX_INDEX);
    scaleW = attrs->GetAttrPointer<float>(SCALEY_INDEX);

    auto temp = tilingContext->GetInputDesc(TEMP_INDEX);
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
    dim = src_shape->GetStorageShape().GetDimNum() - H_INDEX;
    input_shape = src_shape->GetOriginShape();

    auto compileInfo = reinterpret_cast<const UpsampleBicubic2dAAGradCompileInfo *>(tilingContext->GetCompileInfo());
    uint32_t coreNumPlatform = 0;
    if (compileInfo != nullptr) {
        coreNumPlatform = compileInfo->coreNum;
    }

    tilingContext->SetTilingKey(GetTilingKeyVal());
    GetSlideSize();
    SetScale();
    GetOutputShape();
    SetSingleCoreK();
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatform);
    getWorkSpace(needCoreNum);
    tilingContext->SetBlockDim(needCoreNum);

    FillTilingData();

    return ge::GRAPH_SUCCESS;
}

void UpsampleBicubic2dAAGradTiling::SetSingleCoreK()
{
    if (!FloatEqual(realScaleW, (float)0.0)) {
        singleCoreK_w =
            DOUBLE_SIZE * Ceil(slide_size / realScaleW) + DOUBLE_SIZE * Ceil(tilingData.get_max_interp_size_w());
        singleCoreK_w = singleCoreK_w <= input_shapes[W_INDEX] ? singleCoreK_w : input_shapes[W_INDEX];
    } else {
        singleCoreK_w = input_shapes[W_INDEX];
    }
    if (!FloatEqual(realScaleH, (float)0.0)) {
        singleCoreK_h =
            DOUBLE_SIZE * Ceil(slide_size / realScaleH) + DOUBLE_SIZE * Ceil(tilingData.get_max_interp_size_h());
        singleCoreK_h = singleCoreK_h <= input_shapes[H_INDEX] ? singleCoreK_h : input_shapes[H_INDEX];
    } else {
        singleCoreK_h = input_shapes[H_INDEX];
    }
}

uint32_t UpsampleBicubic2dAAGradTiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[2];
    int64_t output_w = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);

    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    if (coreNumPlatform != 0) {
        remainder = slideNum % coreNumPlatform;
        eachCoreSlideNum = slideNum / coreNumPlatform;
    }

    int64_t input_w = output_shapes[3];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = BEST_PERFORMANCE_SIZE;
    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = BEST_PERFORMANCE_SIZE;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(output_w, groupCoreNum), minAvergingRows);
        // 单个batch，每个分组的核数
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_w, tailAvergingRows));
        // 分组数
    }

    uint32_t needCoreNum_h = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_h[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_h[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        if (remainder != 0) {
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = 0;
            if (groupCoreNum != 0) {
                groupIndex = coreIndex / groupCoreNum;
            }

            if (groupIndex < remainder) {
                // 算出第几个分组
                tailSlideStartList_h[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
                tailSlideEndList_h[coreIndex] =
                    std::min(tailSlideStartList_h[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
                int64_t coreIndexInGroup = 0;
                if (groupCoreNum != 0) {
                    coreIndexInGroup = coreIndex % groupCoreNum;
                }
                tailRowStartList_h[coreIndex] = coreIndexInGroup * tailAvergingRows;
                tailRowEndList_h[coreIndex] =
                    std::min(tailRowStartList_h[coreIndex] + tailAvergingRows, static_cast<int64_t>(input_w));
                needCoreNum_h++;
            }
        }
    }

    if (eachCoreSlideNum > 0) {
        needCoreNum_h = coreNumPlatform;
    }

    tilingData.set_need_core_num_h(needCoreNum_h);

    return needCoreNum_h;
}

void UpsampleBicubic2dAAGradTiling::GetTCubeTilingH()
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

void UpsampleBicubic2dAAGradTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    // matmul_tiling::MatmulApiTiling mmTiling_w(platformInfo);
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
void UpsampleBicubic2dAAGradTiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    // 中间tensor
    uint64_t intermediate_matrix_size =
        output_shapes[0] * output_shapes[1] * input_shape.GetDim(2) * output_shapes[3] * dataTypeSize;

    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK_w;
    uint32_t radioMatrixWorkspaceSize_h = slide_size * singleCoreK_h;
    if (workspaces != nullptr) {
        workspaces[0] = (intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum) * BYTE + WORK_SPACE_SIZE;
    }
    tilingData.set_radio_matrix_size(radioMatrixWorkspaceSize);
    tilingData.set_radio_matrix_size_h(radioMatrixWorkspaceSize_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
}

void UpsampleBicubic2dAAGradTiling::GetOutputShape()
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

void UpsampleBicubic2dAAGradTiling::GetSlideSize()
{
    slide_size = BEST_PERFORMANCE_SIZE;
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline T1 UpsampleBicubic2dAAGradTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBicubic2dAAGradTiling::Ceil(T1 x)
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleBicubic2dAAGradTiling::GetDataTypeSize()
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

uint64_t UpsampleBicubic2dAAGradTiling::GetDataTypeVal()
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

uint64_t UpsampleBicubic2dAAGradTiling::GetTilingKeyVal()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return 0;
    }
}
uint32_t UpsampleBicubic2dAAGradTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (needExpandW) {
        needCoreNumW = GetNeedCoreNumW(coreNumPlatform);
        GetTCubeTilingW();
        tilingData.set_need_core_num_w(needCoreNumW);
    }

    if (needExpandH || !needExpandW) {
        needCoreNumH = GetNeedCoreNumH(coreNumPlatform);
        GetTCubeTilingH();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}
uint32_t UpsampleBicubic2dAAGradTiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = 0;
    int64_t remainder = 0;
    if (coreNumPlatform > 0) {
        eachCoreSlideNum = slideNum / coreNumPlatform;
        remainder = slideNum % coreNumPlatform;
    }

    // H维度总数
    int64_t input_h = input_shape.GetDim(0) * input_shape.GetDim(1) * input_shape.GetDim(2);
    // 获取最小分行数
    int64_t minAvergingRows = slide_size;

    // 按照剩余尾块数给核分组，然后每组核再均分行数
    int64_t groupCoreNum = 0;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
    }
    int64_t tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);

    groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));

    int64_t needCoreNum = 0;

    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
    for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
        slideStartList_w[coreIndex] = coreIndex * eachCoreSlideNum * slide_size;
        slideEndList_w[coreIndex] = (std::min((coreIndex + 1) * eachCoreSlideNum, slideNum)) * slide_size;
        int64_t groupIndex = 0;
        // 尾块处理
        if (groupCoreNum > 0) {
            groupIndex = coreIndex / groupCoreNum;
        }

        if (groupIndex < remainder) {
            // 算出第几个分组
            tailSlideStartList_w[coreIndex] = (tailStartSlideNum + groupIndex) * slide_size;
            tailSlideEndList_w[coreIndex] =
                std::min(tailSlideStartList_w[coreIndex] + slide_size, static_cast<int64_t>(outputSize));
            int64_t coreIndexInGroup = 0;
            if (groupCoreNum > 0) {
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

void UpsampleBicubic2dAAGradTiling::FillTilingData()
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

    tilingData.set_dataType(GetDataTypeVal());

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBicubicAAGradTiling(gert::TilingContext *context)
{
    UpsampleBicubic2dAAGradTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = GetCompileInfoPtr<UpsampleBicubic2dAAGradCompileInfo>(context);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleBicubic2dAAGrad)
    .Tiling(tiling4UpsampleBicubicAAGradTiling)
    .TilingParse<UpsampleBicubic2dAAGradCompileInfo>(tilingPrepareTiling);
}  // namespace optiling
