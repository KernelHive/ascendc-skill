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
 * \file upsample_bilinear2d_aa_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_bilinear2d_aa_tiling.h"

using namespace ge;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(op_name, ...)   std::printf(op_name, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define VECTOR_INFER_SHAPE_INNER_ERR_REPORT(op_name, err_msg)        \
  do {                                                               \
    std::printf("op[%s], %s", op_name, err_msg);                     \
  } while (0)
}  // namespace ops
namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {
constexpr uint32_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_4 = 64;

constexpr uint32_t BEST_PERFORMANCE_SCALE_1 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_2 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_3 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_4 = 5;

constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;

constexpr uint32_t OUTPUT_SIZE_ATTR = 0;
constexpr uint32_t ALIGN_CORNERS_ATTR = 1;
constexpr uint32_t SCALE_H_ATTR = 2;
constexpr uint32_t SCALE_W_ATTR = 3;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr float MAX_SUPPORT_SCALE = 50.0f;

class UpsampleBilinearAATiling {
public:
    explicit UpsampleBilinearAATiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    void SetScale();
    inline float compute_scale_value(int64_t input_size, int64_t output_size, bool alignCorners, const float *scale);
    bool getWorkSpace(uint32_t needCoreNum);
    void GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatForm);
    uint32_t GetNeedCoreNumW(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNumH(uint32_t coreNumPlatform);
    void FillTilingData();
    void GetTCubeTilingW();
    void GetTCubeTilingH();
    inline bool CheckScales(const gert::TilingContext *context, float scalesW, float scalesH);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size{16};
    UpsampleBilinearAATilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    uint8_t dim{0};
    const bool *alignCorners{nullptr};
    const float *scaleH{nullptr};
    const float *scaleW{nullptr};
    float realScaleH{0.0f};
    float realScaleW{0.0f};
    const gert::ContinuousVector *output_size{nullptr};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
};

void UpsampleBilinearAATiling::SetScale()
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());

    realScaleH = compute_scale_value(input_shape.GetDim(H_INDEX), output_size_array[0], *alignCorners, scaleH);
    realScaleW = compute_scale_value(input_shape.GetDim(W_INDEX), output_size_array[1], *alignCorners, scaleW);

    tilingData.set_scale_h(realScaleH);
    tilingData.set_scale_w(realScaleW);

    float supportW = (realScaleW >= 1.0) ? realScaleW : 1.0;
    float supportH = (realScaleH >= 1.0) ? realScaleH : 1.0;

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

inline float UpsampleBilinearAATiling::compute_scale_value(
    int64_t input_size, int64_t output_size, bool alignCorner, const float *scale)
{
    if (output_size == input_size) {
        return static_cast<float>(1);
    }
    if (alignCorner) {
        if (output_size > 1) {
            return static_cast<float>(input_size - 1) / (output_size - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return (scale != nullptr && *scale > 0) ? static_cast<float>(*scale)
                                                : (static_cast<float>(input_size) / output_size);
    }
}

inline bool UpsampleBilinearAATiling::CheckScales(const gert::TilingContext *context, float scalesW, float scalesH)
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
    int64_t inputH = input_shape.GetDim(H_INDEX);
    int64_t inputW = input_shape.GetDim(W_INDEX);
    int64_t outputH = output_size_array[0];
    int64_t outputW = output_size_array[1];
    float tmpScalesH = scalesH > 0 ? scalesH : static_cast<float>(inputH / outputH);
    float tmpScalesW = scalesW > 0 ? scalesW : static_cast<float>(inputW / outputW);
    OP_TILING_CHECK((tmpScalesH > MAX_SUPPORT_SCALE || tmpScalesW > MAX_SUPPORT_SCALE),
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
            "Scales should not exceed 50, but got scale (scalesW: %f, scalesH: %f) ",
            tmpScalesW,
            tmpScalesH),
        return false);
    return true;
}

inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleBilinearAATiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    output_size = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_ATTR);
    alignCorners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    scaleH = attrs->GetAttrPointer<float>(SCALE_H_ATTR);
    scaleW = attrs->GetAttrPointer<float>(SCALE_W_ATTR);

    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType srcDtype = ge::DT_UNDEFINED;
    srcDtype = temp->GetDataType();

    // Determine whether all data types are consistent.
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
        dataTypeSize = GetDataTypeSize();
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    auto src_shape = tilingContext->GetInputShape(0);
    dim = src_shape->GetStorageShape().GetDimNum() - 2;  // 其实固定是2

    input_shape = src_shape->GetOriginShape();
    if (!CheckScales(tilingContext, *scaleW, *scaleH)) {
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const UpsampleBilinear2dAACompileInfo *>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreNumPlatForm = compileInfo->coreNum;

    tilingContext->SetTilingKey(1);

    SetScale();
    GetSlideSize();
    GetShapes();
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatForm);
    // 计算workspace，每个核的系数矩阵（是否要乘2，避免doubelBuffer矩阵相互影响？），中间矩阵大小
    if (!getWorkSpace(needCoreNum)) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetBlockDim(needCoreNum);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleBilinearAATiling::GetNeedCoreNum(uint32_t coreNumPlatForm)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScaleW, 1.0)) {
        singleCoreK_w = Ceil(slide_size * realScaleW) + Ceil(tilingData.get_max_interp_size_w());
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatForm);
        GetTCubeTilingW();
    }

    if (!FloatEqual(realScaleH, 1.0) || FloatEqual(realScaleW, 1.0)) {
        singleCoreK_h = Ceil(slide_size * realScaleH) + Ceil(tilingData.get_max_interp_size_h());
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatForm);
        GetTCubeTilingH();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleBilinearAATiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shape[H_INDEX], slide_size, singleCoreK_w);
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

void UpsampleBilinearAATiling::GetTCubeTilingH()
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

// 先只算w方向
bool UpsampleBilinearAATiling::getWorkSpace(uint32_t needCoreNum)
{
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return false;
    }
    // 中间tensor
    uint64_t intermediate_matrix_size =
        output_shapes[0] * output_shapes[1] * input_shapes[2] * output_shapes[3] * dataTypeSize;
    intermediate_matrix_size = (intermediate_matrix_size + ADDR_ALIGN_SIZE - 1) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    // 每个核的系数矩阵，每个核申请两个workspace空间，避免相互覆盖
    int64_t singleCoreK = singleCoreK_w > singleCoreK_h ? singleCoreK_w : singleCoreK_h;

    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
    workspaces[0] = intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size_w(slide_size * singleCoreK_w);
    tilingData.set_radio_matrix_size_h(slide_size * singleCoreK_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    return true;
}

void UpsampleBilinearAATiling::GetShapes()
{
    const int64_t *output_size_array = reinterpret_cast<const int64_t *>(output_size->GetData());
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > C_INDEX) {
            output_shapes[i] = output_size_array[i - H_INDEX];
        }
    }
    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleBilinearAATiling::GetSlideSize()
{
    auto maxScale = realScaleH > realScaleW ? realScaleH : realScaleW;
    if (maxScale <= BEST_PERFORMANCE_SCALE_4) {
        slide_size = BEST_PERFORMANCE_SIZE_4;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_3) {
        slide_size = BEST_PERFORMANCE_SIZE_3;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_2) {
        slide_size = BEST_PERFORMANCE_SIZE_2;
    } else {
        slide_size = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slide_size(slide_size);
}

template <typename T1, typename T2>
inline T1 UpsampleBilinearAATiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleBilinearAATiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleBilinearAATiling::GetDataTypeSize() const
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

uint64_t UpsampleBilinearAATiling::GetDataTypeVal()
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

uint32_t UpsampleBilinearAATiling::GetNeedCoreNumW(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = slideNum / coreNumPlatform;
    int64_t remainder = slideNum % coreNumPlatform;

    // H维度总数
    int64_t input_h = input_shapes[0] * input_shapes[1] * input_shapes[2];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRows = slide_size;

    if (remainder != 0) {
        // 获取最小分行数
        int64_t minAvergingRows = slide_size;
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRows = std::max(CeilA2B(input_h, groupCoreNum), minAvergingRows);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(input_h, tailAvergingRows));
    }
    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    tilingData.set_eachCoreSlideNumW(eachCoreSlideNum);
    tilingData.set_tailStartSlideNumW(tailStartSlideNum);
    tilingData.set_slideNumW(slideNum);
    tilingData.set_groupCoreNumW(groupCoreNum);
    tilingData.set_tailAvergingRowsW(tailAvergingRows);
    tilingData.set_remainderW(remainder);

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t coreIndex = 0; coreIndex < coreNumPlatform; coreIndex++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理
            int64_t groupIndex = coreIndex / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }

    tilingData.set_need_core_num_w(needCoreNum);

    return needCoreNum;
}

uint32_t UpsampleBilinearAATiling::GetNeedCoreNumH(uint32_t coreNumPlatform)
{
    int64_t outputSize = output_shapes[2];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = slideNum / coreNumPlatform;
    int64_t remainder = slideNum % coreNumPlatform;

    // Batch和W维度总数
    int64_t batch = input_shapes[0] * input_shapes[1];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingBatch = slide_size;
    if (remainder != 0) {
        // 按照剩余尾块数给核分组，然后每组核再均分行数
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingBatch = CeilA2B(batch, groupCoreNum);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(batch, tailAvergingBatch));
    }

    int64_t needCoreNum = 0;
    int64_t tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;

    tilingData.set_eachCoreSlideNumH(eachCoreSlideNum);
    tilingData.set_tailStartSlideNumH(tailStartSlideNum);
    tilingData.set_slideNumH(slideNum);
    tilingData.set_groupCoreNumH(groupCoreNum);
    tilingData.set_tailAvergingRowsH(tailAvergingBatch);
    tilingData.set_remainderH(remainder);

    if (eachCoreSlideNum > 0) {
        needCoreNum = coreNumPlatform;
    } else if (remainder != 0) {
        for (uint32_t coreIndexH = 0; coreIndexH < coreNumPlatform; coreIndexH++) {
            groupCoreNum = groupCoreNum == 0 ? 1 : groupCoreNum;
            // 尾块处理, 核数不全都一样
            int64_t groupIndex = coreIndexH / groupCoreNum;
            if (groupIndex < remainder) {
                needCoreNum++;
            }
        }
    }

    tilingData.set_need_core_num_h(needCoreNum);

    return needCoreNum;
}

void UpsampleBilinearAATiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBilinearAATiling(gert::TilingContext *context)
{
    UpsampleBilinearAATiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleBilinear2dAACompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_TILING_CHECK(compileInfo->coreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
            "UpsampleBilinear2dAA GetHardwareInfo Failed, vectorCoreNum:%d",
            compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleBilinear2dAA)
    .Tiling(tiling4UpsampleBilinearAATiling)
    .TilingParse<UpsampleBilinear2dAACompileInfo>(tilingPrepareTiling);

}  // namespace optiling
