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
 * \file upsample_bilinear2d.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "upsample_bilinear2d_tiling.h"

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
constexpr uint32_t BEST_PERFORMANCE_SIZE_16 = 16;
constexpr uint32_t BEST_PERFORMANCE_SIZE_32 = 32;
constexpr uint32_t BEST_PERFORMANCE_SIZE_48 = 48;
constexpr uint32_t BEST_PERFORMANCE_SIZE_64 = 64;
constexpr uint32_t BEST_PERFORMANCE_SIZE_128 = 128;

constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_50 = 50;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_20 = 20;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_8 = 8;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_5 = 5;
constexpr uint32_t BEST_PERFORMANCE_SCALE_BELOW_1 = 1;

constexpr uint32_t BYTE = 8;
constexpr uint32_t BYTE_REPEAT = 256;  // The amount of data that can be processed by a repeat.
constexpr uint32_t BYTE_BASIC_BLOCK = 1024;

constexpr int8_t SHAPE_SIZE = 4;
constexpr int8_t N_INDEX = 0;
constexpr int8_t C_INDEX = 1;
constexpr int8_t H_INDEX = 2;
constexpr int8_t W_INDEX = 3;
constexpr int8_t DIM_ZERO = 0;
constexpr int8_t DIM_ONE = 1;
constexpr int8_t DIM_TWO = 2;
constexpr int8_t DIM_THREE = 3;

constexpr int8_t MODE_LINEAR = 1;
constexpr int8_t MODE_BILINEAR = 2;

constexpr uint32_t ALIGN_CORNERS_ATTR = 0;
constexpr uint32_t SCALES_ATTR = 1;

constexpr uint64_t DATE_TYPE_FLOAT16 = 1;
constexpr uint64_t DATE_TYPE_FLOAT = 2;
constexpr uint64_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t DIM_LEN = 4;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;

constexpr int8_t NUM_ONE = 1;
constexpr int8_t NUM_TWO = 2;
constexpr int8_t NUM_FIVE = 5;
constexpr int64_t NUM_1024 = 1024;

constexpr float MAX_SUPPORT_SHRINK_SCALE = 50.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE = 800.0f;
constexpr float MAX_SUPPORT_ZOOM_SCALE_REV = 0.00125f;

constexpr float SUPPORT = 1.0;
constexpr int64_t max_interp_size = 2;

class UpsampleLinear2dTiling {
public:
    explicit UpsampleLinear2dTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling(const uint8_t mode);

private:
    void setScale(const uint8_t mode);
    void get_scale_from_out(const uint8_t mode);
    inline float compute_scale_value(
        const int64_t input_size, const int64_t output_size, const bool alignCorners, const float scale);
    bool getWorkSpace(const uint32_t needCoreNum);
    void getShapes(const uint8_t mode);
    void setSlideSize(const uint32_t coreNumPlatForm, const uint8_t mode);
    inline int64_t calculateSlideSize(const uint32_t coreNumPlatForm, uint8_t direction);
    inline int64_t getSlideSizeByScale(const uint32_t coreNumPlatForm, uint8_t direction, float real_scale);
    uint8_t GetDataTypeSize() const;
    uint64_t GetDataTypeVal();
    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatForm, const uint8_t mode);
    uint32_t GetNeedCoreNumW(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    uint32_t GetNeedCoreNumH(const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size);
    void FillTilingData();
    void GetTCubeTilingW();
    void GetTCubeTilingH();
    inline bool CheckScales(
        const gert::TilingContext *context, const float scalesW, const float scalesH, const uint8_t mode);

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int32_t Ceil(T1 x) const;

private:
    int64_t slide_size_w{16};
    int64_t slide_size_h{16};
    UpsampleLinear2dTilingData tilingData;
    gert::TilingContext *tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint16_t dataTypeSize{4};
    gert::Shape input_shape;
    gert::Shape output_shape;
    const bool *alignCorners{nullptr};
    float scaleH = 0.0f;
    float scaleW = 0.0f;
    float realScaleH{0.0f};
    float realScaleW{0.0f};

    int64_t output_shapes[4] = {0};
    int64_t input_shapes[4] = {0};

    int64_t slide_size_list[5] = {BEST_PERFORMANCE_SIZE_16,
        BEST_PERFORMANCE_SIZE_32,
        BEST_PERFORMANCE_SIZE_48,
        BEST_PERFORMANCE_SIZE_64,
        BEST_PERFORMANCE_SIZE_128};

    TCubeTiling matmulTiling_w;
    TCubeTiling matmulTiling_h;
    int64_t singleCoreK_w = 0;
    int64_t singleCoreK_h = 0;
    uint32_t coreNumPlatForm = 20;
};

void UpsampleLinear2dTiling::setScale(const uint8_t mode)
{
    if (mode == MODE_BILINEAR) {
        realScaleH = compute_scale_value(input_shapes[H_INDEX], output_shapes[H_INDEX], *alignCorners, scaleH);
        realScaleW = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *alignCorners, scaleW);
    } else {
        realScaleH = 1.0;
        realScaleW = compute_scale_value(input_shapes[W_INDEX], output_shapes[W_INDEX], *alignCorners, scaleW);
    }

    tilingData.set_scale_h(realScaleH);
    tilingData.set_scale_w(realScaleW);
}

void UpsampleLinear2dTiling::get_scale_from_out(const uint8_t mode)
{
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    alignCorners = attrs->GetAttrPointer<bool>(ALIGN_CORNERS_ATTR);
    if (mode == MODE_LINEAR) {
        const float *scales = attrs->GetAttrPointer<float>(SCALES_ATTR);
        scaleH = 1.0f;
        scaleW = *scales;
    } else {
        const gert::ContinuousVector *scalesAttr = attrs->GetAttrPointer<gert::ContinuousVector>(SCALES_ATTR);
        const float *scalesArray = reinterpret_cast<const float *>(scalesAttr->GetData());
        scaleH = scalesArray[DIM_ZERO];
        scaleW = scalesArray[DIM_ONE];
    }
}

inline float UpsampleLinear2dTiling::compute_scale_value(
    const int64_t input_size, const int64_t output_size, const bool alignCorner, const float scale)
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
        return (scale > 0) ? static_cast<float>(scale) : (static_cast<float>(input_size) / output_size);
    }
}

inline bool UpsampleLinear2dTiling::CheckScales(
    const gert::TilingContext *context, const float scalesW, const float scalesH, const uint8_t mode)
{
    if (mode == MODE_LINEAR) {
        // 1D的放大支持800倍，缩小支持50倍
        float maxSupport = scalesW < 1 ? MAX_SUPPORT_ZOOM_SCALE : MAX_SUPPORT_SHRINK_SCALE;
        OP_TILING_CHECK(
            ((scalesW < 1 && scalesW < MAX_SUPPORT_ZOOM_SCALE_REV) || (scalesW > MAX_SUPPORT_SHRINK_SCALE)),
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "Scales should not exceed %f, but got scale (scales: %f) ",
                maxSupport,
                scalesW),
            return false);
    } else {
        // 2D都限制50倍
        OP_TILING_CHECK((scalesH > MAX_SUPPORT_SHRINK_SCALE || scalesW > MAX_SUPPORT_SHRINK_SCALE),
            VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "Scales should not exceed 50, but got scale (scalesW: %f, scalesH: %f) ",
                scalesW,
                scalesH),
            return false);
    }
    return true;
}

inline bool FloatEqual(const float a, const float b)
{
    const float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
};

ge::graphStatus UpsampleLinear2dTiling::RunBigKernelTiling(const uint8_t modeNum)
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    get_scale_from_out(modeNum);

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
    auto dst_shape = tilingContext->GetOutputShape(0);

    input_shape = src_shape->GetOriginShape();
    output_shape = dst_shape->GetOriginShape();

    auto compileInfo = reinterpret_cast<const UpsampleLinear2dCompileInfo *>(tilingContext->GetCompileInfo());
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t coreNumPlatForm = compileInfo->coreNum;
    if (coreNumPlatForm < 1) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(1);
    tilingData.set_mode(modeNum);
    tilingData.set_align_corners(*alignCorners);
    getShapes(modeNum);
    setScale(modeNum);
    if (!CheckScales(tilingContext, realScaleW, realScaleH, modeNum)) {
        return ge::GRAPH_FAILED;
    }

    setSlideSize(coreNumPlatForm, modeNum);
    uint32_t needCoreNum = GetNeedCoreNum(coreNumPlatForm, modeNum);
    // 计算workspace，每个核的系数矩阵（是否要乘2，避免doubelBuffer矩阵相互影响？），中间矩阵大小
    if (!getWorkSpace(needCoreNum)) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetBlockDim(needCoreNum);
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

uint32_t UpsampleLinear2dTiling::GetNeedCoreNum(const uint32_t coreNumPlatForm, const uint8_t mode)
{
    uint32_t needCoreNumW = 0;
    uint32_t needCoreNumH = 0;
    if (!FloatEqual(realScaleW, 1.0) || mode == MODE_LINEAR) {
        singleCoreK_w = Ceil(slide_size_w * realScaleW) + Ceil(max_interp_size);
        if (singleCoreK_w > input_shapes[W_INDEX]) {
            singleCoreK_w = input_shapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatForm, NUM_TWO, slide_size_w);
        GetTCubeTilingW();
    }

    if (mode == MODE_BILINEAR && (!FloatEqual(realScaleH, 1.0) || FloatEqual(realScaleW, 1.0))) {
        singleCoreK_h = Ceil(slide_size_h * realScaleH) + Ceil(max_interp_size);
        if (singleCoreK_h > input_shapes[H_INDEX]) {
            singleCoreK_h = input_shapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatForm, NUM_TWO, slide_size_h);
        GetTCubeTilingH();
    }

    uint32_t needCoreNum = std::max(needCoreNumW, needCoreNumH);
    needCoreNum = needCoreNum < 1 ? 1 : needCoreNum;
    return needCoreNum;
}

void UpsampleLinear2dTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetOrgShape(input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX],
        output_shapes[W_INDEX],
        input_shapes[W_INDEX]);
    mmTiling_w.SetShape(
        input_shapes[N_INDEX] * input_shapes[C_INDEX] * input_shapes[H_INDEX], slide_size_w, singleCoreK_w);
    if (mmTiling_w.GetTiling(tilingData.matmulTiling_w) == -1) {
        return;
    }
}

void UpsampleLinear2dTiling::GetTCubeTilingH()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetOrgShape(output_shapes[H_INDEX], output_shapes[W_INDEX], input_shapes[W_INDEX]);
    mmTiling_h.SetShape(slide_size_h, output_shapes[W_INDEX], singleCoreK_h);

    if (mmTiling_h.GetTiling(tilingData.matmulTiling_h) == -1) {
        return;
    }
}

// 先只算w方向
bool UpsampleLinear2dTiling::getWorkSpace(const uint32_t needCoreNum)
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
    int64_t slide_size = std::max(slide_size_w, slide_size_h);
    uint32_t radioMatrixWorkspaceSize = slide_size * singleCoreK * dataTypeSize;
    workspaces[0] = intermediate_matrix_size + radioMatrixWorkspaceSize * needCoreNum + WORK_SPACE_SIZE;
    tilingData.set_radio_matrix_size_w(slide_size_w * singleCoreK_w);
    tilingData.set_radio_matrix_size_h(slide_size_h * singleCoreK_h);
    tilingData.set_intermediate_matrix_size(intermediate_matrix_size);
    return true;
}

void UpsampleLinear2dTiling::getShapes(const uint8_t mode)
{
    for (int8_t i = 0; i < SHAPE_SIZE; i++) {
        input_shapes[i] = input_shape.GetDim(i);
        output_shapes[i] = input_shape.GetDim(i);
        if (i > C_INDEX && mode == MODE_BILINEAR) {
            output_shapes[i] = output_shape.GetDim(i);
        } else {
            output_shapes[DIM_TWO] = 1;
            output_shapes[DIM_THREE] = output_shape.GetDim(DIM_TWO);
        }
    }

    tilingData.set_input_shapes(input_shapes);
    tilingData.set_output_shapes(output_shapes);
}

void UpsampleLinear2dTiling::setSlideSize(const uint32_t coreNumPlatForm, const uint8_t mode)
{
    slide_size_w = getSlideSizeByScale(coreNumPlatForm, NUM_ONE, realScaleW);
    if (mode == MODE_BILINEAR) {
        slide_size_h = getSlideSizeByScale(coreNumPlatForm, NUM_TWO, realScaleH);
    }
    tilingData.set_slide_size_w(slide_size_w);
    tilingData.set_slide_size_h(slide_size_h);
}

inline int64_t UpsampleLinear2dTiling::getSlideSizeByScale(
    const uint32_t coreNumPlatForm, uint8_t direction, float real_scale)
{
    int64_t slide_size = 16;

    int64_t slideSizeBysize = calculateSlideSize(coreNumPlatForm, direction);
    if (input_shapes[DIM_TWO] == NUM_ONE && input_shapes[DIM_THREE] == NUM_ONE) {
        slideSizeBysize = NUM_1024;
    }
    if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_1) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_128), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_5) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_64), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_8) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_48), slideSizeBysize);
    } else if (real_scale <= BEST_PERFORMANCE_SCALE_BELOW_20) {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_32), slideSizeBysize);
    } else {
        slide_size = std::min(static_cast<int64_t>(BEST_PERFORMANCE_SIZE_16), slideSizeBysize);
    }
    return slide_size;
}

inline int64_t UpsampleLinear2dTiling::calculateSlideSize(const uint32_t coreNumPlatForm, uint8_t direction)
{
    int64_t slide_size = BEST_PERFORMANCE_SIZE_16;
    uint32_t neeCoreNumMax = 0;

    for (uint32_t coreIndex = 0; coreIndex < NUM_FIVE; coreIndex++) {
        uint32_t res = 0;
        if (direction == NUM_ONE) {
            res = GetNeedCoreNumW(coreNumPlatForm, NUM_ONE, slide_size_list[coreIndex]);
        } else {
            res = GetNeedCoreNumH(coreNumPlatForm, NUM_ONE, slide_size_list[coreIndex]);
        }
        slide_size = res >= neeCoreNumMax ? slide_size_list[coreIndex] : slide_size;
        neeCoreNumMax = std::max(res, neeCoreNumMax);
    }
    return slide_size;
}

template <typename T1, typename T2>
inline T1 UpsampleLinear2dTiling::CeilA2B(T1 a, T2 b) const
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int32_t UpsampleLinear2dTiling::Ceil(T1 x) const
{
    int32_t floor_x = int32_t(x);
    if (x == floor_x) {
        return floor_x;
    }
    return floor_x + 1;
}

uint8_t UpsampleLinear2dTiling::GetDataTypeSize() const
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

uint64_t UpsampleLinear2dTiling::GetDataTypeVal()
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

uint32_t UpsampleLinear2dTiling::GetNeedCoreNumW(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[3];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

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

    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumW(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumW(tailStartSlideNum);
        tilingData.set_slideNumW(slideNum);
        tilingData.set_groupCoreNumW(groupCoreNum);
        tilingData.set_tailAvergingRowsW(tailAvergingRows);
        tilingData.set_remainderW(remainder);
        tilingData.set_need_core_num_w(needCoreNum);
    }

    return needCoreNum;
}

uint32_t UpsampleLinear2dTiling::GetNeedCoreNumH(
    const uint32_t coreNumPlatform, uint8_t isCalculate, int64_t slide_size)
{
    int64_t outputSize = output_shapes[2];
    int64_t slideNum = CeilA2B(outputSize, slide_size);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;

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
    if (isCalculate == NUM_TWO) {
        tilingData.set_eachCoreSlideNumH(eachCoreSlideNum);
        tilingData.set_tailStartSlideNumH(tailStartSlideNum);
        tilingData.set_slideNumH(slideNum);
        tilingData.set_groupCoreNumH(groupCoreNum);
        tilingData.set_tailAvergingRowsH(tailAvergingBatch);
        tilingData.set_remainderH(remainder);
        tilingData.set_need_core_num_h(needCoreNum);
    }
    return needCoreNum;
}

void UpsampleLinear2dTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleBilinear2dTiling(gert::TilingContext *context)
{
    UpsampleLinear2dTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling(MODE_BILINEAR);
}

static ge::graphStatus tilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleLinear2dCompileInfo>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_TILING_CHECK(compileInfo->coreNum <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "UpsampleLinear2d GetHardwareInfo Failed, vectorCoreNum:%d",
            compileInfo->coreNum),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleBilinear2d)
    .Tiling(tiling4UpsampleBilinear2dTiling)
    .TilingParse<UpsampleLinear2dCompileInfo>(tilingPrepareTiling);
}  // namespace optiling

namespace ops {
class UpsampleBilinear2d : public OpDef {
public:
    explicit UpsampleBilinear2d(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("scales").AttrType(OPTIONAL).ListFloat();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(UpsampleBilinear2d);
}  // namespace ops