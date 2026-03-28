/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file roi_align_rotated.cpp
 */

#include <cstdint>
#include <cstdio>
#include <cmath>
#include "roi_align_rotated_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_type.h"

using namespace ge;
using namespace std;


// TOOLS API

namespace {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
    if ((ptr) == nullptr) {                                                                        \
        std::printf("nullptr error!");                                                               \
        return ge::GRAPH_FAILED;                                                                     \
    }

template <typename T>
    inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
        return context->GetCompiledInfo<T>();
    }

const uint32_t INPUT_INDEX = 0; // 算子输入2个，输入Tensor序号分别为0和1
const uint32_t ROIS_INDEX = 1;
const uint32_t OUTPUT_INDEX = 0; // 算子输出1个，输出Tensor序号为0
const uint32_t ROIS_NUM_INDEX = 1;

const uint32_t BS_INDEX = 0;
const uint32_t H_INDEX = 1;
const uint32_t W_INDEX = 2;
const uint32_t CHANNEL_INDEX = 3;

const uint32_t PH_INDEX = 0;
const uint32_t PW_INDEX = 1;
const uint32_t SPATIAL_INDEX = 2;
const uint32_t SAMPLING_INDEX = 3;
const uint32_t ALIGNED_INDEX = 4;
const uint32_t CLOCKWISE_INDEX = 5;

const uint32_t ALIGN_VALUE = 8;
const uint32_t TILING_KEY = 1;
const uint32_t TILE_NUM = 8;

const uint32_t DIM_0 = 0;
const uint32_t DIM_1 = 1;
const uint32_t DIM_2 = 2;
const uint32_t DIM_3 = 3;
}  // namespace

//tiling
namespace optiling {

static ge::graphStatus TilingPrepare4RoiAlignRotated(gert::TilingParseContext* context) {
    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform_info = platform_ascendc::PlatformAscendC(platform);
    uint64_t ub_total_size;
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
    uint32_t BLOCK_DIM = platform_info.GetCoreNumAiv();
    if (BLOCK_DIM == 0) {
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = GetCompileInfoPtr<RoiAlignRotatedCompileInfo>(context);
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->totalCoreNum = BLOCK_DIM;
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ub_total_size);

    return ge::GRAPH_SUCCESS;
  }
static ge::graphStatus TilingForRoiAlignRotated(gert::TilingContext* context)
{
    RoiAlignRotatedTilingData tiling;

    auto inputTensorPtr = context->GetInputTensor(INPUT_INDEX);
    auto RoisTensorPtr = context->GetInputTensor(ROIS_INDEX);
    if (inputTensorPtr == nullptr || RoisTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batch_size = inputTensorPtr->GetStorageShape().GetDim(BS_INDEX);
    uint32_t input_h = inputTensorPtr->GetStorageShape().GetDim(H_INDEX);
    uint32_t input_w = inputTensorPtr->GetStorageShape().GetDim(W_INDEX);
    uint32_t channels = inputTensorPtr->GetStorageShape().GetDim(CHANNEL_INDEX);
    uint32_t channels_aligned;
    if (static_cast<uint32_t>(channels % ALIGN_VALUE) == 0) {
        channels_aligned = channels;
    } else {
        channels_aligned = (static_cast<uint32_t>(channels / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    float spatial_scale = *(attrsPtr->GetAttrPointer<float>(SPATIAL_INDEX));
    int32_t sampling_ratio = *(attrsPtr->GetAttrPointer<int32_t>(SAMPLING_INDEX));
    int32_t pooled_height = *(attrsPtr->GetAttrPointer<int32_t>(PH_INDEX));
    int32_t pooled_width = *(attrsPtr->GetAttrPointer<int32_t>(PW_INDEX));
    bool aligned = *(attrsPtr->GetAttrPointer<bool>(ALIGNED_INDEX));
    bool clockwise = *(attrsPtr->GetAttrPointer<bool>(CLOCKWISE_INDEX));

    uint32_t rois_num = RoisTensorPtr->GetStorageShape().GetDim(ROIS_NUM_INDEX);
    if (rois_num == 0) {
        return ge::GRAPH_FAILED;
    }
    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform_info = platform_ascendc::PlatformAscendC(platform);
    uint64_t ub_total_size;
    platform_info.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_total_size);
    uint32_t BLOCK_DIM = platform_info.GetCoreNumAiv();
    if (BLOCK_DIM == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t rois_num_aligned;
    if (static_cast<uint32_t>(rois_num % ALIGN_VALUE) == 0) {
        rois_num_aligned = rois_num;
    } else {
        rois_num_aligned = (static_cast<uint32_t>(rois_num / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    uint32_t tail_num = rois_num_aligned - rois_num; // 获取计算完成后需要丢弃的rois数目
    uint32_t rois_num_per_Score = (rois_num_aligned / BLOCK_DIM / ALIGN_VALUE) * ALIGN_VALUE;
    uint32_t rois_num_per_Lcore = rois_num_per_Score + ALIGN_VALUE;
    uint32_t Score_num = (BLOCK_DIM * (ALIGN_VALUE + rois_num_per_Score) - rois_num_aligned) / ALIGN_VALUE;
    uint32_t Lcore_num = BLOCK_DIM - Score_num;

    if (rois_num_per_Score == 0) {
        BLOCK_DIM = BLOCK_DIM - Score_num;
    }
    if (rois_num_per_Lcore == 0) {
        BLOCK_DIM = BLOCK_DIM - Lcore_num;
    }

    float input_size = float(channels_aligned) / ALIGN_VALUE;
    uint32_t input_buffer_size = static_cast<uint32_t>(ceil(input_size)) * ALIGN_VALUE * sizeof(float);

    tiling.set_blockDim(BLOCK_DIM);
    tiling.set_ub_total_size(ub_total_size);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_batch_size(batch_size);
    tiling.set_channels(channels);
    tiling.set_channels_aligned(channels_aligned);
    tiling.set_input_h(input_h);
    tiling.set_input_w(input_w);
    tiling.set_rois_num_aligned(rois_num_aligned);
    tiling.set_tail_num(tail_num);
    tiling.set_spatial_scale(spatial_scale);
    tiling.set_sampling_ratio(sampling_ratio);
    tiling.set_pooled_height(pooled_height);
    tiling.set_pooled_width(pooled_width);
    tiling.set_aligned(aligned);
    tiling.set_clockwise(clockwise);
    tiling.set_rois_num_per_Lcore(rois_num_per_Lcore);
    tiling.set_rois_num_per_Score(rois_num_per_Score);
    tiling.set_Lcore_num(Lcore_num);
    tiling.set_Score_num(Score_num);
    tiling.set_input_buffer_size(input_buffer_size);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(TILING_KEY);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RoiAlignRotated)
    .Tiling(TilingForRoiAlignRotated)
    .TilingParse<RoiAlignRotatedCompileInfo>(TilingPrepare4RoiAlignRotated);
} // namespace optiling


//proto
namespace ge {


static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto input_shape = context->GetInputShape(INPUT_INDEX);
    auto rois_shape = context->GetInputShape(ROIS_INDEX);
    auto output_shape = context->GetOutputShape(OUTPUT_INDEX);
    if (input_shape == nullptr || rois_shape == nullptr || output_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t rois_num = rois_shape->GetDim(ROIS_NUM_INDEX);
    int32_t channels = input_shape->GetDim(CHANNEL_INDEX);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int32_t* pooled_height = attrsPtr->GetAttrPointer<int32_t>(PH_INDEX);
    const int32_t* pooled_width = attrsPtr->GetAttrPointer<int32_t>(PW_INDEX);

    auto output_shape_length = 4;
    output_shape->SetDimNum(output_shape_length);
    output_shape->SetDim(DIM_0, rois_num);
    output_shape->SetDim(DIM_1, *pooled_height); // 设置输出形状的第二个维度为pooled_height
    output_shape->SetDim(DIM_2, *pooled_width); // 设置输出形状的第三个维度为pooled_width
    output_shape->SetDim(DIM_3, channels); // 设置输出形状的第四个维度为channels

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeRoiAlignRotated(gert::InferDataTypeContext* context)
{
    const ge::DataType value_dtype = context->GetInputDataType(INPUT_INDEX);
    context->SetOutputDataType(OUTPUT_INDEX, value_dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RoiAlignRotated).InferShape(InferShape).InferDataType(InferDataTypeRoiAlignRotated);
} // namespace ge

namespace ops {
class RoiAlignRotated : public OpDef {
public:
    explicit RoiAlignRotated(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rois")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("pooled_h").AttrType(REQUIRED).Int();
        this->Attr("pooled_w").AttrType(REQUIRED).Int();
        this->Attr("spatial_scale").AttrType(REQUIRED).Float();
        this->Attr("sampling_ratio").AttrType(OPTIONAL).Int(0);
        this->Attr("aligned").AttrType(OPTIONAL).Bool(true);
        this->Attr("clockwise").AttrType(OPTIONAL).Bool(false);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(RoiAlignRotated);
} // namespace ops