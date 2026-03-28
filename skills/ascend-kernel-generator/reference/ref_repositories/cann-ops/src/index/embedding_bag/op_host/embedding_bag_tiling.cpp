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
 * \file embedding_bag.cc
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_type.h"
#include "embedding_bag_tiling.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
#define OP_CHECK(cond, log_func, return_expr) \
  if (cond) {                                 \
    log_func;                                 \
    return_expr;                              \
  }
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)

namespace optiling {

constexpr uint32_t WEIGHT_INPUT_INDEX = 0;
constexpr uint32_t INDICES_INPUT_INDEX = 1;
constexpr uint32_t OFFSETS_INPUT_INDEX = 2;
constexpr uint32_t PER_SAMPLER_WEIGHTS_INPUT_INDEX = 3;
constexpr uint32_t INDICES_MAX_MOVE_LENGTH = 1024;
constexpr uint32_t CHECK_TWO_DIM = 2;
constexpr uint32_t CHECK_ONE_DIM = 1;
constexpr uint32_t CHECK_ZERO_DIM = 0;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t MODE_INDEX = 0;
constexpr uint32_t INCLUDE_LAST_OFFSET_INDEX = 3;
constexpr uint32_t PADDING_IDX_INDEX = 4;
constexpr uint32_t ONE_TIMES = 1;
constexpr uint32_t DOUBLE_TIMES = 2;
constexpr uint32_t THREE_TIMES = 3;
constexpr uint32_t FOUR_TIMES = 4;
constexpr uint32_t DOUBLE_BYTE = 4;
constexpr uint32_t FOUR_BYTE = 4;
constexpr uint32_t EIGHT_BYTE = 8;
constexpr uint32_t RESERVED_UB = 15 * 1024;
constexpr uint32_t MAX_DIVIDE_UB_NUM = 3;
constexpr uint32_t OTHER_DIVIDE_UB_NUM = 3;
constexpr uint32_t MODE_SUM = 0;
constexpr uint32_t MODE_MEAN = 1;
constexpr uint32_t MODE_MAX = 2;
constexpr uint32_t TILINGKEY_FP32 = 1;
constexpr uint32_t TILINGKEY_FP16 = 2;
constexpr uint32_t TILINGKEY_BF16 = 3;

const std::string SUM = "sum";
const std::string MEAN = "mean";
const std::string MAX = "max";

class EmbeddingBagTiling {
public:
    explicit EmbeddingBagTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    void GetUsedCore();
    void getTilingKeyAndComputeRepTime(ge::DataType weightDtype, std::string mode);
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:

    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) {
        return (a + b - 1) / b;
    }
    template <typename T1, typename T2>
    inline T1 FloorDiv(T1 a, T2 b) {
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline T1 CeilAlign(T1 a, T2 b) {
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline T1 FloorAlign(T1 a, T2 b) {
        return (a) / b * b;
    }

private:
    EmbeddingBagTilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    int64_t numEmbeddings_ = 0;
    int64_t numOffset_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t usedCoreNum_ = 0;
    int64_t paddingIdx_ = 0;
    int64_t formerOffsetNum_ = 0;
    int64_t tailOffsetNum_ = 0;
    int64_t computeRepTime_ = 0;
    int64_t indicesMaxMoveLength_ = 0;
    int64_t numIndices_ = 0;
    int64_t mode_ = 0;
    int64_t tilingKey_ = 0;          
    uint64_t ubSizePlatForm_ = 0;
    int64_t hasPerSampleWeights_ = 0;
};


ge::graphStatus EmbeddingBagTiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    indicesMaxMoveLength_ = INDICES_MAX_MOVE_LENGTH;

    const gert::StorageShape* weightShape = tilingContext_->GetInputShape(WEIGHT_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext_, weightShape);
    const gert::StorageShape* indicesShape = tilingContext_->GetInputShape(INDICES_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext_, indicesShape);
    const gert::StorageShape* offsetsShape = tilingContext_->GetInputShape(OFFSETS_INPUT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext_, offsetsShape);
    auto perSampleWeights = tilingContext_->GetOptionalInputTensor(PER_SAMPLER_WEIGHTS_INPUT_INDEX);
    if (weightShape->GetStorageShape().GetDimNum() != CHECK_TWO_DIM) {
        OP_LOGD(tilingContext_->GetNodeName(), "weight dim is not 2, please check input");
        return ge::GRAPH_FAILED;
    }
    if (indicesShape->GetStorageShape().GetDimNum() != CHECK_ONE_DIM ||
        offsetsShape->GetStorageShape().GetDimNum() != CHECK_ONE_DIM) {
        OP_LOGD(tilingContext_->GetNodeName(), "indices or offsets dim is not 1, please check input");
        return ge::GRAPH_FAILED;
    }
    if (perSampleWeights != nullptr && perSampleWeights->GetShapeSize() != 0) {
        hasPerSampleWeights_ = 1;   
    }
    numEmbeddings_ = weightShape->GetStorageShape().GetDim(DIM_INDEX1);
    numIndices_ = indicesShape->GetStorageShape().GetDim(DIM_INDEX0);
    numOffset_ = offsetsShape->GetStorageShape().GetDim(DIM_INDEX0);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm_);

    auto attrs = tilingContext_->GetAttrs();
    OP_TILING_CHECK((attrs == nullptr), VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), 
           "Get attrs Failed."), return  ge::GRAPH_FAILED);
    const std::string mode = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
     OP_LOGW(tilingContext_->GetNodeName(), "%s mode is not supported", mode.c_str());
    if (mode == MEAN) {
        mode_ = MODE_MEAN;
    } else if (mode == SUM) {
        mode_ = MODE_SUM;
    } else if (mode == MAX){
        mode_ = MODE_MAX;
    } else {
        return ge::GRAPH_FAILED;
    }
    bool includeLastOffset = *attrs->GetAttrPointer<bool>(INCLUDE_LAST_OFFSET_INDEX);
    if (includeLastOffset) {
        numOffset_ -= 1;
    }
    OP_TILING_CHECK((numOffset_ == 0), VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), 
           "If include_last_offset is true, offset size should greater than 1"), return  ge::GRAPH_FAILED);
    paddingIdx_ = *attrs->GetAttrPointer<int>(PADDING_IDX_INDEX);
    GetUsedCore();
    
    ge::DataType weightDatatype = tilingContext_->GetInputDesc(WEIGHT_INPUT_INDEX)->GetDataType();
    
    getTilingKeyAndComputeRepTime(weightDatatype, mode);

    OP_TILING_CHECK((computeRepTime_ <= 0), VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), 
        "computeRepTime  less than 0"), return  ge::GRAPH_FAILED);

    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

void EmbeddingBagTiling::GetUsedCore() {
    if (this->numOffset_ <= this->coreNum_) {
        this->usedCoreNum_ = this->numOffset_;
        this->formerOffsetNum_ = 1;
        this->tailOffsetNum_ = 1;
        return;
    }
    if (this->coreNum_ != 0) {
        this->formerOffsetNum_ = FloorDiv(this->numOffset_, this->coreNum_);
    }
    this->usedCoreNum_ = this->coreNum_;
    this->tailOffsetNum_ = this->numOffset_ - this->formerOffsetNum_ * (this->usedCoreNum_ - 1);
}

void EmbeddingBagTiling::getTilingKeyAndComputeRepTime(ge::DataType weightDtype, std::string mode) {
    auto resiveUbSize = ubSizePlatForm_ - RESERVED_UB - formerOffsetNum_ * sizeof(float) * DOUBLE_TIMES - 
            INDICES_MAX_MOVE_LENGTH * sizeof(float) * DOUBLE_TIMES;

    if (weightDtype == ge::DT_FLOAT) {
        tilingKey_ = TILINGKEY_FP32;
    } else if (weightDtype == ge::DT_FLOAT16) {
        tilingKey_ = TILINGKEY_FP16;
    } else {
        tilingKey_ = TILINGKEY_BF16;
    }
    if (mode == MAX) {
        computeRepTime_ = resiveUbSize / (MAX_DIVIDE_UB_NUM * sizeof(float));
    } else {
        computeRepTime_ = resiveUbSize / (OTHER_DIVIDE_UB_NUM * sizeof(float));
    }
}

ge::graphStatus EmbeddingBagTiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingData_.set_formerOffsetNum(formerOffsetNum_);
    tilingData_.set_tailOffsetNum(tailOffsetNum_);
    tilingData_.set_numEmbeddings(numEmbeddings_);
    tilingData_.set_computeRepTime(computeRepTime_);
    tilingData_.set_indicesMaxMoveLength(indicesMaxMoveLength_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_paddingIdx(paddingIdx_);
    tilingData_.set_numIndices(numIndices_);
    tilingData_.set_mode(mode_);
    tilingData_.set_hasPerSampleWeights(hasPerSampleWeights_);
    tilingData_.set_tilingKey(tilingKey_);
    
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->SetTilingKey(tilingKey_);
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void EmbeddingBagTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_->GetNodeName(),  "start print             ");
    OP_LOGD(tilingContext_->GetNodeName(),  "usedCoreNum_:            %d", usedCoreNum_);
    OP_LOGD(tilingContext_->GetNodeName(),  "formerOffsetNum_:       %ld", formerOffsetNum_);
    OP_LOGD(tilingContext_->GetNodeName(),  "tailOffsetNum_:         %ld", tailOffsetNum_);
    OP_LOGD(tilingContext_->GetNodeName(),  "numEmbeddings_:         %ld", numEmbeddings_);
    OP_LOGD(tilingContext_->GetNodeName(),  "computeRepTime_:        %ld", computeRepTime_);
    OP_LOGD(tilingContext_->GetNodeName(),  "indicesMaxMoveLength_:  %ld", indicesMaxMoveLength_);
    OP_LOGD(tilingContext_->GetNodeName(),  "paddingIdx_:            %ld", paddingIdx_);
    OP_LOGD(tilingContext_->GetNodeName(),  "numIndices_:            %ld", numIndices_);
    OP_LOGD(tilingContext_->GetNodeName(),  "mode_:                  %ld", mode_);
    OP_LOGD(tilingContext_->GetNodeName(),  "hasPerSampleWeights_:   %ld", hasPerSampleWeights_); 
    OP_LOGD(tilingContext_->GetNodeName(),  "tilingKey_:             %ld", tilingKey_);      
}

ge::graphStatus TilingEmbeddingBag(gert::TilingContext* context)
{
    EmbeddingBagTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

IMPL_OP_OPTILING(EmbeddingBag)
    .Tiling(TilingEmbeddingBag);
} // namespace optiling

