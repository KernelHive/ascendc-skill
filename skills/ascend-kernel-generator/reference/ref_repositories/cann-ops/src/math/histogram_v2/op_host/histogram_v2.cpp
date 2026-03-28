/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file histogram_v2.cc
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "histogram_v2_tiling.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
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

// tiling
namespace optiling {
  constexpr int64_t SIZE_OF_FP32 = 4;
  constexpr int64_t BYTE_BLOCK = 32;

  constexpr int64_t HISTOGRAM_V2_FP32 = 0;
  constexpr int64_t HISTOGRAM_V2_INT32 = 1;
  constexpr int64_t HISTOGRAM_V2_INT8 = 2;
  constexpr int64_t HISTOGRAM_V2_UINT8 = 3;
  constexpr int64_t HISTOGRAM_V2_INT16 = 4;
  constexpr int64_t HISTOGRAM_V2_INT64 = 5;
  constexpr int64_t HISTOGRAM_V2_FP16 = 6;
  constexpr int64_t HISTOGRAM_V2_NOT_SUPPORT = -1;

  constexpr int64_t UB_SELF_LENGTH = 16320; // 64 * 255
  constexpr int64_t UB_BINS_LENGTH = 16320; // 16320 * 4 = 65280 < 65535，结果可一次性搬出
  constexpr int64_t UB_SELF_LENGTH_310P = 16000;
  constexpr int64_t UB_BINS_LENGTH_310P = 16320;

  class HistogramV2Tiling {
    public:
        explicit HistogramV2Tiling(gert::TilingContext* context) : tilingContext(context){};
        ge::graphStatus Init();
        ge::graphStatus SetKernelTiling();
        void TilingDataPrint() const;
    private:
        inline void SetTilingKeyMode(ge::DataType dType) const;
        inline void TilingDataForCore();
        inline void TilingDataInCore(ge::DataType dType);

        HistogramV2TilingData tilingData;
        gert::TilingContext* tilingContext = nullptr;
        int64_t coreNum = 0;
        int64_t totalLength = 0; // the length of input
        int64_t tailNum = 0;
        int64_t userWorkspaceSize = 0;
    private:
        // kernel needed.
        int64_t bins = 0;
        int64_t ubSelfLength = UB_SELF_LENGTH;
        int64_t ubBinsLength = UB_BINS_LENGTH;

        int64_t formerNum = 0;
        int64_t formerLength = 0;
        int64_t formerLengthAligned = 0;
        int64_t tailLength = 0;
        int64_t tailLengthAligned = 0;

        int64_t formerTileNum = 0;
        int64_t formerTileDataLength = 0;
        int64_t formerTileLeftDataLength = 0;
        int64_t formerTileLeftDataLengthAligned = 0;

        int64_t tailTileNum = 0;
        int64_t tailTileDataLength = 0;
        int64_t tailTileLeftDataLength = 0;
        int64_t tailTileLeftDataLengthAligned = 0;
  };

  inline void HistogramV2Tiling::SetTilingKeyMode(ge::DataType dType) const {
    switch (dType) {
        case ge::DT_FLOAT:
            tilingContext->SetTilingKey(HISTOGRAM_V2_FP32);
            break;
        case ge::DT_INT32:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT32);
            break;
        case ge::DT_INT8:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT8);
            break;
        case ge::DT_UINT8:
            tilingContext->SetTilingKey(HISTOGRAM_V2_UINT8);
            break;
        case ge::DT_INT16:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT16);
            break;
        case ge::DT_INT64:
            tilingContext->SetTilingKey(HISTOGRAM_V2_INT64);
            break;
        case ge::DT_FLOAT16:
            tilingContext->SetTilingKey(HISTOGRAM_V2_FP16);
            break;
        default:
            tilingContext->SetTilingKey(HISTOGRAM_V2_NOT_SUPPORT);
            break;
    }
  }

  inline void HistogramV2Tiling::TilingDataForCore() {
    OP_LOGD(tilingContext->GetNodeName(), "TilingDataForCore start.");
    auto alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    formerNum = 1;
    tailNum = coreNum - formerNum;
    tailLength = totalLength / coreNum;
    formerLength = totalLength - (tailLength * coreNum) + tailLength;
    formerLengthAligned = ((formerLength + alignNum -1) / alignNum) * alignNum;
    tailLengthAligned = ((tailLength + alignNum -1) / alignNum) * alignNum;
    if (tailLength == 0) {
        coreNum = 1;
    }
    if (totalLength == 0) {
        coreNum = 0;
    }
    OP_LOGD(tilingContext->GetNodeName(), "TilingDataForCore end.");
  }

  inline void HistogramV2Tiling::TilingDataInCore(ge::DataType dType) {
    int64_t tileLength = ubSelfLength;
    switch (dType) {
        case ge::DT_INT64:
            tileLength = ubSelfLength / HISTOGRAM_V2_INT8;
            break;
        default:
            break;
    }
    int64_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
    formerTileNum = formerLength / tileLength;
    formerTileDataLength = tileLength;
    formerTileLeftDataLength = formerLength - formerTileNum * formerTileDataLength;
    formerTileLeftDataLengthAligned = ((formerTileLeftDataLength + alignNum -1) / alignNum) * alignNum;

    tailTileNum = tailLength / tileLength;
    tailTileDataLength = tileLength;
    tailTileLeftDataLength = tailLength - tailTileNum * tailTileDataLength;
    tailTileLeftDataLengthAligned = ((tailTileLeftDataLength + alignNum -1) / alignNum) * alignNum;
  }

  ge::graphStatus HistogramV2Tiling::Init() {
    OP_LOGD(tilingContext->GetNodeName(), "Tiling initing.");

    auto selfShape = tilingContext->GetInputShape(0)->GetStorageShape();
    totalLength = selfShape.GetShapeSize();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(tilingContext->GetNodeName(), "coreNum %ld.", coreNum);
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    auto dType = tilingContext->GetInputDesc(0)->GetDataType();
    
    auto attrs = tilingContext->GetAttrs();
    int32_t binsIndex = 0;
    bins = *(attrs->GetAttrPointer<int64_t>(binsIndex));
    SetTilingKeyMode(dType);
    tilingContext->SetNeedAtomic(true);
    TilingDataForCore();
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
      ubSelfLength = UB_SELF_LENGTH_310P;
      ubBinsLength = UB_BINS_LENGTH_310P;
    }
    TilingDataInCore(dType);
    // Sync workspace size and kernel result size
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
      int64_t alignNum = BYTE_BLOCK / SIZE_OF_FP32;
      userWorkspaceSize = coreNum * BYTE_BLOCK + coreNum * (bins + alignNum) * SIZE_OF_FP32;
    } 
    size_t *currentWorkSpace = tilingContext->GetWorkspaceSizes(1);
    currentWorkSpace[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + userWorkspaceSize;
    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus HistogramV2Tiling::SetKernelTiling() {
    tilingContext->SetBlockDim(coreNum);

    tilingData.set_bins(bins);
    tilingData.set_ubBinsLength(ubBinsLength);

    tilingData.set_formerNum(formerNum);
    tilingData.set_formerLength(formerLength);
    tilingData.set_formerLengthAligned(formerLengthAligned);
    tilingData.set_tailLength(tailLength);
    tilingData.set_tailLengthAligned(tailLengthAligned);

    tilingData.set_formerTileNum(formerTileNum);
    tilingData.set_formerTileDataLength(formerTileDataLength);
    tilingData.set_formerTileLeftDataLength(formerTileLeftDataLength);
    tilingData.set_formerTileLeftDataLengthAligned(formerTileLeftDataLengthAligned);
    
    tilingData.set_tailTileNum(tailTileNum);
    tilingData.set_tailTileDataLength(tailTileDataLength);
    tilingData.set_tailTileLeftDataLength(tailTileLeftDataLength);
    tilingData.set_tailTileLeftDataLengthAligned(tailTileLeftDataLengthAligned);

    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
  }

  void HistogramV2Tiling::TilingDataPrint() const {
    OP_LOGD(tilingContext->GetNodeName(), "coreNum: %ld.", coreNum);
    OP_LOGD(tilingContext->GetNodeName(), "totalLength: %ld.", totalLength);
    OP_LOGD(tilingContext->GetNodeName(), "bins: %ld.", bins);
    OP_LOGD(tilingContext->GetNodeName(), "formerNum: %ld.", formerNum);
    OP_LOGD(tilingContext->GetNodeName(), "formerLength: %ld.", formerLength);
    OP_LOGD(tilingContext->GetNodeName(), "formerLengthAligned: %ld.", formerLengthAligned);
    OP_LOGD(tilingContext->GetNodeName(), "tailLength: %ld.", tailLength);
    OP_LOGD(tilingContext->GetNodeName(), "tailLengthAligned: %ld.", tailLengthAligned);
    OP_LOGD(tilingContext->GetNodeName(), "formerTileNum: %ld.", formerTileNum);
    OP_LOGD(tilingContext->GetNodeName(), "formerTileDataLength: %ld.", formerTileDataLength);
    OP_LOGD(tilingContext->GetNodeName(), "formerTileLeftDataLength: %ld.", formerTileLeftDataLength);
    OP_LOGD(tilingContext->GetNodeName(), "formerTileLeftDataLengthAligned: %ld.", formerTileLeftDataLengthAligned);
    OP_LOGD(tilingContext->GetNodeName(), "tailTileNum: %ld.", tailTileNum);
    OP_LOGD(tilingContext->GetNodeName(), "tailTileDataLength: %ld.", tailTileDataLength);
    OP_LOGD(tilingContext->GetNodeName(), "tailTileLeftDataLength: %ld.", tailTileLeftDataLength);
    OP_LOGD(tilingContext->GetNodeName(), "tailTileLeftDataLengthAligned: %ld.", tailTileLeftDataLengthAligned);
  }

  static ge::graphStatus TilingHistogramV2(gert::TilingContext* context) {
    HistogramV2Tiling tilingObject(context);
    tilingObject.Init();
    return tilingObject.SetKernelTiling();
  }

  IMPL_OP_OPTILING(HistogramV2)
      .Tiling(TilingHistogramV2);
};  // namespace optiling

// proto
namespace ge{
namespace ops {
static constexpr int64_t BINS_IDX = 0;
static constexpr int64_t OUTPUT_IDX = 0;

static ge::graphStatus HistogramV2InferShapeFunc(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do HistogramV2InferShapeFunc");
  // 获取bins
  auto attr = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT(context, attr);
  int64_t bins = *(attr->GetAttrPointer<int64_t>(BINS_IDX));

  gert::Shape* y_shape = context->GetOutputShape(OUTPUT_IDX);
  OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  y_shape->SetDimNum(1);
  y_shape->SetDim(0, bins);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HistogramV2InferDataTypeFunc (gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do HistogramV2InferDataTypeFunc");

  auto inputDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, ge::DT_INT32);
  OP_LOGD(context->GetNodeName(), "End to do HistogramV2InferDataTypeFunc end");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HistogramV2).InferShape(HistogramV2InferShapeFunc).InferDataType(HistogramV2InferDataTypeFunc);
} // namespace ops
} //namespace ge

