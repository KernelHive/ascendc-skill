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
 * \file is_finite.cpp
 * \brief
 */
#include "is_inf_tiling.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
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

using namespace ge;

namespace optiling {
constexpr uint32_t DATA_BLOCK = 32;
constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;


class IsInfTiling {
public:
    explicit IsInfTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    void AssignDataToEachCore();
    void FillTilingData();

    uint8_t GetDataTypeSize();
    uint64_t GetTilingKeyVal();

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    uint32_t GetUsableUbMemory(uint64_t ubSizePlatForm);
    
private:
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;
    IsInfTilingData tilingData;

    uint8_t dataBlockSize = 0;

    uint32_t totalDataCount = 1;
    uint32_t usableUbSize = 0;
    uint32_t needCoreNum = 0;
    uint32_t perCoreDataCount = 0;
    uint32_t tailDataCoreNum = 0;
    uint32_t lastCoreDataCount = 0;
};

ge::graphStatus IsInfTiling::RunBigKernelTiling() {
    if (tilingContext == nullptr) {
        OP_LOGE(context->GetNodeName(), "tilingContext is nullptr!");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

    uint64_t ubSizePlatForm = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    // Get dtype information, and the total number of data.
    if (tilingContext != nullptr && tilingContext->GetInputDesc(0) != nullptr) {
        dataType = tilingContext->GetInputDesc(0)->GetDataType();
    }
    uint8_t dataTypeSize = GetDataTypeSize();
    dataBlockSize = DATA_BLOCK * dataTypeSize;
    const gert::StorageShape* shape = tilingContext->GetInputShape(0);
    uint16_t dimNumOfShape = 0;
    if (shape != nullptr) {
        dimNumOfShape = shape->GetStorageShape().GetDimNum();
    }
    for (uint16_t i = 0; i < dimNumOfShape; i++) {
        if (shape != nullptr) {
            totalDataCount *= shape->GetStorageShape().GetDim(i);
        }
    }

    usableUbSize = GetUsableUbMemory(ubSizePlatForm);
    needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    AssignDataToEachCore();
    FillTilingData();

    tilingContext->SetTilingKey(GetTilingKeyVal());
    tilingContext->SetBlockDim(needCoreNum);
    
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = WORK_SPACE_SIZE;
    }
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

template <typename T1, typename T2>
inline T1 IsInfTiling::CeilA2B(T1 a, T2 b) const {
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

uint8_t IsInfTiling::GetDataTypeSize() {
    switch (dataType) {
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return BYTE_LEN_4;
    }
}

uint64_t IsInfTiling::GetTilingKeyVal() {
    switch (dataType) {
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_BF16:
            return TILING_KEY_BFLOAT16;
        default:
            return 0;
    }
}

uint32_t IsInfTiling::GetNeedCoreNum(uint32_t coreNumPlatform) {
    uint32_t tempCoreNum = (uint32_t)CeilA2B(totalDataCount, DATA_BLOCK);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void IsInfTiling::AssignDataToEachCore() {
    perCoreDataCount = totalDataCount / needCoreNum;
    perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;
    uint32_t tempTailDataCount = totalDataCount - perCoreDataCount * needCoreNum;
    tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
    lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;
}


uint32_t IsInfTiling::GetUsableUbMemory(uint64_t ubSizePlatForm) {
    // The remaining UB size is split in two, double buffer enabled, input and output, and rounded down 32 data.
    uint32_t canUseUbSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize()) / UB_DIVIDER_FOR_TEMP_CASTING;
    canUseUbSize = canUseUbSize / dataBlockSize * dataBlockSize;
    return canUseUbSize;
}

void IsInfTiling::FillTilingData() {
    tilingData.set_totalDataCount(totalDataCount);
    tilingData.set_usableUbSize(usableUbSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_perCoreDataCount(perCoreDataCount);
    tilingData.set_tailDataCoreNum(tailDataCoreNum);
    tilingData.set_lastCoreDataCount(lastCoreDataCount);
}

static ge::graphStatus TilingIsInfTiling(gert::TilingContext* context) {
    IsInfTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(IsInf)
    .Tiling(TilingIsInfTiling);
}  // namespace optiling

// proto
namespace ge{
namespace ops {
static ge::graphStatus InferShape4IsInf(gert::InferShapeContext* context) {
  OP_LOGD(context, "Begin to do IsInfInferShape");
  auto inShape = context->GetInputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, inShape);
  auto outShape = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

  *outShape = *inShape;

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4IsInf(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4IsInf");

  auto inputDtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, ge::DT_BOOL);
  OP_LOGD(context->GetNodeName(), "End to do InferDataType4IsInf");
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(IsInf)
    .InferShape(InferShape4IsInf).InferDataType(InferDataType4IsInf);
} // namespace ops
} // namespace ge


