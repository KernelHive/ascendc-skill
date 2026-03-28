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
 * \file non_finite_check.cpp
 * \brief
 */
#include "non_finite_check_tiling.h"
#include <cmath>
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

template <typename T>
T CeilAlign(T a, T b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b * b;
}

using namespace ge;

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;
constexpr size_t WORKSPACE_SIZE = 1;

constexpr uint8_t DTYPE_SIZE_FLOAT = 4;
constexpr uint8_t NUM_TWO = 2;
constexpr uint32_t COEFFICIENT_1 = 128;

class NonFiniteCheckTiling {
public:
    explicit NonFiniteCheckTiling(gert::TilingContext* context)
        : tilingContext(context), nodeName(context->GetNodeName()) {};

    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    void InitTilingDataItems();
    ge::graphStatus CheckParams() const;
    ge::graphStatus FillCompileInfo();
    bool CalcNeedCoreNum();
    void AssignDataToEachCore();
    bool DivideUbMemory();
    uint32_t GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const;
    uint64_t GetTilingKeyVal() const;
    void FillTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    std::string nodeName = "NonFiniteCheck";
    NonFiniteCheckTilingData tilingData;
    NonFiniteCheckCompileInfo compileInfo;

    uint32_t maxProcCount = 0;
    uint32_t tempValUbSize = 0;
    int64_t tensorDataCountAlignedList[MAX_TENSOR_COUNT] = {0};
    int64_t* tensorDataCountList = nullptr;
    uint16_t* tensorStartList = nullptr;
    uint16_t* tensorEndList = nullptr;
    int64_t* tensorStartOffsetList = nullptr;
    int64_t* tensorEndOffsetList = nullptr;
    int64_t totalDataCountAligned = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    int32_t dataTypeSize = 0;
    int32_t elementsPerBlock = 0;
    int32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;
};

ge::graphStatus NonFiniteCheckTiling::Init() {
    InitTilingDataItems();
    totalTensorCount = int32_t(tilingContext->GetComputeNodeInputNum());
    OP_TILING_CHECK(
        totalTensorCount > MAX_TENSOR_COUNT || totalTensorCount <= 0,
        OP_LOGE(nodeName, "The number of input tensors [%d] not in (0, %hu].", totalTensorCount, MAX_TENSOR_COUNT),
        return ge::GRAPH_FAILED);
    // Get shape, dtype information, and the total number of data.
    for (int32_t i = 0; i < totalTensorCount; i++) {
        auto descPtr = tilingContext->GetDynamicInputDesc(0, i);
        OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, descPtr);
        auto tempDtype = descPtr->GetDataType();
        // Determine whether all data types are consistent.
        if (i == 0) {
            dataType = tempDtype;
            dataTypeSize = ge::GetSizeByDataType(dataType);
            OP_TILING_CHECK(dataTypeSize <= 0, OP_LOGE(nodeName, "dataTypeSize[%d] is invalid.", dataTypeSize),
                            return ge::GRAPH_FAILED);
            elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        } else if (tempDtype != dataType) {
            OP_LOGE(nodeName, "All tensor data types must be consistent.");
            return ge::GRAPH_FAILED;
        }
        auto shapePtr = tilingContext->GetDynamicInputShape(0, i);
        OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, shapePtr);
        tensorDataCountList[i] = shapePtr->GetStorageShape().GetShapeSize();
        OP_TILING_CHECK(tensorDataCountList[i] == 0, OP_LOGE(nodeName, "The input shape not support empty tensor."),
                        return ge::GRAPH_FAILED);
        // Make a 32-byte alignment for each Tensor
        tensorDataCountAlignedList[i] = CeilAlign(tensorDataCountList[i], int64_t(elementsPerBlock));
        totalDataCountAligned += tensorDataCountAlignedList[i];
    }

    return CheckParams();
}

ge::graphStatus NonFiniteCheckTiling::RunBigKernelTiling() {
    OP_LOGD(nodeName, "Start.");
    OP_TILING_CHECK(FillCompileInfo() != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "FillCompileInfo error."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "Platform info, totalCoreNum:%d, ubSizePlatForm:%lu.", compileInfo.totalCoreNum,
            compileInfo.ubSizePlatForm);
    OP_TILING_CHECK(compileInfo.totalCoreNum > MAX_CORE_COUNT,
                    OP_LOGE(nodeName, "The number of totalCoreNum exceeds the limit(%hu).", MAX_CORE_COUNT),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CalcNeedCoreNum() == false, OP_LOGE(nodeName, "Param needCoreNum is zero."),
                    return ge::GRAPH_FAILED);
    AssignDataToEachCore();
    OP_TILING_CHECK(DivideUbMemory() == false, OP_LOGE(nodeName, "DivideUbMemory failed."), return ge::GRAPH_FAILED);

    FillTilingData();

    tilingContext->SetTilingKey(GetTilingKeyVal());
    tilingContext->SetBlockDim(needCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(nodeName, "Success.");
    return ge::GRAPH_SUCCESS;
}

void NonFiniteCheckTiling::InitTilingDataItems() {
    tensorDataCountList = tilingData.get_tensorDataCountList();
    tensorStartList = tilingData.get_tensorStartList();
    tensorEndList = tilingData.get_tensorEndList();
    tensorStartOffsetList = tilingData.get_tensorStartOffsetList();
    tensorEndOffsetList = tilingData.get_tensorEndOffsetList();
}

ge::graphStatus NonFiniteCheckTiling::CheckParams() const {
    OP_LOGD(nodeName, "dataType:%d, totalTensorCount:%d, totalDataCountAligned:%ld.", static_cast<int32_t>(dataType),
            totalTensorCount, totalDataCountAligned);
    OP_TILING_CHECK(dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT,
                    OP_LOGE(nodeName, "The input dtype not in [float16, bfloat16, float]."), return ge::GRAPH_FAILED);

    auto flagDescPtr = tilingContext->GetOutputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(tilingContext, flagDescPtr);
    OP_TILING_CHECK(flagDescPtr->GetDataType() != ge::DT_FLOAT, OP_LOGE(nodeName, "The output dtype must be float."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonFiniteCheckTiling::FillCompileInfo() {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    compileInfo.totalCoreNum = int32_t(ascendcPlatform.GetCoreNumAiv());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

bool NonFiniteCheckTiling::CalcNeedCoreNum() {
    needCoreNum = uint32_t(totalDataCountAligned / elementsPerBlock);
    if (needCoreNum > uint32_t(compileInfo.totalCoreNum)) {
        needCoreNum = compileInfo.totalCoreNum;
    }
    if (needCoreNum == 0) {
        return false;
    } else {
        return true;
    }
}

void NonFiniteCheckTiling::AssignDataToEachCore() {
    int64_t blockCount = totalDataCountAligned / elementsPerBlock;
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum;  // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (uint16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount != 0 && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountAlignedList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i;  // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount != 0) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountAlignedList[totalTensorCount - 1] - 1;
    }
}

bool NonFiniteCheckTiling::DivideUbMemory() {
    // A 32-byte alignment is performed on the UB available space.
    uint32_t canUseUbSize = uint32_t(compileInfo.ubSizePlatForm / BYTE_BLOCK * BYTE_BLOCK);
    uint32_t dtypeSizeTemp = dataTypeSize;
    if (dataType == ge::DT_BF16) {
        dtypeSizeTemp = DTYPE_SIZE_FLOAT;
    }
    uint32_t predictSGUbSize =
        uint32_t((canUseUbSize - BYTE_BLOCK) * COEFFICIENT_1 * 1.0 / (BYTE_REPEAT + dtypeSizeTemp));
    uint32_t maxDataUbSize = predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    maxProcCount = maxDataUbSize / dtypeSizeTemp;
    tempValUbSize = GetReduceRetValSize(maxDataUbSize, dtypeSizeTemp);
    if ((NUM_TWO * maxDataUbSize + tempValUbSize) > compileInfo.ubSizePlatForm) {
        return false;
    } else {
        return true;
    }
}

uint32_t NonFiniteCheckTiling::GetReduceRetValSize(uint32_t srcDataSize, uint32_t dtypeSize) const {
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint8_t perBlockCount = BYTE_BLOCK / dtypeSize;
    uint32_t iter1OutputCount = uint32_t(std::ceil(NUM_TWO * 1.0 * srcDataSize / BYTE_REPEAT));
    uint32_t iter1AlignEnd = CeilAlign(iter1OutputCount, uint32_t(perBlockCount));
    return iter1AlignEnd * dtypeSize;
}

uint64_t NonFiniteCheckTiling::GetTilingKeyVal() const {
    switch (dataType) {
        case ge::DT_FLOAT:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT);
        case ge::DT_FLOAT16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_FLOAT16);
        case ge::DT_BF16:
            return static_cast<uint64_t>(NonFiniteCheckTilingKey::KEY_BF16);
        default:
            return 0;
    }
}

void NonFiniteCheckTiling::FillTilingData() {
    tilingData.set_maxProcCount(maxProcCount);
    tilingData.set_tempValUbSize(tempValUbSize);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus Tiling4NonFiniteCheck(gert::TilingContext* context) {
    NonFiniteCheckTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Init tiling object return failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingObject.RunBigKernelTiling() != ge::GRAPH_SUCCESS) {
        VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Run big kernel tiling return failed.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NonFiniteCheck)
    .Tiling(Tiling4NonFiniteCheck);
}  // namespace optiling

// proto
namespace ge{
namespace ops {
const int64_t OUTPUT_IDX = 0;

static ge::graphStatus InferShapeForNonFiniteCheck(gert::InferShapeContext* context) {
    auto out_shape = context->GetOutputShape(OUTPUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    out_shape->SetDimNum(1);
    out_shape->SetDim(0, 1);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForNonFiniteCheck(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForNonFiniteCheck");
  context->SetOutputDataType(OUTPUT_IDX, ge::DT_FLOAT);
  OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForNonFiniteCheck");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NonFiniteCheck)
    .InferShape(InferShapeForNonFiniteCheck)
    .InferDataType(InferDataTypeForNonFiniteCheck);
}  // namespace ops
} // namespace ge


