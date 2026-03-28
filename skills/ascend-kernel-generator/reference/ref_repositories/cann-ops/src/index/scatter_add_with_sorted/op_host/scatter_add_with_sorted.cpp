/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include "scatter_add_with_sorted_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

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

using namespace std;

namespace {
    const int DT_FLOAT32_TYPE = 1;
    const int DT_FLOAT16_TYPE = 2;
    const int DT_INT32_TYPE = 3;
    const int DT_UINT8_TYPE = 4;
    const int DT_INT8_TYPE = 5;
    const int DT_BF16_TYPE = 6;

    const int ADD_MODE = 10;

    const int SIZE_OF_FP16 = 2;
    const int SIZE_OF_FP32 = 4;
    const int SIZE_OF_INT32 = 4;
    const int SIZE_OF_UINT8 = 1;
    const int SIZE_OF_INT8 = 1;
    const int SIZE_OF_BF16 = 2;

    const int BUFFER_NUM = 2;
    const int MAX_SIZE = 64;

    const int INPUT_0 = 0;
    const int INPUT_1 = 1;
    const int INPUT_2 = 2;
    const int INPUT_3 = 3;
}

namespace optiling {
  class ScatterAddWithSortedTiling {
    public:
        explicit ScatterAddWithSortedTiling(gert::TilingContext* context) : tilingContext(context) {};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void TilingDataPrint() const;
    private:
        ScatterAddWithSortedTilingData tilingData;
        gert::TilingContext* tilingContext = nullptr;
        uint32_t tilingKey = 0;
        uint64_t usedCoreNum = 0;
        uint64_t extraTaskCore = 0;
        uint64_t eachCount = 1;
        uint64_t lastCount = 1;
        uint64_t inputCount = 1;
        uint64_t indicesCount = 1;
        uint64_t updatesCount = 1;
        uint64_t inputOneTime = 0;
        uint64_t updatesOneTime = 0;
        uint64_t updatesAlign = 0;
        uint64_t updatesLoop = 0;
        uint64_t updatesEach = 0;
        uint64_t updatesLast = 0;
        uint64_t eachNum = 0;
        uint64_t eachLoop = 0;
        uint64_t eachTail = 0;
        uint64_t lastNum = 0;
        uint64_t lastLoop = 0;
        uint64_t lastTail = 0;
        uint64_t maxSize = 0;
        uint64_t dataAlign = 32;
        uint64_t workspaceSize = 1024 * 1024 * 16;
        uint64_t max_ub = 20480;
  };

  ge::graphStatus ScatterAddWithSortedTiling::Init(){
    if (tilingContext == nullptr) {
        OP_LOGE("ScatterAddWithSorted", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t totalWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    static uint32_t coreNum = totalCoreNum;
    if (coreNum == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    workspaceSize = totalWorkspaceSize;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext->GetNodeName(), "ubSizePlatForm: %lu.", ubSizePlatForm);

    auto attrs = tilingContext->GetAttrs();
    if (attrs == nullptr || tilingContext->GetInputShape(INPUT_0) == nullptr
        || tilingContext->GetInputShape(INPUT_1) == nullptr || tilingContext->GetInputShape(INPUT_2) == nullptr
        || tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(), "tilingContext inputshape or outputshape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const char* reduce = attrs->GetAttrPointer<char>(0);
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto valueDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    uint32_t inputSize = 0;
    uint32_t isAdd = 0;
    if (strcmp(reduce, "add") == 0) {
        isAdd = 1;
    }

    if (ge::DT_FLOAT == inputDtype) {
        tilingKey = DT_FLOAT32_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_FP32 + SIZE_OF_FP32 + SIZE_OF_FP32;
        dataAlign = dataAlign / SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        tilingKey = DT_FLOAT16_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_FP16 + SIZE_OF_FP16 + SIZE_OF_FP16;
        dataAlign = dataAlign / SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        tilingKey = DT_INT32_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_UINT8 == inputDtype) {
        tilingKey = DT_UINT8_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        tilingKey = DT_INT8_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        tilingKey = DT_BF16_TYPE + isAdd * ADD_MODE;
        inputSize = SIZE_OF_BF16 + SIZE_OF_BF16 + SIZE_OF_FP16;
        dataAlign = dataAlign / SIZE_OF_BF16;
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "var only support float, float16, int32, uint8, int8, bf16.");
        return ge::GRAPH_FAILED;
    }

    if (inputDtype != valueDtype) {
        OP_LOGE(tilingContext->GetNodeName(), "value dtype must be same as var.");
        return ge::GRAPH_FAILED;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_2)->GetDataType()) {
        OP_LOGE(tilingContext->GetNodeName(), "sorted_index only support int32.");
        return ge::GRAPH_FAILED;
    }

    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    
    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum != updatesShape.GetDimNum()) {
        OP_LOGE(tilingContext->GetNodeName(), "the dimNum of input must equal the dimNum of updates.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (i == inputDimNum - 1) {
            inputOneTime = dimInput;
            updatesOneTime = dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = indicesShape.GetDim(0);
    if (inputOneTime == 0 || updatesOneTime == 0 || inputCount == 0 || updatesCount == 0 || indicesCount == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "shape cannot equal 0.");
        return ge::GRAPH_FAILED;
    }

    uint32_t inputDataAlign = dataAlign / inputSize;
    
    // atomic_add scene，self dtype in int32, int8, uint8
    if (tilingContext->GetInputDesc(INPUT_3) == nullptr) {
        if (indicesCount < coreNum) {
            usedCoreNum = indicesCount;
            eachNum = 1;
        } else {
            usedCoreNum = coreNum;
            eachNum = indicesCount / coreNum;
            extraTaskCore = indicesCount - eachNum * coreNum;
        }
        updatesEach = max_ub / inputSize;
        updatesEach = updatesEach > updatesOneTime ? updatesOneTime : updatesEach;
        updatesLoop = (updatesOneTime - 1) / updatesEach + 1;
        updatesLast = updatesOneTime - updatesEach * (updatesLoop - 1);
        updatesAlign = ((updatesEach - 1) / inputDataAlign + 1) * inputDataAlign;
        
        OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }

    if (ge::DT_INT32 != tilingContext->GetInputDesc(INPUT_3)->GetDataType()) {
        OP_LOGE(tilingContext->GetNodeName(), "pos only support int32.");
        return ge::GRAPH_FAILED;
    }

    inputSize += (SIZE_OF_FP32 + SIZE_OF_FP32) / BUFFER_NUM;
    updatesAlign = (updatesOneTime + dataAlign - 1) / dataAlign * dataAlign;
    if (updatesAlign * inputSize > max_ub) {
        maxSize = MAX_SIZE;
        max_ub = max_ub - maxSize * (SIZE_OF_INT32 + SIZE_OF_INT32);
        updatesEach = max_ub / inputSize;
        updatesEach = updatesEach > updatesOneTime ? updatesOneTime : updatesEach;
        updatesLoop = (updatesOneTime - 1) / updatesEach + 1;
        updatesLast = updatesOneTime - updatesEach * (updatesLoop - 1);
        updatesAlign = (updatesEach + dataAlign - 1) / dataAlign * dataAlign;;
    } else {
        maxSize = (max_ub - updatesAlign * inputSize) / (SIZE_OF_INT32 + SIZE_OF_INT32);
        updatesEach = updatesLast = updatesOneTime;
        updatesLoop = 1;
    }

    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachCount - 1) / eachCount;
    lastCount = indicesCount - eachCount * (usedCoreNum - 1);

    eachNum = eachCount;
    eachLoop = 1;
    eachTail = eachCount;
    if (eachCount > maxSize) {
        eachNum = maxSize;
        eachLoop = (eachCount + maxSize - 1) / maxSize;
        eachTail = eachCount - (eachLoop - 1) * eachNum;
    }

    lastNum = lastCount;
    lastLoop = 1;
    lastTail = lastCount;
    if (lastCount > maxSize) {
        lastNum = maxSize;
        lastLoop = (lastCount + maxSize - 1) / maxSize;
        lastTail = lastCount - (lastLoop - 1) * lastNum;
    }

    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ScatterAddWithSortedTiling::RunKernelTiling(){
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_extraTaskCore(extraTaskCore);
    tilingData.set_eachCount(eachCount);
    tilingData.set_lastCount(lastCount);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_inputOneTime(inputOneTime);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_updatesAlign(updatesAlign);
    tilingData.set_maxSize(maxSize);
    tilingData.set_eachNum(eachNum);
    tilingData.set_eachLoop(eachLoop);
    tilingData.set_eachTail(eachTail);
    tilingData.set_lastNum(lastNum);
    tilingData.set_lastLoop(lastLoop);
    tilingData.set_lastTail(lastTail);
    tilingData.set_updatesLoop(updatesLoop);
    tilingData.set_updatesEach(updatesEach);
    tilingData.set_updatesLast(updatesLast);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetTilingKey(tilingKey);
    tilingContext->SetBlockDim(usedCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;
    TilingDataPrint();
    OP_LOGD(tilingContext->GetNodeName(), "Tiling end.");
    return ge::GRAPH_SUCCESS;
  }

  void ScatterAddWithSortedTiling::TilingDataPrint() const {
    OP_LOGD(tilingContext->GetNodeName(), "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext->GetNodeName(), "extraTaskCore: %lu.", extraTaskCore);
    OP_LOGD(tilingContext->GetNodeName(), "eachCount: %lu.", eachCount);
    OP_LOGD(tilingContext->GetNodeName(), "lastCount: %lu.", lastCount);
    OP_LOGD(tilingContext->GetNodeName(), "inputCount: %lu.", inputCount);
    OP_LOGD(tilingContext->GetNodeName(), "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext->GetNodeName(), "updatesCount: %lu.", updatesCount);
    OP_LOGD(tilingContext->GetNodeName(), "inputOneTime: %lu.", inputOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "updatesOneTime: %lu.", updatesOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "updatesAlign: %lu.", updatesAlign);
    OP_LOGD(tilingContext->GetNodeName(), "maxSize: %lu.", maxSize);
    OP_LOGD(tilingContext->GetNodeName(), "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext->GetNodeName(), "eachLoop: %lu.", eachLoop);
    OP_LOGD(tilingContext->GetNodeName(), "eachTail: %lu.", eachTail);
    OP_LOGD(tilingContext->GetNodeName(), "lastNum: %lu.", lastNum);
    OP_LOGD(tilingContext->GetNodeName(), "lastLoop: %lu.", lastLoop);
    OP_LOGD(tilingContext->GetNodeName(), "lastTail: %lu.", lastTail);
    OP_LOGD(tilingContext->GetNodeName(), "updatesLoop: %lu.", updatesLoop);
    OP_LOGD(tilingContext->GetNodeName(), "updatesEach: %lu.", updatesEach);
    OP_LOGD(tilingContext->GetNodeName(), "updatesLast: %lu.", updatesLast);
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext->GetNodeName(), "max_ub: %lu.", max_ub);
  }

  ge::graphStatus TilingScatterAddWithSorted(gert::TilingContext* context) {
    ScatterAddWithSortedTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
  }

  IMPL_OP_OPTILING(ScatterAddWithSorted)
      .Tiling(TilingScatterAddWithSorted);
} // namespace optiling
