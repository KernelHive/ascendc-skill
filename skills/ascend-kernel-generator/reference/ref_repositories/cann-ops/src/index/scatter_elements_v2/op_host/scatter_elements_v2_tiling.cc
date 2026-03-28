/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file scatter_elements_v2_tiling.cc
 * \brief
 */

#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "scatter_elements_v2_tiling.h"

using namespace std;

// tools api
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops
namespace optiling {
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling


namespace {
    const int INPUT_TYPE = 100;
    const int INDIC_TYPE = 10;
    const int OPERATOR_TYPE = 1;

    const int SIZE_OF_FP16 = 2;
    const int SIZE_OF_FP32 = 4;
    const int SIZE_OF_INT32 = 4;
    const int SIZE_OF_INT64 = 8;
    const int SIZE_OF_UINT8 = 1;
    const int SIZE_OF_INT8 = 1;
    const int SIZE_OF_BF16 = 2;

    const int DT_FLOAT32_TYPE = 1;
    const int DT_FLOAT16_TYPE = 2;
    const int DT_INT32_TYPE = 3;
    const int DT_UINT8_TYPE = 4;
    const int DT_INT8_TYPE = 5;
    const int DT_BF16_TYPE = 6;
    const int DT_INT32_INDEX_TYPE = 1;
    const int DT_INT64_INDEX_TYPE = 2;

    const int NONE = 1;
    const int ADD = 2;
    const int MUL = 3;
    const int BUFFER_NUM = 1;
    const int HALF_UB = 2;

    const int INPUT_0 = 0;
    const int INPUT_1 = 1;
    const int INPUT_2 = 2;
    const int WORK_SPACE_SIZE = 1024 * 1024 * 16;

    const int SMALL_MODE = 1;
}

namespace optiling {
  class ScatterElementsV2Tiling {
    public:
        explicit ScatterElementsV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
        ge::graphStatus Init();
        ge::graphStatus RunKernelTiling();
        void TilingDataPrint() const;
    private:
        ScatterElementsV2TilingData tilingData;
        gert::TilingContext* tilingContext = nullptr;
        uint32_t tilingKey = 0;
        uint64_t usedCoreNum = 0;
        uint64_t eachNum = 1;
        uint64_t extraTaskCore = 0;
        uint64_t inputCount = 1;
        uint64_t indicesCount = 1;
        uint64_t updatesCount = 1;
        uint64_t inputOneTime = 0;
        uint64_t indicesOneTime = 0;
        uint64_t updatesOneTime = 0;
        uint64_t inputLoop = 0;
        uint64_t indicesLoop = 0;
        uint64_t inputEach = 0;
        uint64_t indicesEach = 0;
        uint64_t inputLast = 0;
        uint64_t indicesLast = 0;
        uint64_t eachPiece = 1;
        uint64_t inputAlign = 8;
        uint64_t indicesAlign = 8;
        uint64_t updatesAlign = 8;
        uint64_t inputOnePiece = 0;
        uint64_t modeFlag = 0;
        uint64_t lastIndicesLoop = 1;
        uint64_t lastIndicesEach = 1;
        uint64_t lastIndicesLast = 1;
        uint64_t oneTime = 1;
        uint64_t lastOneTime = 1;
        uint64_t workspaceSize = 1024 * 1024 * 16;
        uint64_t max_ub = 20480;
  };

  ge::graphStatus ScatterElementsV2Tiling::Init(){
    if (tilingContext == nullptr) {
        OP_LOGE("ScatterElementsV2", "tilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "coreNum must greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm = 0;
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
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    uint32_t inputSize = 0;
    if (ge::DT_FLOAT == inputDtype) {
        tilingKey += INPUT_TYPE * DT_FLOAT32_TYPE;
        inputSize = SIZE_OF_FP32;
    } else if (ge::DT_FLOAT16 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_FLOAT16_TYPE;
        inputSize = SIZE_OF_FP16;
    } else if (ge::DT_INT32 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT32_TYPE;
        inputSize = SIZE_OF_INT32;
    } else if (ge::DT_UINT8 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_UINT8_TYPE;
        inputSize = SIZE_OF_UINT8;
    } else if (ge::DT_INT8 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_INT8_TYPE;
        inputSize = SIZE_OF_INT8;
    } else if (ge::DT_BF16 == inputDtype) {
        tilingKey += INPUT_TYPE * DT_BF16_TYPE;
        inputSize = SIZE_OF_BF16;
    }
    else {
        OP_LOGE(tilingContext->GetNodeName(), "var only support float, float16, int32, uint8, int8, bf16.");
        return ge::GRAPH_FAILED;
    }
    uint32_t indicesSize = 0;
    auto indicesDtype = tilingContext->GetInputDesc(1)->GetDataType();
    if (ge::DT_INT32 == indicesDtype) {
        tilingKey += INDIC_TYPE * DT_INT32_INDEX_TYPE;
        indicesSize = SIZE_OF_INT32;
    } else if (ge::DT_INT64 == indicesDtype) {
        tilingKey += INDIC_TYPE * DT_INT64_INDEX_TYPE;
        indicesSize = SIZE_OF_INT64;
    }
    else {
        OP_LOGE(tilingContext->GetNodeName(), "indices only support int64, int32.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dataAlign = 32;
    uint32_t inputDataAlign = dataAlign / inputSize;
    uint32_t indexDataAlign = dataAlign / indicesSize;

    int dim = *(attrs->GetAttrPointer<int>(0));
    const char* reduce = attrs->GetAttrPointer<char>(1);
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    if (strcmp(reduce, "none") == 0) {
        tilingKey += OPERATOR_TYPE * NONE;
    } else if (strcmp(reduce, "add") == 0) {
        tilingKey += OPERATOR_TYPE * ADD;
    } else if (strcmp(reduce, "mul") == 0) {
        tilingKey += OPERATOR_TYPE * MUL;
        OP_LOGE(tilingContext->GetNodeName(), "scatter_elements_v2 not support mul.");
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "scatter_elements_v2 only support none, add.");
        return ge::GRAPH_FAILED;
    }

    if (inputDimNum != indicesShape.GetDimNum() || inputDimNum != tilingContext->GetInputShape(INPUT_2)->GetStorageShape().GetDimNum()) {
        OP_LOGE(tilingContext->GetNodeName(), "the dimNum of input must equal the dimNum of indices.");
        return ge::GRAPH_FAILED;
    }

    uint32_t realDim = 0;
    if (dim < 0) {
        realDim = dim + inputDimNum;
    } else {
        realDim = dim;
    }

    if (realDim != inputDimNum - 1) {
        OP_LOGE(tilingContext->GetNodeName(), "scatter_elements_v2 not support dim != -1.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < inputDimNum; ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimIndices = indicesShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (dimUpdates < dimIndices) {
            OP_LOGE(tilingContext->GetNodeName(), "the dim of updates must greater than or equal the dim of indices.");
            return ge::GRAPH_FAILED;
        }
        if (realDim == i) {
            inputOneTime = dimInput;
            indicesOneTime = dimIndices;
            updatesOneTime = dimUpdates;
        }
        inputCount *= dimInput;
        indicesCount *= dimIndices;
        updatesCount *= dimUpdates;
    }

    if (inputOneTime == 0 || indicesOneTime == 0 || updatesOneTime == 0
        || inputCount == 0 || indicesCount == 0 || updatesCount == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "shape cannot equal 0.");
        return ge::GRAPH_FAILED;
    }

    uint32_t isDeterministicKey = tilingContext->GetDeterministic() == 1 ? 1 : 0;

    if (ge::DT_INT64 == indicesDtype) {
        indicesSize += SIZE_OF_INT32;
    }
    if ((strcmp(reduce, "add") == 0) && (ge::DT_FLOAT16 == inputDtype || ge::DT_BF16 == inputDtype)) {
        inputSize += sizeof(float) / BUFFER_NUM;
        indicesSize += sizeof(float) / BUFFER_NUM;
    }

    uint32_t times = indicesCount / indicesOneTime;
    uint32_t totalSize = inputSize * inputOneTime + indicesSize * (updatesOneTime + indicesOneTime);
    // 小包场景，一次性可以搬入一轮尾轴，按ub上限尽可能的搬多轮，特殊处理走small分支
    if (totalSize <= max_ub) {
        modeFlag = SMALL_MODE;
        max_ub = max_ub / totalSize;
        if (times < coreNum) {
            usedCoreNum = times;
            indicesEach = 1;
            indicesLoop = 1;
            indicesLast = 1;
        } else {
            oneTime = (times + coreNum - 1) / coreNum;
            usedCoreNum = (times + oneTime - 1) / oneTime;
            indicesLoop = (oneTime + max_ub - 1) / max_ub;
            indicesEach = indicesLoop == 1 ? oneTime : max_ub;
            indicesLast = oneTime - (indicesLoop - 1) * indicesEach;

            lastOneTime = times - oneTime * (usedCoreNum - 1);
            lastIndicesLoop = (lastOneTime + max_ub - 1) / max_ub;
            lastIndicesEach = lastIndicesLoop == 1 ? lastOneTime : max_ub;
            lastIndicesLast = lastOneTime - (lastIndicesLoop - 1) * lastIndicesEach;
        }
        indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;

        OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
        return ge::GRAPH_SUCCESS;
    }
    
    inputOnePiece = inputOneTime;
    if (times < coreNum) { // 一个任务可以分给多个核
        uint32_t need = (inputOneTime + dataAlign - 1) / dataAlign; // 每个核至少处理32个数，每个任务需要多少个核
        eachPiece = coreNum / times; // 每个任务可用的核数
        eachPiece = eachPiece > need ? need : eachPiece;
        if (isDeterministicKey == 1) {
            eachPiece = 1;
        }
        eachNum = eachPiece == 1 ? 1 : 0;
        extraTaskCore = 0;
        inputOnePiece = (inputOneTime + eachPiece - 1) / eachPiece; // 每个核需要处理的数量
        usedCoreNum = (inputOneTime + inputOnePiece - 1) / inputOnePiece * times; // 再次根据每个核需要处理的数量重新计算需要的核数
    } else { // 一个任务单独由一个核处理
        usedCoreNum = coreNum;
        eachNum = times / coreNum;
        extraTaskCore = times - eachNum * coreNum;
    }

    uint32_t indicesSum = indicesOneTime * (inputSize + indicesSize);
    uint32_t inputSum = inputOnePiece * inputSize;
    if (inputSum + indicesSum > static_cast<uint32_t>(max_ub)) {
        if (indicesSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            inputEach = (max_ub - indicesSum) / inputSize;
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesLoop = 1;
            indicesEach = indicesLast = indicesOneTime;
        } else if (inputSum < static_cast<uint32_t>(max_ub / HALF_UB)) {
            indicesEach = (max_ub - inputSum) / (inputSize + indicesSize);
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);

            inputLoop = 1;
            inputEach = inputLast = inputOnePiece;
        } else {
            inputEach = max_ub / (HALF_UB * inputSize);
            inputEach = inputEach > inputOnePiece ? inputOnePiece : inputEach;
            inputLoop = (inputOnePiece - 1) / inputEach + 1;
            inputLast = inputOnePiece - inputEach * (inputLoop - 1);

            indicesEach = max_ub / (HALF_UB * (inputSize + indicesSize));
            indicesEach = indicesEach > indicesOneTime ? indicesOneTime : indicesEach;
            indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
            indicesLast = indicesOneTime - indicesEach * (indicesLoop - 1);
        }
    }
    else {
        inputLoop = indicesLoop = 1;
        inputEach = inputLast = inputOnePiece;
        indicesEach = indicesLast = indicesOneTime;
    }

    inputAlign = ((inputEach - 1) / inputDataAlign + 1) * inputDataAlign;
    indicesAlign = ((indicesEach - 1) / indexDataAlign + 1) * indexDataAlign;
    updatesAlign = ((indicesEach - 1) / inputDataAlign + 1) * inputDataAlign;

    OP_LOGD(tilingContext->GetNodeName(), "Tiling inited.");
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ScatterElementsV2Tiling::RunKernelTiling(){
    OP_LOGD(tilingContext->GetNodeName(), "Tiling start.");

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_eachNum(eachNum);
    tilingData.set_extraTaskCore(extraTaskCore);
    tilingData.set_eachPiece(eachPiece);
    tilingData.set_inputAlign(inputAlign);
    tilingData.set_indicesAlign(indicesAlign);
    tilingData.set_updatesAlign(updatesAlign);
    tilingData.set_inputCount(inputCount);
    tilingData.set_indicesCount(indicesCount);
    tilingData.set_updatesCount(updatesCount);
    tilingData.set_inputOneTime(inputOneTime);
    tilingData.set_indicesOneTime(indicesOneTime);
    tilingData.set_updatesOneTime(updatesOneTime);
    tilingData.set_inputEach(inputEach);
    tilingData.set_indicesEach(indicesEach);
    tilingData.set_inputLast(inputLast);
    tilingData.set_indicesLast(indicesLast);
    tilingData.set_inputLoop(inputLoop);
    tilingData.set_indicesLoop(indicesLoop);
    tilingData.set_inputOnePiece(inputOnePiece);
    tilingData.set_modeFlag(modeFlag);
    tilingData.set_lastIndicesLoop(lastIndicesLoop);
    tilingData.set_lastIndicesEach(lastIndicesEach);
    tilingData.set_lastIndicesLast(lastIndicesLast);
    tilingData.set_oneTime(oneTime);
    tilingData.set_lastOneTime(lastOneTime);
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

  void ScatterElementsV2Tiling::TilingDataPrint() const {
    OP_LOGD(tilingContext->GetNodeName(), "usedCoreNum: %lu.", usedCoreNum);
    OP_LOGD(tilingContext->GetNodeName(), "eachNum: %lu.", eachNum);
    OP_LOGD(tilingContext->GetNodeName(), "extraTaskCore: %lu.", extraTaskCore);
    OP_LOGD(tilingContext->GetNodeName(), "eachPiece: %lu.", eachPiece);
    OP_LOGD(tilingContext->GetNodeName(), "inputAlign: %lu.", inputAlign);
    OP_LOGD(tilingContext->GetNodeName(), "indicesAlign: %lu.", indicesAlign);
    OP_LOGD(tilingContext->GetNodeName(), "updatesAlign: %lu.", updatesAlign);
    OP_LOGD(tilingContext->GetNodeName(), "inputCount: %lu.", inputCount);
    OP_LOGD(tilingContext->GetNodeName(), "indicesCount: %lu.", indicesCount);
    OP_LOGD(tilingContext->GetNodeName(), "updatesCount: %lu.", updatesCount);
    OP_LOGD(tilingContext->GetNodeName(), "inputOneTime: %lu.", inputOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "indicesOneTime: %lu.", indicesOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "updatesOneTime: %lu.", updatesOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "inputEach: %lu.", inputEach);
    OP_LOGD(tilingContext->GetNodeName(), "indicesEach: %lu.", indicesEach);
    OP_LOGD(tilingContext->GetNodeName(), "inputLast: %lu.", inputLast);
    OP_LOGD(tilingContext->GetNodeName(), "indicesLast: %lu.", indicesLast);
    OP_LOGD(tilingContext->GetNodeName(), "inputLoop: %lu.", inputLoop);
    OP_LOGD(tilingContext->GetNodeName(), "indicesLoop: %lu.", indicesLoop);
    OP_LOGD(tilingContext->GetNodeName(), "inputOnePiece: %lu.", inputOnePiece);
    OP_LOGD(tilingContext->GetNodeName(), "modeFlag: %lu.", modeFlag);
    OP_LOGD(tilingContext->GetNodeName(), "lastIndicesLoop: %lu.", lastIndicesLoop);
    OP_LOGD(tilingContext->GetNodeName(), "lastIndicesEach: %lu.", lastIndicesEach);
    OP_LOGD(tilingContext->GetNodeName(), "lastIndicesLast: %lu.", lastIndicesLast);
    OP_LOGD(tilingContext->GetNodeName(), "oneTime: %lu.", oneTime);
    OP_LOGD(tilingContext->GetNodeName(), "lastOneTime: %lu.", lastOneTime);
    OP_LOGD(tilingContext->GetNodeName(), "tilingKey: %u.", tilingKey);
    OP_LOGD(tilingContext->GetNodeName(), "max_ub: %lu.", max_ub);
  }

  ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context) {
    ScatterElementsV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
  }

  IMPL_OP_OPTILING(ScatterElementsV2)
      .Tiling(TilingScatterElementsV2);
}
