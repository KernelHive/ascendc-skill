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
 * \file ctc_loss_v3_grad.cpp
 * \brief
 */
#include "ctc_loss_v3_grad.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace std;
using namespace AscendC;

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

namespace optiling {
constexpr uint64_t RESERVE_SAPCE = 1.5 * 1024;
constexpr int64_t SELECT_SPACE = 8 * 1024;
constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
constexpr int64_t T_DIM = 0;
constexpr int64_t N_DIM = 1;
constexpr int64_t C_DIM = 2;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t ALPHA_T_DIM = 1;
constexpr int64_t ALPHA_DIM = 2;
constexpr int64_t S_DIM = 2;
constexpr int64_t S_COE = 2;
constexpr int64_t ALIGN = 8;
constexpr int64_t INPUT_GRAD_OUT_IDX = 0;
constexpr int64_t INPUT_LOG_PROB_IDX = 1;
constexpr int64_t INPUT_TRAGETS_IDX = 2;
constexpr int64_t INPUT_INPUT_LENGTHS_IDX = 3;
constexpr int64_t INPUT_TARGET_LENGTHS_IDX = 4;
constexpr int64_t INPUT_LOSS_IDX = 5;
constexpr int64_t INPUT_LOG_ALPHA_IDX = 6;
constexpr int64_t LOSS_DIM_NUM = 1;
constexpr int64_t GRAD_DIM_NUM = 3;
constexpr int64_t TARGETS_DIM_NUM = 2;
constexpr int64_t GRAD_RESERVED_UB = 40 * 1024;
constexpr int64_t MAX_SYMBOL_SET = 200000;
constexpr int64_t MAX_LABEL_LEN = 1000;
constexpr int64_t MAX_BATCH = 10000;
constexpr int64_t LARGE_BATCH = 1500;
constexpr int64_t NUM_PER_REPEAT = 64;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t FLOAT_DSIZE = 4;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t MASKNUM_PER_BYTES = 8;
constexpr int64_t WORKSPACE_ALPHA_NUM = 4;
constexpr int64_t TARGETS_GM_NUM = 2;
constexpr int64_t TARGETS_UB_NUM = 4;
constexpr int64_t TARGETS_LENGTH_UB_NUM = 5;
constexpr int64_t MASK_UB_NUM = 5;
constexpr int64_t TARGETS_LENGTH_MASK_UB_NUM = 2;
constexpr int64_t ALPHA_LENGTH_NUM = 8;
const std::string OP_NAME = "CTCLossV3Grad";

template <typename T1, typename T2>
static ge::graphStatus CeilAlign(T1 a, T2 b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

class CTCLossV3GradTiling {
public:
    explicit CTCLossV3GradTiling(gert::TilingContext* context) : context(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
    bool CheckShapeInfo();
    bool CheckShapeInfoForN();
    bool CheckShapeInfoForCT();
    void PrintTilingData();
    void TilingDataPrint();
private:
    CTCLossV3GradTilingData tilingData;
    gert::TilingContext* context = nullptr;
    int64_t sysWorkspaceSize = 16 * 1024 *1024;
    int64_t useWorkspaceSize = 0;
    int64_t coreUsed = 48;
    int64_t sliceLength = 1;
    int64_t sliceLengthTail = 1;
    int64_t probSliceNum = 1;
    int64_t alphaLength = 0;
    
    int64_t maxInputLength = 1;
    int64_t sDimRange = 1;
    int64_t maxTargetLength = 1;
    int64_t symbolSet = 1;

    int64_t batchSize = 192 * 1024;
    int64_t targetsDimNum = 1;
    int64_t targetsNum = 1;
    int64_t taskPerCore = 1;
    int64_t taskTailCore = 1;
    int64_t BLANK = 0;
    int64_t zeroInfinity = 0;
    int64_t targetDsize = 0;
    int64_t gradDsize = 0;
    int64_t gradOutN = 0;
    int64_t targetsN = 0;
    int64_t inputLengthsN = 0;
    int64_t targetLengthsN = 0;
    int64_t lossN = 0;
    int64_t logAlphaN = 0;
    int64_t gradN = 0;
    int64_t gradT = 0;
    int64_t gradC = 0;
    int64_t logAlphaT = 0;
    uint64_t ubSizePlatForm = 0;
};

bool CTCLossV3GradTiling::CheckShapeInfoForN()
{
    // Check the shape info for some inputs, focusing mainly on batchSize.
    auto nodeName = context->GetNodeName();
    auto const gradOutShape = context->GetInputShape(INPUT_GRAD_OUT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradOutShape);
    auto const gradOutShapeVal = gradOutShape->GetStorageShape();
    OP_TILING_CHECK(
        gradOutShapeVal.GetDimNum() != LOSS_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check grad_out shape failed, the dims of grad_out not equal 1."),
        return false);
    gradOutN = gradOutShapeVal.GetDim(BATCH_DIM);
    
    auto const inputLengthsShape = context->GetInputShape(INPUT_INPUT_LENGTHS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputLengthsShape);
    auto const inputLengthsShapeVal = inputLengthsShape->GetStorageShape();
    OP_TILING_CHECK(
        inputLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check input_lengths shape failed, the dims of input_lengths not equal 1."),
        return false);
    inputLengthsN = inputLengthsShapeVal.GetDim(BATCH_DIM);

    auto const targetLengthsShape = context->GetInputShape(INPUT_TARGET_LENGTHS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, targetLengthsShape);
    auto const targetLengthsShapeVal = targetLengthsShape->GetStorageShape();
    OP_TILING_CHECK(
        targetLengthsShapeVal.GetDimNum() != LOSS_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check target_lengths shape failed, the dims of target_lengths not equal 1."),
        return false);
    targetLengthsN = targetLengthsShapeVal.GetDim(BATCH_DIM);

    auto const lossShape = context->GetInputShape(INPUT_LOSS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, targetLengthsShape);
    auto const lossShapeVal = lossShape->GetStorageShape();
    OP_TILING_CHECK(
        lossShapeVal.GetDimNum() != LOSS_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check neg_log_likelihood shape failed, the dims of neg_log_likelihood not equal 1."),
        return false);
    lossN = lossShapeVal.GetDim(BATCH_DIM);
    return true;
}

bool CTCLossV3GradTiling::CheckShapeInfoForCT() {
    // Check the shape information for some inputs and outputs, focusing mainly on symbolset and time.
    auto nodeName = context->GetNodeName();
    auto const logProbsShape = context->GetInputShape(INPUT_LOG_PROB_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, logProbsShape);
    auto const logProbsShapeVal = logProbsShape->GetStorageShape();
    OP_TILING_CHECK(
        logProbsShapeVal.GetDimNum() != GRAD_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check log_probs shape failed, the dims of log_probs not equal 3."),
        return false);
    maxInputLength = logProbsShapeVal.GetDim(T_DIM);
    batchSize = logProbsShapeVal.GetDim(N_DIM);
    symbolSet = logProbsShapeVal.GetDim(C_DIM);

    auto const targetsShape = context->GetInputShape(INPUT_TRAGETS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, targetsShape);
    auto const targetsShapeVal = targetsShape->GetStorageShape();
    targetsDimNum = targetsShapeVal.GetDimNum();
    OP_TILING_CHECK(
        targetsDimNum != LOSS_DIM_NUM && targetsDimNum != TARGETS_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check targets shape failed, the dims of targets not equal 1 or 2."),
        return false);

    targetsN = targetsNum = targetsShapeVal.GetDim(BATCH_DIM);
    if (targetsDimNum > LOSS_DIM_NUM) {
        targetsNum = targetsNum * targetsShapeVal.GetDim(1);
        sDimRange = targetsShapeVal.GetDim(1);
    }

    auto const logAlphaShape = context->GetInputShape(INPUT_LOG_ALPHA_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, logAlphaShape);
    auto const logAlphaShapeVal = logAlphaShape->GetStorageShape();
    OP_TILING_CHECK(
        logAlphaShapeVal.GetDimNum() != GRAD_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check log_alpha shape failed, the dims of log_alpha not equal 3."),
        return false);
    logAlphaN = logAlphaShapeVal.GetDim(BATCH_DIM);
    logAlphaT = logAlphaShapeVal.GetDim(ALPHA_T_DIM);
    alphaLength = logAlphaShapeVal.GetDim(ALPHA_DIM);
    maxTargetLength = (alphaLength - 1) / S_COE;

    auto const gradShape = context->GetOutputShape(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    auto const gradShapeVal = gradShape->GetStorageShape();
    OP_TILING_CHECK(
        gradShapeVal.GetDimNum() != GRAD_DIM_NUM,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check grad shape failed, the dims of grad not equal 3."),
        return false);
    gradN = gradShapeVal.GetDim(N_DIM);
    gradT = gradShapeVal.GetDim(T_DIM);
    gradC = gradShapeVal.GetDim(C_DIM);
    return true;
}

bool CTCLossV3GradTiling::CheckShapeInfo()
{   
    // Check the shape information for all the inputs and outputs.
    auto nodeName = context->GetNodeName();
    if (!CheckShapeInfoForN() || !CheckShapeInfoForCT()) {
        return false;
    }

    bool NCheck = gradOutN == batchSize && batchSize == inputLengthsN && inputLengthsN == targetLengthsN &&
                  targetLengthsN == lossN && lossN == logAlphaN && logAlphaN == gradN;
    NCheck = targetsDimNum > 1 ? (NCheck && (batchSize == targetsN)) : NCheck;
    OP_TILING_CHECK(!NCheck,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check batchSize failed."),
                    return false);

    OP_TILING_CHECK(batchSize > MAX_BATCH,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "BatchSize is too large, AICPU recommended."),
                    return false);

    bool TCheck = maxInputLength == gradT && gradT == logAlphaT;
    OP_TILING_CHECK(!TCheck,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check max time failed."),
                    return false);

    bool CCheck = symbolSet == gradC;
    OP_TILING_CHECK(!CCheck,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "Check symbolSet failed."),
                    return false);

    OP_TILING_CHECK(
        symbolSet > MAX_SYMBOL_SET || maxTargetLength > MAX_LABEL_LEN,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "SymbolSet or max targetLength is too large, AICPU recommended."),
        return false);
    return true;
}

ge::graphStatus CTCLossV3GradTiling::Init()
{   
    auto nodeName = context->GetNodeName();
    OP_LOGD(context->GetNodeName(), "CTCLossV3Grad tiling starts running"); 
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    int64_t ubSize = static_cast<int64_t>(ubSizePlatForm) - RESERVE_SAPCE - SELECT_SPACE;
    if (!CheckShapeInfo()) {
        return ge::GRAPH_FAILED;
    }
    
    targetDsize = GetSizeByDataType(context->GetInputDesc(INPUT_TRAGETS_IDX)->GetDataType());
    gradDsize = GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    auto* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* blankPtr = attrs->GetAttrPointer<int64_t>(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, blankPtr);
    BLANK = *blankPtr;
    OP_TILING_CHECK(BLANK < 0 || BLANK >= symbolSet,
        VECTOR_INNER_ERR_REPORT_TILIING(nodeName, "BLANK is out of the range, please input the right value."),
        return ge::GRAPH_FAILED);
    const auto* zeroInfinityPtr = attrs->GetAttrPointer<int64_t>(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, zeroInfinityPtr);
    zeroInfinity = *zeroInfinityPtr;
    coreUsed = batchSize > coreNum ? coreNum : batchSize;
    taskPerCore = batchSize / coreNum;
    taskTailCore = batchSize % coreNum;

    int64_t alphaLengthAlign = CeilAlign(alphaLength, NUM_PER_REPEAT);
    int64_t maxTargetLengthAlign = CeilAlign(maxTargetLength, NUM_PER_REPEAT);
    int64_t targetLengthPlusAlign = CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT);
    // Calculate the ubsize needed for targetsLengths and alphaLength
    int64_t needSize = alphaLength * FLOAT_DSIZE + alphaLengthAlign * FLOAT_DSIZE * ALPHA_LENGTH_NUM +
        CeilAlign(alphaLengthAlign / MASKNUM_PER_BYTES, BLOCK_SIZE) * MASK_UB_NUM +
        targetLengthPlusAlign * INT64_DSIZE * TARGETS_UB_NUM +
        maxTargetLengthAlign * (FLOAT_DSIZE * TARGETS_LENGTH_UB_NUM + INT64_DSIZE) +
        CeilAlign(maxTargetLengthAlign / MASKNUM_PER_BYTES, BLOCK_SIZE) * TARGETS_LENGTH_MASK_UB_NUM;
    needSize = batchSize <= LARGE_BATCH ? (needSize + batchSize * (targetDsize + INT64_DSIZE)) : needSize;
    sliceLength = (ubSize - needSize) / INT64_DSIZE; // NeedSize is smaller than ubSize
    sliceLength = sliceLength > symbolSet ? symbolSet : sliceLength;
    sliceLength = CeilAlign(sliceLength, NUM_PER_REPEAT);
    probSliceNum = sliceLength == 0 ? 0 : (symbolSet - 1 + sliceLength) / sliceLength;
    sliceLengthTail = symbolSet - (probSliceNum - 1) * sliceLength;
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 
        batchSize > LARGE_BATCH ? (sysWorkspaceSize + alphaLength * FLOAT_DSIZE * WORKSPACE_ALPHA_NUM * batchSize +
        FLOAT_DSIZE * batchSize * TARGETS_GM_NUM) :
        (sysWorkspaceSize + alphaLength * FLOAT_DSIZE * WORKSPACE_ALPHA_NUM * batchSize);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CTCLossV3GradTiling::RunKernelTiling()
{   
    context->SetTilingKey(0); // only one tilingKey now
    context->SetBlockDim(coreUsed);
    tilingData.set_sliceLength(sliceLength);
    tilingData.set_sliceLengthTail(sliceLengthTail);
    tilingData.set_probSliceNum(probSliceNum);
    tilingData.set_alphaLength(alphaLength);
    tilingData.set_maxInputLength(maxInputLength);
    tilingData.set_symbolSet(symbolSet);
    tilingData.set_batchSize(batchSize);
    tilingData.set_sDimRange(sDimRange);
    tilingData.set_targetsDimNum(targetsDimNum);
    tilingData.set_targetsNum(targetsNum);
    tilingData.set_taskPerCore(taskPerCore);
    tilingData.set_taskTailCore(taskTailCore);
    tilingData.set_BLANK(BLANK);
    tilingData.set_zeroInfinity(zeroInfinity);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void CTCLossV3GradTiling::PrintTilingData() {
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Start printing");
    OP_LOGD(nodeName, "sliceLength is %ld.", sliceLength);
    OP_LOGD(nodeName, "sliceLengthTail is %ld.", sliceLengthTail);
    OP_LOGD(nodeName, "probSliceNum is %ld.", probSliceNum);
    OP_LOGD(nodeName, "alphaLength is %ld.", alphaLength);
    OP_LOGD(nodeName, "maxInputLength is %ld.", maxInputLength);
    OP_LOGD(nodeName, "symbolSet is %ld.", symbolSet);
    OP_LOGD(nodeName, "batchSize is %ld.", batchSize);
    OP_LOGD(nodeName, "targetsDimNum is %ld.", targetsDimNum);
    OP_LOGD(nodeName, "sDimRange is %ld.", sDimRange);
    OP_LOGD(nodeName, "targetsNum is %ld.", targetsNum);
    OP_LOGD(nodeName, "taskPerCore is %ld.", taskPerCore);
    OP_LOGD(nodeName, "taskTailCore is %ld.", taskTailCore);
    OP_LOGD(nodeName, "BLANK is %ld.", BLANK);
    OP_LOGD(nodeName, "zeroInfinity is %ld.", zeroInfinity);
    OP_LOGD(nodeName, "End printing");
    OP_LOGD(nodeName, "CTCLossV3Grad tiling end running");
}

static ge::graphStatus TilingFunc4CTCLossV3Grad(gert::TilingContext* context)
{
    CTCLossV3GradTiling tilingObject(context);
    if (tilingObject.Init()!=ge::GRAPH_SUCCESS){
        OP_LOGD(context->GetNodeName(),  "Init failed!");
        return ge::GRAPH_FAILED;
    }

    return tilingObject.RunKernelTiling();
}

IMPL_OP_OPTILING(CTCLossV3Grad)
    .Tiling(TilingFunc4CTCLossV3Grad);
}