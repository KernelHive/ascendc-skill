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
 * \file apply_fused_ema_adam_tiling.cpp
 * \brief
 */
#include "apply_fused_ema_adam_tiling.h"

using namespace std;
namespace optiling {
constexpr uint32_t INPUT_GRAD_IDX = 0;
constexpr uint32_t INPUT_VAR_IDX = 1;
constexpr uint32_t INPUT_M_IDX = 2;
constexpr uint32_t INPUT_V_IDX = 3;
constexpr uint32_t INPUT_S_IDX = 4;
constexpr uint32_t INPUT_STEP_IDX = 5;
constexpr uint32_t ATTR_LR_IDX = 0;
constexpr uint32_t ATTR_EMA_DECAY_IDX = 1;
constexpr uint32_t ATTR_BETA1_IDX = 2;
constexpr uint32_t ATTR_BETA2_IDX = 3;
constexpr uint32_t ATTR_EPS_IDX = 4;
constexpr uint32_t ATTR_MODE_IDX = 5;
constexpr uint32_t ATTR_BIAS_CORRECTION_IDX = 6;
constexpr uint32_t ATTR_WEIGHT_DECAY_IDX = 7;
constexpr uint32_t ONE_BLK_NUM = 16;
constexpr uint32_t ONE_BLK_NUM_FP32 = 8;
constexpr uint32_t BYTE_ONE_BLK = 32;
constexpr uint32_t TBUFFER_NUM = 2;
constexpr uint32_t QUEUE_NUM = 9;
constexpr uint32_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint32_t FP32_DTYPE_SIZE = 4;
ApplyFusedEmaAdamTilingData fusedEmaAdamTiling;

void GetTilingAttr(gert::TilingContext* context) {
    auto* attrs = context->GetAttrs();
    const float* attrLr = attrs->GetAttrPointer<float>(ATTR_LR_IDX);
    fusedEmaAdamTiling.set_lr(static_cast<float>(*attrLr));
    const float* attrEmaDecay = attrs->GetAttrPointer<float>(ATTR_EMA_DECAY_IDX);
    fusedEmaAdamTiling.set_emaDecay(static_cast<float>(*attrEmaDecay));
    const float* attrBeta1 = attrs->GetAttrPointer<float>(ATTR_BETA1_IDX);
    fusedEmaAdamTiling.set_beta1(static_cast<float>(*attrBeta1));
    const float* attrBeta2 = attrs->GetAttrPointer<float>(ATTR_BETA2_IDX);
    fusedEmaAdamTiling.set_beta2(static_cast<float>(*attrBeta2));
    const float* attrEps = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    fusedEmaAdamTiling.set_eps(static_cast<float>(*attrEps));

    const uint64_t* attrMode = attrs->GetAttrPointer<uint64_t>(ATTR_MODE_IDX);
    fusedEmaAdamTiling.set_mode(static_cast<uint64_t>(*attrMode));

    const bool* attrBiasCorrection = attrs->GetAttrPointer<bool>(ATTR_BIAS_CORRECTION_IDX);
    auto biasCorrection = *attrBiasCorrection;
    fusedEmaAdamTiling.set_biasCorrection(static_cast<uint64_t>(biasCorrection ? 1 : 0));

    const float* attrWeightDecay = attrs->GetAttrPointer<float>(ATTR_WEIGHT_DECAY_IDX);
    fusedEmaAdamTiling.set_weightDecay(static_cast<float>(*attrWeightDecay));
}

void DtypeTilingKey(gert::TilingContext* context) {
    auto nodeName = context->GetNodeName();

    auto dtypeInput = context->GetInputDesc(INPUT_GRAD_IDX);
    auto gradDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_VAR_IDX);
    auto varDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_M_IDX);
    auto mDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_V_IDX);
    auto vDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_S_IDX);
    auto sDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_STEP_IDX);
    auto stepDtype = dtypeInput->GetDataType();

    uint32_t tilingKey = 100;
    if (gradDtype == ge::DT_FLOAT) {
        tilingKey += 2;
    } else if (gradDtype == ge::DT_FLOAT16) {
        tilingKey += 1;
    }
    context->SetTilingKey(tilingKey);
}

void TilingCompute(gert::TilingContext* context) {
    auto nodeName = context->GetNodeName();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t totalDataNum = context->GetInputShape(INPUT_V_IDX)->GetStorageShape().GetShapeSize();
    uint64_t dtypeSize = context->GetInputDesc(INPUT_V_IDX)->GetDataType() == ge::DT_FLOAT ?
                         FP32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
    uint64_t frontCoreNum = totalDataNum % coreNum != 0 ? totalDataNum % coreNum : coreNum;
    uint64_t tailCoreNum = totalDataNum <= coreNum ? 0 : coreNum - frontCoreNum;
    uint64_t blockDim = frontCoreNum + tailCoreNum;
    uint64_t coreCalcNum = (totalDataNum + coreNum -1) / coreNum;

    fusedEmaAdamTiling.set_frontCoreNum(frontCoreNum);
    fusedEmaAdamTiling.set_tailCoreNum(tailCoreNum);
    fusedEmaAdamTiling.set_coreCalcNum(coreCalcNum);
    context->SetBlockDim(blockDim);

    uint64_t tBuffersize = TBUFFER_NUM * BYTE_ONE_BLK;
    uint64_t bufferSize = ubSize - tBuffersize;
    uint64_t coreOnesize = (dtypeSize == FP32_DTYPE_SIZE) ? dtypeSize * QUEUE_NUM :
                           (dtypeSize + FP32_DTYPE_SIZE) * QUEUE_NUM;
    if (fusedEmaAdamTiling.get_mode() == 1) {
        coreOnesize -= (dtypeSize == FP32_DTYPE_SIZE) ? dtypeSize : (dtypeSize + FP32_DTYPE_SIZE);
    }
    uint64_t alignSize = dtypeSize == FP32_DTYPE_SIZE ? ONE_BLK_NUM_FP32 : ONE_BLK_NUM;
    uint64_t coreCalcMax = bufferSize / coreOnesize / alignSize * alignSize;
    uint64_t loopNum = coreCalcMax < coreCalcNum ? (coreCalcNum + coreCalcMax - 1) / coreCalcMax : 1;
    uint64_t frontCalcExtra = loopNum == 1 ? coreCalcNum : coreCalcNum - coreCalcMax * (loopNum - 1);
    uint64_t tailCalcExtra = frontCalcExtra -1;

    fusedEmaAdamTiling.set_coreCalcMax(coreCalcMax);
    fusedEmaAdamTiling.set_loopNum(loopNum);
    fusedEmaAdamTiling.set_frontCalcExtra(frontCalcExtra);
    fusedEmaAdamTiling.set_tailCalcExtra(tailCalcExtra);
}

ge::graphStatus Tiling4ApplyFusedEmaAdam(gert::TilingContext* context) {
    auto nodeName = context->GetNodeName();
    GetTilingAttr(context);
    DtypeTilingKey(context);
    TilingCompute(context);

    fusedEmaAdamTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(fusedEmaAdamTiling.GetDataSize());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ApplyFusedEmaAdam(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

struct ApplyFusedEmaAdamCompileInfo {};

IMPL_OP_OPTILING(ApplyFusedEmaAdam)
    .Tiling(Tiling4ApplyFusedEmaAdam)
    .TilingParse<ApplyFusedEmaAdamCompileInfo>(TilingPrepare4ApplyFusedEmaAdam);
} // namespace optiling