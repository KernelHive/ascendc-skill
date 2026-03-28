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
 * \file max_pool3d_grad_with_argmax_tiling_splitw.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace TilingKeys {
    //1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=float
    constexpr uint64_t TILING_KEY_SPLITW_FLOAT = 111100;
    //no overlap(1), splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_SPLITW_HALF_NO_OVERLAP = 111101;
    //overlap (2), splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_SPLITW_HALF_OVERLAP = 211101;
    //1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=bfloat16
    constexpr uint64_t TILING_KEY_SPLITW_BF16 = 111102;
}

using namespace AscendC;

namespace optiling {

bool MaxPool3DGradWithArgmaxSplitWTiling::IsCapable() {
    splitData.partD = 1;
    splitData.partOutD = inputData.dilation[D_DIM] * (inputData.kernelSize[D_DIM] - 1) + 1;
    splitData.partH = 1;
    splitData.partOutH = inputData.dilation[H_DIM] * (inputData.kernelSize[H_DIM] - 1) + 1;
    std::array<uint64_t, DHW_DIMS> parts{splitData.partD, splitData.partH, inputData.inputShape[W_DIM]};
    std::array<uint64_t, DHW_DIMS> partOuts{splitData.partOutD, splitData.partOutH, padOutputData.padOutputShape[W_DIM]};

    FindSplitParts(parts, partOuts, bufSizes, W_DIM);
    
    splitData.partW = parts[W_DIM];
    splitData.partOutW = partOuts[W_DIM];
    maxPoolGradParams.workspaceSize = maxPoolGradParams.totalCoreNum * blockLength * padOutputData.padOutputShape[D_DIM] * padOutputData.padOutputShape[H_DIM] * padOutputData.padOutputShape[W_DIM] * sizeof(float);
    bool defaultDilation = ((inputData.dilation[D_DIM] == 1) && (inputData.dilation[H_DIM] == 1) && (inputData.dilation[W_DIM] == 1));
    return ((splitData.partW != 0) && (splitData.partOutW != 0) && (!defaultDilation));
}

uint64_t MaxPool3DGradWithArgmaxSplitWTiling::GetTilingKey() const {
    if (dtype == ge::DataType::DT_FLOAT) {
        return TilingKeys::TILING_KEY_SPLITW_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        if (!inputData.isOverlap) {
            return TilingKeys::TILING_KEY_SPLITW_HALF_NO_OVERLAP;
        } else {
            return TilingKeys::TILING_KEY_SPLITW_HALF_OVERLAP;
        }
    } else {
        return TilingKeys::TILING_KEY_SPLITW_BF16;
    }
}

void MaxPool3DGradWithArgmaxSplitWTiling::DoUBTiling() {
    uint64_t batchesBlock = inputData.batches / blockLength; 
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / maxPoolGradParams.totalCoreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % maxPoolGradParams.totalCoreNum;
    splitData.batchesPerCore = batchesBlockPerCore * blockLength;
    splitData.leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
}

void MaxPool3DGradWithArgmaxSplitWTiling::SetTilingData() {
    tiling.set_inputShapes(&(inputData.inputShape[0]));
    tiling.set_outShapes(&(inputData.outShape[0]));
    tiling.set_kD(inputData.kernelSize[D_DIM]);
    tiling.set_kH(inputData.kernelSize[H_DIM]);
    tiling.set_kW(inputData.kernelSize[W_DIM]);
    tiling.set_sD(inputData.stride[D_DIM]);
    tiling.set_sH(inputData.stride[H_DIM]);
    tiling.set_sW(inputData.stride[W_DIM]);
    tiling.set_pD(inputData.pad[D_DIM]);
    tiling.set_pH(inputData.pad[H_DIM]);
    tiling.set_pW(inputData.pad[W_DIM]);
    tiling.set_dD(inputData.dilation[D_DIM]);
    tiling.set_dH(inputData.dilation[H_DIM]);
    tiling.set_dW(inputData.dilation[W_DIM]);
    tiling.set_batchesPerCore(splitData.batchesPerCore);
    tiling.set_leftOverBatches(splitData.leftOverBatches);
    tiling.set_partD(splitData.partD);
    tiling.set_partH(splitData.partH);
    tiling.set_partW(splitData.partW);
    tiling.set_partOutD(splitData.partOutD);
    tiling.set_partOutH(splitData.partOutH);
    tiling.set_partOutW(splitData.partOutW);
    tiling.set_ceilD(padOutputData.ceil[D_DIM]);
    tiling.set_ceilH(padOutputData.ceil[H_DIM]);
    tiling.set_ceilW(padOutputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DGradWithArgmaxSplitWTiling::DoOpTiling() {
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxSplitWTiling::PostTiling() {
    context_->SetBlockDim(maxPoolGradParams.totalCoreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxSplitWTiling, 5);

} // namespace optiling
