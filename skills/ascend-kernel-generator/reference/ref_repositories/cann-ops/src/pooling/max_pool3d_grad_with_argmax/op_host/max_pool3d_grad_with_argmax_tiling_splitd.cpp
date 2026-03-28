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
 * \file max_pool3d_grad_with_argmax_tiling_splitd.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace TilingKeys {
    //1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=float
    constexpr uint64_t TILING_KEY_SPLITD_FLOAT = 110000;
    //no overlap (1), splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_SPLITD_HALF_NO_OVERLAP = 110001;
    //overlap (2), splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_SPLITD_HALF_OVERLAP = 210001;
    //1, splitD=1, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16
    constexpr uint64_t TILING_KEY_SPLITD_BF16 = 110002;
}

using namespace AscendC;

namespace optiling {

bool MaxPool3DGradWithArgmaxSplitDTiling::IsCapable() {
    std::array<uint64_t, DHW_DIMS> parts{inputData.inputShape[D_DIM], inputData.inputShape[H_DIM], inputData.inputShape[W_DIM]};
    std::array<uint64_t, DHW_DIMS> partOuts{padOutputData.padOutputShape[D_DIM], padOutputData.padOutputShape[H_DIM], padOutputData.padOutputShape[W_DIM]};

    FindSplitParts(parts, partOuts, bufSizes, D_DIM);
    
    splitData.partD = parts[D_DIM];
    splitData.partOutD = partOuts[D_DIM];
    maxPoolGradParams.workspaceSize = maxPoolGradParams.totalCoreNum * blockLength * padOutputData.padOutputShape[D_DIM] * padOutputData.padOutputShape[H_DIM] * padOutputData.padOutputShape[W_DIM] * sizeof(float);
    bool defaultDilation = ((inputData.dilation[D_DIM] == 1) && (inputData.dilation[H_DIM] == 1) && (inputData.dilation[W_DIM] == 1));
    return ((splitData.partD != 0) && (splitData.partOutD != 0) && (!defaultDilation));
}

uint64_t MaxPool3DGradWithArgmaxSplitDTiling::GetTilingKey() const {
    if (dtype == ge::DataType::DT_FLOAT) {
        return TilingKeys::TILING_KEY_SPLITD_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        if (!inputData.isOverlap) {
            return TilingKeys::TILING_KEY_SPLITD_HALF_NO_OVERLAP;
        } else {
            return TilingKeys::TILING_KEY_SPLITD_HALF_OVERLAP;
        }
    } else {
        return TilingKeys::TILING_KEY_SPLITD_BF16;
    }
}

void MaxPool3DGradWithArgmaxSplitDTiling::DoUBTiling() {
    uint64_t batchesBlock = inputData.batches / blockLength; 
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / maxPoolGradParams.totalCoreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % maxPoolGradParams.totalCoreNum;
    splitData.batchesPerCore = batchesBlockPerCore * blockLength;
    splitData.leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
}

void MaxPool3DGradWithArgmaxSplitDTiling::SetTilingData() {
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
    tiling.set_partH(inputData.inputShape[H_DIM]);
    tiling.set_partW(inputData.inputShape[W_DIM]);
    tiling.set_partOutD(splitData.partOutD);
    tiling.set_partOutH(padOutputData.padOutputShape[H_DIM]);
    tiling.set_partOutW(padOutputData.padOutputShape[W_DIM]);	
    tiling.set_ceilD(padOutputData.ceil[D_DIM]);
    tiling.set_ceilH(padOutputData.ceil[H_DIM]);
    tiling.set_ceilW(padOutputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DGradWithArgmaxSplitDTiling::DoOpTiling() {
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxSplitDTiling::PostTiling() {
    context_->SetBlockDim(maxPoolGradParams.totalCoreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxSplitDTiling, 3);

} // namespace optiling
