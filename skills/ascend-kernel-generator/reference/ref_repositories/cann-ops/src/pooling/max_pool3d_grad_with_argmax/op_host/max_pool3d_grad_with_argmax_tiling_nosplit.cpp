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
 * \file max_pool3d_grad_with_argmax_tiling_nosplit.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3d_grad_with_argmax_tiling.h"

namespace TilingKeys {
    //1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=float
    constexpr uint64_t TILING_KEY_NOSPLIT_FLOAT = 100000;
    //no overlap (1), splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_NOSPLIT_HALF_NO_OVERLAP = 100001;
    //overlap (2), splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half
    constexpr uint64_t TILING_KEY_NOSPLIT_HALF_OVERLAP = 200001;
    //1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16
    constexpr uint64_t TILING_KEY_NOSPLIT_BF16 = 100002;
}

using namespace AscendC;

namespace optiling {
constexpr uint64_t DHW_TOP = 256;
constexpr uint64_t BATCHES = 1000;
constexpr uint64_t BLOCK_KERNEL = 8;

bool MaxPool3DGradWithArgmaxNoSplitTiling::IsCapable() {
    auto summaryMemory = CalcBufferSizes(inputData.inputShape, padOutputData.padOutputShape,
                                         padOutputData.padOutputShape[W_DIM] * padOutputData.padOutputShape[H_DIM], bufSizes, 0);
    uint64_t dhwShape = inputData.outShape[D_DIM] * inputData.outShape[H_DIM] * inputData.outShape[W_DIM];
    uint64_t dhwKernel = inputData.kernelSize[D_DIM] * inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    bool defaultParams = ((inputData.ceilMode == false) && (inputData.kernelSize[D_DIM] == inputData.stride[D_DIM]) && (inputData.kernelSize[H_DIM] == inputData.stride[H_DIM]) && 
                          (inputData.kernelSize[W_DIM] == inputData.stride[W_DIM]) && (inputData.pad[D_DIM] == 0) && (inputData.pad[H_DIM] == 0) && (inputData.pad[W_DIM] == 0));
    
    return ((summaryMemory < ubSizeNew) && (((dhwShape <= DHW_TOP) && (inputData.batches <= BATCHES)) || (dhwShape >= DHW_TOP)) && (dhwKernel <= BLOCK_KERNEL) && defaultParams);
}

uint64_t MaxPool3DGradWithArgmaxNoSplitTiling::GetTilingKey() const {
    if (dtype == ge::DataType::DT_FLOAT) {
        return TilingKeys::TILING_KEY_NOSPLIT_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        if (!inputData.isOverlap) {
            return TilingKeys::TILING_KEY_NOSPLIT_HALF_NO_OVERLAP;
        } else {
            return TilingKeys::TILING_KEY_NOSPLIT_HALF_OVERLAP;
        }
    } else {
        return TilingKeys::TILING_KEY_NOSPLIT_BF16;
    }
}

void MaxPool3DGradWithArgmaxNoSplitTiling::DoUBTiling() {
    uint64_t batchesBlock = inputData.batches / blockLength; 
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / maxPoolGradParams.totalCoreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % maxPoolGradParams.totalCoreNum;
    batchesPerCore = batchesBlockPerCore * blockLength;
    leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
}

void MaxPool3DGradWithArgmaxNoSplitTiling::SetTilingData() {
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
    tiling.set_batchesPerCore(batchesPerCore);
    tiling.set_leftOverBatches(leftOverBatches);
    tiling.set_partD(inputData.inputShape[D_DIM]);
    tiling.set_partH(inputData.inputShape[H_DIM]);
    tiling.set_partW(inputData.inputShape[W_DIM]);
    tiling.set_partOutD(padOutputData.padOutputShape[D_DIM]);
    tiling.set_partOutH(padOutputData.padOutputShape[H_DIM]);
    tiling.set_partOutW(padOutputData.padOutputShape[W_DIM]);
    tiling.set_ceilD(padOutputData.ceil[D_DIM]);
    tiling.set_ceilH(padOutputData.ceil[H_DIM]);
    tiling.set_ceilW(padOutputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DGradWithArgmaxNoSplitTiling::DoOpTiling() {
    DoUBTiling();
    SetTilingData();
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxNoSplitTiling::PostTiling() {
    context_->SetBlockDim(maxPoolGradParams.totalCoreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DGradWithArgmax", MaxPool3DGradWithArgmaxNoSplitTiling, 1);

} // namespace optiling