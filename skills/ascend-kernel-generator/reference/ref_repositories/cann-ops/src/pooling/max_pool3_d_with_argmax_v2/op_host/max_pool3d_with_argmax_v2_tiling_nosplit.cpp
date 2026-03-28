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
 * \file max_pool3d_with_argmax_v2_tiling_nosplit.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3_d_with_argmax_v2_tiling.h"

//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=float
constexpr uint64_t TILING_KEY_NOSPLIT_FLOAT = 100000;
//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=half
constexpr uint64_t TILING_KEY_NOSPLIT_HALF = 100001;
//1, splitD=0, splitH=0, splitW=0, splitKernel = 0, dtype=bfloat16
constexpr uint64_t TILING_KEY_NOSPLIT_BF16 = 100002;

using namespace AscendC;

namespace optiling {


bool MaxPool3DWithArgmaxV2NoSplitTiling::IsCapable() {
    array<uint64_t, DHW_DIMS> tmpOutShape{inputData.outShape[D_DIM], inputData.outShape[H_DIM], inputData.outShape[W_DIM]};
    auto summaryMemory = CalcBufferSizes(padInputData.padInputShape, tmpOutShape,
                                         padInputData.padInputShape[W_DIM] * padInputData.padInputShape[H_DIM], bufSizes);
    return (summaryMemory < ubSizeNew);
}

uint64_t MaxPool3DWithArgmaxV2NoSplitTiling::GetTilingKey() const {
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_NOSPLIT_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_NOSPLIT_HALF;
    } else {
        return TILING_KEY_NOSPLIT_BF16;
    }
}

void MaxPool3DWithArgmaxV2NoSplitTiling::DoUBTiling() {
    uint64_t batchesBlock = inputData.batches / blockLength; 
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / coreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % coreNum;
    batchesPerCore = batchesBlockPerCore * blockLength;
    leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
}

void MaxPool3DWithArgmaxV2NoSplitTiling::SetTilingData() {
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
    tiling.set_partD(padInputData.padInputShape[D_DIM]);
    tiling.set_partH(padInputData.padInputShape[H_DIM]);
    tiling.set_partW(padInputData.padInputShape[W_DIM]);
    tiling.set_minFloat(-std::numeric_limits<float>::infinity());
    tiling.set_ceilD(padInputData.ceil[D_DIM]);
    tiling.set_ceilH(padInputData.ceil[H_DIM]);
    tiling.set_ceilW(padInputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
}

ge::graphStatus MaxPool3DWithArgmaxV2NoSplitTiling::DoOpTiling() {
    DoUBTiling();
    SetTilingData();
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NoSplitTiling::PostTiling() {
    context_->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2NoSplitTiling, 2);

} // namespace optiling
