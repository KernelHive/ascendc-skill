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
 * \file max_pool3d_with_argmax_v2_tiling_splitw.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3_d_with_argmax_v2_tiling.h"

//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=float
constexpr uint64_t TILING_KEY_SPLITW_FLOAT = 111100;
//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=half
constexpr uint64_t TILING_KEY_SPLITW_HALF = 111101;
//1, splitD=1, splitH=1, splitW=1, splitKernel = 0, dtype=bfloat16
constexpr uint64_t TILING_KEY_SPLITW_BF16 = 111102;

using namespace AscendC;

namespace optiling {

bool MaxPool3DWithArgmaxV2SplitWTiling::IsCapable() {
    splitData.partD = inputData.dilation[D_DIM] * (inputData.kernelSize[D_DIM] - 1) + 1;
    splitData.partOutD = 1;
    splitData.partH = inputData.dilation[H_DIM] * (inputData.kernelSize[H_DIM] - 1) + 1;
    splitData.partOutH = 1;
    array<uint64_t, DHW_DIMS> parts{splitData.partD, splitData.partH, padInputData.padInputShape[W_DIM]};
    array<uint64_t, DHW_DIMS> partOuts{splitData.partOutD, splitData.partOutH, inputData.outShape[W_DIM]};

    FindSplitParts(parts, partOuts, bufSizes, W_DIM);
    
    splitData.partW = parts[W_DIM];
    splitData.partOutW = partOuts[W_DIM];
    return ((splitData.partW != 0) && (splitData.partOutW != 0));
}

uint64_t MaxPool3DWithArgmaxV2SplitWTiling::GetTilingKey() const {
    if (dtype == ge::DataType::DT_FLOAT) {
        return TILING_KEY_SPLITW_FLOAT;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_SPLITW_HALF;
    } else {
        return TILING_KEY_SPLITW_BF16;
    }
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitWTiling::SmallBatchesTiling(void) {
    if (splitData.partOutW == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t batchesBlock = splitData.leftOverBatches / blockLength;
    uint64_t batchesRem = splitData.leftOverBatches % blockLength;
    uint64_t countW = RoundDownBlock(inputData.outShape[W_DIM], splitData.partOutW) / splitData.partOutW + ((inputData.outShape[W_DIM] % splitData.partOutW) != 0 ? 1 : 0);
    
    uint64_t wholeCount = inputData.outShape[D_DIM] * inputData.outShape[H_DIM] * countW;

    splitData.partsPerCore = (wholeCount * (batchesBlock + (batchesRem != 0 ? 1 : 0))) / coreNum;
    splitData.leftOverParts = (wholeCount * (batchesBlock + (batchesRem != 0 ? 1 : 0))) % coreNum;
    uint64_t count = 0;
    uint64_t curPos = 0;
    uint64_t offW = splitData.partW - (inputData.kernelSize[W_DIM] - 1) * inputData.dilation[W_DIM] + inputData.stride[W_DIM] - 1;
    for (uint64_t b = 0; b < (batchesBlock + (batchesRem != 0)); b++) {
        if ((count != 0) && (curPos > 0)) {
            if (b < batchesBlock) {
                splitData.sizeIn[curPos-1] += blockLength;
            }
            else {
                splitData.sizeIn[curPos-1] += batchesRem;
            }
        }
        for (uint64_t d = 0, dOut = 0; dOut < inputData.outShape[D_DIM]; d += inputData.stride[D_DIM], dOut++) {
            for (uint64_t h = 0, hOut = 0; hOut < inputData.outShape[H_DIM]; h += inputData.stride[H_DIM], hOut++) {
                for (uint64_t w = 0, wOut = 0; wOut < inputData.outShape[W_DIM]; w += offW, wOut += splitData.partOutW) {
                    count++;                                                                            
                    if (count == 1) {
                        splitData.splitPointDIn[curPos] = d;
                        splitData.splitPointHIn[curPos] = h;
                        splitData.splitPointWIn[curPos] = w;
                        splitData.splitPointDOut[curPos] = dOut;
                        splitData.splitPointHOut[curPos] = hOut;
                        splitData.splitPointWOut[curPos] = wOut;
                        splitData.sizeIn[curPos] = b < batchesBlock ? blockLength : batchesRem;
                        splitData.batchStart[curPos] = b * blockLength;
                        curPos++;
                    }
                    if (count > (splitData.partsPerCore + (curPos <= splitData.leftOverParts ? 1 : 0) - 1)) {
                        count = 0;
                    }
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MaxPool3DWithArgmaxV2SplitWTiling::DoUBTiling() {
    uint64_t batchesBlock = inputData.batches / blockLength; 
    uint64_t batchesRem = inputData.batches % blockLength;
    const uint64_t batchesBlockPerCore = batchesBlock / coreNum;
    const uint64_t leftOverBatchesBlock = batchesBlock % coreNum;
    splitData.batchesPerCore = batchesBlockPerCore * blockLength;
    splitData.leftOverBatches = leftOverBatchesBlock * blockLength + batchesRem;
    
    if (splitData.batchesPerCore == 0) {
        SmallBatchesTiling(); 
    }
}

void MaxPool3DWithArgmaxV2SplitWTiling::SetTilingData() {
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
    tiling.set_minFloat(-std::numeric_limits<float>::infinity());
    tiling.set_ceilD(padInputData.ceil[D_DIM]);
    tiling.set_ceilH(padInputData.ceil[H_DIM]);
    tiling.set_ceilW(padInputData.ceil[W_DIM]);
    tiling.set_sizeUb1(bufSizes.sizeUb1);
    tiling.set_sizeUb2(bufSizes.sizeUb2);
    tiling.set_sizeValues(bufSizes.valSize);
    
    tiling.set_partsPerCore(splitData.partsPerCore);
    tiling.set_leftOverParts(splitData.leftOverParts);
    tiling.set_splitPointDIn(splitData.splitPointDIn);
    tiling.set_splitPointHIn(splitData.splitPointHIn);
    tiling.set_splitPointWIn(splitData.splitPointWIn);
    tiling.set_splitPointDOut(splitData.splitPointDOut);
    tiling.set_splitPointHOut(splitData.splitPointHOut);
    tiling.set_splitPointWOut(splitData.splitPointWOut);
    tiling.set_sizeIn(splitData.sizeIn);
    tiling.set_batchStart(splitData.batchStart);
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitWTiling::DoOpTiling() {
    DoUBTiling();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2SplitWTiling::PostTiling() {
    context_->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2SplitWTiling, 5);

} // namespace optiling
