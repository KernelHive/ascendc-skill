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
 * \file max_pool3d_with_argmax_v2_tiling_basesplit.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "platform/platform_info.h"
#include "max_pool3_d_with_argmax_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

const uint32_t BLOCK_LEN_UINT8 = 32;
const uint32_t BITS_UINT8 = 8;
const uint32_t RESERVED_FOR_SELECT = 8 * 1024;
const uint32_t MASK_COUNT = 2; // use 2 masks: mask for max values and for nan values

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::GetPlatformInfo() {
    auto ret = MaxPool3DWithArgmaxV2BaseTiling::GetPlatformInfo();
    if (ret == ge::GRAPH_FAILED) {
        return ret;
    }
    ubSizeNew = ubSize - RESERVED_FOR_SELECT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::GetShapeAttrsInfo() {
    auto ret = MaxPool3DWithArgmaxV2BaseTiling::GetShapeAttrsInfo();
    if (ret == ge::GRAPH_FAILED) {
        return ret;
    }

    blockLength = (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_BF16) ? BLOCK_LEN_FP32 : BLOCK_LEN_FP16;
    blockLengthS = (dtype == ge::DataType::DT_FLOAT) ? BLOCK_LEN_FP32 : BLOCK_LEN_FP16;
    
    return PadCalc();
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::GetWorkspaceSize() {
    return MaxPool3DWithArgmaxV2BaseTiling::GetWorkspaceSize();
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::PadCalc() {
    if (inputData.kernelSize[D_DIM] == 0 || inputData.kernelSize[H_DIM] == 0 || inputData.kernelSize[W_DIM] == 0 ||
        inputData.stride[D_DIM] == 0 || inputData.stride[H_DIM] == 0 || inputData.stride[W_DIM] == 0) {
        return ge::GRAPH_FAILED;
    }
    
    const array<uint64_t, DHW_DIMS> inputCalc{InputCalc(D_DIM), InputCalc(H_DIM), InputCalc(W_DIM)};
    if (inputCalc[D_DIM] == 0 || inputCalc[H_DIM] == 0 || inputCalc[W_DIM] == 0) {
        return ge::GRAPH_FAILED;
    }

    const array<int64_t, DHW_DIMS> inputDiff{(int64_t)inputCalc[D_DIM] - (int64_t)inputData.inputShape[D_DIM], 
                                             (int64_t)inputCalc[H_DIM] - (int64_t)inputData.inputShape[H_DIM],
                                             (int64_t)inputCalc[W_DIM] - (int64_t)inputData.inputShape[W_DIM]};
    
    return InputPadCalc(inputDiff);
}

uint64_t MaxPool3DWithArgmaxV2BaseSplitTiling::InputCalc(uint64_t dim) {
    if (dim != D_DIM && dim != H_DIM && dim != W_DIM) {
        return 0;
    }
    return (inputData.outShape[dim] - 1) * inputData.stride[dim] + inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + 1;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::InputPadCalc(const array<int64_t, DHW_DIMS> inpDiff) {
    padInputData.ceil[D_DIM] = max(inpDiff[D_DIM] - (int64_t)inputData.pad[D_DIM], (int64_t)0);
    padInputData.ceil[H_DIM] = max(inpDiff[H_DIM] - (int64_t)inputData.pad[H_DIM], (int64_t)0);
    padInputData.ceil[W_DIM] = max(inpDiff[W_DIM] - (int64_t)inputData.pad[W_DIM], (int64_t)0);

    padInputData.padInputShape[D_DIM] = inputData.inputShape[D_DIM] + inputData.pad[D_DIM] + padInputData.ceil[D_DIM];
    padInputData.padInputShape[H_DIM] = inputData.inputShape[H_DIM] + inputData.pad[H_DIM] + padInputData.ceil[H_DIM];
    padInputData.padInputShape[W_DIM] = inputData.inputShape[W_DIM] + inputData.pad[W_DIM] + padInputData.ceil[W_DIM];

    if (padInputData.padInputShape[D_DIM] == 0 || padInputData.padInputShape[H_DIM] == 0 || padInputData.padInputShape[W_DIM] == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DWithArgmaxV2BaseSplitTiling::CalcBufferSizes(const array<uint64_t, DHW_DIMS> part,
                                                               const array<uint64_t, DHW_DIMS> partOut,
                                                               const uint64_t partHwInp,
                                                               UBBufferSize& bufferSizes) {
    if (inputData.kernelSize[D_DIM] == 0 || inputData.kernelSize[H_DIM] == 0 || inputData.kernelSize[W_DIM] == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t kernelDHW = inputData.kernelSize[D_DIM] * inputData.kernelSize[H_DIM] * inputData.kernelSize[W_DIM];
    const uint64_t halfKernelDHW = kernelDHW / MAX_DIV;

    uint64_t partRoundDhwInp = part[D_DIM] * partHwInp;
    uint64_t transDataSize = partRoundDhwInp < NCHW_CONV_ADDR_LIST_SIZE ? (MIN_TRANSPOSE_ROWS + blockLength) * blockLength : RoundUpBlock(partRoundDhwInp, blockLengthS) * blockLength;
    bufferSizes.sizeUb1 = partRoundDhwInp < NCHW_CONV_ADDR_LIST_SIZE ? (MIN_TRANSPOSE_ROWS + blockLength) * blockLength : RoundUpBlock(partRoundDhwInp, MIN_TRANSPOSE_ROWS) * blockLength;
    bufferSizes.valSize = RoundUpBlock(partOut[D_DIM] * partOut[H_DIM] * partOut[W_DIM], MIN_TRANSPOSE_ROWS) * blockLength;
    
    if (inputData.stride[D_DIM] >= inputData.kernelSize[D_DIM] && inputData.stride[H_DIM] >= inputData.kernelSize[H_DIM] && \
        inputData.stride[W_DIM] >= inputData.kernelSize[W_DIM]) {
        bufferSizes.sizeUb2 = transDataSize;
        // if kD=kH=kW=1 2 masks will require more space than original input due to alignment
        if (inputData.kernelSize[D_DIM] == 1 && inputData.kernelSize[H_DIM] == 1 && inputData.kernelSize[W_DIM] == 1) {
            bufSizes.sizeUb1 = std::max(bufSizes.sizeUb1, MASK_COUNT * RoundUpBlock((kernelDHW * blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * partOut[D_DIM] * partOut[H_DIM] * partOut[W_DIM] / (uint32_t)sizeof(float));
        }
    } else {
        if (partOut[D_DIM] * partOut[H_DIM] > 1) {
            transDataSize = kernelDHW * partOut[W_DIM] * blockLength > transDataSize ? kernelDHW * partOut[W_DIM] * blockLength : transDataSize;
            bufferSizes.sizeUb2 = transDataSize + std::max((kernelDHW - halfKernelDHW) * blockLength * partOut[W_DIM], MASK_COUNT * RoundUpBlock((kernelDHW * blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * partOut[W_DIM] / (uint32_t)sizeof(float));
        } else {
            bufferSizes.sizeUb1 = (kernelDHW + MAX_DIV - 1) / MAX_DIV * partOut[W_DIM] * blockLength > bufSizes.sizeUb1 ? (kernelDHW + MAX_DIV - 1) / MAX_DIV * partOut[W_DIM] * blockLength : bufSizes.sizeUb1;
            bufferSizes.sizeUb1 = std::max(bufSizes.sizeUb1, MASK_COUNT * RoundUpBlock((kernelDHW * blockLength) / BITS_UINT8, BLOCK_LEN_UINT8) * partOut[W_DIM] / (uint32_t)sizeof(float));
            bufferSizes.sizeUb2 = kernelDHW * partOut[W_DIM] * blockLength > transDataSize ? kernelDHW * partOut[W_DIM] * blockLength : transDataSize;
            transDataSize = kernelDHW * partOut[W_DIM] * blockLength > transDataSize ? kernelDHW * partOut[W_DIM] * blockLength : transDataSize;
        }
    }
    bufferSizes.sizeUb2 = kernelDHW <= blockLength ? bufferSizes.sizeUb2 + blockLength * blockLength : bufferSizes.sizeUb2;

    //extend buffer if output > input
    bufferSizes.sizeUb2 = std::max(bufferSizes.sizeUb2, bufferSizes.valSize);

    return (bufferSizes.sizeUb2 + bufferSizes.valSize + bufferSizes.valSize + bufferSizes.sizeUb1 + RoundUpBlock(kernelDHW, blockLength) * blockLength + DHW_DIMS * RoundUpBlock(partOut[D_DIM] * partOut[H_DIM] * partOut[W_DIM], blockLength)) * sizeof(float);
}

uint64_t MaxPool3DWithArgmaxV2BaseSplitTiling::RoundUpBlock(const uint64_t& src, const uint64_t& blockLen) {
    if (blockLen != 0) {
        return src != 0 ? src + (blockLen - src % blockLen) % blockLen : blockLen;
    }
    return blockLen;
}

uint64_t MaxPool3DWithArgmaxV2BaseSplitTiling::RoundDownBlock(const uint64_t& src, const uint64_t& blockLen) {
    if (blockLen != 0) {
        return (src / blockLen) * blockLen;
    }
    return blockLen;
}

ge::graphStatus MaxPool3DWithArgmaxV2BaseSplitTiling::FindSplitParts(array<uint64_t, DHW_DIMS> &inParts, array<uint64_t, DHW_DIMS> &outParts,
                                                                     UBBufferSize& bufferSizes, uint64_t dim) {                 
    if (dim != D_DIM && dim != H_DIM && dim != W_DIM) {
        return ge::GRAPH_FAILED;
    }

    uint64_t left = 0;
    uint64_t right = inputData.outShape[dim];
    uint64_t summaryMemory = 0;
    uint64_t partHwInp = inParts[W_DIM] * inParts[H_DIM];
    while (left < right - 1) {
        outParts[dim] = (left + right) / BINARY_SEARCH_COEFF;
        inParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (outParts[dim] - 1) + 1;

        if (dim == H_DIM) {
            partHwInp = RoundUpBlock(inParts[W_DIM] * inParts[H_DIM], blockLengthS);
        } else if (dim == W_DIM) {
            partHwInp = RoundUpBlock(inParts[W_DIM], blockLengthS) * inParts[H_DIM];
        }

        summaryMemory = CalcBufferSizes(inParts, outParts, partHwInp, bufferSizes);
        if (summaryMemory <= ubSizeNew) {
            left = outParts[dim];
        } else {
            right = outParts[dim];
        }
        if ((left == right - 1) && (left != 0) && (summaryMemory > ubSizeNew)) {
            outParts[dim] -= 1;
            inParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (outParts[dim] - 1) + 1;

            if (dim == H_DIM) {
                partHwInp = RoundUpBlock(inParts[W_DIM] * inParts[H_DIM], blockLengthS);
            } else if (dim == W_DIM) {
                partHwInp = RoundUpBlock(inParts[W_DIM], blockLengthS) * inParts[H_DIM];
            }
            
            summaryMemory = CalcBufferSizes(inParts, outParts, partHwInp, bufferSizes);
        }
    }
    
    if (left == 0) {
        inParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (outParts[dim] - 1) + 1;
        if (dim == H_DIM) {
            partHwInp = RoundUpBlock(inParts[W_DIM] * inParts[H_DIM], blockLengthS);
        } else if (dim == W_DIM) {
            partHwInp = RoundUpBlock(inParts[W_DIM], blockLengthS) * inParts[H_DIM];
        }

        summaryMemory = CalcBufferSizes(inParts, outParts, partHwInp, bufferSizes);
    }
    if (summaryMemory > ubSizeNew) {
        inParts[dim] = 0;
        outParts[dim] = 0;
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling