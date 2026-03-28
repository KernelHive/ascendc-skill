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
 * \file max_pool3d_grad_with_argmax_tiling_basesplit.cpp
 * \brief
 */

#include "max_pool3d_grad_with_argmax_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

const uint32_t BLOCK_LEN_FP32 = 8;
const uint32_t BLOCK_LEN_FP16 = 16;
const uint32_t BLOCK_LEN_UINT8 = 32;
const uint32_t RESERVED_FOR_SELECT = 8 * 1024;
const uint32_t QUE_NUM = 3;
const uint32_t GENERAL_OFFSETS_NUM = 3;
const uint32_t BITS_UINT8 = 8;
const uint32_t CDHW_DIMS = 4;
const uint32_t NCDHW_DIMS = 5;

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::ParamsDefinition() {
    auto attrs = context_->GetAttrs();
    
    inputData.inputShape = {static_cast<uint64_t>(maxPoolGradParams.doDim),static_cast<uint64_t>(maxPoolGradParams.hoDim), static_cast<uint64_t>(maxPoolGradParams.woDim)};
    inputData.outShape = {static_cast<uint64_t>(maxPoolGradParams.diDim),static_cast<uint64_t>(maxPoolGradParams.hiDim), static_cast<uint64_t>(maxPoolGradParams.wiDim)};
    inputData.kernelSize = {static_cast<uint64_t>(maxPoolGradParams.kd),static_cast<uint64_t>(maxPoolGradParams.kh), static_cast<uint64_t>(maxPoolGradParams.kw)};
    inputData.stride = {static_cast<uint64_t>(maxPoolGradParams.sd),static_cast<uint64_t>(maxPoolGradParams.sh), static_cast<uint64_t>(maxPoolGradParams.sw)};
    inputData.pad = {static_cast<uint64_t>(maxPoolGradParams.pDTop),static_cast<uint64_t>(maxPoolGradParams.pHTop), static_cast<uint64_t>(maxPoolGradParams.pWTop)};
    inputData.dilation = {static_cast<uint64_t>(maxPoolGradParams.dilationD),static_cast<uint64_t>(maxPoolGradParams.dilationH), static_cast<uint64_t>(maxPoolGradParams.dilationW)};
    inputData.ceilMode = *attrs->GetAttrPointer<bool>(CEIL_MODE_ATTR_INDEX);
    inputData.batches = maxPoolGradParams.ncDim;

    if (inputData.kernelSize[D_DIM] == 1) {
        inputData.dilation[D_DIM] = 1;
    } 
    if (inputData.kernelSize[H_DIM] == 1) {
        inputData.dilation[H_DIM] = 1;
    } 
    if (inputData.kernelSize[W_DIM] == 1) {
        inputData.dilation[W_DIM] = 1;
    }
    
    auto isOverlapDim = [&](const uint32_t& curDim) {
        return inputData.stride[curDim] <= (inputData.dilation[curDim] * (inputData.kernelSize[curDim] - 1));
    };

    inputData.isOverlap = isOverlapDim(D_DIM) || isOverlapDim(H_DIM) || isOverlapDim(W_DIM);
    return ge::GRAPH_SUCCESS;
}
    
ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::GetPlatformInfo() {
    auto ret = MaxPool3DGradWithArgmaxTilingBase::GetPlatformInfo();
    
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != ret,
                    OP_LOGE(context_->GetNodeName(), "GetPlatformInfo failed."), return ret);
    ubSizeNew = maxPoolGradParams.maxUbSize - RESERVED_FOR_SELECT;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::GetShapeAttrsInfo() {
    auto inputDesc = context_->GetInputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();

    if (dtype != ge::DataType::DT_BF16 && dtype != ge::DataType::DT_FLOAT16 && dtype != ge::DataType::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPool3DGradWithArgmax: invalid dtype");
        return ge::GRAPH_FAILED;
    }
    
    auto ret = MaxPool3DGradWithArgmaxTilingBase::GetShapeAttrsInfo();
	
    OP_TILING_CHECK(ge::GRAPH_SUCCESS != ret,
                    OP_LOGE(context_->GetNodeName(), "GetShapeAttrsInfo failed."), return ret);

    OP_TILING_CHECK(ge::GRAPH_SUCCESS != ParamsDefinition(),
                    OP_LOGE(context_->GetNodeName(), "Set params failed."), return ge::GRAPH_FAILED);
                    
    blockLength = (dtype == ge::DataType::DT_FLOAT || dtype == ge::DataType::DT_BF16) ? BLOCK_LEN_FP32 : inputData.isOverlap ? BLOCK_LEN_FP32 : BLOCK_LEN_FP16;
    blockLengthS = (dtype == ge::DataType::DT_FLOAT) ? BLOCK_LEN_FP32 : BLOCK_LEN_FP16;
    
    return PadCalc();
}

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::GetWorkspaceSize()  {
    return MaxPool3DGradWithArgmaxTilingBase::GetWorkspaceSize();
}

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::PadCalc() {
    bool zeroInput = inputData.kernelSize[D_DIM] == 0 || inputData.kernelSize[H_DIM] == 0 || 
                     inputData.kernelSize[W_DIM] == 0 || inputData.stride[D_DIM] == 0 || 
                     inputData.stride[H_DIM] == 0 || inputData.stride[W_DIM] == 0;
    
    OP_TILING_CHECK(zeroInput, OP_LOGE(context_->GetNodeName(), "Kernel size or stride can not be 0."), return ge::GRAPH_FAILED);
    
    const std::array<uint64_t, DHW_DIMS> outputCalc{InputCalc(D_DIM), InputCalc(H_DIM), InputCalc(W_DIM)};
    bool zeroOutput = outputCalc[D_DIM] == 0 || outputCalc[H_DIM] == 0 || outputCalc[W_DIM] == 0;
    OP_TILING_CHECK(zeroOutput, OP_LOGE(context_->GetNodeName(), "Onput data can not be 0."), return ge::GRAPH_FAILED);
    
    const std::array<int64_t, DHW_DIMS> outputDiff{(int64_t)outputCalc[D_DIM] - (int64_t)inputData.outShape[D_DIM],
                                                   (int64_t)outputCalc[H_DIM] - (int64_t)inputData.outShape[H_DIM],
                                                   (int64_t)outputCalc[W_DIM] - (int64_t)inputData.outShape[W_DIM]};
    return InputPadCalc(outputDiff);
}

uint64_t MaxPool3DGradWithArgmaxBaseSplitTiling::InputCalc(uint64_t dim) {
    if (dim != D_DIM && dim != H_DIM && dim != W_DIM) {
        return 0;
    }
    return (inputData.inputShape[dim] - 1) * inputData.stride[dim] + inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + 1;
}

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::InputPadCalc(const std::array<int64_t, DHW_DIMS> outDiff) {
    padOutputData.ceil[D_DIM] = std::max(outDiff[D_DIM] - (int64_t)inputData.pad[D_DIM], (int64_t)0);
    padOutputData.ceil[H_DIM] = std::max(outDiff[H_DIM] - (int64_t)inputData.pad[H_DIM], (int64_t)0);
    padOutputData.ceil[W_DIM] = std::max(outDiff[W_DIM] - (int64_t)inputData.pad[W_DIM], (int64_t)0);

    padOutputData.padOutputShape[D_DIM] = inputData.outShape[D_DIM] + inputData.pad[D_DIM] + padOutputData.ceil[D_DIM];
    padOutputData.padOutputShape[H_DIM] = inputData.outShape[H_DIM] + inputData.pad[H_DIM] + padOutputData.ceil[H_DIM];
    padOutputData.padOutputShape[W_DIM] = inputData.outShape[W_DIM] + inputData.pad[W_DIM] + padOutputData.ceil[W_DIM];

    bool zeroPadOut = padOutputData.padOutputShape[D_DIM] == 0 || padOutputData.padOutputShape[H_DIM] == 0 || 
                      padOutputData.padOutputShape[W_DIM] == 0;

    OP_TILING_CHECK(zeroPadOut, OP_LOGE(context_->GetNodeName(), "Padded output data can not be 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DGradWithArgmaxBaseSplitTiling::CalcBufferSizes(const std::array<uint64_t, DHW_DIMS> part, const std::array<uint64_t, DHW_DIMS> partOut,
                                                                 const uint64_t partHwOut, UBBufferSize& bufSizes, uint64_t dim) {
    bool zeroKernel = inputData.kernelSize[D_DIM] == 0 || inputData.kernelSize[H_DIM] == 0 || 
                      inputData.kernelSize[W_DIM] == 0;
    OP_TILING_CHECK(zeroKernel, OP_LOGE(context_->GetNodeName(), "Kernel size can not be 0."), return ge::GRAPH_FAILED);
 
    uint64_t kernelDHW = inputData.kernelSize[D_DIM] + ((inputData.kernelSize[D_DIM] - 1) * (inputData.dilation[D_DIM] - 1));
    kernelDHW *= inputData.kernelSize[H_DIM] + ((inputData.kernelSize[H_DIM] - 1) * (inputData.dilation[H_DIM] - 1));
    kernelDHW *= inputData.kernelSize[W_DIM] + ((inputData.kernelSize[W_DIM] - 1) * (inputData.dilation[W_DIM] - 1));
    const uint64_t alKernelDHW = RoundUpBlock(kernelDHW, blockLength);
    const uint64_t blockAlKernelDHW = alKernelDHW * blockLength;
    const uint64_t dhwGradOutputSize = part[D_DIM] * part[H_DIM] * part[W_DIM];
    const uint64_t blockAlDhwGradInput = (dim == W_DIM) ? partOut[D_DIM] * partOut[H_DIM] * RoundUpBlock(partOut[W_DIM], MIN_TRANSPOSE_ROWS) * blockLength:
                                                          partOut[D_DIM] * RoundUpBlock(partHwOut, MIN_TRANSPOSE_ROWS) * blockLength;
    uint64_t transDataSize = dhwGradOutputSize < NCHW_CONV_ADDR_LIST_SIZE ? (MIN_TRANSPOSE_ROWS + blockLength) * blockLength : RoundUpBlock(dhwGradOutputSize, MIN_TRANSPOSE_ROWS) * blockLength;
    uint64_t bufferSize = std::max(blockAlKernelDHW * dhwGradOutputSize, blockAlDhwGradInput);
    uint64_t transAlignBufferSize = RoundUpBlock(bufferSize, MIN_TRANSPOSE_ROWS);
    uint64_t generalOffsetsSize = GENERAL_OFFSETS_NUM * RoundUpBlock(dhwGradOutputSize, blockLength);
    uint64_t masksBufferSize = RoundUpBlock(blockLength * kernelDHW * dhwGradOutputSize / BITS_UINT8, BLOCK_LEN_UINT8) +
                               (inputData.pad[D_DIM] * partHwOut + inputData.pad[H_DIM] * partOut[W_DIM] + inputData.pad[W_DIM]) * blockLength;
    
    bufSizes.sizeUb1 = transDataSize;
    bufSizes.sizeUb2 = transAlignBufferSize;
    bufSizes.valSize = generalOffsetsSize + blockAlKernelDHW + masksBufferSize + transAlignBufferSize;
    
    uint64_t memoryQueInVals = bufSizes.sizeUb1 * sizeof(float);
    uint64_t memoryQueInInds = (QUE_NUM - 1) * bufSizes.sizeUb1 * sizeof(float);
    uint64_t memoryQueOutVals = bufSizes.sizeUb2 * sizeof(float);
    uint64_t memoryTmpBuf = bufSizes.valSize * sizeof(float);
    
    return memoryQueInVals + memoryQueInInds + memoryQueOutVals + memoryTmpBuf;
}

uint64_t MaxPool3DGradWithArgmaxBaseSplitTiling::RoundUpBlock(const uint64_t& src, const uint64_t& blockLen) {
    if (blockLen != 0) {
        return src != 0 ? src + (blockLen - src % blockLen) % blockLen : blockLen;
    }
    return blockLen;
}

uint64_t MaxPool3DGradWithArgmaxBaseSplitTiling::RoundDownBlock(const uint64_t& src, const uint64_t& blockLen) {
    if (blockLen != 0) {
        return (src / blockLen) * blockLen;
    }
    return blockLen;
}

ge::graphStatus MaxPool3DGradWithArgmaxBaseSplitTiling::FindSplitParts(std::array<uint64_t, DHW_DIMS> &inParts,
                                                                       std::array<uint64_t, DHW_DIMS> &outParts,
                                                                       UBBufferSize& bufSizes, uint64_t dim) {                 
    bool wrongSplit = dim != D_DIM && dim != H_DIM && dim != W_DIM;
    OP_TILING_CHECK(wrongSplit, OP_LOGE(context_->GetNodeName(), "Kernel can only have split by D (0),\
                                                                  by H (1), and by W(2)."), return ge::GRAPH_FAILED);
    
    uint64_t left = 0;
    uint64_t right = inputData.inputShape[dim];
    uint64_t summaryMemory = 0;
    uint64_t partHwOut = outParts[W_DIM] * outParts[H_DIM];
    while (left < right - 1) {
        inParts[dim] = (left + right) / BINARY_SEARCH_COEFF;
        outParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (inParts[dim] - 1) + 1;
		
        if (dim == H_DIM) {
            partHwOut = RoundUpBlock(outParts[W_DIM] * outParts[H_DIM], blockLengthS);
        } else if (dim == W_DIM) {
            partHwOut = RoundUpBlock(outParts[W_DIM], blockLengthS) * outParts[H_DIM];
        }
    
        summaryMemory = CalcBufferSizes(inParts, outParts, partHwOut, bufSizes, dim);
        if (summaryMemory <= ubSizeNew) {
            left = inParts[dim];
        } else {
            right = inParts[dim];
        }
        if ((left == right - 1) && (left != 0) && (summaryMemory > ubSizeNew)) {
            inParts[dim] -= 1;
            outParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (inParts[dim] - 1) + 1;
		
            if (dim == H_DIM) {
                partHwOut = RoundUpBlock(outParts[W_DIM] * outParts[H_DIM], blockLengthS);
            } else if (dim == W_DIM) {
                partHwOut = RoundUpBlock(outParts[W_DIM], blockLengthS) * outParts[H_DIM];
            }
            summaryMemory = CalcBufferSizes(inParts, outParts, partHwOut, bufSizes, dim);
        }
    }
    
    if (left == 0) {
        outParts[dim] = inputData.dilation[dim] * (inputData.kernelSize[dim] - 1) + inputData.stride[dim] * (inParts[dim] - 1) + 1;
        if (dim == H_DIM) {
            partHwOut = RoundUpBlock(outParts[W_DIM] * outParts[H_DIM], blockLengthS);
        } else if (dim == W_DIM) {
            partHwOut = RoundUpBlock(outParts[W_DIM], blockLengthS) * outParts[H_DIM];
        }
        
        summaryMemory = CalcBufferSizes(inParts, outParts, partHwOut, bufSizes, dim);
    }
    if (summaryMemory > ubSizeNew) {
        inParts[dim] = 0;
        outParts[dim] = 0;
    }

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
