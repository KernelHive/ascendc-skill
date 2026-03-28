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
 * \file max_pool3d_with_argmax_v2_no_expand_indices_tiling.cpp
 * \brief
 */

#include "tiling/tiling_templates_registry.h"
#include "max_pool3_d_with_argmax_v2_tiling.h"

namespace optiling {
  static constexpr uint64_t BLOCK_SIZE = 32;
  static constexpr uint64_t BLOCK_COUNT = 8;
  static constexpr uint64_t FLOAT16_SIZE = 2;
  static constexpr uint64_t FLOAT32_SIZE = 4;
  static constexpr uint64_t INT32_SIZE = 4;
  static constexpr uint64_t OR_MASK_RESVERVED = 256;
  static constexpr uint64_t UB_RESVERVED_SIZE = 8 * 1024;
  static constexpr uint64_t UB_SIZE = 192 * 1024 - UB_RESVERVED_SIZE - OR_MASK_RESVERVED;
  static constexpr uint64_t MASK_DTYPE_BIT = 16;
  static constexpr uint64_t MASK_DTYPE_SIZE = MASK_DTYPE_BIT / 8;
  static constexpr uint64_t TRANS_ALIGN = 16;
  static constexpr uint64_t CACHE_ALIGN_SIZE = 256;
  static constexpr uint64_t TWO_KB_SIZE = 2 * 1024;
  static constexpr uint64_t PAD_F = 0;
  static constexpr uint64_t PAD_B = 1;
  static constexpr uint64_t PAD_T = 2;
  static constexpr uint64_t PAD_D = 3;
  static constexpr uint64_t PAD_L = 4;
  static constexpr uint64_t PAD_R = 5;
  static constexpr uint64_t NO_PADDING_TILING_KEY = 300001;
  static constexpr uint64_t PADDING_TILING_KEY = 300002;
  static constexpr uint64_t NUM_OF_INPUT_PARTS = 2;
  static constexpr uint64_t NUM_OF_OUTPUT_MAX_PARTS = 2;
  static constexpr uint64_t NUM_OF_OUTPUT_INDICES_PARTS = 2;
  static constexpr uint64_t NUM_OF_TEMP_INDICES_PARTS = 2;
  static constexpr uint64_t NUM_OF_MASK_PARTS = 2;
  static constexpr uint64_t MID_FACTOR = 2;

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoDAdjustment() {
  splitData.outShape[D_DIM] = (inputData.inputShape[D_DIM] + splitData.pad[PAD_F] + splitData.pad[PAD_B] - inputData.kernelSize[D_DIM] + (inputData.stride[D_DIM] - 1)) / inputData.stride[D_DIM] + 1;
  uint64_t usedDx = (splitData.outShape[D_DIM] - 1) * inputData.stride[D_DIM] + inputData.kernelSize[D_DIM];
  if (usedDx - inputData.kernelSize[D_DIM] >= inputData.inputShape[D_DIM] + splitData.pad[PAD_F]) {
    splitData.outShape[D_DIM] -= 1;
    usedDx = (splitData.outShape[D_DIM] - 1) * inputData.stride[D_DIM] - (inputData.stride[D_DIM] - 1) + inputData.kernelSize[D_DIM];
    if (usedDx < splitData.pad[PAD_F] + inputData.inputShape[D_DIM]) {
      splitData.pad[PAD_B] = 0;
    } else {
      splitData.pad[PAD_B] = usedDx - splitData.pad[PAD_F] - inputData.inputShape[D_DIM];
    }
  } else if (!splitData.pad[PAD_F] && !splitData.pad[PAD_B] && usedDx > inputData.inputShape[D_DIM]) {
    splitData.pad[PAD_B] = usedDx - inputData.inputShape[D_DIM];
  }
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoHAdjustment() {
  splitData.outShape[H_DIM] = (inputData.inputShape[H_DIM] + splitData.pad[PAD_T] + splitData.pad[PAD_D] - inputData.kernelSize[H_DIM] + (inputData.stride[H_DIM] - 1)) / inputData.stride[H_DIM] + 1;
  uint64_t usedHx = (splitData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
  if (usedHx - inputData.kernelSize[H_DIM] >= inputData.inputShape[H_DIM] + splitData.pad[PAD_T]) {
    splitData.outShape[H_DIM] -= 1;
    usedHx = (splitData.outShape[H_DIM] - 1) * inputData.stride[H_DIM] - (inputData.stride[H_DIM] - 1) + inputData.kernelSize[H_DIM];
    if (usedHx < splitData.pad[PAD_T] + inputData.inputShape[H_DIM]) {
      splitData.pad[PAD_D] = 0;
    } else {
      splitData.pad[PAD_D] = usedHx - splitData.pad[PAD_T] - inputData.inputShape[H_DIM];
    }
  } else if (!splitData.pad[PAD_T] && !splitData.pad[PAD_D] && usedHx > inputData.inputShape[H_DIM]) {
    splitData.pad[PAD_D] = usedHx - inputData.inputShape[H_DIM];
  }
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoWAdjustment() {
  splitData.outShape[W_DIM] = (inputData.inputShape[W_DIM] + splitData.pad[PAD_L] + splitData.pad[PAD_R] - inputData.kernelSize[W_DIM] + (inputData.stride[W_DIM] - 1)) / inputData.stride[W_DIM] + 1;
  uint64_t usedWx = (splitData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM];
  if (usedWx - inputData.kernelSize[W_DIM] >= inputData.inputShape[W_DIM] + splitData.pad[PAD_L]) {
    splitData.outShape[W_DIM] -= 1;
    usedWx = (splitData.outShape[W_DIM] - 1) * inputData.stride[W_DIM] - (inputData.stride[W_DIM] - 1) + inputData.kernelSize[W_DIM];
    if (usedWx < splitData.pad[PAD_L] + inputData.inputShape[W_DIM]) {
      splitData.pad[PAD_R] = 0;
    } else {
      splitData.pad[PAD_R] = usedWx - splitData.pad[PAD_L] - inputData.inputShape[W_DIM];
    }
  } else if (!splitData.pad[PAD_L] && !splitData.pad[PAD_R] && usedWx > inputData.inputShape[W_DIM]) {
    splitData.pad[PAD_R] = usedWx - inputData.inputShape[W_DIM];
  }
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoOutputPadAdjustment() {
  splitData.outShape[D_DIM] = inputData.outShape[D_DIM];
  splitData.outShape[H_DIM] = inputData.outShape[H_DIM];
  splitData.outShape[W_DIM] = inputData.outShape[W_DIM];
    // infershape dy hy wy
  if (inputData.ceilMode) {
    DoDAdjustment();
    DoHAdjustment();
    DoWAdjustment();
  }
}

uint64_t CeilAlign(uint64_t blockToAlign, uint64_t alignSize) {
  return ((blockToAlign + (alignSize - 1)) & ~(alignSize - 1));
}

bool MaxPool3DWithArgmaxV2NoExpandIndicesTiling::IsCapable() {
  if (inputData.dilation[D_DIM] != 1 || inputData.dilation[H_DIM] != 1 || inputData.dilation[W_DIM] != 1) {
    return false;
  }
  inputBytes = dtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
  bufferData.inputCastBytes = dtype == ge::DT_FLOAT16 ? FLOAT16_SIZE : FLOAT32_SIZE;
  
  splitData.pad[PAD_F] = inputData.pad[D_DIM];
  splitData.pad[PAD_B] = inputData.pad[D_DIM];
  splitData.pad[PAD_T] = inputData.pad[H_DIM];
  splitData.pad[PAD_D] = inputData.pad[H_DIM];
  splitData.pad[PAD_L] = inputData.pad[W_DIM];
  splitData.pad[PAD_R] = inputData.pad[W_DIM];

  DoOutputPadAdjustment();
  bufferData.blockDataNum = BLOCK_SIZE / bufferData.inputCastBytes;
  bufferData.ncMaxFactor = bufferData.blockDataNum * BLOCK_COUNT;
  splitData.ncFactor = std::min(inputData.batches, bufferData.ncMaxFactor);
  bufferData.ncFactorAlign = CeilAlign(splitData.ncFactor, TRANS_ALIGN);
  splitData.ncTail = inputData.batches % splitData.ncFactor;
  splitData.ncTail = splitData.ncTail == 0 ? splitData.ncFactor : splitData.ncTail;
  splitData.ncOuter = (inputData.batches + splitData.ncFactor - 1) / splitData.ncFactor;

  // initializa left boundary and right boundary for binary searching optimum ub allocation
  bufferData.lbDyFactor = 1;
  bufferData.lbHyFactor = 1;
  bufferData.lbWyFactor = 1;
  bufferData.ubDyFactor = splitData.outShape[D_DIM];
  bufferData.ubHyFactor = splitData.outShape[H_DIM];
  bufferData.ubWyFactor = splitData.outShape[W_DIM];
  bufferData.tmpDyFactor = bufferData.lbDyFactor;
  bufferData.tmpHyFactor = bufferData.lbHyFactor;
  bufferData.tmpWyFactor = bufferData.lbWyFactor;
  bufferData.tmpWyFactorAlign = CeilAlign(bufferData.tmpWyFactor, TRANS_ALIGN);
  DoBufferCalculate();
  // return false if we cant even put in one output element(wy=1)
  return bufferData.tmpTotalSize <= UB_SIZE;
}

uint64_t MaxPool3DWithArgmaxV2NoExpandIndicesTiling::GetTilingKey() const {
  uint64_t tilingKey = NO_PADDING_TILING_KEY;
  if (splitData.pad[PAD_F] || splitData.pad[PAD_B] || splitData.pad[PAD_T] || splitData.pad[PAD_D] || splitData.pad[PAD_L] || splitData.pad[PAD_R]) {
    tilingKey = PADDING_TILING_KEY;
  }
  return tilingKey;
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoBufferCalculate() {
  // for float32/bfloat16, transpose takes 16x8 as smallest pattern,
  // for float16, transpose takes 16x16 as smallest pattern,
  // for input nc -> 16 (num of elements), wx -> 32( byte, 2x16, 4x8)
  // for output wy -> 16 (num of elements), nc -> 32(byte, already aligned in input)

  bufferData.outputMaxPoolSize = bufferData.ncFactorAlign * bufferData.tmpDyFactor * bufferData.tmpHyFactor * bufferData.tmpWyFactorAlign * bufferData.inputCastBytes;
  bufferData.outputMaxPoolSize = CeilAlign(bufferData.outputMaxPoolSize, CACHE_ALIGN_SIZE);
  // different when dtype is float16
  bufferData.outputIndicePoolSize = bufferData.ncMaxFactor * bufferData.tmpDyFactor * bufferData.tmpHyFactor * bufferData.tmpWyFactorAlign * INT32_SIZE;
  bufferData.outputIndicePoolSize = CeilAlign(bufferData.outputIndicePoolSize, CACHE_ALIGN_SIZE);
  bufferData.tmpDxFactor = (bufferData.tmpDyFactor - 1) * inputData.stride[D_DIM] + inputData.kernelSize[D_DIM];
  bufferData.tmpHxFactor = (bufferData.tmpHyFactor - 1) * inputData.stride[H_DIM] + inputData.kernelSize[H_DIM];
  bufferData.tmpWxFactorAlign = CeilAlign((bufferData.tmpWyFactor - 1) * inputData.stride[W_DIM] + inputData.kernelSize[W_DIM], BLOCK_SIZE / inputBytes);
  bufferData.inputPoolSize = bufferData.ncFactorAlign * bufferData.tmpDxFactor * bufferData.tmpHxFactor * bufferData.tmpWxFactorAlign * bufferData.inputCastBytes;
  bufferData.inputPoolSize = CeilAlign(bufferData.inputPoolSize, CACHE_ALIGN_SIZE);
  bufferData.indiceTempPoolSize = bufferData.ncMaxFactor * bufferData.tmpWyFactorAlign * INT32_SIZE;
  bufferData.indiceTempPoolSize = CeilAlign(bufferData.indiceTempPoolSize, CACHE_ALIGN_SIZE);
  bufferData.maskPoolSize = CeilAlign(bufferData.blockDataNum * BLOCK_COUNT / MASK_DTYPE_BIT * MASK_DTYPE_SIZE, BLOCK_SIZE) * bufferData.tmpWyFactorAlign;
  bufferData.maskPoolSize = CeilAlign(bufferData.maskPoolSize, CACHE_ALIGN_SIZE);
  bufferData.tmpTotalSize = bufferData.inputPoolSize * NUM_OF_INPUT_PARTS + bufferData.outputMaxPoolSize * NUM_OF_OUTPUT_MAX_PARTS + bufferData.outputIndicePoolSize * NUM_OF_OUTPUT_INDICES_PARTS + bufferData.indiceTempPoolSize * NUM_OF_TEMP_INDICES_PARTS + bufferData.maskPoolSize * NUM_OF_MASK_PARTS;
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoBoundaryAdjustment() {
  if (bufferData.tmpTotalSize < UB_SIZE) {
    if (bufferData.lbWyFactor < bufferData.ubWyFactor) {
      bufferData.lbWyFactor = bufferData.tmpWyFactor;
    } else if (bufferData.lbHyFactor < bufferData.ubHyFactor) {
      bufferData.lbHyFactor = bufferData.tmpHyFactor;
    } else {
      bufferData.lbDyFactor = bufferData.tmpDyFactor;
    }
  } else {
    // for the wy restriction, when upper bound needs adjusting,
    // need to adjust in the d, h, w order.
    if (bufferData.lbDyFactor < bufferData.ubDyFactor) {
      bufferData.ubDyFactor = (bufferData.lbDyFactor + bufferData.ubDyFactor) / MID_FACTOR;
    } else if (bufferData.lbHyFactor < bufferData.ubHyFactor) {
      bufferData.ubHyFactor = (bufferData.lbHyFactor + bufferData.ubHyFactor) / MID_FACTOR;
    } else {
      bufferData.ubWyFactor = bufferData.tmpWyFactor - 1;
    }
  }
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoUBTiling() {
  // for float32/bfloat16, transpose takes 16x8 as smallest pattern,
  // for float16, transpose takes 16x16 as smallest pattern,
  // for input nc -> 16 (num of elements), wx -> 32( byte, 2x16, 4x8)
  // for output wy -> 16 (num of elements), nc -> 32(byte, already aligned in input)

  while (bufferData.lbDyFactor * bufferData.lbHyFactor * bufferData.lbWyFactor < bufferData.ubDyFactor * bufferData.ubHyFactor * bufferData.ubWyFactor) {
    // for left boundary, we update in the order of w -> h -> d, for we need to maximize w first, then increase h.
    // the order is reversed for right boundary, decrease in the order of d -> h -> w.
    if (bufferData.lbWyFactor < bufferData.ubWyFactor) {
      bufferData.tmpWyFactor = (bufferData.lbWyFactor + bufferData.ubWyFactor + 1) / MID_FACTOR;
      bufferData.tmpWyFactorAlign = CeilAlign(bufferData.tmpWyFactor, TRANS_ALIGN);
    } else if (bufferData.lbHyFactor < bufferData.ubHyFactor) {
      bufferData.tmpHyFactor = (bufferData.lbHyFactor + bufferData.ubHyFactor + 1) / MID_FACTOR;
    } else {
      bufferData.tmpDyFactor = (bufferData.lbDyFactor + bufferData.ubDyFactor + 1) / MID_FACTOR;
    }
    DoBufferCalculate();
    if (bufferData.tmpTotalSize <= UB_SIZE && bufferData.tmpTotalSize > UB_SIZE - TWO_KB_SIZE) {
      // the total size has reached a rather optimal range.
      break;
    }
    DoBoundaryAdjustment();
  }

  // when binary search terminates, it cannot guarantee that the finishing state is the optimal allocation.
  // even so, it is very close to the optimum state, so we reduce the value in d -> h -> w order one by one
  // till we reach the optimal point.
  while (bufferData.tmpTotalSize > UB_SIZE) {
    if (bufferData.tmpDyFactor != 1) {
      bufferData.tmpDyFactor -= 1;
    } else if (bufferData.tmpHyFactor != 1) {
      bufferData.tmpHyFactor -= 1;
    } else {
      bufferData.tmpWyFactor -= 1;
      bufferData.tmpWyFactorAlign = CeilAlign(bufferData.tmpWyFactor, TRANS_ALIGN);
    }
    DoBufferCalculate();
  }

  bufferData.inputBufferSize = bufferData.inputPoolSize;
  bufferData.outputMaxBufferSize = bufferData.outputMaxPoolSize;
  bufferData.outputIndiceBufferSize = bufferData.outputIndicePoolSize;
  bufferData.indiceTempBufferSize = bufferData.indiceTempPoolSize;
  bufferData.maskBufferSize = bufferData.maskPoolSize;
  // calc w dimension
  splitData.wyFactor = bufferData.tmpWyFactor;
  splitData.wyTail = splitData.outShape[W_DIM] % splitData.wyFactor;
  splitData.wyTail = splitData.wyTail == 0 ? splitData.wyFactor : splitData.wyTail;
  splitData.wyOuter = (splitData.outShape[W_DIM] + splitData.wyFactor - 1) / splitData.wyFactor;

  // calc h dimension
  splitData.hyFactor = bufferData.tmpHyFactor;
  splitData.hyTail = splitData.outShape[H_DIM] % splitData.hyFactor;
  splitData.hyTail = splitData.hyTail == 0 ? splitData.hyFactor : splitData.hyTail;
  splitData.hyOuter = (splitData.outShape[H_DIM] + splitData.hyFactor - 1) / splitData.hyFactor;

  // calc d dimension
  splitData.dyFactor = bufferData.tmpDyFactor;
  splitData.dyFactor = splitData.dyFactor == 0 ? 1 : splitData.dyFactor;
  splitData.dyTail = splitData.outShape[D_DIM] % splitData.dyFactor;
  splitData.dyTail = splitData.dyTail == 0 ? splitData.dyFactor : splitData.dyTail;
  splitData.dyOuter = (splitData.outShape[D_DIM] + splitData.dyFactor - 1) / splitData.dyFactor;
  return;
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoBlockTiling() {
  // cal baseBlock nums
  splitData.totalIdx = splitData.ncOuter * splitData.dyOuter * splitData.hyOuter * splitData.wyOuter;
  //
  splitData.blockFactor = (splitData.totalIdx + coreNum - 1) / coreNum;
  splitData.blockTail = splitData.totalIdx % splitData.blockFactor;
  splitData.blockTail = splitData.blockTail == 0 ? splitData.blockFactor : splitData.blockTail;
  splitData.coreNums = (splitData.totalIdx + splitData.blockFactor - 1) / splitData.blockFactor;
  return;
}

void MaxPool3DWithArgmaxV2NoExpandIndicesTiling::SetTilingData() {
  tiling.set_nc(inputData.batches);
  tiling.set_dx(inputData.inputShape[D_DIM]);
  tiling.set_hx(inputData.inputShape[H_DIM]);
  tiling.set_wx(inputData.inputShape[W_DIM]);
  tiling.set_kd(inputData.kernelSize[D_DIM]);
  tiling.set_kh(inputData.kernelSize[H_DIM]);
  tiling.set_kw(inputData.kernelSize[W_DIM]);
  tiling.set_sd(inputData.stride[D_DIM]);
  tiling.set_sh(inputData.stride[H_DIM]);
  tiling.set_sw(inputData.stride[W_DIM]);
  tiling.set_pf(splitData.pad[PAD_F]);
  tiling.set_pb(splitData.pad[PAD_B]);
  tiling.set_pt(splitData.pad[PAD_T]);
  tiling.set_pd(splitData.pad[PAD_D]);
  tiling.set_pl(splitData.pad[PAD_L]);
  tiling.set_pr(splitData.pad[PAD_R]);
  tiling.set_dy(splitData.outShape[D_DIM]);
  tiling.set_hy(splitData.outShape[H_DIM]);
  tiling.set_wy(splitData.outShape[W_DIM]);
  tiling.set_ncFactor(splitData.ncFactor);
  tiling.set_ncTail(splitData.ncTail);
  tiling.set_ncOuter(splitData.ncOuter);
  tiling.set_dyFactor(splitData.dyFactor);
  tiling.set_dyTail(splitData.dyTail);
  tiling.set_dyOuter(splitData.dyOuter);
  tiling.set_hyFactor(splitData.hyFactor);
  tiling.set_hyTail(splitData.hyTail);
  tiling.set_hyOuter(splitData.hyOuter);
  tiling.set_wyFactor(splitData.wyFactor);
  tiling.set_wyTail(splitData.wyTail);
  tiling.set_wyOuter(splitData.wyOuter);
  tiling.set_blockFactor(splitData.blockFactor);
  tiling.set_blockTail(splitData.blockTail);
  tiling.set_totalIdx(splitData.totalIdx);
  tiling.set_coreNums(splitData.coreNums);
  tiling.set_inputBufferSize(bufferData.inputBufferSize);
  tiling.set_outputMaxBufferSize(bufferData.outputMaxBufferSize);
  tiling.set_outputIndiceBufferSize(bufferData.outputIndiceBufferSize);
  tiling.set_indiceTempBufferSize(bufferData.indiceTempBufferSize);
  tiling.set_maskBufferSize(bufferData.maskBufferSize);
}

ge::graphStatus MaxPool3DWithArgmaxV2NoExpandIndicesTiling::DoOpTiling() {
  DoUBTiling();
  DoBlockTiling();
  SetTilingData();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DWithArgmaxV2NoExpandIndicesTiling::PostTiling() {
  context_->SetBlockDim(tiling.get_coreNums());
  tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaxPool3DWithArgmaxV2", MaxPool3DWithArgmaxV2NoExpandIndicesTiling, 0);

}  // namespace optiling
