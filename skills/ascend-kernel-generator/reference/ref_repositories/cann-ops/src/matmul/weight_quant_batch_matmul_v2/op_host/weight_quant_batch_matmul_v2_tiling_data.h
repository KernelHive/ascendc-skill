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
 * \file weight_quant_batch_matmul_v2_tiling_data.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H


namespace optiling {
// tiling data for custom and splitK
BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2TilingData)
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, vecBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(uint8_t, kPadSize);
TILING_DATA_FIELD_DEF(uint8_t, nPadSize);
TILING_DATA_FIELD_DEF(uint8_t, haveBatchA);
TILING_DATA_FIELD_DEF(uint8_t, haveBatchB);
TILING_DATA_FIELD_DEF(uint8_t, reserve1);
TILING_DATA_FIELD_DEF(uint8_t, reserve2);
TILING_DATA_FIELD_DEF(uint8_t, reserve3);
TILING_DATA_FIELD_DEF(uint16_t, vecSingleKGroupNum);
TILING_DATA_FIELD_DEF(uint16_t, vecSingleKTailGroupNum);
TILING_DATA_FIELD_DEF(uint16_t, AL1Pingpong);
TILING_DATA_FIELD_DEF(uint16_t, BL1Pingpong);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleK);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleN);
TILING_DATA_FIELD_DEF(uint32_t, vec2SingleM);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleKTail);
TILING_DATA_FIELD_DEF(uint32_t, vecSingleNTail);
TILING_DATA_FIELD_DEF(uint32_t, wInQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, offsetInQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, scaleInQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, wOutQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, antiQuantTmpBufferSize);
TILING_DATA_FIELD_DEF(uint32_t, vecCubeNRatio);
TILING_DATA_FIELD_DEF(uint32_t, vecCubeTailNRatio);
TILING_DATA_FIELD_DEF(uint32_t, vecCubeKRatio);
TILING_DATA_FIELD_DEF(uint32_t, vecCubeTailKRatio);
TILING_DATA_FIELD_DEF(uint32_t, cubeTailM);
TILING_DATA_FIELD_DEF(uint32_t, cubeTailN);
TILING_DATA_FIELD_DEF(uint32_t, cubeSingleNLoop);
TILING_DATA_FIELD_DEF(uint32_t, cubeSingleNTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, repeatAxisMax);
TILING_DATA_FIELD_DEF(uint64_t, vecSingleKLoop);
TILING_DATA_FIELD_DEF(uint64_t, vecSingleNLoop);
TILING_DATA_FIELD_DEF(uint64_t, vecSingleKTailLoop);
TILING_DATA_FIELD_DEF(uint64_t, vecSingleNTailLoop);
TILING_DATA_FIELD_DEF(uint64_t, vec2SingleMLoop);

TILING_DATA_FIELD_DEF(uint64_t, kAlign);
TILING_DATA_FIELD_DEF(uint64_t, nAlign);
TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF(uint64_t, mSize);

TILING_DATA_FIELD_DEF(uint64_t, blockBatch);
TILING_DATA_FIELD_DEF(uint64_t, shapeBatch);
TILING_DATA_FIELD_DEF(uint64_t, mAubSize);
TILING_DATA_FIELD_DEF(uint64_t, kAubSize);
TILING_DATA_FIELD_DEF(uint64_t, nBubSize);
TILING_DATA_FIELD_DEF(uint64_t, kBubSize);
TILING_DATA_FIELD_DEF(uint64_t, mCubSize);
TILING_DATA_FIELD_DEF(uint64_t, nCubSize);
TILING_DATA_FIELD_DEF(uint64_t, mAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kAL1Size);
TILING_DATA_FIELD_DEF(uint64_t, nBL1Size);
TILING_DATA_FIELD_DEF(uint64_t, kBL1Size);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2, WeightQuantBatchMatmulV2TilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2TilingDataOp, WeightQuantBatchMatmulV2TilingData)
}  // namespace optiling
#endif  // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_DATA_H

