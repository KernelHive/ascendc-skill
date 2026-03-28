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
 * \file stft_tiling_base.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_STFT_BASE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_STFT_BASE_H_

#include "stft_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(STFTPlanTilingData)
TILING_DATA_FIELD_DEF(uint32_t, oneRowLen);
TILING_DATA_FIELD_DEF(uint32_t, totalLine);
TILING_DATA_FIELD_DEF(uint32_t, tailLine);
TILING_DATA_FIELD_DEF(uint32_t, ubMaxLine);
TILING_DATA_FIELD_DEF(uint32_t, totalInCol);
TILING_DATA_FIELD_DEF(uint32_t, tailInCol);
TILING_DATA_FIELD_DEF(uint32_t, tailBlockIdx);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, matmulM);
TILING_DATA_FIELD_DEF(uint32_t, matmulN);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(STFTPlanTilingDataOp, STFTPlanTilingData);


BEGIN_TILING_DATA_DEF(STFTTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, nfft);
TILING_DATA_FIELD_DEF(uint32_t, hop);
TILING_DATA_FIELD_DEF(uint32_t, frameCount);
TILING_DATA_FIELD_DEF(uint32_t, frameCountAlign);
TILING_DATA_FIELD_DEF(uint32_t, blkFrame);
TILING_DATA_FIELD_DEF(uint32_t, matmulM);
TILING_DATA_FIELD_DEF(uint32_t, sizePerRepeat);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, aicBatchFactor);
TILING_DATA_FIELD_DEF(uint32_t, aicBatchLoop);
TILING_DATA_FIELD_DEF(uint32_t, aicTotalLen);
TILING_DATA_FIELD_DEF(uint32_t, aicTailLen);
TILING_DATA_FIELD_DEF(uint32_t, aicMatmulMCore);
TILING_DATA_FIELD_DEF(uint32_t, aivBatchLoop);
TILING_DATA_FIELD_DEF(uint32_t, aivTotalEvenRow);
TILING_DATA_FIELD_DEF(uint32_t, aivTotalOddRow);
TILING_DATA_FIELD_DEF(uint32_t, aivTailEvenRow);
TILING_DATA_FIELD_DEF(uint32_t, aivTailOddRow);
TILING_DATA_FIELD_DEF(uint32_t, aivWindowLoop);
TILING_DATA_FIELD_DEF(uint32_t, aivGatherUbGap);
TILING_DATA_FIELD_DEF(uint32_t, aicBatchTailIdx);
TILING_DATA_FIELD_DEF(uint32_t, aivBatchTailIdx);
TILING_DATA_FIELD_DEF(uint32_t, aicMTailIdx);
TILING_DATA_FIELD_DEF(uint32_t, aivMTailIdx);
TILING_DATA_FIELD_DEF(uint32_t, aicTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, aivTailLoop);
TILING_DATA_FIELD_DEF(uint32_t, reservedData);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(STFT, STFTTilingData);

BEGIN_TILING_DATA_DEF(STFTGeneralizedTilingData)
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, inputSize);
TILING_DATA_FIELD_DEF(uint32_t, nfft);
TILING_DATA_FIELD_DEF(uint32_t, nfftAlign);
TILING_DATA_FIELD_DEF(uint32_t, hopLength);
TILING_DATA_FIELD_DEF(uint32_t, matmulM);  // nfft or nfft / 2 + 1
TILING_DATA_FIELD_DEF(uint32_t, matmulN);  // windows count
TILING_DATA_FIELD_DEF(uint32_t, normalized);
TILING_DATA_FIELD_DEF(float, rootNfft);
TILING_DATA_FIELD_DEF(uint32_t, copyUBSize);  // UB used in DataCopy in split window
TILING_DATA_FIELD_DEF(uint32_t, splitWindowSkipNum);
TILING_DATA_FIELD_DEF(uint32_t, maskUBSize);  // size of gather mask in UB
// number of cores batches are splited to
TILING_DATA_FIELD_DEF(uint32_t, batchCoreNum);
// number of batchs splited to each core, could be batchCoreFactor or batchCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, batchCoreFactor);
// numbef of cores on which batchs number is batchCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, batchTailCoreNum);
// number of cores matmulM are splited to
TILING_DATA_FIELD_DEF(uint32_t, matmulMCoreNum);
// number of matmulM splited to each core, could be matmulMCoreFactor or matmulMCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, matmulMCoreFactor);
// numbef of cores on which matmulM number is matmulMCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, matmulMTailCoreNum);
// number of cores matmulN are splited to
TILING_DATA_FIELD_DEF(uint32_t, matmulNCoreNum);
// number of matmulN splited to each core, could be matmulNCoreFactor or matmulNCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, matmulNCoreFactor);
// numbef of cores on which matmulN number is matmulMCoreFactor - 1
TILING_DATA_FIELD_DEF(uint32_t, matmulNTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, nFactorUbFormer);
TILING_DATA_FIELD_DEF(uint32_t, nFactorUbTail);
TILING_DATA_FIELD_DEF(uint32_t, nFactorUbLoop);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(STFTPlanTilingData, planTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(STFT_1, STFTGeneralizedTilingData);
REGISTER_TILING_DATA_CLASS(STFT_2, STFTGeneralizedTilingData);
REGISTER_TILING_DATA_CLASS(STFT_3, STFTGeneralizedTilingData);

class STFTBaseTiling : public TilingBaseClass {
 public:
  explicit STFTBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {
  }

 protected:
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;

 protected:
  int64_t ubSize{0};
  int64_t l1Size{0};
  int64_t l0ASize{0};
  int64_t l0BSize{0};
  int64_t l0CSize{0};
  int64_t sysWorkspaceSize{0};
  int32_t batch{0};
  int32_t inputSize{0};
  int64_t winLength{0};
  int64_t nfft{0};
  int64_t hop{0};
  int64_t coreNum{0};
  int64_t aivCoreNum{0};
  int64_t aicCoreNum{0};
  bool normalized{true};
  bool onesided{true};
  bool returnComplex{true};
  ge::DataType dtype;
};
}  // namespace optiling

#endif
