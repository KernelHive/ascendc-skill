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
 * \file dequant_swiglu_quant_tiling.h
 * \brief
 */
#ifndef DEQUANT_SWIGLU_QUANT_TILING_H
#define DEQUANT_SWIGLU_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {
const int64_t STATIC_FLOAT16_X = 10000;
const int64_t STATIC_BFLOAT16_X = 10001;
const int64_t STATIC_FLOAT16_XD = 10002;
const int64_t STATIC_BFLOAT16_XD = 10003;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_ONE = 10004;
const int64_t STATIC_INT_X_INT_BIAS_QUANT_D = 10005;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 10006;
const int64_t STATIC_INT_X_FLOAT16_BIAS_QUANT_D = 10007;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 10008;
const int64_t STATIC_INT_X_FLOAT32_BIAS_QUANT_D = 10009;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 10010;
const int64_t STATIC_INT_X_BFLOAT16_BIAS_QUANT_D = 10011;

const int64_t DYNAMIC_FLOAT16_X = 30009;
const int64_t DYNAMIC_BFLOAT16_X = 30011;
const int64_t DYNAMIC_FLOAT16_XD = 30010;
const int64_t DYNAMIC_BFLOAT16_XD = 30012;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_ONE = 30001;
const int64_t DYNAMIC_INT_X_INT_BIAS_QUANT_D = 30005;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_ONE = 30003;
const int64_t DYNAMIC_INT_X_FLOAT16_BIAS_QUANT_D = 30007;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_ONE = 30002;
const int64_t DYNAMIC_INT_X_FLOAT32_BIAS_QUANT_D = 30006;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_ONE = 30004;
const int64_t DYNAMIC_INT_X_BFLOAT16_BIAS_QUANT_D = 30008;

BEGIN_TILING_DATA_DEF(DequantSwigluQuantBaseTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_0, DequantSwigluQuantBaseTilingData)
REGISTER_TILING_DATA_CLASS(DequantSwigluQuant_1, DequantSwigluQuantBaseTilingData)

BEGIN_TILING_DATA_DEF(SwiGluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
    TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
    TILING_DATA_FIELD_DEF(uint64_t, rowLen);
    TILING_DATA_FIELD_DEF(uint64_t, colLen);
    TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);
    TILING_DATA_FIELD_DEF(uint32_t, baseColLen);
    TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
    TILING_DATA_FIELD_DEF(uint32_t, biasIsEmpty);
    TILING_DATA_FIELD_DEF(uint32_t, quantScaleIsEmpty);
    TILING_DATA_FIELD_DEF(uint32_t, activateScaleIsEmpty);
    TILING_DATA_FIELD_DEF(uint64_t, swiColLen);
    TILING_DATA_FIELD_DEF(uint64_t, perRowLen);
    TILING_DATA_FIELD_DEF(uint64_t, modRowLen);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSwigluQuant, SwiGluTilingData)

struct DequantSwigluQuantCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

class DequantSwigluQuantDskTiling : public TilingBaseClass {
 public:
  explicit DequantSwigluQuantDskTiling(gert::TilingContext* context_) : TilingBaseClass(context_) {
  }
  ~DequantSwigluQuantDskTiling() override {
  }
  uint64_t coreNum_ = 0;
  uint64_t ubSize_ = 0;
  int64_t groupNum_ = 0;
  int64_t actRight_ = 0;
  int64_t quantMode_ = 0;
  uint64_t workspaceSize_ = 0;
  int64_t maxPreCore_ = 0;
  bool hasWeightScale_ = false;
  bool hasActivationScale_ = false;
  bool hasBias_ = false;
  bool hasQuantScale_ = false;
  bool hasQuantOffset_ = false;
  bool hasGroupIndex_ = false;

 protected:
  bool IsCapable() override;
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoOpTiling() override;
  ge::graphStatus DoLibApiTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus GetWorkspaceSize() override;
  ge::graphStatus PostTiling() override;
  void DumpTilingInfo() override;
  ge::graphStatus GetAttr();
  ge::graphStatus CheckDtype();
  ge::graphStatus CheckForModeDynamic();
  bool IsDSKCase();

 private:
  uint64_t tilingKey_ = 0;
  DequantSwigluQuantBaseTilingData tilingData_;
  int64_t inDimx_ = 0;
  int64_t inDimy_ = 0;
  int64_t outDimy_ = 0;
};

template<typename T>
inline auto AlignUp(T num, T rnd) -> decltype(num)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}
// align num to multiples of rnd, round down
template<typename T>
inline auto AlignDown(T num, T rnd) -> decltype(num)
{
    return ((((rnd) == 0) || ((num) < (rnd))) ? 0 : ((num) / (rnd) * (rnd)));
}

template<typename T>
inline auto DivCeil(T num, T div) -> decltype(num)
{
    return (((div) == 0) ? 0 : (((num) + (div) - 1) / (div)));
}

inline bool GetLengthByType(int32_t dtype, uint32_t& dsize)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            dsize = sizeof(int16_t);
            return true;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            dsize = sizeof(int32_t);
            return true;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            dsize = sizeof(int64_t);
            return true;
        default:
            return false;
    }
}

}  // namespace optiling
#endif  // DEQUANT_SWIGLU_QUANT_TILING_H
