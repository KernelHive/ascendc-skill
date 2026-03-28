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
 * \file ascend_quant_v2_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ASCEND_QUANT_V2_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ASCEND_QUANT_V2_H

#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AscendQuantV2TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numCore);
  TILING_DATA_FIELD_DEF(int32_t, blockAxis);
  TILING_DATA_FIELD_DEF(int32_t, ubAxis);
  TILING_DATA_FIELD_DEF(int64_t, dim0);
  TILING_DATA_FIELD_DEF(int64_t, dim1);
  TILING_DATA_FIELD_DEF(int64_t, dim2);
  TILING_DATA_FIELD_DEF(int64_t, blockUnion);
  TILING_DATA_FIELD_DEF(int64_t, blockFactor);
  TILING_DATA_FIELD_DEF(int64_t, blockTailFactor);
  TILING_DATA_FIELD_DEF(int64_t, baseN);
  TILING_DATA_FIELD_DEF(int64_t, baseLen);
  TILING_DATA_FIELD_DEF(int16_t, hasOffset);
  TILING_DATA_FIELD_DEF(int16_t, sqrtMode);
  TILING_DATA_FIELD_DEF(int16_t, roundMode);
  TILING_DATA_FIELD_DEF(int16_t, reserve1);

  END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendQuantV2, AscendQuantV2TilingData)

namespace ascendquantv2 {
enum class RoundMode {
  MODE_ROUND = 0,
  MODE_FLOOR = 1,
  MODE_CEIL = 2,
  MODE_TRUNC = 3,
  MODE_UNDEFINED = -1,
};

enum class QuantKey {
  KEY_PER_CHANNEL = 0,
  KEY_PER_TENSOR = 1,
  KEY_PER_HEAD = 2,
};

class AscendQuantV2 {
public:
  explicit AscendQuantV2(gert::TilingContext* context) : context_(context){};
  ge::graphStatus DoAscendQuantV2Tiling();

protected:
  ge::graphStatus GetCompileInfo();
  ge::graphStatus GetOpParam();
  ge::graphStatus CheckInputValid(const gert::Shape& input1, const gert::Shape& input2, int32_t axis) const;
  ge::graphStatus CheckAttrs(const gert::Shape& xShape);
  RoundMode GetRoundMode(std::string &roundMode);
  void MergeInputShape(const gert::Shape& input);
  uint32_t GetCoreNum(int64_t factor, int64_t coreNum) const;
  uint32_t GetCoreNumDoubleCut(int64_t shape0, int64_t shape1, int64_t coreNum) const;
  void CalcBlockFactor(int64_t size);
  void CalcBlockFactorDoubleCut(int64_t shape0, int64_t size);
  void CalcUBFactor(int64_t cacheLineNum);
  void CalcUBFactorDoubleCut(int64_t cacheLineNum);
  int64_t GetNewShape0(int64_t shape0, int64_t shape1);
  void CalcTiling();
  void CalcTilingKey();
  int64_t CalcMaxBaseLen(int64_t ubSize) const;
  int64_t CalcMaxN(int64_t ubSize, int64_t base) const;
  void WriteTilingData();
  void CalcPerHeadTiling();
  void CalcPerHeadBlockFactor();
  void CalcPerHeadUBFactor(int64_t cacheLineNum);
  ge::graphStatus CheckSupport310P();
  ge::graphStatus CheckShapeEqual(const gert::Shape& shape1, const gert::Shape& shape2) const;

private:
  gert::TilingContext* context_ = nullptr;
  AscendQuantV2TilingData tilingData_;

  uint32_t coreNum_{0};
  uint64_t ubSize_{0};
  bool isAscend910B_{false};
  int64_t reserveUb_{2048};
  int64_t cacheLine_{256};

  gert::Shape xInputShape_;
  ge::DataType xDtype_{ge::DT_UNDEFINED};
  ge::DataType yDtype_{ge::DT_UNDEFINED};
  bool hasOffset_{true};
  bool useDoubleCut{false};
  int16_t sqrtMode_ = 0;
  RoundMode roundMode_ = RoundMode::MODE_UNDEFINED;
  int32_t dstType_ = 0;
  int32_t axis_{-1};

  uint32_t actCoreNum_{0};
  int64_t blockUnion_{1};
  int64_t blockBind_{1};
  int32_t blockAxis_{-1};
  int64_t blockFactor_{-1};
  int64_t blockTailFactor_{-1};
  int32_t ubAxis_{-1};
  int64_t ubFactor_{-1};
  int64_t baseN_{1};
  int64_t baseLen_{1};
  uint64_t tilingKey_{0};
  bool isPerTensor_{false};
  bool isPerHead_{false};
};
} // namespace ascendquantv2
} // namespace optiling
#endif