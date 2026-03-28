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
 * \file dx_cache_tiling_impl.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DX_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DX_CACHE_TILING_IMPL_H_

#include "cube/algorithm/cache_tiling_impl.h"
#include "cube/algorithm/calculator/conv2d_bp_input_cycle_calculator.h"
#include "cube/algorithm/calculator/conv2d_bp_input_block_dims_calculator.h"
#include "cube/algorithm/calculator/conv2d_bp_input_l0_calculator.h"
#include "cube/algorithm/calculator/conv2d_bp_input_l1_calculator.h"
#include "cube/algorithm/calculator/conv2d_bp_input_ub_calculator.h"

namespace optiling {
namespace cachetiling {
enum TilingIdOffset : uint32_t {
  kBinaryModeOffset = 1,
  kLoad3dSpecialOffset = 3,
  kConv1dFlagOffset = 5,
  kBl1AttachFlagOffset = 6,
  kAl1AttachFlagOffset = 8,
  kAbkl1AttachFlagOffset = 10,
  kDbCubOffset = 12,
  kDbL0cOffset = 13,
  kDbBl1Offset = 14,
  kDbAl1Offset = 15,
  kExtendTilingOffset = 16,
  kMinKl1DivKl0Is1Offset = 18,
  kSplitWOffSet = 19,
  kGroupsGt1Offset = 20,
};

struct AttachFlag
{
  bool condition;
  int32_t al1_attach_flag;
  int32_t bl1_attach_flag;
  int32_t abkl1_attach_flag;
};

class DxCacheTilingImpl : public CacheTilingImpl {
 public:
  explicit DxCacheTilingImpl(const CubeTilingParam &params);
  ~DxCacheTilingImpl() override{};
  bool GenTiling(CubeTiling &tiling) override;
  bool Init(const CubeTilingParam &params) override;
  void Clear() override;

 private:
  void SetOrigShape();
  void CheckSpecialTemplate();
  void CalcTilingId(Conv2DBpInputTiling &tiling) const;
  void SetTiling(Conv2DBpInputTiling &tiling) const;
  void GetAttachFlag(const Conv2DBpInputTiling &tiling, int32_t &al1_attach_flag,
                     int32_t &bl1_attach_flag, int32_t &abkl1_attach_flag) const;

  int32_t h_num_ = 0;
  const Conv2DBpInputTilingParam *dx_param_ = nullptr;
  Conv2DBpInputCycleCalculator cycle_calculator_;
  Conv2DBpInputBlockDimsCalculator block_dims_calculator_;
  Conv2DBpInputL0Calculator l0_calculator_;
  Conv2DBpInputL1Calculator l1_calculator_;
  Conv2DBpInputUbCalculator ub_calculator_;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_DX_CACHE_TILING_IMPL_H_
