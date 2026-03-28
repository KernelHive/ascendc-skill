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
 * \file conv3ddx_cache_tiling_impl.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3DDX_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CONV3DDX_CACHE_TILING_IMPL_H_

#include "cube/algorithm/cache_tiling_impl.h"
#include "cube/algorithm/calculator/conv3d_bp_input_tiling_calculator.h"

namespace optiling {
namespace cachetiling {

class Conv3DDxCacheTilingImpl : public CacheTilingImpl {
 public:
  explicit Conv3DDxCacheTilingImpl(const CubeTilingParam &params);
  ~Conv3DDxCacheTilingImpl() override = default;
  bool GenTiling(CubeTiling &tiling) override;
  bool Init(const CubeTilingParam &params) override;
  void Clear() override;

 private:
  void ShowResourceStatistics() const;
  void SetOrigShape();
  void SetTiling(Conv3DBpInputTiling &tiling) const;

  const Conv3DBpInputTilingParam *conv3ddx_params_ = nullptr;
  Conv3DBpInputTilingCalculator conv3ddx_tiling_calculator_;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_Conv3DDX_CACHE_TILING_IMPL_H_
