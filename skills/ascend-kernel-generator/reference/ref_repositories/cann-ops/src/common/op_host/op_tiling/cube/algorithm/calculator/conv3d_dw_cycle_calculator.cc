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
 * \file conv3d_dw_cycle_calculator.cc
 * \brief
 */
#include "cube/algorithm/calculator/conv3d_dw_cycle_calculator.h"

#include <algorithm>

#include "cube/util/cube_util.h"
#include "cube/util/math_util.h"

namespace optiling {
namespace cachetiling {
Conv3DDwCycleCalculator::Conv3DDwCycleCalculator(SingleCoreStatus &core_status)
    : CycleCalculator(core_status) {}

bool Conv3DDwCycleCalculator::Init(const CubeTilingParam &params) {
  dout_dims_factors_.clear();
  kd_dims_factors_.clear();
  dout_dims_factors_.reserve(static_cast<size_t>(params.platform_info.core_num()));
  kd_dims_factors_.reserve(static_cast<size_t>(params.platform_info.core_num()));
  return CycleCalculator::Init(params);
}

void Conv3DDwCycleCalculator::GenBlockDimsMapFactors() {
  mapped_shape_.batch = MathUtil::GetNonFactorMap(batch_dims_factors_, orig_shape_.batch, core_num_);
  MathUtil::AddCoreFactor(orig_shape_.batch, core_num_, batch_dims_factors_);
  MathUtil::GetFactors(dout_dims_factors_, params_->a_shape.d, core_num_);

  // m = co1 * co0, only cut co1
  mapped_shape_.m = MathUtil::GetNonFactorMap(m_dims_factors_, params_->a_shape.c1, core_num_);
  MathUtil::AddCoreFactor(params_->a_shape.c1, core_num_, m_dims_factors_);

  // if kd > 1, not enable non factor to split cin1_g
  if (params_->c_shape.d == 1) {
    mapped_shape_.n = MathUtil::GetNonFactorMap(n_dims_factors_, params_->b_shape.c1, core_num_);
    MathUtil::AddCoreFactor(params_->b_shape.c1, core_num_, n_dims_factors_);
  } else {
    MathUtil::GetFactors(n_dims_factors_, params_->b_shape.c1, core_num_);
  }
  MathUtil::GetFactors(kd_dims_factors_, params_->c_shape.d, core_num_);

  MathUtil::GetFactors(g_dims_factors_, params_->real_g, core_num_);
}

bool Conv3DDwCycleCalculator::IsValidBatchDim(int32_t batch_dim) const {
  if (params_->a_shape.d > 1 && d_dim_ > 1) {
    int64_t batch_single_core = MathUtil::CeilDivision(params_->a_shape.batch, static_cast<int64_t>(batch_dim));
    int64_t dout_single_core = params_->a_shape.d / static_cast<int64_t>(d_dim_);
    int64_t batch_dout_single_core = batch_single_core * dout_single_core;
    if ((batch_dout_single_core > params_->a_shape.d && batch_dout_single_core % params_->a_shape.d != 0) ||
        (batch_dout_single_core < params_->a_shape.d && params_->a_shape.d % batch_dout_single_core != 0) ||
        (params_->type == kConv3DBackpropFilter && batch_dout_single_core > std::numeric_limits<int32_t>::max())) {
      OP_LOGD(params_->op_type, "skip batch_dim: %d, batch: %ld, dout: %ld", batch_dim, params_->a_shape.batch,
              params_->a_shape.d);
      return false;
    }
  }

  return true;
}

void Conv3DDwCycleCalculator::UpdateTilingShape() {
  CycleCalculator::UpdateTilingShape();

  total_n_l0_ = shape_.n * (params_->c_shape.d / kd_dim_);
  total_n_l1_ = total_n_l0_;
  auto conv3d_dw = dynamic_cast<const Conv3DBpFilterTilingParam *>(params_);
  if (conv3d_dw != nullptr && conv3d_dw->kernel_d > 1) {
    // if real_g > 1, then kd can only be 1 in L1
    // if real_g > 1 or D dim pads non zero, then kd can only be 1 in L0
    if (conv3d_dw->real_g > 1 || conv3d_dw->pad_f > 0 || conv3d_dw->pad_b > 0) {
      total_n_l0_ = shape_.n;
      if (conv3d_dw->real_g > 1) {
        total_n_l1_ = total_n_l0_;
      }
    }
  }

  block_dims_.d = d_dim_;
  block_dims_.n *= kd_dim_;
  shape_.batch *= (params_->a_shape.d / d_dim_);
  shape_.n *= (params_->c_shape.d / kd_dim_);
}

bool Conv3DDwCycleCalculator::LoopBlockDims(bool prune, int32_t used_core) {
  for (auto dout_factor : dout_dims_factors_) {
    for (auto kd_factor : kd_dims_factors_) {
      used_core = dout_factor * kd_factor;
      if (IsInvalidFactor(used_core)) {
        break;
      }

      d_dim_ = dout_factor;
      kd_dim_ = kd_factor;
      OP_LOGE_IF(!CycleCalculator::LoopBlockDims(prune, used_core), false, params_->op_type, "LoopBlockDims failed");
    }
  }

  return true;
}
}  // namespace cachetiling
}  // namespace optiling
