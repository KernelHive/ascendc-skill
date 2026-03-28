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
 * \file dw_cache_tiling_impl.cc
 * \brief
 */
#include "cube/algorithm/conv3d_dw_cache_tiling_impl.h"
#include "cube/algorithm/calculator/conv3d_dw_cycle_calculator.h"

namespace optiling {
namespace cachetiling {
namespace {
enum TilingIdOffset : uint32_t {
  kBinaryModeOffset = 0,
  kLoadModeOffset = 2,
  kConv1dFlagOffset = 4,
  kLoad3dSpecialOffset = 5,
  kMinKl1CmpKl0Offset = 6,
  kBl1AttachOffset = 7,
  kAl1AttachOffset = 9,
  kAbkl1AttachOffset = 11,
  kDbL0cOffset = 13,
  kDbBl1Offset = 14,
  kDbAl1Offset = 15,
  kStridehReadFlagOffset = 16,
  kLinearEmbeddingOptiFlagOffset = 17,
};
}

Conv3DDwCacheTilingImpl::Conv3DDwCacheTilingImpl(const CubeTilingParam &params)
    : DwCacheTilingImpl(params) {}

bool Conv3DDwCacheTilingImpl::Init(const CubeTilingParam &params) {
  (void)CacheTilingImpl::Init(params);

  if (cycle_calculator_ == nullptr) {
    cycle_calculator_ =
        std::unique_ptr<CycleCalculator>(new (std::nothrow) Conv3DDwCycleCalculator(single_core_status_));
    if (cycle_calculator_ == nullptr) {
      return false;
    }
  }

  (void)cycle_calculator_->Init(params);
  return true;
}

void Conv3DDwCacheTilingImpl::SetOrigShape() {
  TilingShape shape;
  shape.batch = params_->a_shape.batch;
  shape.m = params_->a_shape.c1;
  shape.k = MathUtil::Align(params_->a_shape.h * params_->a_shape.w, kBlockSize) / params_->k0;
  shape.n = params_->b_shape.c1 * params_->c_shape.d * params_->kernel_h * params_->kernel_w;
  shape.group = params_->real_g;
  single_core_status_.UpdateOrigShape(shape);
  OP_LOGD(params_->op_type, "[orig_shape][%s]", shape.ToString().c_str());
}

bool Conv3DDwCacheTilingImpl::CheckCycleModelUnsupport() const {
  return false; // all cases go to cycle model
}

void Conv3DDwCacheTilingImpl::SetTiling(CubeTiling &tiling) const {
  UpdateTiling(tiling);

  auto &dw_tiling = reinterpret_cast<Conv3DBpFilterTiling &>(tiling);
  dw_tiling.d_dim = single_core_status_.block_dims().d;

  if (tiling.m_al1 == kNone) {
    tiling.db_al1 = 1;
  }
  if (tiling.n_bl1 == kNone) {
    tiling.db_bl1 = 1;
  }
  OP_LOGD(params_->op_type, "Tiling params [tiling][%s]", tiling.ToString().c_str());
}

int32_t Conv3DDwCacheTilingImpl::CalcTilingId(const CubeTiling &tiling, const TilingIdParam &id_param) const {
  int32_t tiling_id = 0;
  tiling_id += (params_->binary_mode - 1) << kBinaryModeOffset;
  tiling_id += id_param.load_mode() << kLoadModeOffset;
  tiling_id += static_cast<int32_t>(params_->conv1d_flag) << kConv1dFlagOffset;
  tiling_id += (params_->load3d_special - 1) << kLoad3dSpecialOffset;
  tiling_id += id_param.min_kl1_cmp_kl0() << kMinKl1CmpKl0Offset;
  tiling_id += id_param.bl1_attach_flag() << kBl1AttachOffset;
  tiling_id += id_param.al1_attach_flag() << kAl1AttachOffset;
  tiling_id += id_param.abkl1_attach_flag() << kAbkl1AttachOffset;
  tiling_id += (tiling.db_l0c - 1) << kDbL0cOffset;
  tiling_id += (tiling.db_bl1 - 1) << kDbBl1Offset;
  tiling_id += (tiling.db_al1 - 1) << kDbAl1Offset;
  tiling_id += params_->strideh_read_flag << kStridehReadFlagOffset;
  tiling_id += params_->linear_embedding_opti_flag << kLinearEmbeddingOptiFlagOffset;
  OP_LOGD(params_->op_type, "tiling_id %d", tiling_id);
  return tiling_id;
}

REGISTER_TILING_GENERATOR(kConv3DBackpropFilter, Conv3DDwCacheTilingImpl);
REGISTER_TILING_GENERATOR(kConv3DBackpropFilterV2, Conv3DDwCacheTilingImpl);
}  // namespace cachetiling
}  // namespace optiling
