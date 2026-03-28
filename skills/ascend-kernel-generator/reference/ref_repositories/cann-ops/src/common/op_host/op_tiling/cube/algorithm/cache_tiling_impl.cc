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
 * \file cache_tiling_impl.cc
 * \brief
 */
#include "cube/algorithm/cache_tiling_impl.h"

#include "cube/util/cube_util.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {
CacheTilingImpl::CacheTilingImpl(const CubeTilingParam &params) : params_(&params) {}

bool CacheTilingImpl::Init(const CubeTilingParam &params) {
  params_ = &params;
  single_core_status_.Init();
  return true;
}

void CacheTilingImpl::Clear() {
  params_ = nullptr;
}

void CacheTilingImpl::UpdateTiling(CubeTiling &tiling) const {
  UpdateTilingBlockDims(tiling);
  UpdateTilingL1Status(tiling);
  UpdateTilingL0Status(tiling);
  UpdateTilingUbStatus(tiling);
}

void CacheTilingImpl::UpdateTilingBlockDims(CubeTiling &tiling) const {
  const DimFactor &block_dims = single_core_status_.block_dims();
  tiling.batch_dim = block_dims.batch;
  tiling.n_dim = block_dims.n;
  tiling.k_dim = block_dims.k;
  tiling.m_dim = block_dims.m;
  tiling.group_dim = block_dims.group;
}

void CacheTilingImpl::UpdateTilingL1Status(CubeTiling &tiling) const {
  const L1Status &l1_status = single_core_status_.l1_status();
  tiling.m_al1 = l1_status.m_al1;
  tiling.k_al1 = l1_status.k_al1;
  tiling.k_bl1 = l1_status.k_bl1;
  tiling.n_bl1 = l1_status.n_bl1;
  tiling.db_al1 = l1_status.db_al1;
  tiling.db_bl1 = l1_status.db_bl1;
  tiling.ho_bl1 = l1_status.ho;
  tiling.al1_bound = l1_status.al1_bound;
  tiling.bl1_bound = l1_status.bl1_bound;
}

void CacheTilingImpl::UpdateTilingL0Status(CubeTiling &tiling) const {
  const L0Status &l0_status = single_core_status_.l0_status();
  tiling.m_l0 = l0_status.m;
  tiling.k_l0 = l0_status.k;
  tiling.n_l0 = l0_status.n;
  tiling.db_l0c = l0_status.db_l0c;
}

void CacheTilingImpl::UpdateTilingUbStatus(CubeTiling &tiling) const {
  const UbStatus &ub_status = single_core_status_.ub_status();
  tiling.n_cub = ub_status.n_cub;
  tiling.db_cub = ub_status.db_cub;
  tiling.k_aub = ub_status.k_aub;
  tiling.m_aub = ub_status.m_aub;
  tiling.db_aub = ub_status.db_aub;
  tiling.k_bub = ub_status.k_bub;
  tiling.n_bub = ub_status.n_bub;
  tiling.db_bub = ub_status.db_bub;
}

DEFINE_REGISTRY_TYPE(CacheTilingFactory, CacheTilingImpl, const CubeTilingParam &);
}  // namespace cachetiling
}  // namespace optiling