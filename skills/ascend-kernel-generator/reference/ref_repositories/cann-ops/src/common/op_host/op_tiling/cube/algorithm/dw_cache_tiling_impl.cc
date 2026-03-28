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
#include "cube/algorithm/dw_cache_tiling_impl.h"

#include "cube/util/configuration.h"
#include "cube/util/cube_util.h"
#include "op_log.h"

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
  kStridehReadFlagOffset = 13,
  kLinearEmbeddingOptiFlagOffset = 14,
};
}

void TilingIdParam::Calc(const CubeTilingParam *params, const SingleCoreStatus &status, const CubeTiling &tiling) {
  auto attach_l1_helper_func = [&status](int32_t k_l1, int32_t l1) -> int32_t {
    const TilingShape &shape = status.shape();
    bool k_full_load = shape.batch == 1 && k_l1 == shape.k;
    if (l1 == kNone) {
      return kAttachFullLoad;
    } else if (k_full_load) {
      return kAttachEqual;
    } else {
      return kAttachLess;
    }
  };

  al1_attach_flag_ = attach_l1_helper_func(tiling.k_al1 / params->load3d_special, tiling.m_al1);
  bl1_attach_flag_ = attach_l1_helper_func(tiling.k_bl1 / params->load3d_special, tiling.n_bl1);
  abkl1_attach_flag_ = tiling.k_al1 == tiling.k_bl1  ? kAttachFullLoad
                       : tiling.k_al1 > tiling.k_bl1 ? kAttachEqual
                                                     : kAttachLess;
  load_mode_ = static_cast<int32_t>(params->load2d_flag); // 0: load3d, 1: load2d
  if (params->split_w_flag) {
    load_mode_ = 2; // 2 means load3d w split case
    al1_attach_flag_ = kAttachLess;
    bl1_attach_flag_ = kAttachLess;
  } else if (params->dma_flag) {
    load_mode_ = 3; // 3 means dma_flag case
    if (!params->linear_embedding_opti_flag) {
      al1_attach_flag_ = kAttachLess;
      bl1_attach_flag_ = kAttachLess;
    } else {
      if (al1_attach_flag_ == kAttachEqual ||
          (al1_attach_flag_ == kAttachFullLoad && abkl1_attach_flag_ != kAttachEqual)) {
        al1_attach_flag_ = kAttachLess;
      }
      bl1_attach_flag_ = kAttachLess;
    }
  }

  min_kl1_cmp_kl0_ = std::min(tiling.k_al1, params->load3d_special * tiling.k_bl1) == tiling.k_l0 ? 0 : 1;
}

DwCacheTilingImpl::DwCacheTilingImpl(const CubeTilingParam &params)
    : CacheTilingImpl(params),
      block_dims_calculator_(single_core_status_),
      l0_calculator_(single_core_status_),
      l1_calculator_(single_core_status_),
      ub_calculator_(single_core_status_) {}

bool DwCacheTilingImpl::Init(const CubeTilingParam &params) {
  (void)CacheTilingImpl::Init(params);

  if (cycle_calculator_ == nullptr) {
    cycle_calculator_ = std::unique_ptr<CycleCalculator>(new (std::nothrow) CycleCalculator(single_core_status_));
    if (cycle_calculator_ == nullptr) {
      return false;
    }
  }
  (void)cycle_calculator_->Init(params);
  (void)block_dims_calculator_.Init(params);
  (void)l0_calculator_.Init(params);
  (void)l1_calculator_.Init(params);
  (void)ub_calculator_.Init(params);
  return true;
}

void DwCacheTilingImpl::Clear() {
  CacheTilingImpl::Clear();
  cycle_calculator_->Clear();
  block_dims_calculator_.Clear();
  l0_calculator_.Clear();
  l1_calculator_.Clear();
  ub_calculator_.Clear();
}

void DwCacheTilingImpl::CalcOrigShape(TilingShape &shape) const {
  shape.batch = params_->a_shape.batch;
  shape.m = params_->a_shape.c1;
  // for fp32 scene, k0 is 8, but K align to 16(restricted by load3d)
  if (params_->split_w_flag) {
    // cut ho as k_dim when split w
    shape.k = params_->a_shape.h;
  } else {
    shape.k = MathUtil::Align(params_->a_shape.h * params_->a_shape.w, kBlockSize) / params_->k0;
  }
  shape.n = params_->b_shape.c1 * params_->kernel_h * params_->kernel_w;
  shape.group = params_->real_g;
}

void DwCacheTilingImpl::SetOrigShape() {
  TilingShape shape;
  CalcOrigShape(shape);
  single_core_status_.UpdateOrigShape(shape);
  OP_LOGD(params_->op_type, "[orig_shape][%s]", shape.ToString().c_str());
}

void DwCacheTilingImpl::ShowResourceStatistics() const {
  if (!Configuration::Instance().IsDebugMode()) {
    return;
  }

  ResourceStatistic resource_statistic;
  const DimFactor &block_dims = single_core_status_.block_dims();
  const L1Status &l1_status = single_core_status_.l1_status();
  const L0Status &l0_status = single_core_status_.l0_status();
  const UbStatus &ub_status = single_core_status_.ub_status();

  // calc core used
  resource_statistic.set_core_used(block_dims.ReduceMul());

  // calc l1 buffer
  resource_statistic.set_l1_used(CubeUtil::CalcL1Size(params_, l1_status, l0_status));

  // calc l0 buffer, in fp32 scene fractal in L0A/L0B is 16 * 8
  int32_t l0a_used = l0_status.m * l0_status.k * l0_status.db_l0a * params_->a_dtype_bytes * (kBlockSize * params_->k0);
  int32_t l0b_used = l0_status.k * l0_status.n * l0_status.db_l0b * params_->b_dtype_bytes * (kBlockSize * params_->k0);
  int32_t l0c_used = l0_status.m * l0_status.n * l0_status.db_l0c * params_->c_dtype_bytes * kCubeTileNumSize;
  resource_statistic.set_l0a_used(l0a_used);
  resource_statistic.set_l0b_used(l0b_used);
  resource_statistic.set_l0c_used(l0c_used);

  // calc ub buffer
  int32_t aub_size = 0;
  int32_t bub_size = 0;
  if (params_->binary_mode == kBinaryModeNCHW || params_->binary_mode == kBinaryModeNHWC) {
    int32_t aub_min_size = (params_->aub_fused_num + 1) * kCubeTileNumSize * params_->a_dtype_bytes * ub_status.db_aub;
    aub_size = ub_status.m_aub * ub_status.k_aub * aub_min_size;
    bub_size = (params_->bub_fused_num + 1) * ub_status.n_bub * kBlockSize * params_->b_dtype_bytes *
               MathUtil::Align(ub_status.k_bub * params_->b_shape.w, kBlockSize) * ub_status.db_bub;
  }
  int32_t cub_min_size =
      (params_->cub_fused_num + 1) * l0_status.m * kCubeTileNumSize * params_->c_dtype_bytes * ub_status.db_cub;
  int32_t cub_size = ub_status.n_cub * cub_min_size;
  int32_t ub_used = aub_size + bub_size + cub_size;
  resource_statistic.set_ub_used(ub_used);
  OP_LOGD(params_->op_type, "[ResourceStatistic][%s]", resource_statistic.Show(params_->platform_info).c_str());
}

std::string DwCacheTilingImpl::InputArgsToString(const tuningtiling::Conv2DDwInputArgs& input_args) const {
  std::stringstream ss;
  ss << "a_shape_n: " << input_args.a_shape_n << ", a_shape_h: " << input_args.a_shape_h
     << ", a_shape_w: " << input_args.a_shape_w << ", b_shape_h: " << input_args.b_shape_h
     << ", b_shape_w: " << input_args.b_shape_w << ", c_shape_n: " << input_args.c_shape_n
     << ", c_shape_c: " << input_args.c_shape_c << ", c_shape_h: " << input_args.c_shape_h
     << ", c_shape_w: " << input_args.c_shape_w << ", groups: " << input_args.groups << ", stride_h: "
     << input_args.stride_h << ", stride_w: " << input_args.stride_w << ", dilation_h: " << input_args.dilation_h
     << ", dilation_w: " << input_args.dilation_w << ", pad_u: " << input_args.pad_u << ", pad_d: " << input_args.pad_d
     << ", pad_l: " << input_args.pad_l << ", pad_r: " << input_args.pad_r << ", a_dtype: " << input_args.a_dtype
     << ", b_dtype: " << input_args.b_dtype << ", c_dtype: " << input_args.c_dtype
     << ", binary_mode: " << input_args.binary_mode << ", hf32_flag: " << input_args.hf32_flag
     << ", reserved_params1: " << input_args.reserved_params1 << ", reserved_params2: " << input_args.reserved_params2
     << ", reserved_params3: " << input_args.reserved_params3 << ", reserved_params4: " << input_args.reserved_params4
     << ", reserved_params5: " << input_args.reserved_params5;
  return ss.str();
}

void DwCacheTilingImpl::BuildRepoQueryParams(tuningtiling::Conv2DDwInputArgs &input_args) const {
  input_args.a_shape_n = params_->a_shape.batch;
  input_args.a_shape_h = params_->a_shape.h;
  input_args.a_shape_w = params_->a_shape.w;
  input_args.b_shape_h = params_->b_shape.h;
  input_args.b_shape_w = params_->b_shape.w;
  input_args.c_shape_n = params_->a_shape.c;
  input_args.c_shape_c = params_->b_shape.c;
  input_args.c_shape_h = params_->kernel_h;
  input_args.c_shape_w = params_->kernel_w;
  input_args.groups = params_->groups;
  input_args.stride_h = params_->stride_h;
  input_args.stride_w = params_->stride_w;
  input_args.dilation_h = params_->dilation_h;
  input_args.dilation_w = params_->dilation_w;
  input_args.pad_u = params_->pad_u;
  input_args.pad_d = params_->pad_d;
  input_args.pad_l = params_->pad_l;
  input_args.pad_r = params_->pad_r;
  input_args.a_dtype = params_->a_dtype;
  input_args.b_dtype = params_->b_dtype;
  input_args.c_dtype = params_->c_dtype;
  input_args.binary_mode = params_->binary_mode;
  input_args.hf32_flag = params_->hf32_flag;
  input_args.reserved_params1 = 0;
  input_args.reserved_params2 = 0;
  input_args.reserved_params3 = 0;
  input_args.reserved_params4 = 0;
  input_args.reserved_params5 = 0;

  OP_LOGD(params_->op_type, "info_dict to aoe input_args: %s", InputArgsToString(input_args).c_str());
}

bool DwCacheTilingImpl::TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuning_tiling) {
  auto aoe_dw_tiling = std::dynamic_pointer_cast<tuningtiling::Conv2DDwTunnerTiling>(tuning_tiling);
  if (aoe_dw_tiling == nullptr) {
    OP_LOGE(params_->op_type, "dynamic_pointer_cast TuningTilingDefPtr to Conv2DDwTunnerTiling failed");
    return false;
  }
  DimFactor block_dims(aoe_dw_tiling->batch_dim, aoe_dw_tiling->m_dim, aoe_dw_tiling->k_dim, aoe_dw_tiling->n_dim,
                       aoe_dw_tiling->group_dim);
  single_core_status_.UpdateBlockDims(block_dims);

  L0Status l0_status = single_core_status_.l0_status();
  l0_status.m = aoe_dw_tiling->m_l0;
  l0_status.k = aoe_dw_tiling->k_l0;
  l0_status.n = aoe_dw_tiling->n_l0;
  l0_status.db_l0a = aoe_dw_tiling->db_l0a;
  l0_status.db_l0b = aoe_dw_tiling->db_l0b;
  l0_status.db_l0c = aoe_dw_tiling->db_l0c;
  single_core_status_.UpdateL0Status(l0_status);

  L1Status l1_status = single_core_status_.l1_status();
  l1_status.m_al1 = aoe_dw_tiling->m_al1;
  l1_status.k_al1 = aoe_dw_tiling->k_al1;
  l1_status.k_bl1 = aoe_dw_tiling->k_bl1;
  l1_status.n_bl1 = aoe_dw_tiling->n_bl1;
  l1_status.db_al1 = aoe_dw_tiling->db_al1;
  l1_status.db_bl1 = aoe_dw_tiling->db_bl1;
  // calc bl1_bound
  CubeUtil::CalcL1RemainStatus(params_, l0_status, l1_status);
  single_core_status_.UpdateL1Status(l1_status);

  UbStatus ub_status = single_core_status_.ub_status();
  ub_status.n_cub = aoe_dw_tiling->n_cub;
  ub_status.m_aub = aoe_dw_tiling->m_aub;
  ub_status.k_aub = aoe_dw_tiling->k_aub;
  ub_status.k_bub = aoe_dw_tiling->k_bub;
  ub_status.n_bub = aoe_dw_tiling->n_bub;
  ub_status.db_aub = aoe_dw_tiling->db_aub;
  ub_status.db_bub = aoe_dw_tiling->db_bub;
  ub_status.db_cub = aoe_dw_tiling->db_cub;
  single_core_status_.UpdateUbStatus(ub_status);

  TilingShape shape;
  CalcOrigShape(shape);
  shape.batch =
      MathUtil::CeilDivision(shape.batch, static_cast<int32_t>(aoe_dw_tiling->batch_dim));
  shape.group = shape.group / static_cast<int32_t>(aoe_dw_tiling->group_dim);
  shape.m = MathUtil::CeilDivision(shape.m, static_cast<int32_t>(aoe_dw_tiling->m_dim));
  shape.n = MathUtil::CeilDivision(params_->b_shape.c1, static_cast<int32_t>(aoe_dw_tiling->n_dim)) *
            params_->kernel_h * params_->kernel_w;
  shape.k = MathUtil::CeilDivision(shape.k, static_cast<int32_t>(aoe_dw_tiling->k_dim));
  single_core_status_.UpdateShape(shape);
  return true;
}

bool DwCacheTilingImpl::GetTilingFromRepo() {
  // std::shared_ptr<tuningtiling::TuningTilingDef> repo_tiling = nullptr;
  // RuntimeKb::PlatformInfo plt{.core_num = params_->platform_info.core_num(),
  //                                          .soc_version = params_->platform_info.soc_version()};
  // if (!plt.soc_version.empty()) {
  //   try {
  //     tuningtiling::Conv2DDwInputArgs input_args;
  //     BuildRepoQueryParams(input_args);
  //     (void)RuntimeKb::RuntimeBankManager::Instance().Query(&input_args, sizeof(input_args),
  //                                                           std::string(params_->op_type), plt, repo_tiling);
  //   } catch(...) {
  //     OP_LOGE(params_->op_type, "Query knowledge bank failed");
  //   }
  // } else {
  //   OP_LOGD(params_->op_type, "Get soc_version fail, can't query tiling from repo.");
  // }

  // if (repo_tiling != nullptr) {
  //   return TranslateAoeTiling(repo_tiling);
  // }
  return false;
}

bool DwCacheTilingImpl::CheckCycleModelUnsupport() const {
  return !params_->platform_info.support_l0c2out() || params_->binary_mode != kBinaryModeNC1HWC0 ||
         params_->conv1d_flag || params_->b_shape.c0 == kSmallChannelSize;
}

bool DwCacheTilingImpl::GenTiling(CubeTiling &tiling) {
  SetOrigShape();

  if (GetTilingFromRepo()) {
    OP_LOGI(params_->op_type,
            "Get tiling by repo. Shape[Backprop[%s], Fmap[%s], Output[%s], SingleCoreStatus[%s]",
            params_->a_shape.ToString().c_str(), params_->b_shape.ToString().c_str(),
            params_->c_shape.ToString().c_str(), single_core_status_.ToString().c_str());
  } else {
    if (CheckCycleModelUnsupport()) {
      OP_LOGE_IF(!block_dims_calculator_.Exec(), false, params_->op_type, "fail to exec block_dims_calculator");
      OP_LOGE_IF(!l0_calculator_.Exec(), false, params_->op_type, "fail to exec l0_calculator");
      OP_LOGE_IF(!l1_calculator_.Exec(), false, params_->op_type, "fail to exec l1_calculator");
      OP_LOGE_IF(!ub_calculator_.Exec(), false, params_->op_type, "fail to exec l1_calculator");
    } else {
      OP_LOGE_IF(!cycle_calculator_->Exec(), false, params_->op_type, "fail to exec cycle_calculator");
    }
  }

  CheckSpecialTemplate();
  SetTiling(tiling);

  TilingIdParam tiling_id_param;
  tiling_id_param.Calc(params_, single_core_status_, tiling);
  FixTilingParam(tiling);

  tiling.tiling_id = CalcTilingId(tiling, tiling_id_param);
  OP_LOGD(params_->op_type, "[single core status][%s]", single_core_status_.ToString().c_str());
  ShowResourceStatistics();
  return true;
}

void DwCacheTilingImpl::CheckSpecialTemplate() {
  const TilingShape &shape = single_core_status_.shape();
  L1Status l1_status = single_core_status_.l1_status();
  const L0Status &l0_status = single_core_status_.l0_status();
  const DimFactor &block_dims = single_core_status_.block_dims();

  if (shape.batch != 1 || shape.group != 1) {
    return;
  }

  if ((l1_status.m_al1 * l0_status.m) == shape.m  && l1_status.k_al1 == params_->load3d_special * shape.k) {
    l1_status.m_al1 = kNone;
    OP_LOGD(params_->op_type, "check special template, tiling al1 changed to full load");
  }

  // for fp32 scene, calculate c1 with c0(16) in conv2d_backprop_filter.cc, need to calculate again with 8
  int64_t cin1_g =
      MathUtil::Align(params_->mag_factor * params_->b_shape.c / params_->groups, params_->b_shape.c0) / params_->k0;
  int64_t total_n = cin1_g * params_->kernel_h * params_->kernel_w;
  int64_t n_single_core = MathUtil::CeilDivision(total_n, block_dims.n * l1_status.n_bl1 * l0_status.n);
  // in fp32 scene, caculate n_single_core with new cin1_g value to avoid full load when n_single_core > 1
  if ((l1_status.n_bl1 * l0_status.n) == shape.n && l1_status.k_bl1 == params_->load3d_special * shape.k &&
    (params_->b_dtype != ge::DT_FLOAT || n_single_core == 1)) {
    l1_status.n_bl1 = kNone;
    OP_LOGD(params_->op_type, "check special template, tiling bl1 changed to full load");
  }

  single_core_status_.UpdateL1Status(l1_status);
}


void DwCacheTilingImpl::SetTiling(CubeTiling &tiling) const {
  auto &dw_tiling = reinterpret_cast<Conv2DBpFilterTiling &>(tiling);
  dw_tiling.wi_bub = ub_calculator_.GetTilingWiBub();
  UpdateTiling(tiling);
  if (tiling.m_al1 == kNone) {
    tiling.db_al1 = 1;
  }
  if (tiling.n_bl1 == kNone) {
    tiling.db_bl1 = 1;
  }
  OP_LOGD(params_->op_type, "Tiling params [tiling][%s]", tiling.ToString().c_str());
}

void DwCacheTilingImpl::FixTilingParam(CubeTiling &tiling) {
  // l0a_m = l0c_m * m_l0
  if (tiling.m_al1 == kNone) {
    tiling.m_al1 = MathUtil::CeilDivision(single_core_status_.shape().m, tiling.m_l0);
  }

  if (tiling.n_bl1 == kNone) {
    tiling.n_bl1 = MathUtil::CeilDivision(single_core_status_.shape().n, tiling.n_l0);
  }

  L1Status l1_status = single_core_status_.l1_status();
  l1_status.m_al1 = tiling.m_al1;
  l1_status.n_bl1 = tiling.n_bl1;
  single_core_status_.UpdateL1Status(l1_status);
  OP_LOGD(params_->op_type, "Fix tiling params [tiling][%s]", tiling.ToString().c_str());
}

int32_t DwCacheTilingImpl::CalcTilingId(const CubeTiling &tiling, const TilingIdParam &id_param) const {
  int32_t tiling_id = 0;
  tiling_id += (params_->binary_mode - 1) << kBinaryModeOffset;
  tiling_id += id_param.load_mode() << kLoadModeOffset;
  tiling_id += static_cast<int32_t>(params_->conv1d_flag) << kConv1dFlagOffset;
  tiling_id += (params_->load3d_special - 1) << kLoad3dSpecialOffset;
  tiling_id += id_param.min_kl1_cmp_kl0() << kMinKl1CmpKl0Offset;
  tiling_id += id_param.bl1_attach_flag() << kBl1AttachOffset;
  tiling_id += id_param.al1_attach_flag() << kAl1AttachOffset;
  tiling_id += id_param.abkl1_attach_flag() << kAbkl1AttachOffset;
  tiling_id += params_->strideh_read_flag << kStridehReadFlagOffset;
  tiling_id += params_->linear_embedding_opti_flag << kLinearEmbeddingOptiFlagOffset;
  OP_LOGD(params_->op_type, "tiling_id %d", tiling_id);
  return tiling_id;
}
REGISTER_TILING_GENERATOR(kConv2DBackpropFilter, DwCacheTilingImpl);
REGISTER_TILING_GENERATOR(kConv2DBackpropFilterV2, DwCacheTilingImpl);
}  // namespace cachetiling
}  // namespace optiling
