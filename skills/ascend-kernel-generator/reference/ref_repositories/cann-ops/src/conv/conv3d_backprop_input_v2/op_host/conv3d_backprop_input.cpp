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
 * \file conv3d_backprop_input.cpp
 * \brief tiling function of conv3d_backprop_input and conv3d_transpose
 */
#include "conv3d_backprop_input.h"
#include "op_log.h"

#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGW(nodeName, fmt, ...)  \
    std::printf(fmt, ##__VA_ARGS__);  \
    std::printf("\n")
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)       \
    if ((ptr) == nullptr) {                               \
        std::printf("nullptr error!");                  \
        return ge::GRAPH_FAILED;                        \
    }

namespace {
const size_t kOriShapeDim = 5;
const size_t kStridesDim = 5;
const size_t kPadsDim = 6;
const size_t kDilationsDim = 5;

const size_t kPaddingConv3dBpInputIdx = 5;
const size_t kPaddingConv3dTransposeIdx = 7;
const size_t K_OUTPUT_PADDING_CONV3D_TRANSPOSE_IDX = 5;
const size_t K_OFFSET_X_CONV3D_TRANSPOSE_IDX = 6;
// NCDHW
const size_t K_N_DIM_NCDHW = 0;
const size_t K_C_DIM_NCDHW = 1;
const size_t K_D_DIM_NCDHW = 2;
const size_t K_H_DIM_NCDHW = 3;
const size_t K_W_DIM_NCDHW = 4;
// NDHWC
const size_t K_N_DIM_NDHWC = 0;
const size_t K_D_DIM_NDHWC = 1;
const size_t K_H_DIM_NDHWC = 2;
const size_t K_W_DIM_NDHWC = 3;
const size_t K_C_DIM_NDHWC = 4;
// NDC1HWC0
const size_t kNDimNDC1HWC0Idx = 0;
const size_t kDDimNDC1HWC0Idx = 1;
// FRACTAL_Z_3D
const size_t kDkCin1HkWkFRACTALZ3DIdx = 0;
const size_t kCo1FRACTALZ3DIdx = 1;
const size_t kCo0FRACTALZ3DIdx = 2;
const size_t kCin0FRACTALZ3DIdx = 3;
// pad
const size_t K_CONV3D_PAD_HEAD_IDX = 0;
const size_t K_CONV3D_PAD_TAIL_IDX = 1;
const size_t K_CONV3D_PAD_UP_IDX = 2;
const size_t K_CONV3D_PAD_DOWN_IDX = 3;
const size_t K_CONV3D_PAD_LEFT_IDX = 4;
const size_t K_CONV3D_PAD_RIGHT_IDX = 5;
// for param check
const int32_t kDimUp = 2147483647;
const int32_t kDimBatchUp = ((1UL << 31) - 1);
const int32_t kDimLow = 1;
const int32_t kFilterDimHWUp = 511;
const int32_t K_DEFAULT_DILATIONS = 1;
const int32_t K_DEFAULT_STRIDES = 1;
const int32_t kDilationLow = 1;
const int32_t kDilationUp = 255;
const int32_t kPadUp = 255;
const int32_t kDimWNormalUp = 4096;
const int32_t kNumTwo = 2;
const int32_t kBlockSize = 16;
const int32_t kInputIndexOne = 1;
const int32_t kInputIndexTwo = 2;

const int64_t kDataSizeMax = ((1UL << 63) - 1);

int32_t kConv3dBpInputDedyInputIndex = 2;
}  // namespace

namespace optiling {
bool SetRunInfoConv3DDx(const Conv3DDxParas &conv3ddx_paras, const cachetiling::Conv3DBpInputTiling &tiling,
                       RunInfoPara &run, gert::TilingContext *context) {
  run.batch_n = conv3ddx_paras.tiling_param.a_shape.batch;
  run.dedy_cout = conv3ddx_paras.tiling_param.a_shape.c;
  run.dedy_d = conv3ddx_paras.tiling_param.a_shape.d;
  run.dedy_h = conv3ddx_paras.tiling_param.a_shape.h;
  run.dedy_w = conv3ddx_paras.tiling_param.a_shape.w;
  run.dedx_cin = conv3ddx_paras.tiling_param.c_shape.c;
  run.dedx_d = conv3ddx_paras.tiling_param.c_shape.d;
  run.dedx_h = conv3ddx_paras.tiling_param.c_shape.h;
  run.dedx_w = conv3ddx_paras.tiling_param.c_shape.w;
  run.kernel_d = conv3ddx_paras.tiling_param.b_shape.d;
  run.kernel_h = conv3ddx_paras.tiling_param.b_shape.h;
  run.kernel_w = conv3ddx_paras.tiling_param.b_shape.w;
  run.dedy_cout1 = conv3ddx_paras.tiling_param.a_shape.c1;
  run.dedx_cin1 = conv3ddx_paras.tiling_param.c_shape.c1;
  run.real_g = conv3ddx_paras.tiling_param.real_g;
  run.dedy_cout1_g = conv3ddx_paras.tiling_param.co1g;
  run.dedx_cin1_g = conv3ddx_paras.tiling_param.ci1g;
  run.kernel_g_dk_cin1g_hk_wk = conv3ddx_paras.filter_gdkci1ghw;
  // attr vars 28+22+26
  run.stride_d = conv3ddx_paras.tiling_param.stride_d;
  run.stride_h = conv3ddx_paras.tiling_param.stride_h;
  run.stride_w = conv3ddx_paras.tiling_param.stride_w;
  run.pad_h = conv3ddx_paras.tiling_param.pad_h;
  run.pad_t = conv3ddx_paras.tiling_param.pad_t;
  run.pad_u = conv3ddx_paras.tiling_param.pad_u;
  run.pad_d = conv3ddx_paras.tiling_param.pad_d;
  run.pad_l = conv3ddx_paras.tiling_param.pad_l;
  run.pad_r = conv3ddx_paras.tiling_param.pad_r;
  run.dilation_d = conv3ddx_paras.tiling_param.dilation_d;
  run.dilation_h = conv3ddx_paras.tiling_param.dilation_h;
  run.dilation_w = conv3ddx_paras.tiling_param.dilation_w;
  run.shape_up_modify = conv3ddx_paras.shape_up_modify;
  run.shape_left_modify = conv3ddx_paras.shape_left_modify;
  run.shape_down_modify = conv3ddx_paras.shape_down_modify;
  run.shape_right_modify = conv3ddx_paras.shape_right_modify;
  run.backprop_pad_h = conv3ddx_paras.tiling_param.backprop_pad_h;
  run.backprop_pad_t = conv3ddx_paras.tiling_param.backprop_pad_t;
  run.backprop_pad_u = conv3ddx_paras.tiling_param.backprop_pad_u;
  run.backprop_pad_d = conv3ddx_paras.tiling_param.backprop_pad_d;
  run.backprop_pad_l = conv3ddx_paras.tiling_param.backprop_pad_l;
  run.backprop_pad_r = conv3ddx_paras.tiling_param.backprop_pad_r;
  // tiling vars
  run.batch_dim = tiling.batch_dim;
  run.n_dim = tiling.n_dim;
  run.m_dim = tiling.m_dim;
  run.group_dim = tiling.group_dim;
  run.d_dim = tiling.d_dim;
  run.m_al1 = tiling.m_al1;
  run.n_bl1 = tiling.n_bl1;
  run.m_l0 = tiling.m_l0;
  int32_t max_kl1 = std::max(tiling.k_al1, tiling.k_bl1);
  int32_t min_kl1 = std::min(tiling.k_al1, tiling.k_bl1);
  OP_TILING_CHECK(tiling.k_l0 == 0 || min_kl1 == 0 || tiling.n_cub == 0,
                  CUBE_INNER_ERR_REPORT(context,
                                        "Divisor min_kl1 or k_l0 or n_cub equal 0, check tiling form cache tiling."),
                  return false);
  run.n_l0_div_ub = tiling.n_l0 / tiling.n_cub;
  run.n_cub = tiling.n_cub;
  run.k_l0 = tiling.k_l0;
  run.min_kl1_div_kl0 = std::max(
      min_kl1 * conv3ddx_paras.tiling_param.b_shape.h * conv3ddx_paras.tiling_param.b_shape.w / tiling.k_l0, 1L);
  run.max_kl1_div_min_kl1 = max_kl1 / min_kl1;
  run.k_div_max_kl1 = cachetiling::MathUtil::CeilDivision(conv3ddx_paras.tiling_param.co1g_reduce, max_kl1);
  run.d_al1 = tiling.d_al1;
  run.d_bl1 = tiling.d_bl1;
  run.d_al0 = tiling.d_al0;
  run.d_bl0 = tiling.d_bl0;
  run.d_cl0 = tiling.d_cl0;
  run.k_aub = tiling.k_aub;
  run.m_aub = tiling.m_aub;
  run.wo_aub = tiling.wo_aub;
  run.al1_bound = tiling.al1_bound;
  run.bl1_bound = tiling.bl1_bound;
  run.aub_bound = tiling.aub_bound;

  run.load3d_special = conv3ddx_paras.tiling_param.load3d_special;
  run.hf32_flag = conv3ddx_paras.tiling_param.hf32_flag;
  return true;
}

static inline int32_t Align(int32_t param1, int32_t param2) {
  if (param2 == 0) {
    return 0;
  }
  return ((param1 + param2 - 1) / param2) * param2;
}

static inline bool IsOverflowInt32(int64_t value) {
  if (value > INT32_MAX || value < INT32_MIN) {
    return true;
  }
  return false;
}

static inline bool CheckRange(int32_t value, int32_t value_low, int32_t value_up) {
  if (value < value_low || value > value_up) {
    return false;
  }
  return true;
}

static inline bool CheckRangeInt64(int64_t value, int32_t value_low, int32_t value_up) {
  if (value < value_low || value > value_up) {
    return false;
  }
  return true;
}

static inline bool CheckLowerBound(int32_t value, int32_t value_low) { return value >= value_low; }

static inline bool CheckValue(int32_t value, int32_t value_temp) { return value == value_temp; }

static inline string IntToBinary(uint64_t &n) {
  string ans = "";
  do {
    uint64_t t = n % 2UL;
    ans += (t + '0');
    n /= 2UL;
  } while (n);
  return ans;
}

static inline void OutputErrorMsg(const vector<string> &error_info, string &error_flag) {
  string msg;
  for (size_t i = 0; i < error_flag.length(); i++) {
    if (error_flag[i] == '1' && i < error_info.size()) {
      msg = error_info[i];
      OP_LOGE("Conv3DBackpropInput", "Error msg is: %s", msg.c_str());
      break;
    }
  }
}

static bool CheckL1SizeLimit(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context) {
  int64_t w_value = conv3ddx_paras.tiling_param.a_shape.w * conv3ddx_paras.tiling_param.stride_w;
  int64_t h_value_max =
      (conv3ddx_paras.tiling_param.filter_h_dilation - 1) + kBlockSize / conv3ddx_paras.tiling_param.c_shape.w + 2;
  if (kBlockSize < conv3ddx_paras.tiling_param.c_shape.w) {
    h_value_max = conv3ddx_paras.tiling_param.filter_h_dilation + 1;
  } else if (kBlockSize % conv3ddx_paras.tiling_param.c_shape.w == 0) {
    h_value_max =
        (conv3ddx_paras.tiling_param.filter_h_dilation - 1) + kBlockSize / conv3ddx_paras.tiling_param.c_shape.w;
  }

  h_value_max = std::min(
      h_value_max, conv3ddx_paras.tiling_param.a_shape.h * conv3ddx_paras.tiling_param.stride_h);

  // 计算最小载入时 b_l1_dk == 1, a_l1_d 需要根据载入情况反推
  int64_t b_l1_dk = 1;

  int64_t l1_real_dk_check = conv3ddx_paras.tiling_param.a_shape.c1 == 1 ? conv3ddx_paras.tiling_param.b_shape.d : 1;
  int64_t a_l1_d = cachetiling::CubeUtil::GetDfactor(l1_real_dk_check, &conv3ddx_paras.tiling_param, 1);

  int64_t a_l1_size = h_value_max * w_value * a_l1_d * conv3ddx_paras.tiling_param.a_shape.c0 *
                       conv3ddx_paras.tiling_param.a_dtype_bytes;
  int64_t b_l1_size = b_l1_dk * conv3ddx_paras.tiling_param.b_shape.h * conv3ddx_paras.tiling_param.filter_co0 *
                       conv3ddx_paras.tiling_param.b_shape.w * conv3ddx_paras.tiling_param.filter_ci0 *
                       conv3ddx_paras.tiling_param.b_dtype_bytes;
  // 在stride_d > k_d，或者d方向需要额外补零时，v1算子需要在l1上预留一块大小为baseM*baseN的buffer，置零之后再写出去
  int64_t fill_zero_size = kBlockSize * kBlockSize * conv3ddx_paras.tiling_param.b_dtype_bytes;
  if (conv3ddx_paras.tiling_param.b_dtype == ge::DT_FLOAT) {
    // fp32一定走v2，v2算子不需要预留buffer置零
    fill_zero_size = 0;
  }

  bool w_size_limit = std::max(conv3ddx_paras.tiling_param.c_shape.w, w_value) <= kDimWNormalUp;
  // 不支持切W
  if (a_l1_size + b_l1_size + fill_zero_size > static_cast<int64_t>(conv3ddx_paras.tiling_param.platform_info.l1_size()) ||
      !w_size_limit) {
    std::stringstream ss;
    ss << "Minimum load size may exceed L1 buffer, a_l1_size is " << a_l1_size << "B, b_l1_size is " << b_l1_size
      << "B, fill_zero_size is " << fill_zero_size << "B, L1size is " << conv3ddx_paras.tiling_param.platform_info.l1_size()
      << "B, width may exceed limits, actual width: " << std::max(conv3ddx_paras.tiling_param.c_shape.w, w_value)
      << ", width limit: " << kDimWNormalUp;
    CUBE_INNER_ERR_REPORT(context, "Error msg: %s", ss.str().c_str());
    return false;
  }

  return true;
}

bool CheckPadParamsWithLog(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context) {
  const auto op_name = context->GetNodeName();
  cachetiling::Conv3DBpInputTilingParam &tilingParam = conv3ddx_paras.tiling_param;
  OP_LOGE_IF(!CheckRange(tilingParam.pad_h, 0, kDimUp), false, op_name,
    "pad_h value [%d] is invalid, support range [%d, %d]", tilingParam.pad_h, 0, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.pad_t, 0, kDimUp), false, op_name,
    "pad_t value [%d] is invalid, support range [%d, %d]", tilingParam.pad_t, 0, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.pad_u, 0, kPadUp), false, op_name,
    "pad_u value [%d] is invalid, support range [%d, %d]", tilingParam.pad_u, 0, kPadUp);
  OP_LOGE_IF(!CheckRange(tilingParam.pad_d, 0, kPadUp), false, op_name,
    "pad_d value [%d] is invalid, support range [%d, %d]", tilingParam.pad_d, 0, kPadUp);
  OP_LOGE_IF(!CheckRange(tilingParam.pad_l, 0, kPadUp), false, op_name,
    "pad_l value [%d] is invalid, support range [%d, %d]", tilingParam.pad_l, 0, kPadUp);
  OP_LOGE_IF(!CheckRange(tilingParam.pad_r, 0, kPadUp), false, op_name,
    "pad_r value [%d] is invalid, support range [%d, %d]", tilingParam.pad_r, 0, kPadUp);
  return true;
}

bool CheckShapeValidWithLog(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context) {
  const auto op_name = context->GetNodeName();
  cachetiling::Conv3DBpInputTilingParam &tilingParam = conv3ddx_paras.tiling_param;
  OP_LOGE_IF(!CheckRange(tilingParam.groups, kDimLow, kDimUp), false, op_name,
    "group value [%d] is invalid, support range [%d, %d]", tilingParam.groups, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.b_shape.h, kDimLow, kFilterDimHWUp), false, op_name,
    "the H dim of filter [%ld] is invalid, support range [%d, %d]", tilingParam.b_shape.h, kDimLow, kFilterDimHWUp);
  OP_LOGE_IF(!CheckRange(tilingParam.b_shape.w, kDimLow, kFilterDimHWUp), false, op_name,
    "the W dim of filter [%ld] is invalid, support range [%d, %d]", tilingParam.b_shape.w, kDimLow, kFilterDimHWUp);
  OP_LOGE_IF(!CheckRange(tilingParam.b_shape.d, kDimLow, kDimBatchUp), false, op_name,
    "the D dim of filter [%ld] is invalid, support range [%d, %d]", tilingParam.b_shape.d, kDimLow, kDimBatchUp);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.batch, kDimLow, kDimBatchUp), false, op_name,
    "batch value [%ld] is invalid, support range [%d, %d]", tilingParam.a_shape.batch, kDimLow, kDimBatchUp);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.d, kDimLow, kDimUp), false, op_name,
    "dout value [%ld] is invalid, support range [%d, %d]", tilingParam.a_shape.d, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.h, kDimLow, kDimUp), false, op_name,
    "hout value [%ld] is invalid, support range [%d, %d]", tilingParam.a_shape.h, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.w, kDimLow, kDimUp), false, op_name,
    "wout value [%ld] is invalid, support range [%d, %d]", tilingParam.a_shape.w, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckLowerBound(tilingParam.a_shape.c, kDimLow), false, op_name,
    "cout value [%ld] is invalid, should not be less than [%d]", tilingParam.a_shape.c, kDimLow);
  OP_LOGE_IF(!CheckLowerBound(tilingParam.a_shape.c1, kDimLow), false, op_name,
    "cout1 value [%ld] is invalid, should not be less than [%d]", tilingParam.a_shape.c1, kDimLow);
  OP_LOGE_IF(!CheckLowerBound(tilingParam.c_shape.c1, kDimLow), false, op_name,
    "cin1 value [%ld] is invalid, should not be less than [%d]", tilingParam.c_shape.c1, kDimLow);
  OP_LOGE_IF(!CheckLowerBound(tilingParam.c_shape.c, kDimLow), false, op_name,
    "cin value [%ld] is invalid, should not be less than [%d]", tilingParam.c_shape.c, kDimLow);
  OP_LOGE_IF(!CheckLowerBound(tilingParam.c_shape.d, kDimLow), false, op_name,
    "din value [%ld] is invalid, should not be less than [%d]", tilingParam.c_shape.d, kDimLow);
  return true;
}

bool CheckParamsWithLog(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context) {
  const auto op_name = context->GetNodeName();
  cachetiling::Conv3DBpInputTilingParam &tilingParam = conv3ddx_paras.tiling_param;

  OP_LOGE_IF(!CheckRange(tilingParam.dilation_h, kDilationLow, kDilationUp), false, op_name,
    "dilation_h value [%d] is invalid, support range [%d, %d]", tilingParam.dilation_h, kDilationLow, kDilationUp);
  OP_LOGE_IF(!CheckRange(tilingParam.dilation_w, kDilationLow, kDilationUp), false, op_name,
    "dilation_w value [%d] is invalid, support range [%d, %d]", tilingParam.dilation_w, kDilationLow, kDilationUp);
  OP_LOGE_IF(!CheckRange(tilingParam.dilation_d, kDilationLow, kDilationUp), false, op_name,
    "dilation_d value [%d] is invalid, support range [%d, %d]", tilingParam.dilation_d, kDilationLow, kDilationUp);
  OP_LOGE_IF(!CheckRange(tilingParam.stride_h, kDimLow, kDimUp), false, op_name,
    "stride_h value [%d] is invalid, support range [%d, %d]", tilingParam.stride_h, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.stride_w, kDimLow, kDimUp), false, op_name,
    "stride_w value [%d] is invalid, support range [%d, %d]", tilingParam.stride_w, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.stride_d, kDimLow, kDimUp), false, op_name,
    "stride_d value [%d] is invalid, support range [%d, %d]", tilingParam.stride_d, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckPadParamsWithLog(conv3ddx_paras, context), false, op_name, "check pad params failed");
  OP_LOGE_IF((tilingParam.filter_d_dilation > conv3ddx_paras.fmap_d_padding), false, op_name,
    "((filter_d - 1) * dilation_d + 1)=[%ld] must less than or equal to (fmap_d + pad_h + pad_t)=[%ld]",
    tilingParam.filter_d_dilation, conv3ddx_paras.fmap_d_padding);
  OP_LOGE_IF((tilingParam.filter_h_dilation > conv3ddx_paras.fmap_h_padding), false, op_name,
    "((filter_h - 1) * dilation_h + 1)=[%ld] must less than or equal to (fmap_h + pad_u + pad_d)=[%ld]",
    tilingParam.filter_h_dilation, conv3ddx_paras.fmap_h_padding);
  OP_LOGE_IF((tilingParam.filter_w_dilation > conv3ddx_paras.fmap_w_padding), false, op_name,
    "((filter_w - 1) * dilation_w + 1)=[%ld] must less than or equal to (fmap_w + pad_l + pad_r)=[%ld]",
    tilingParam.filter_w_dilation, conv3ddx_paras.fmap_w_padding);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.w * tilingParam.stride_w, kDimLow, kDimUp),false, op_name,
    "out_backprop's W after expands [%ld] is invalid, support range [%d, %d]",
    tilingParam.a_shape.w * tilingParam.stride_w, kDimLow, kDimUp);
  OP_LOGE_IF(!CheckRange(tilingParam.a_shape.h * tilingParam.stride_h, kDimLow, kDimUp),
    false, op_name, "out_backprop's H after expands [%ld] is invalid, support range [%d, %d]",
    tilingParam.a_shape.h * tilingParam.stride_h, kDimLow, kDimUp);
  return true;
}

bool CheckParams(Conv3DDxParas &conv3ddx_paras, gert::TilingContext *context) {
  int32_t dedy_c_align = Align(conv3ddx_paras.tiling_param.a_shape.c, conv3ddx_paras.tiling_param.a_shape.c0);
  int32_t dedx_c_align = Align(conv3ddx_paras.tiling_param.c_shape.c, conv3ddx_paras.tiling_param.c_shape.c0);
  int32_t filter_c_align = Align(conv3ddx_paras.tiling_param.b_shape.c, conv3ddx_paras.tiling_param.filter_ci0);
  int32_t filter_n_align = Align(conv3ddx_paras.tiling_param.b_shape.batch, conv3ddx_paras.tiling_param.filter_co0);
  int64_t dedy_size = conv3ddx_paras.tiling_param.a_shape.batch * dedy_c_align * conv3ddx_paras.tiling_param.a_shape.d *
                      conv3ddx_paras.tiling_param.a_shape.w * conv3ddx_paras.tiling_param.a_shape.h *
                      conv3ddx_paras.tiling_param.a_dtype_bytes;
  int64_t dedx_size = conv3ddx_paras.tiling_param.a_shape.batch * dedx_c_align * conv3ddx_paras.tiling_param.c_shape.d *
                      conv3ddx_paras.tiling_param.c_shape.w * conv3ddx_paras.tiling_param.c_shape.h *
                      conv3ddx_paras.tiling_param.c_dtype_bytes;
  int64_t filter_size = filter_n_align * filter_c_align * conv3ddx_paras.tiling_param.filter_d_dilation *
                        conv3ddx_paras.tiling_param.b_shape.w * conv3ddx_paras.tiling_param.b_shape.h *
                        conv3ddx_paras.tiling_param.b_dtype_bytes;
  conv3ddx_paras.fmap_d_padding =
      conv3ddx_paras.tiling_param.c_shape.d + conv3ddx_paras.tiling_param.pad_h + conv3ddx_paras.tiling_param.pad_t;
  conv3ddx_paras.fmap_h_padding =
      conv3ddx_paras.tiling_param.c_shape.h + conv3ddx_paras.tiling_param.pad_u + conv3ddx_paras.tiling_param.pad_d;
  conv3ddx_paras.fmap_w_padding =
      conv3ddx_paras.tiling_param.c_shape.w + conv3ddx_paras.tiling_param.pad_l + conv3ddx_paras.tiling_param.pad_r;

  if (!CheckParamsWithLog(conv3ddx_paras, context) || !CheckShapeValidWithLog(conv3ddx_paras, context)) {
    return false;
  }

  uint32_t shift = 0;
  uint64_t invalid = (!CheckRange(conv3ddx_paras.tiling_param.groups, kDimLow, kDimUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.b_shape.h, kDimLow, kFilterDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.b_shape.w, kDimLow, kFilterDimHWUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.b_shape.d, kDimLow, kDimBatchUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.a_shape.batch, kDimLow, kDimBatchUp) << shift++);
  invalid = invalid + (!CheckLowerBound(conv3ddx_paras.tiling_param.a_shape.c1, kDimLow) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.a_shape.d, kDimLow, kDimUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.a_shape.h, kDimLow, kDimUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.a_shape.w, kDimLow, kDimUp) << shift++);
  invalid = invalid + (!CheckLowerBound(conv3ddx_paras.tiling_param.a_shape.c, kDimLow) << shift++);
  invalid = invalid + (!CheckLowerBound(conv3ddx_paras.tiling_param.c_shape.c1, kDimLow) << shift++);
  invalid = invalid + (!CheckLowerBound(conv3ddx_paras.tiling_param.c_shape.c, kDimLow) << shift++);
  invalid = invalid + (!CheckLowerBound(conv3ddx_paras.tiling_param.c_shape.d, kDimLow) << shift++);
  invalid = invalid + ((!CheckRange(conv3ddx_paras.tiling_param.dilation_h, kDilationLow, kDilationUp) ||
                        !CheckRange(conv3ddx_paras.tiling_param.dilation_w, kDilationLow, kDilationUp) ||
                        !CheckRange(conv3ddx_paras.tiling_param.dilation_d, kDilationLow, kDilationUp))
                       << shift++);
  invalid = invalid +
            (!CheckRange(conv3ddx_paras.tiling_param.a_shape.h * conv3ddx_paras.tiling_param.stride_h, kDimLow, kDimUp)
             << shift++);
  invalid = invalid +
            (!CheckRange(conv3ddx_paras.tiling_param.a_shape.w * conv3ddx_paras.tiling_param.stride_w, kDimLow, kDimUp)
             << shift++);
  // Co % g == 0
  invalid =
      invalid + (!CheckValue(conv3ddx_paras.tiling_param.a_shape.c % conv3ddx_paras.tiling_param.groups, 0) << shift++);
  // Ci % g == 0
  invalid =
      invalid + (!CheckValue(conv3ddx_paras.tiling_param.c_shape.c % conv3ddx_paras.tiling_param.groups, 0) << shift++);
  // Ci == Ci / g * g
  invalid = invalid + (!CheckValue(conv3ddx_paras.tiling_param.c_shape.c,
                                   conv3ddx_paras.tiling_param.b_shape.c * conv3ddx_paras.tiling_param.groups)
                       << shift++);
  invalid = invalid +
            (!CheckValue(conv3ddx_paras.tiling_param.a_shape.c, conv3ddx_paras.tiling_param.b_shape.batch) << shift++);
  invalid = invalid + (!CheckValue(conv3ddx_paras.tiling_param.a_shape.batch, conv3ddx_paras.tiling_param.c_shape.batch)
                       << shift++);
  invalid = invalid + ((conv3ddx_paras.tiling_param.filter_d_dilation > conv3ddx_paras.fmap_d_padding) << shift++);
  invalid = invalid + ((conv3ddx_paras.tiling_param.filter_h_dilation > conv3ddx_paras.fmap_h_padding) << shift++);
  invalid = invalid + ((conv3ddx_paras.tiling_param.filter_w_dilation > conv3ddx_paras.fmap_w_padding) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.c_shape.h, kDimLow, kDimUp) << shift++);
  invalid = invalid + (!CheckRange(conv3ddx_paras.tiling_param.c_shape.w, kDimLow, kDimUp) << shift++);
  int64_t do_temp = (conv3ddx_paras.fmap_d_padding - conv3ddx_paras.tiling_param.filter_d_dilation) /
                    conv3ddx_paras.tiling_param.stride_d + 1;
  int64_t ho_temp = (conv3ddx_paras.fmap_h_padding - conv3ddx_paras.tiling_param.filter_h_dilation) /
                    conv3ddx_paras.tiling_param.stride_h + 1;
  int64_t wo_temp = (conv3ddx_paras.fmap_w_padding - conv3ddx_paras.tiling_param.filter_w_dilation) /
                    conv3ddx_paras.tiling_param.stride_w + 1;
  invalid = invalid + ((do_temp != conv3ddx_paras.tiling_param.a_shape.d) << shift++);
  invalid = invalid + ((ho_temp != conv3ddx_paras.tiling_param.a_shape.h) << shift++);
  invalid = invalid + ((wo_temp != conv3ddx_paras.tiling_param.a_shape.w) << shift++);
  invalid =
      invalid + ((dedy_c_align == 0 || dedx_c_align == 0 || filter_c_align == 0 || filter_n_align == 0) << shift++);
  invalid = invalid + ((dedy_size > kDataSizeMax) << shift++);
  invalid = invalid + ((dedx_size > kDataSizeMax) << shift++);
  // 左移flag超过int32范围，需要加int64强转
  invalid = invalid + (static_cast<int64_t>((filter_size > kDataSizeMax)) << shift++);
  invalid = invalid + (static_cast<int64_t>(!CheckL1SizeLimit(conv3ddx_paras, context)) << shift++);
  invalid = invalid +
            (static_cast<int64_t>(conv3ddx_paras.pad_up_before > kPadUp || conv3ddx_paras.pad_left_before > kPadUp ||
                                  conv3ddx_paras.pad_down_after > kPadUp || conv3ddx_paras.pad_right_after > kPadUp)
             << shift++);
  if (invalid != 0) {
    vector<string> error_info = {"groups must be equal 1",
                                  "kh value invalid",
                                  "kw value invalid",
                                  "kd value invalid",
                                  "batch value invalid",
                                  "co1 value invalid",
                                  "dout value invalid",
                                  "ho value invalid",
                                  "wo value invalid",
                                  "co value invalid",
                                  "c1 value invalid",
                                  "cin value invalid",
                                  "din value invalid",
                                  "h, w, d dilations value invalid",
                                  "out_backprop's H after expands is invalid",
                                  "out_backprop's W after expands is invalid",
                                  "c dim of out_backprop must be div by groups",
                                  "c dim of fmap must be div by groups",
                                  "c dim of fmap must be equal with filter c multi groups",
                                  "c dim of out_backprop must be equal with filter n",
                                  "fmap batch not equal with out_backprop batch",
                                  "filter_d_dilation or fmap_d_padding invalid",
                                  "filter_h_dilation or fmap_h_padding invalid",
                                  "filter_w_dilation or fmap_w_padding invalid",
                                  "hin value invalid",
                                  "win value invalid",
                                  "fmap_d does not match out_backprop_d",
                                  "fmap_h does not match out_backprop_h",
                                  "fmap_w does not match out_backprop_w",
                                  "ci or co is invalid",
                                  "out_backprop size larger than int64",
                                  "fmap size larger than int64",
                                  "filter size larger than int64",
                                  "this case may exceed size",
                                  "backprop pad value invalid"};
    string error_flag = IntToBinary(invalid);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }
  return true;
}

template <typename T>
static void GetNCDHWShape(const T &origin_shape, cachetiling::Shape &ncdhw_shape, const ge::Format &origin_format) {
  // caller already checked buffer size
  if (origin_format == ge::FORMAT_NDHWC) {
    ncdhw_shape.batch = origin_shape[0];  // 0: N
    ncdhw_shape.c = origin_shape[4];  // 4: C
    ncdhw_shape.d = origin_shape[1];  // 1: D
    ncdhw_shape.h = origin_shape[2];  // 2: H
    ncdhw_shape.w = origin_shape[3];  // 3: W
  } else if (origin_format == ge::FORMAT_NCDHW) {
    ncdhw_shape.batch = origin_shape[0];  // 0: N
    ncdhw_shape.c = origin_shape[1];  // 1: C
    ncdhw_shape.d = origin_shape[2];  // 2: D
    ncdhw_shape.h = origin_shape[3];  // 3: H
    ncdhw_shape.w = origin_shape[4];  // 4: W
  } else if (origin_format == ge::FORMAT_DHWCN) {
    ncdhw_shape.batch = origin_shape[4];  // 4: N
    ncdhw_shape.c = origin_shape[3];  // 3: C
    ncdhw_shape.d = origin_shape[0];  // 0: D
    ncdhw_shape.h = origin_shape[1];  // 1: H
    ncdhw_shape.w = origin_shape[2];  // 2: W
  }
}

static bool UpdateDtypeParams(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras,
                              cachetiling::OpType op_type) {
  const auto op_name = context->GetNodeName();

  if (op_type == cachetiling::kConv3DTranspose) {
    // Conv3DTranspose, index of x is 1, index of filter is 2
    conv3ddx_paras.tiling_param.a_dtype = context->GetInputDesc(1)->GetDataType();
    conv3ddx_paras.tiling_param.b_dtype = context->GetInputDesc(2)->GetDataType();
    conv3ddx_paras.tiling_param.c_dtype = context->GetOutputDesc(0)->GetDataType();
  } else {
    // Conv3dBackpropInput, index of out_backprop is 2, index of filter is 1
    conv3ddx_paras.tiling_param.a_dtype = context->GetInputDesc(2)->GetDataType();
    conv3ddx_paras.tiling_param.b_dtype = context->GetInputDesc(1)->GetDataType();
    conv3ddx_paras.tiling_param.c_dtype = context->GetOutputDesc(0)->GetDataType();
  }

  if (conv3ddx_paras.tiling_param.a_dtype == ge::DT_BF16 && conv3ddx_paras.tiling_param.b_dtype == ge::DT_BF16 &&
      conv3ddx_paras.tiling_param.c_dtype == ge::DT_BF16) {
      conv3ddx_paras.tiling_param.a_dtype = ge::DT_FLOAT16;
      conv3ddx_paras.tiling_param.b_dtype = ge::DT_FLOAT16;
      conv3ddx_paras.tiling_param.c_dtype = ge::DT_FLOAT16;
  }
  OP_LOGE_IF(!((conv3ddx_paras.tiling_param.a_dtype == ge::DT_FLOAT16 &&
                 conv3ddx_paras.tiling_param.b_dtype == ge::DT_FLOAT16 &&
                 conv3ddx_paras.tiling_param.c_dtype == ge::DT_FLOAT16) ||
                 (conv3ddx_paras.tiling_param.a_dtype == ge::DT_FLOAT &&
                 conv3ddx_paras.tiling_param.b_dtype == ge::DT_FLOAT &&
                 conv3ddx_paras.tiling_param.c_dtype == ge::DT_FLOAT)),
             false, op_name, "out_backprop/fitler/y dtype only support fp16 and fp32 now.");

  conv3ddx_paras.tiling_param.a_dtype_bytes = ge::GetSizeByDataType(conv3ddx_paras.tiling_param.a_dtype);
  OP_LOGE_IF(conv3ddx_paras.tiling_param.a_dtype_bytes == -1, false, op_name, "out_backprop dtype size is invalid");
  conv3ddx_paras.tiling_param.b_dtype_bytes = ge::GetSizeByDataType(conv3ddx_paras.tiling_param.b_dtype);
  OP_LOGE_IF(conv3ddx_paras.tiling_param.b_dtype_bytes == -1, false, op_name, "filter dtype size is invalid");
  conv3ddx_paras.tiling_param.c_dtype_bytes = ge::GetSizeByDataType(conv3ddx_paras.tiling_param.c_dtype);
  OP_LOGE_IF(conv3ddx_paras.tiling_param.c_dtype_bytes == -1, false, op_name, "y dtype size is invalid");

  return true;
}

static bool CheckAttrRangeDilations(gert::TilingContext *context, const int64_t *dilations) {
  const auto op_name = context->GetNodeName();
  auto y_ori_format = context->GetOutputDesc(0)->GetOriginFormat();
  if (y_ori_format == ge::FORMAT_NCDHW) {
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_N_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS), false, op_name,
      "dilation_n value [%ld] is invalid, support range [%d, %d]", dilations[K_N_DIM_NCDHW],
      K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_C_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS), false, op_name,
      "dilation_c value [%ld] is invalid, support range [%d, %d]", dilations[K_C_DIM_NCDHW],
      K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_D_DIM_NCDHW], kDilationLow, kDilationUp), false, op_name,
      "dilation_d value [%ld] is invalid, support range [%d, %d]", dilations[K_D_DIM_NCDHW],
      kDilationLow, kDilationUp);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_H_DIM_NCDHW], kDilationLow, kDilationUp), false, op_name,
      "dilation_h value [%ld] is invalid, support range [%d, %d]", dilations[K_H_DIM_NCDHW],
      kDilationLow, kDilationUp);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_W_DIM_NCDHW], kDilationLow, kDilationUp), false, op_name,
      "dilation_w value [%ld] is invalid, support range [%d, %d]", dilations[K_W_DIM_NCDHW],
      kDilationLow, kDilationUp);
  } else {
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_N_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS), false, op_name,
      "dilation_n value [%ld] is invalid, support range [%d, %d]", dilations[K_N_DIM_NDHWC],
      K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_C_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS), false, op_name,
      "dilation_c value [%ld] is invalid, support range [%d, %d]", dilations[K_C_DIM_NDHWC],
      K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_D_DIM_NDHWC], kDilationLow, kDilationUp), false, op_name,
      "dilation_d value [%ld] is invalid, support range [%d, %d]", dilations[K_D_DIM_NDHWC],
      kDilationLow, kDilationUp);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_H_DIM_NDHWC], kDilationLow, kDilationUp), false, op_name,
      "dilation_h value [%ld] is invalid, support range [%d, %d]", dilations[K_H_DIM_NDHWC],
      kDilationLow, kDilationUp);
    OP_LOGE_IF(!CheckRangeInt64(dilations[K_W_DIM_NDHWC], kDilationLow, kDilationUp), false, op_name,
      "dilation_w value [%ld] is invalid, support range [%d, %d]", dilations[K_W_DIM_NDHWC],
      kDilationLow, kDilationUp);
  }

  return true;
}

static bool CheckAttrRangeStrides(gert::TilingContext *context, const int64_t *strides) {
  const auto op_name = context->GetNodeName();
  auto y_ori_format = context->GetOutputDesc(0)->GetOriginFormat();
  if (y_ori_format == ge::FORMAT_NCDHW) {
    OP_LOGE_IF(!CheckRangeInt64(strides[K_N_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES), false, op_name,
      "stride_n value [%ld] is invalid, support range [%d, %d]", strides[K_N_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_C_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES), false, op_name,
      "stride_c value [%ld] is invalid, support range [%d, %d]", strides[K_C_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_D_DIM_NCDHW], kDimLow, kDimUp), false, op_name,
      "stride_d value [%ld] is invalid, support range [%d, %d]", strides[K_D_DIM_NCDHW], kDimLow, kDimUp);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_H_DIM_NCDHW], kDimLow, kDimUp), false, op_name,
      "stride_h value [%ld] is invalid, support range [%d, %d]", strides[K_H_DIM_NCDHW], kDimLow, kDimUp);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_W_DIM_NCDHW], kDimLow, kDimUp), false, op_name,
      "stride_w value [%ld] is invalid, support range [%d, %d]", strides[K_W_DIM_NCDHW], kDimLow, kDimUp);
  } else {
    OP_LOGE_IF(!CheckRangeInt64(strides[K_N_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES), false, op_name,
      "stride_n value [%ld] is invalid, support range [%d, %d]", strides[K_N_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_C_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES), false, op_name,
      "stride_c value [%ld] is invalid, support range [%d, %d]", strides[K_C_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_D_DIM_NDHWC], kDimLow, kDimUp), false, op_name,
      "stride_d value [%ld] is invalid, support range [%d, %d]", strides[K_D_DIM_NDHWC], kDimLow, kDimUp);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_H_DIM_NDHWC], kDimLow, kDimUp), false, op_name,
      "stride_h value [%ld] is invalid, support range [%d, %d]", strides[K_H_DIM_NDHWC], kDimLow, kDimUp);
    OP_LOGE_IF(!CheckRangeInt64(strides[K_W_DIM_NDHWC], kDimLow, kDimUp), false, op_name,
      "stride_w value [%ld] is invalid, support range [%d, %d]", strides[K_W_DIM_NDHWC], kDimLow, kDimUp);
  }

  return true;
}

static bool CheckAttrRangePads(gert::TilingContext *context, const int64_t *pads) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_HEAD_IDX], 0, kDimUp), false, op_name,
    "pad_h value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_HEAD_IDX], 0, kDimUp);
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_TAIL_IDX], 0, kDimUp), false, op_name,
    "pad_t value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_TAIL_IDX], 0, kDimUp);
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_UP_IDX], 0, kPadUp), false, op_name,
    "pad_u value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_UP_IDX], 0, kPadUp);
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_DOWN_IDX], 0, kPadUp), false, op_name,
    "pad_d value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_DOWN_IDX], 0, kPadUp);
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_LEFT_IDX], 0, kPadUp), false, op_name,
    "pad_l value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_LEFT_IDX], 0, kPadUp);
  OP_LOGE_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_RIGHT_IDX], 0, kPadUp), false, op_name,
    "pad_r value [%ld] is invalid, support range [%d, %d]", pads[K_CONV3D_PAD_RIGHT_IDX], 0, kPadUp);

  return true;
}

static bool CheckAttrRange(gert::TilingContext *context, const int64_t *strides, const int64_t *pads,
  const int64_t *dilations, const int64_t *groups) {
  const auto op_name = context->GetNodeName();

  OP_LOGE_IF(!CheckAttrRangeDilations(context, dilations), false, op_name, "check dilations range failed");
  OP_LOGE_IF(!CheckAttrRangeStrides(context, strides), false, op_name, "check strides range failed");
  OP_LOGE_IF(!CheckAttrRangePads(context, pads), false, op_name, "check pads range failed");
  // groups
  if (groups != nullptr) {
    OP_LOGE_IF(!CheckRangeInt64(*groups, kDimLow, kDimUp), false, op_name,
      "group value [%ld] is invalid, support range [%d, %d]", *groups, kDimLow, kDimUp);
  }

  return true;
}

static bool CheckTransposeAttr(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  auto yDesc = context->GetOutputDesc(0);
  auto attrs = context->GetAttrs();
  auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(K_OUTPUT_PADDING_CONV3D_TRANSPOSE_IDX);
  OP_LOGE_IF(outputPadding == nullptr, false, context, "failed to get output_padding attrs");
  OP_LOGE_IF(outputPadding->GetSize() != kOriShapeDim, false, context,
             "The output_padding should be 5d, actual dim num: %zu", outputPadding->GetSize());
  const auto outputPaddingData = reinterpret_cast<const int64_t *>(outputPadding->GetData());
  if (yDesc->GetOriginFormat() == ge::FORMAT_NCDHW) {
    OP_LOGE_IF(outputPaddingData[K_N_DIM_NCDHW] != 0 || outputPaddingData[K_C_DIM_NCDHW] != 0, false, context,
      "N/D output padding should be zero.\n");
    conv3ddx_paras.output_padding_d = outputPaddingData[K_D_DIM_NCDHW];
    conv3ddx_paras.output_padding_h = outputPaddingData[K_H_DIM_NCDHW];
    conv3ddx_paras.output_padding_w = outputPaddingData[K_W_DIM_NCDHW];
  } else {  // yDesc->GetOriginFormat() == ge::FORMAT_NDHWC
    OP_LOGE_IF(outputPaddingData[K_N_DIM_NDHWC] != 0 || outputPaddingData[K_C_DIM_NDHWC] != 0, false, context,
      "N/D output padding should be zero.\n");
    conv3ddx_paras.output_padding_d = outputPaddingData[K_D_DIM_NDHWC];
    conv3ddx_paras.output_padding_h = outputPaddingData[K_H_DIM_NDHWC];
    conv3ddx_paras.output_padding_w = outputPaddingData[K_W_DIM_NDHWC];
  }
  if (attrs->GetAttrNum() > K_OFFSET_X_CONV3D_TRANSPOSE_IDX) {
    const auto offsetX = attrs->GetAttrPointer<int64_t>(K_OFFSET_X_CONV3D_TRANSPOSE_IDX);
    OP_LOGE_IF(offsetX == nullptr, false, context, "failed to get offsetX attrs");
    OP_LOGE_IF(*offsetX != 0, false, context, "offsetX:%ld is invalid, it should be 0", *offsetX);
  }
  return true;
}

static bool CheckTransposeOutputdingRange(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) 
{
  // outputPadding值需要小于同维度dilation或stride
  OP_LOGE_IF((conv3ddx_paras.output_padding_d >= conv3ddx_paras.tiling_param.stride_d && 
    conv3ddx_paras.output_padding_d >= conv3ddx_paras.tiling_param.dilation_d), false, context,
    "output_padding_d value[%d] should smaller than dilation_d[%d] or stride_d[%d]",
    conv3ddx_paras.output_padding_d, conv3ddx_paras.tiling_param.dilation_d, conv3ddx_paras.tiling_param.stride_d);
  OP_LOGE_IF((conv3ddx_paras.output_padding_h >= conv3ddx_paras.tiling_param.stride_h && 
    conv3ddx_paras.output_padding_h >= conv3ddx_paras.tiling_param.dilation_h), false, context,
    "output_padding_h value[%d] should smaller than dilation_h[%d] or stride_h[%d]",
    conv3ddx_paras.output_padding_h, conv3ddx_paras.tiling_param.dilation_h, conv3ddx_paras.tiling_param.stride_h);
  OP_LOGE_IF((conv3ddx_paras.output_padding_w >= conv3ddx_paras.tiling_param.stride_w && 
    conv3ddx_paras.output_padding_w >= conv3ddx_paras.tiling_param.dilation_w), false, context,
    "output_padding_w value[%d] should smaller than dilation_w[%d] or stride_w[%d]",
    conv3ddx_paras.output_padding_w, conv3ddx_paras.tiling_param.dilation_w, conv3ddx_paras.tiling_param.stride_w);

  return true;
}

static void SetConv3ddxParas(Conv3DDxParas &conv3ddx_paras, const int64_t *pads_data, cachetiling::Shape &strides_ncdhw,
  cachetiling::Shape &dilations_ncdhw, const int64_t *groups) 
{
  size_t idx = 0;
  conv3ddx_paras.tiling_param.pad_h = pads_data[idx++];
  conv3ddx_paras.tiling_param.pad_t = pads_data[idx++];
  conv3ddx_paras.tiling_param.pad_u = pads_data[idx++];
  conv3ddx_paras.tiling_param.pad_d = pads_data[idx++];
  conv3ddx_paras.tiling_param.pad_l = pads_data[idx++];
  conv3ddx_paras.tiling_param.pad_r = pads_data[idx++];
  conv3ddx_paras.tiling_param.stride_d = strides_ncdhw.d;
  conv3ddx_paras.tiling_param.stride_h = strides_ncdhw.h;
  conv3ddx_paras.tiling_param.stride_w = strides_ncdhw.w;
  conv3ddx_paras.tiling_param.dilation_d = dilations_ncdhw.d;
  conv3ddx_paras.tiling_param.dilation_h = dilations_ncdhw.h;
  conv3ddx_paras.tiling_param.dilation_w = dilations_ncdhw.w;
  conv3ddx_paras.tiling_param.groups = *groups;

  conv3ddx_paras.tiling_param.stride_expand_flag = (conv3ddx_paras.tiling_param.stride_h != 1 ||
    conv3ddx_paras.tiling_param.stride_w != 1 || conv3ddx_paras.tiling_param.stride_d != 1);

  conv3ddx_paras.tiling_param.dilation_d_gt_one_flag = dilations_ncdhw.d > 1 ? 1: 0;
}

static bool GetAttrAndDtypeParams(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras,
                                  cachetiling::OpType op_type) {
  auto attrs = context->GetAttrs();
  OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context, "failed to get runtime attrs"), return false);

  size_t idx = 0;
  const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const auto groups = attrs->GetAttrPointer<int64_t>(idx++);

  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(strides == nullptr, false, op_name, "get strides from context fail.");
  OP_LOGE_IF(strides->GetSize() != kStridesDim, false, op_name, "strides of context dim len is invalid.");
  OP_LOGE_IF(pads == nullptr, false, op_name, "get pads from context fail.");
  OP_LOGE_IF(pads->GetSize() != kPadsDim, false, op_name, "pads of context dim len is invalid.");
  OP_LOGE_IF(dilations == nullptr, false, op_name, "get dilations from context fail.");
  OP_LOGE_IF(dilations->GetSize() != kDilationsDim, false, op_name, "dilations of context dim len is invalid.");
  OP_LOGE_IF(groups == nullptr, false, op_name, "get groups from context fail.");

  const auto strides_data = reinterpret_cast<const int64_t *>(strides->GetData());
  const auto pads_data = reinterpret_cast<const int64_t *>(pads->GetData());
  const auto dilations_data = reinterpret_cast<const int64_t *>(dilations->GetData());
  OP_LOGE_IF(!CheckAttrRange(context, strides_data, pads_data, dilations_data, groups),
    false, context, "check attr range failed");
  cachetiling::Shape strides_ncdhw;
  cachetiling::Shape dilations_ncdhw;
  ge::Format data_format = context->GetInputDesc(2)->GetOriginFormat();
  if (op_type == cachetiling::kConv3DTranspose) {
    OP_LOGE_IF(!CheckTransposeAttr(context, conv3ddx_paras), false, context, "check transpose attr failed");
    data_format = context->GetInputDesc(1)->GetOriginFormat();
  }
  GetNCDHWShape(strides_data, strides_ncdhw, data_format);
  GetNCDHWShape(dilations_data, dilations_ncdhw, data_format);

  OP_LOGE_IF(strides_ncdhw.batch != 1, false, op_name, "strides N[%ld] is invalid, must be 1.", strides_ncdhw.batch);
  OP_LOGE_IF(strides_ncdhw.c != 1, false, op_name, "strides C[%ld] is invalid, must be 1.", strides_ncdhw.c);

  SetConv3ddxParas(conv3ddx_paras, pads_data, strides_ncdhw, dilations_ncdhw, groups);
  if (op_type == cachetiling::kConv3DTranspose) {
    OP_LOGE_IF(!CheckTransposeOutputdingRange(context, conv3ddx_paras), false, context, "check transpose attr failed");
  }

  return UpdateDtypeParams(context, conv3ddx_paras, op_type);
}

static void UpdateShapeParams(Conv3DDxParas &conv3ddx_paras,
                              cachetiling::Shape &out_backprop_shape_ncdhw, cachetiling::Shape &filter_shape_ncdhw,
                              cachetiling::Shape &y_shape_ncdhw) {
  // a_shape means out_backprop shape
  conv3ddx_paras.tiling_param.a_shape.c = out_backprop_shape_ncdhw.c;
  conv3ddx_paras.tiling_param.a_shape.d = out_backprop_shape_ncdhw.d;
  conv3ddx_paras.tiling_param.a_shape.h = out_backprop_shape_ncdhw.h;
  conv3ddx_paras.tiling_param.a_shape.w = out_backprop_shape_ncdhw.w;
  // only support fp16 now
  conv3ddx_paras.tiling_param.a_shape.c0 = cachetiling::kDtypeBlockReduceMap.at(conv3ddx_paras.tiling_param.a_dtype);
  conv3ddx_paras.tiling_param.a_shape.c1 = cachetiling::MathUtil::CeilDivision(conv3ddx_paras.tiling_param.a_shape.c,
                                                                               conv3ddx_paras.tiling_param.a_shape.c0);
  // c_shape means y shape
  conv3ddx_paras.tiling_param.c_shape.c = y_shape_ncdhw.c;
  conv3ddx_paras.tiling_param.c_shape.d = y_shape_ncdhw.d;
  conv3ddx_paras.tiling_param.c_shape.h = y_shape_ncdhw.h;
  conv3ddx_paras.tiling_param.c_shape.w = y_shape_ncdhw.w;
  conv3ddx_paras.tiling_param.c_shape.c0 = cachetiling::kDtypeBlockReduceMap.at(conv3ddx_paras.tiling_param.c_dtype);
  conv3ddx_paras.tiling_param.c_shape.c1 = cachetiling::MathUtil::CeilDivision(conv3ddx_paras.tiling_param.c_shape.c,
                                                                               conv3ddx_paras.tiling_param.c_shape.c0);
  // b_shape means filter shape
  conv3ddx_paras.tiling_param.b_shape.batch = filter_shape_ncdhw.batch;
  conv3ddx_paras.tiling_param.b_shape.c = filter_shape_ncdhw.c;
  conv3ddx_paras.tiling_param.b_shape.d = filter_shape_ncdhw.d;
  conv3ddx_paras.tiling_param.b_shape.h = static_cast<int32_t>(filter_shape_ncdhw.h);
  conv3ddx_paras.tiling_param.b_shape.w = static_cast<int32_t>(filter_shape_ncdhw.w);
  conv3ddx_paras.tiling_param.filter_d_dilation =
      (conv3ddx_paras.tiling_param.b_shape.d - 1) * conv3ddx_paras.tiling_param.dilation_d + 1;
  conv3ddx_paras.tiling_param.filter_h_dilation =
      (conv3ddx_paras.tiling_param.b_shape.h - 1) * conv3ddx_paras.tiling_param.dilation_h + 1;
  conv3ddx_paras.tiling_param.filter_w_dilation =
      (conv3ddx_paras.tiling_param.b_shape.w - 1) * conv3ddx_paras.tiling_param.dilation_w + 1;
  conv3ddx_paras.tiling_param.b_shape.c0 = cachetiling::kDtypeBlockReduceMap.at(conv3ddx_paras.tiling_param.b_dtype);
  conv3ddx_paras.tiling_param.b_shape.c1 = cachetiling::MathUtil::CeilDivision(conv3ddx_paras.tiling_param.b_shape.c,
                                                                               conv3ddx_paras.tiling_param.b_shape.c0);
}

static bool CalShapeInfoFromDesc(gert::TilingContext *context, size_t filter_input_index,
                                 size_t out_backprop_input_index, Conv3DDxParas &conv3ddx_paras) {
  auto filter_desc = context->GetInputDesc(filter_input_index);
  auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  auto y_desc = context->GetOutputDesc(0);

  auto filter_shape = context->GetInputShape(filter_input_index);
  auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
  auto y_shape = context->GetOutputShape(0);

  conv3ddx_paras.tiling_param.a_shape.batch = out_backprop_shape->GetStorageShape().GetDim(kNDimNDC1HWC0Idx);
  conv3ddx_paras.tiling_param.c_shape.batch = y_shape->GetStorageShape().GetDim(kNDimNDC1HWC0Idx);
  conv3ddx_paras.filter_gdkci1ghw = filter_shape->GetStorageShape().GetDim(kDkCin1HkWkFRACTALZ3DIdx);
  // NOTE only support shape of filter is (g'*dk*ci1_g'*hk*wk, co1', co0, ci0)
  conv3ddx_paras.tiling_param.co1g = filter_shape->GetStorageShape().GetDim(kCo1FRACTALZ3DIdx);
  if (context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT && conv3ddx_paras.tiling_param.groups > 1) {
      conv3ddx_paras.tiling_param.co1g *= 2; // 2: BLOCK_NUM / FP32_C0
  }
  conv3ddx_paras.tiling_param.co1g_reduce = conv3ddx_paras.tiling_param.co1g;
  conv3ddx_paras.tiling_param.filter_co0 = filter_shape->GetStorageShape().GetDim(kCo0FRACTALZ3DIdx);
  conv3ddx_paras.tiling_param.filter_ci0 = filter_shape->GetStorageShape().GetDim(kCin0FRACTALZ3DIdx);

  auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto y_ori_format = y_desc->GetOriginFormat();

  const auto &out_backprop_ori_shape = out_backprop_shape->GetOriginShape();
  const auto &filter_ori_shape = filter_shape->GetOriginShape();
  const auto &y_ori_shape = y_shape->GetOriginShape();
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(out_backprop_ori_shape.GetDimNum() != kOriShapeDim, false, op_name,
             "out_backprop ori shape dim nums is invalid.");
  OP_LOGE_IF(filter_ori_shape.GetDimNum() != kOriShapeDim, false, op_name, "filter ori shape dim nums is invalid.");
  OP_LOGE_IF(y_ori_shape.GetDimNum() != kOriShapeDim, false, op_name, "y ori shape dim nums is invalid.");

  cachetiling::Shape out_backprop_shape_ncdhw;
  cachetiling::Shape filter_shape_ncdhw;
  cachetiling::Shape y_shape_ncdhw;
  GetNCDHWShape(out_backprop_ori_shape, out_backprop_shape_ncdhw, out_backprop_ori_format);
  GetNCDHWShape(filter_ori_shape, filter_shape_ncdhw, filter_ori_format);
  GetNCDHWShape(y_ori_shape, y_shape_ncdhw, y_ori_format);

  UpdateShapeParams(conv3ddx_paras, out_backprop_shape_ncdhw, filter_shape_ncdhw, y_shape_ncdhw);
  return true;
}

static bool GetShapeParams(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras, cachetiling::OpType op_type, bool isV2Impl) {
  const auto op_name = context->GetNodeName();
  size_t out_backprop_input_index = static_cast<size_t>(kInputIndexTwo);
  size_t filter_input_index = static_cast<size_t>(kInputIndexOne);
  if (op_type == cachetiling::kConv3DTranspose) {
    out_backprop_input_index = kInputIndexOne;
    filter_input_index = kInputIndexTwo;
  }
  const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  const auto filter_desc = context->GetInputDesc(filter_input_index);
  const auto y_desc = context->GetOutputDesc(0);

  const auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
  const auto filter_shape = context->GetInputShape(filter_input_index);
  const auto y_shape = context->GetOutputShape(0);
  OP_LOGE_IF(out_backprop_desc == nullptr || filter_desc == nullptr || y_desc == nullptr, false, op_name,
             "failed to get out_backprop/filter/y tensor desc from context.");
  OP_LOGE_IF(out_backprop_shape == nullptr || filter_shape == nullptr || y_shape == nullptr, false, op_name,
             "failed to get out_backprop/filter/y shape from context.");

  auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto y_ori_format = y_desc->GetOriginFormat();
  OP_LOGE_IF(out_backprop_ori_format != y_ori_format, false, op_name,
             "y(dedx) ori format should be same with out_backprop.");
  OP_LOGE_IF(out_backprop_ori_format != ge::FORMAT_NDHWC && out_backprop_ori_format != ge::FORMAT_NCDHW, false, op_name,
             "out_backprop ori format should be NDHWC or NCDHW.");
  OP_LOGE_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW &&
                 filter_ori_format != ge::FORMAT_DHWCN,
             false, op_name, "filter ori format should be NDHWC or NCDHW or DHWCN.");

  auto out_backprop_format = static_cast<ge::Format>(ge::GetPrimaryFormat(out_backprop_desc->GetStorageFormat()));
  auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
  auto y_format = static_cast<ge::Format>(ge::GetPrimaryFormat(y_desc->GetStorageFormat()));
  OP_LOGE_IF(out_backprop_format != ge::FORMAT_NDC1HWC0 || filter_format != ge::FORMAT_FRACTAL_Z_3D,
            false, op_name,
            "out_backprop format should be NDC1HWC0, filter format should be FRACTAL_Z_3D.");
  if (!isV2Impl || op_type == cachetiling::kConv3DTranspose) {
    OP_LOGE_IF(y_format != ge::FORMAT_NDC1HWC0,
              false, op_name,
              "y format should be NDC1HWC0.");
  } else {
    OP_LOGE_IF(y_format != ge::FORMAT_NDC1HWC0 && y_format != ge::FORMAT_NCDHW,
              false, op_name,
              "y format should be NDC1HWC0 or NCDHW.");
  }

  OP_LOGE_IF(!CalShapeInfoFromDesc(context, filter_input_index, out_backprop_input_index, conv3ddx_paras), false,
             op_name, "Cal Shape Info From Desc fail.");
  return true;
}

static void ReCalDilation(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  // if kernelD/H/W is equal to 1, dilationD/H/W should be set to 1
  if (conv3ddx_paras.tiling_param.b_shape.d == 1) {
      OP_LOGD(context, "kernel_d is equal to 1, dilation_d will be set to 1");
      conv3ddx_paras.tiling_param.dilation_d = 1;
  }

  if (conv3ddx_paras.tiling_param.b_shape.h == 1) {
      OP_LOGD(context, "kernel_h is equal to 1, dilation_h will be set to 1");
      conv3ddx_paras.tiling_param.dilation_h = 1;
  }

  if (conv3ddx_paras.tiling_param.b_shape.w == 1) {
      OP_LOGD(context, "kernel_w is equal to 1, dilation_w will be set to 1");
      conv3ddx_paras.tiling_param.dilation_w = 1;
  }
}

static bool CalGroups(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  if (conv3ddx_paras.tiling_param.b_shape.c == 0 ||
      conv3ddx_paras.tiling_param.c_shape.c % conv3ddx_paras.tiling_param.b_shape.c != 0) {
    OP_LOGE(context, "fmap_channel(%ld) %% filter_channel(%ld) != 0", conv3ddx_paras.tiling_param.c_shape.c,
            conv3ddx_paras.tiling_param.b_shape.c);
    return false;
  }

  int32_t groups = conv3ddx_paras.tiling_param.c_shape.c / conv3ddx_paras.tiling_param.b_shape.c;
  if (conv3ddx_paras.tiling_param.groups == 1) {
    conv3ddx_paras.tiling_param.groups = groups;
  } else if (groups != conv3ddx_paras.tiling_param.groups) {
    OP_LOGE(context, "fmap_channel(%ld) / filter_channel(%ld) != groups(%d)", conv3ddx_paras.tiling_param.c_shape.c,
            conv3ddx_paras.tiling_param.b_shape.c, conv3ddx_paras.tiling_param.groups);
    return false;
  }
  return true;
}

template <class T>
static bool CheckAllZero(const T *tensor_data, size_t dim_size) {
  if (tensor_data == nullptr) {
    // 获取不到data的场景，非onnx模型，input_size一定非零
    return false;
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    if (tensor_data[idx] != 0) {
      return false;
    }
  }
  return true;
}

static bool CheckInputSizeAllZero(gert::TilingContext *context, bool &allzero) {
  auto input_size = context->GetInputTensor(0);
  OP_LOGE_IF(input_size == nullptr, false, context, "get input size fail");
  size_t input_size_dim_num = static_cast<size_t>(input_size->GetOriginShape().GetShapeSize());
  OP_LOGE_IF(input_size_dim_num != kOriShapeDim, false, context, "input_size must be 5d");
  auto dtype = context->GetInputDesc(0)->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = input_size->GetData<int32_t>();
    allzero = CheckAllZero(tensor_data, input_size_dim_num);
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = input_size->GetData<int64_t>();
    allzero = CheckAllZero(tensor_data, input_size_dim_num);
  } else {
    OP_LOGE(context, "input_size dtype only support int32 or int64");
    return false;
  }
  return true;
}

/*
 * 该函数对 Conv3DTranspose 的处理包括两部分:
 * 1. 将 output_padding 属性计入 filter_[x]_dilation 统一处理,
 *    从 PyTorch 对 torch.nn.ConvTranspose3d 的介绍可知，这种处理在形式上是成立的
 * 2. ONNX 模型中的 Conv3DTranspose 算子 input_size 全为零，因此需要在此计算 c_shape
 */
static bool HandleConv3DTranspose(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  conv3ddx_paras.tiling_param.filter_d_dilation += conv3ddx_paras.output_padding_d;
  conv3ddx_paras.tiling_param.filter_h_dilation += conv3ddx_paras.output_padding_h;
  conv3ddx_paras.tiling_param.filter_w_dilation += conv3ddx_paras.output_padding_w;

  bool all_zero = false;
  if (CheckInputSizeAllZero(context, all_zero) && all_zero) {
    int64_t standard_d = conv3ddx_paras.tiling_param.stride_d * (conv3ddx_paras.tiling_param.a_shape.d - 1) +
                         ((conv3ddx_paras.tiling_param.b_shape.d - 1) * conv3ddx_paras.tiling_param.dilation_d + 1);
    conv3ddx_paras.tiling_param.c_shape.d = standard_d + conv3ddx_paras.output_padding_d -
                                            conv3ddx_paras.tiling_param.pad_h - conv3ddx_paras.tiling_param.pad_t;

    int64_t standard_h = conv3ddx_paras.tiling_param.stride_h * (conv3ddx_paras.tiling_param.a_shape.h - 1) +
                         ((conv3ddx_paras.tiling_param.b_shape.h - 1) * conv3ddx_paras.tiling_param.dilation_h + 1);
    conv3ddx_paras.tiling_param.c_shape.h = standard_h + conv3ddx_paras.output_padding_h -
                                            conv3ddx_paras.tiling_param.pad_u - conv3ddx_paras.tiling_param.pad_d;

    int64_t standard_w = conv3ddx_paras.tiling_param.stride_w * (conv3ddx_paras.tiling_param.a_shape.w - 1) +
                         ((conv3ddx_paras.tiling_param.b_shape.w - 1) * conv3ddx_paras.tiling_param.dilation_w + 1);
    conv3ddx_paras.tiling_param.c_shape.w = standard_w + conv3ddx_paras.output_padding_w -
                                            conv3ddx_paras.tiling_param.pad_l - conv3ddx_paras.tiling_param.pad_r;
    conv3ddx_paras.tiling_param.c_shape.batch = conv3ddx_paras.tiling_param.a_shape.batch;
    conv3ddx_paras.tiling_param.c_shape.c = conv3ddx_paras.tiling_param.b_shape.c;
    conv3ddx_paras.tiling_param.c_shape.c1 = cachetiling::MathUtil::CeilDivision(
        conv3ddx_paras.tiling_param.c_shape.c, conv3ddx_paras.tiling_param.c_shape.c0);
  }
  return true;
}

bool CheckCalPads(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  int64_t do_expect = (conv3ddx_paras.tiling_param.c_shape.d + conv3ddx_paras.tiling_param.pad_h +
                      conv3ddx_paras.tiling_param.pad_t - conv3ddx_paras.tiling_param.filter_d_dilation) /
                      conv3ddx_paras.tiling_param.stride_d + 1;
  int64_t ho_expect = (conv3ddx_paras.tiling_param.c_shape.h + conv3ddx_paras.tiling_param.pad_u +
                      conv3ddx_paras.tiling_param.pad_d - conv3ddx_paras.tiling_param.filter_h_dilation) /
                      conv3ddx_paras.tiling_param.stride_h + 1;
  int64_t wo_expect = (conv3ddx_paras.tiling_param.c_shape.w + conv3ddx_paras.tiling_param.pad_l +
                      conv3ddx_paras.tiling_param.pad_r - conv3ddx_paras.tiling_param.filter_w_dilation) /
                      conv3ddx_paras.tiling_param.stride_w + 1;
  OP_TILING_CHECK(
      do_expect != conv3ddx_paras.tiling_param.a_shape.d || ho_expect != conv3ddx_paras.tiling_param.a_shape.h ||
          wo_expect != conv3ddx_paras.tiling_param.a_shape.w,
      CUBE_INNER_ERR_REPORT(
          context, "out_backprop's shape[%ld,%ld,%ld,%ld,%ld] is not equal with inferred shape[%ld,%ld,%ld,%ld,%ld]",
          conv3ddx_paras.tiling_param.a_shape.batch, conv3ddx_paras.tiling_param.a_shape.c,
          conv3ddx_paras.tiling_param.a_shape.d, conv3ddx_paras.tiling_param.a_shape.h,
          conv3ddx_paras.tiling_param.a_shape.w,
          conv3ddx_paras.tiling_param.a_shape.batch, conv3ddx_paras.tiling_param.a_shape.c,
          do_expect, ho_expect, wo_expect),
      return false);
  return true;
}

static bool CalPads(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras, cachetiling::OpType op_type) {
  auto attrs = context->GetAttrs();
  size_t padding_attr_idx = kPaddingConv3dBpInputIdx;
  if (op_type == cachetiling::kConv3DTranspose) {
    padding_attr_idx = kPaddingConv3dTransposeIdx;
  }
  if (attrs->GetAttrNum() <= padding_attr_idx) {
    OP_LOGD(context, "no padding attr, skip calc and check");
    conv3ddx_paras.tiling_param.filter_d_dilation += conv3ddx_paras.output_padding_d;
    conv3ddx_paras.tiling_param.filter_h_dilation += conv3ddx_paras.output_padding_h;
    conv3ddx_paras.tiling_param.filter_w_dilation += conv3ddx_paras.output_padding_w;
    return true;
  }

  auto padding = attrs->GetAttrPointer<char>(padding_attr_idx);
  if (padding != nullptr && (padding[0] == 'S')) {
    int32_t pad_d = std::max(Align(conv3ddx_paras.tiling_param.c_shape.d, conv3ddx_paras.tiling_param.stride_d) -
                                 conv3ddx_paras.tiling_param.stride_d + conv3ddx_paras.tiling_param.filter_d_dilation -
                                 conv3ddx_paras.tiling_param.c_shape.d,
                             0L);
    int32_t pad_head = (pad_d >> 1L);
    int32_t pad_tail = pad_d - pad_head;
    int32_t pad_h = std::max(Align(conv3ddx_paras.tiling_param.c_shape.h, conv3ddx_paras.tiling_param.stride_h) -
                                 conv3ddx_paras.tiling_param.stride_h + conv3ddx_paras.tiling_param.filter_h_dilation -
                                 conv3ddx_paras.tiling_param.c_shape.h,
                             0L);
    int32_t pad_up = (pad_h >> 1L);
    int32_t pad_down = pad_h - pad_up;
    int32_t pad_w = std::max(Align(conv3ddx_paras.tiling_param.c_shape.w, conv3ddx_paras.tiling_param.stride_w) -
                                 conv3ddx_paras.tiling_param.stride_w + conv3ddx_paras.tiling_param.filter_w_dilation -
                                 conv3ddx_paras.tiling_param.c_shape.w,
                             0L);
    int32_t pad_left = (pad_w >> 1L);
    int32_t pad_right = pad_w - pad_left;
    conv3ddx_paras.tiling_param.pad_h = pad_head;
    conv3ddx_paras.tiling_param.pad_t = pad_tail;
    conv3ddx_paras.tiling_param.pad_u = pad_up;
    conv3ddx_paras.tiling_param.pad_d = pad_down;
    conv3ddx_paras.tiling_param.pad_l = pad_left;
    conv3ddx_paras.tiling_param.pad_r = pad_right;
  }

  if (op_type == cachetiling::kConv3DTranspose) {
    OP_LOGE_IF(!HandleConv3DTranspose(context, conv3ddx_paras), false, context, "Failed to process Conv3DTranspose.");
  }

  return CheckCalPads(context, conv3ddx_paras);
}

static bool CalRealG(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  // calc real g and check shape
  int32_t dy_c_ori = conv3ddx_paras.tiling_param.a_shape.c / conv3ddx_paras.tiling_param.groups;
  OP_TILING_CHECK(dy_c_ori == 0,
                  CUBE_INNER_ERR_REPORT(context, "Given groups %d , expected out_backporp to be at least %d at dimension 1,  \
                  but got out_backporp of size %ld  instead",
                                        conv3ddx_paras.tiling_param.groups, conv3ddx_paras.tiling_param.groups,
                                        conv3ddx_paras.tiling_param.a_shape.c),
                  return false);                                      
  int32_t dx_c_extend = Lcm(conv3ddx_paras.tiling_param.b_shape.c, conv3ddx_paras.tiling_param.c_shape.c0) /
                        conv3ddx_paras.tiling_param.b_shape.c;
  int32_t dy_c_extend = Lcm(dy_c_ori, kBlockSize) / dy_c_ori;
  conv3ddx_paras.multiple_extend = cachetiling::MathUtil::Min(Lcm(dx_c_extend, dy_c_extend),
      conv3ddx_paras.tiling_param.groups);
  conv3ddx_paras.tiling_param.real_g =
      (static_cast<int64_t>(conv3ddx_paras.tiling_param.groups) + conv3ddx_paras.multiple_extend - 1) /
      conv3ddx_paras.multiple_extend;
  conv3ddx_paras.tiling_param.ci1g = cachetiling::MathUtil::CeilDivision(
      conv3ddx_paras.multiple_extend * conv3ddx_paras.tiling_param.b_shape.c, conv3ddx_paras.tiling_param.c_shape.c0);
  int32_t co1g = (conv3ddx_paras.multiple_extend * dy_c_ori + kBlockSize - 1) / kBlockSize;
  if (context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT && conv3ddx_paras.tiling_param.groups > 1) {
      co1g *= 2; // 2: BLOCK_NUM / FP32_C0
  }
  return true;
}

static int32_t CalBackpropPadBefore(int32_t filter, int32_t dilation, int32_t pad) {
  return (filter - 1) * dilation - pad;
}

static int64_t CalBackpropPadAfter(int64_t inputDim, int64_t outputDim, int32_t stride, int32_t pad) {
  // orginal formula is inputDim = (outputDim * stride + 1) - padBefore + filterDilation, it can be simplified as follow.
  return inputDim - outputDim * stride + pad;
}

static bool CalModifyBackpropPadD(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  cachetiling::Shape &dedyShape = conv3ddx_paras.tiling_param.a_shape;
  cachetiling::Shape &filterShape = conv3ddx_paras.tiling_param.b_shape;
  cachetiling::Shape &dedxShape = conv3ddx_paras.tiling_param.c_shape;
  cachetiling::Conv3DBpInputTilingParam &tilingParam = conv3ddx_paras.tiling_param;

  conv3ddx_paras.pad_head_before = CalBackpropPadBefore(filterShape.d, tilingParam.dilation_d, tilingParam.pad_h);
  int64_t pad_tail_after = CalBackpropPadAfter(dedxShape.d, dedyShape.d, tilingParam.stride_d, tilingParam.pad_h);
  OP_LOGE_IF(IsOverflowInt32(pad_tail_after) || !CheckRange(static_cast<int32_t>(pad_tail_after), -kDimUp, kDimUp),
    false, context, "pad_tail_after = (inputD - outputD * strideD + padHead)=%ld is invalid, it should be in[%d, %d]",
    pad_tail_after, -kDimUp, kDimUp);
  pad_tail_after = (pad_tail_after + abs(pad_tail_after)) / kNumTwo;
  conv3ddx_paras.pad_tail_after = pad_tail_after;
  conv3ddx_paras.tiling_param.backprop_pad_h = conv3ddx_paras.pad_head_before;
  conv3ddx_paras.tiling_param.backprop_pad_t = conv3ddx_paras.pad_tail_after;
  return true;
}

static bool CalModifyBackpropPadHW(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  cachetiling::Shape &dedyShape = conv3ddx_paras.tiling_param.a_shape;
  cachetiling::Shape &filterShape = conv3ddx_paras.tiling_param.b_shape;
  cachetiling::Shape &dedxShape = conv3ddx_paras.tiling_param.c_shape;
  cachetiling::Conv3DBpInputTilingParam &tilingParam = conv3ddx_paras.tiling_param;

  conv3ddx_paras.pad_left_before = CalBackpropPadBefore(filterShape.w, tilingParam.dilation_w, tilingParam.pad_l);
  conv3ddx_paras.pad_up_before = CalBackpropPadBefore(filterShape.h, tilingParam.dilation_h, tilingParam.pad_u);

  OP_LOGE_IF(!CheckRange(conv3ddx_paras.pad_left_before, 0, kPadUp), false, context,
    "backprop_pad_left=((kw - 1) * dilation_w - pad_left)=[%d] is invalid, it should be in [%d, %d]",
    conv3ddx_paras.pad_left_before, 0 , kPadUp);
  OP_LOGE_IF(!CheckRange(conv3ddx_paras.pad_up_before, 0, kPadUp), false, context,
    "backprop_pad_up=((kh - 1) * dilation_h - pad_up)=[%d] is invalid, it should be in [%d, %d]",
    conv3ddx_paras.pad_up_before, 0 , kPadUp);

  conv3ddx_paras.shape_left_modify = (conv3ddx_paras.pad_left_before - abs(conv3ddx_paras.pad_left_before)) / kNumTwo;
  conv3ddx_paras.shape_up_modify = (conv3ddx_paras.pad_up_before - abs(conv3ddx_paras.pad_up_before)) / kNumTwo;

  int64_t pad_right_after = CalBackpropPadAfter(dedxShape.w, dedyShape.w, tilingParam.stride_w, tilingParam.pad_l);
  int64_t pad_down_after = CalBackpropPadAfter(dedxShape.h, dedyShape.h, tilingParam.stride_h, tilingParam.pad_u);

  OP_LOGE_IF(IsOverflowInt32(pad_right_after) || !CheckRange(static_cast<int32_t>(pad_right_after), -kPadUp, kPadUp),
    false, context, "backprop_right_pad = (inputW - outputW * strideW + padLeft)=%ld is invalid, it should be in[%d, %d]",
    pad_right_after, -kPadUp, kPadUp);

  OP_LOGE_IF(IsOverflowInt32(pad_down_after) || !CheckRange(static_cast<int32_t>(pad_down_after), -kPadUp, kPadUp),
    false, context, "backprop_down_pad = (inputH - outputH * strideH + padUp)=%ld is invalid, it should be in[%d, %d]",
    pad_down_after, -kPadUp, kPadUp);

  int64_t shape_down_modify = (pad_down_after - abs(pad_down_after)) / kNumTwo;
  int64_t shape_right_modify = (pad_right_after - abs(pad_right_after)) / kNumTwo;

  conv3ddx_paras.pad_left_before = (conv3ddx_paras.pad_left_before + abs(conv3ddx_paras.pad_left_before)) / kNumTwo;
  pad_down_after = (pad_down_after + abs(pad_down_after)) / kNumTwo;
  pad_right_after = (pad_right_after + abs(pad_right_after)) / kNumTwo;

  conv3ddx_paras.pad_right_after = pad_right_after;
  conv3ddx_paras.pad_down_after = pad_down_after;
  conv3ddx_paras.shape_right_modify = shape_right_modify;
  conv3ddx_paras.shape_down_modify = shape_down_modify;

  conv3ddx_paras.tiling_param.backprop_pad_u = conv3ddx_paras.pad_up_before;
  conv3ddx_paras.tiling_param.backprop_pad_d = conv3ddx_paras.pad_down_after;
  conv3ddx_paras.tiling_param.backprop_pad_l = conv3ddx_paras.pad_left_before;
  conv3ddx_paras.tiling_param.backprop_pad_r = conv3ddx_paras.pad_right_after;
  return true;
}

static bool CalModify(gert::TilingContext *context, Conv3DDxParas &conv3ddx_paras) {
  OP_LOGE_IF(!CalModifyBackpropPadD(context, conv3ddx_paras), false, context, "Cal backprop pad d invalid");
  OP_LOGE_IF(!CalModifyBackpropPadHW(context, conv3ddx_paras), false, context, "Cal backprop pad h,w invalid");
  return true;
}

bool Conv3DBackpropInputParseFunc(gert::TilingContext *context, cachetiling::OpType op_type,
                                         Conv3DDxParas &conv3ddx_paras, bool isV2Impl) {
  const auto op_name = context->GetNodeName();
  OP_LOGE_IF(!GetAttrAndDtypeParams(context, conv3ddx_paras, op_type), false, op_name, "Get attrs and dtype Failed.");
  OP_LOGE_IF(!GetShapeParams(context, conv3ddx_paras, op_type, isV2Impl), false, op_name, "Set shape params failed.");
  ReCalDilation(context, conv3ddx_paras);
  OP_LOGE_IF(!CalGroups(context, conv3ddx_paras), false, op_name, "Calc groups failed.");
  OP_LOGE_IF(!CalPads(context, conv3ddx_paras, op_type), false, op_name, "Calc pads failed.");
  OP_LOGE_IF(!CalRealG(context, conv3ddx_paras), false, op_name, "Calc real_g failed.");
  OP_LOGE_IF(!CalModify(context, conv3ddx_paras), false, op_name, "Modify pad failed.");
  return true;
}

static bool UpdateRunInfoBinary(Conv3DDxParas &conv3ddx_paras, const cachetiling::Conv3DBpInputTiling &tiling,
                                gert::TilingContext *context) {
  auto tiling_data = context->GetRawTilingData();
  size_t capacity = tiling_data->GetCapacity();
  OP_TILING_CHECK(capacity < sizeof(RunInfoPara),
                  CUBE_INNER_ERR_REPORT(context, "insufficient tiling data capacity %zu", capacity), return false);
  auto run = reinterpret_cast<RunInfoPara *>(tiling_data->GetData());
  OP_TILING_CHECK(!SetRunInfoConv3DDx(conv3ddx_paras, tiling, *run, context),
                  CUBE_INNER_ERR_REPORT(context, "Set run info failed."), return false);
  tiling_data->SetDataSize(sizeof(RunInfoPara));
  context->SetBlockDim(
      static_cast<uint32_t>(tiling.group_dim * tiling.batch_dim * tiling.d_dim * tiling.n_dim * tiling.m_dim));
  context->SetTilingKey(static_cast<uint64_t>(tiling.tiling_id));
  return true;
}

bool cachetiling::TransRepoTiling(tuningtiling::TuningTilingDefPtr &repo_tiling,
                                  cachetiling::Conv3DBpInputTiling &tiling, gert::TilingContext *context) {
  auto repo_dx_tiling = std::dynamic_pointer_cast<tuningtiling::Conv3DDxTunnerTiling>(repo_tiling);
  if (repo_dx_tiling == nullptr) {
    const auto op_name = context->GetNodeName();
    OP_LOGI(op_name, "Unable to get repo tiling");
    return false;
  }
  tiling.group_dim = repo_dx_tiling->group_dim;
  tiling.batch_dim = repo_dx_tiling->batch_dim;
  tiling.d_dim = repo_dx_tiling->d_dim;
  tiling.n_dim = repo_dx_tiling->n_dim;
  tiling.m_dim = repo_dx_tiling->m_dim;
  tiling.m_al1 = repo_dx_tiling->m_al1;
  tiling.n_bl1 = repo_dx_tiling->n_bl1;
  tiling.d_al1 = repo_dx_tiling->d_al1;
  tiling.d_bl1 = repo_dx_tiling->d_bl1;
  tiling.k_aub = repo_dx_tiling->k_aub;
  tiling.m_aub = repo_dx_tiling->m_aub;
  tiling.wo_aub = repo_dx_tiling->wo_aub;
  tiling.m_l0 = repo_dx_tiling->m_l0;
  tiling.n_l0 = repo_dx_tiling->n_l0;
  tiling.d_al0 = repo_dx_tiling->d_al0;
  tiling.d_bl0 = repo_dx_tiling->d_bl0;
  tiling.d_cl0 = repo_dx_tiling->d_cl0;
  tiling.n_cub = repo_dx_tiling->n_cub;
  tiling.k_l0 = repo_dx_tiling->k_l0;
  tiling.k_al1 = repo_dx_tiling->k_al1;
  tiling.k_bl1 = repo_dx_tiling->k_bl1;
  tiling.al1_bound = repo_dx_tiling->al1_bound;
  tiling.bl1_bound = repo_dx_tiling->bl1_bound;
  tiling.aub_bound = repo_dx_tiling->aub_bound;
  tiling.tiling_id = repo_dx_tiling->tiling_id;
  return true;
}

bool cachetiling::GetTilingFromRepo(const cachetiling::CubeTilingParam &params,
                                    cachetiling::Conv3DBpInputTiling &tiling, gert::TilingContext *context,
                                    cachetiling::OpType op_type) {
  // std::shared_ptr<tuningtiling::TuningTilingDef> repo_tiling = nullptr;
  // if (params.platform_info.support_l0c2out()) {
  //   if (params.platform_info.soc_version() != "") {
  //     static RuntimeKb::PlatformInfo plt(params.platform_info.core_num(), params.platform_info.soc_version());
  //     if (op_type == cachetiling::kConv3DBackpropInput) {
  //       (void)RuntimeKb::RuntimeBankManager::Instance().Query(context, "Conv3DBackpropInput", plt, repo_tiling);
  //     } else if (op_type == cachetiling::kConv3DTranspose) {
  //       (void)RuntimeKb::RuntimeBankManager::Instance().Query(context, "Conv3DTranspose", plt, repo_tiling);
  //     }
  //   } else {
  //     OP_LOGD(context, "Unable to determine platform.");
  //   }
  // }

  // if (repo_tiling != nullptr) {
  //   return TransRepoTiling(repo_tiling, tiling, context);
  // }
  return true;
}

ge::graphStatus TilingForConv3DDx(gert::TilingContext *context, cachetiling::OpType op_type) {
  auto compile_info = reinterpret_cast<const Conv3DBackPropInputCompileInfo *>(context->GetCompileInfo());
  OP_TILING_CHECK(compile_info == nullptr, CUBE_INNER_ERR_REPORT(context, "compile_info is null"),
                  return ge::GRAPH_FAILED);
  // cachetiling::Timer timer(context->GetNodeType(), "OpTiling");
  // OP_LOGI(context, "%s", optiling::DebugTilingContext(context).c_str());

  if (compile_info->repo_binary_flag) {
    Conv3DDxParas conv3ddx_paras(cachetiling::kConv3DBackpropInput);
    conv3ddx_paras.tiling_param.binary_mode = compile_info->binary_mode;
    conv3ddx_paras.tiling_param.platform_info.SetRuntimePlatformInfo(*compile_info);
    OP_TILING_CHECK(!Conv3DBackpropInputParseFunc(context, op_type, conv3ddx_paras),
                    CUBE_INNER_ERR_REPORT(context, "failed to parse context"), return ge::GRAPH_FAILED);
    if (!CheckParams(conv3ddx_paras, context)) {
      OP_LOGE(context, "params is invalid");
      return ge::GRAPH_FAILED;
    }
    cachetiling::Conv3DBpInputTiling tiling;
    cachetiling::Conv3DBpInputTilingParam &conv3ddxTilingParam = conv3ddx_paras.tiling_param;
    bool cache_tiling_invalid =
        !cachetiling::GetTiling<cachetiling::Conv3DBpInputTilingParam, cachetiling::Conv3DBpInputTiling,
                                cachetiling::Conv3DBpInputHashParam, cachetiling::Conv3DBpInputHashItem>(
            conv3ddxTilingParam, tiling, context, op_type) ||
        !UpdateRunInfoBinary(conv3ddx_paras, tiling, context);
    if (cache_tiling_invalid) {
      OP_LOGE(context, "binary mode failed");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  if (op_type == cachetiling::kConv3DTranspose) {
    kConv3dBpInputDedyInputIndex = 1;
  }
  return TilingForConv3DBpInput(context, kConv3dBpInputDedyInputIndex, true);
}

static ge::graphStatus TilingForConv3DBpInput(gert::TilingContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT(context, "context is null"), return ge::GRAPH_FAILED);
  return TilingForConv3DDx(context, cachetiling::kConv3DBackpropInput);
};

IMPL_OP_OPTILING(Conv3DBackpropInput)
    .Tiling(TilingForConv3DBpInput)
    .TilingParse<Conv3DBackPropInputCompileInfo>(ParseCubeCompileInfo<Conv3DBackPropInputCompileInfo, 4>);  // 4: ndhw
}  // namespace optiling
