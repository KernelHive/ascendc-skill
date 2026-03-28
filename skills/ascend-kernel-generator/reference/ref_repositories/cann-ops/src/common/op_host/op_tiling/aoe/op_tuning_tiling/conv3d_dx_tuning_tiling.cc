/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aoe/op_tuning_tiling/conv3d_dx_tuning_tiling.h"

#include "aoe/runtime_kb/common/kb_common.h"
#include "aoe/runtime_kb/common/kb_log.h"
#include "nlohmann/json.hpp"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
static void GetAttrsInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DDxInputArgs> &conv3d_dx_args) {
  auto attrs = context->GetAttrs();
  const gert::ContinuousVector *strides_list = nullptr;
  const gert::ContinuousVector *pads_list = nullptr;
  const gert::ContinuousVector *dilations_list = nullptr;
  const int64_t *groups = nullptr;
  size_t idx = 0;

  strides_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  pads_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  dilations_list = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  groups = attrs->GetAttrPointer<int64_t>(idx++);

  conv3d_dx_args->groups = static_cast<int32_t>(*groups);
  const int64_t *strides_list_data = reinterpret_cast<const int64_t *>(strides_list->GetData());
  const int64_t *pads_list_data = reinterpret_cast<const int64_t *>(pads_list->GetData());
  const int64_t *dilations_list_data = reinterpret_cast<const int64_t *>(dilations_list->GetData());

  if (context->GetOutputDesc(0)->GetOriginFormat() == ge::FORMAT_NCDHW) {
    // data format is NCDHW; 2, 3 is index of strides_d, strides_h, strides_w and dilation_d, dilation_h, dilation_w
    conv3d_dx_args->stride_expand_d = static_cast<int32_t>(strides_list_data[2]);
    conv3d_dx_args->stride_expand_h = static_cast<int32_t>(strides_list_data[3]);
    conv3d_dx_args->stride_expand_w = static_cast<int32_t>(strides_list_data[4]);
    conv3d_dx_args->dilation_d = static_cast<int32_t>(dilations_list_data[2]);
    conv3d_dx_args->dilation_h = static_cast<int32_t>(dilations_list_data[3]);
    conv3d_dx_args->dilation_w = static_cast<int32_t>(dilations_list_data[4]);
  } else {
    // data format is NDHWC
    conv3d_dx_args->stride_expand_d = static_cast<int32_t>(strides_list_data[1]);
    conv3d_dx_args->stride_expand_h = static_cast<int32_t>(strides_list_data[2]);
    conv3d_dx_args->stride_expand_w = static_cast<int32_t>(strides_list_data[3]);
    conv3d_dx_args->dilation_d = static_cast<int32_t>(dilations_list_data[1]);
    conv3d_dx_args->dilation_h = static_cast<int32_t>(dilations_list_data[2]);
    conv3d_dx_args->dilation_w = static_cast<int32_t>(dilations_list_data[3]);
  }

  // 0, 1, 2, 3, 4, 5 is pad index for pad_h, pad_t, pad_u, pad_b, pad_l, pad_r
  conv3d_dx_args->pad_h = static_cast<int32_t>(pads_list_data[0]);
  conv3d_dx_args->pad_t = static_cast<int32_t>(pads_list_data[1]);
  conv3d_dx_args->pad_u = static_cast<int32_t>(pads_list_data[2]);
  conv3d_dx_args->pad_d = static_cast<int32_t>(pads_list_data[3]);
  conv3d_dx_args->pad_l = static_cast<int32_t>(pads_list_data[4]);
  conv3d_dx_args->pad_r = static_cast<int32_t>(pads_list_data[5]);
  // 3ddx unsupport fp32 now, hf32 value default to 0
  conv3d_dx_args->hf32_flag = 0;

  conv3d_dx_args->cub_double_num = 0;
  conv3d_dx_args->fused_double_operand_num = 0;
  conv3d_dx_args->reserved_params1 = 0;
  conv3d_dx_args->reserved_params2 = 0;
  conv3d_dx_args->reserved_params3 = 0;
  conv3d_dx_args->reserved_params4 = 0;
  conv3d_dx_args->reserved_params5 = 0;
}

static void GetDedxInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DDxInputArgs> &conv3d_dx_args) {
  auto dedx_desc = context->GetOutputDesc(0);
  conv3d_dx_args->c_dtype = dedx_desc->GetDataType();

  auto dedx_ori_format = dedx_desc->GetOriginFormat();
  conv3d_dx_args->c_format = dedx_ori_format;

  auto dedx_shape = context->GetOutputShape(0);
  auto &dedx_ori_shape = dedx_shape->GetOriginShape();

  if (dedx_ori_format == ge::FORMAT_NCDHW) {
    // output format is NCDHW; 2, 3, 4 is dim index for dedx_d, dedx_h, dedx_w
    conv3d_dx_args->c_shape_d = dedx_ori_shape.GetDim(2);
    conv3d_dx_args->c_shape_h = dedx_ori_shape.GetDim(3);
    conv3d_dx_args->c_shape_w = dedx_ori_shape.GetDim(4);
  } else {
    // output format is NDHWC
    conv3d_dx_args->c_shape_d = dedx_ori_shape.GetDim(1);
    conv3d_dx_args->c_shape_h = dedx_ori_shape.GetDim(2);
    conv3d_dx_args->c_shape_w = dedx_ori_shape.GetDim(3);
  }
}

static void GetDedyInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DDxInputArgs> &conv3d_dx_args,
                 size_t dedy_input_index) {
  auto dedy_desc = context->GetInputDesc(dedy_input_index);
  conv3d_dx_args->a_dtype = dedy_desc->GetDataType();

  auto dedy_ori_format = dedy_desc->GetOriginFormat();
  conv3d_dx_args->a_format = dedy_ori_format;

  auto dedy_shape = context->GetInputShape(dedy_input_index);
  auto &dedy_ori_shape = dedy_shape->GetOriginShape();

  if (dedy_ori_format == ge::FORMAT_NCDHW) {
    // dedy format is NCDHW; 0, 2, 3, 4 is dim index for dedy_n, dedy_d, dedy_h, dedy_w
    conv3d_dx_args->a_shape_n = dedy_ori_shape.GetDim(0);
    conv3d_dx_args->a_shape_d = dedy_ori_shape.GetDim(2);
    conv3d_dx_args->a_shape_h = dedy_ori_shape.GetDim(3);
    conv3d_dx_args->a_shape_w = dedy_ori_shape.GetDim(4);
  } else {
    // NDHWC
    conv3d_dx_args->a_shape_n = dedy_ori_shape.GetDim(0);
    conv3d_dx_args->a_shape_d = dedy_ori_shape.GetDim(1);
    conv3d_dx_args->a_shape_h = dedy_ori_shape.GetDim(2);
    conv3d_dx_args->a_shape_w = dedy_ori_shape.GetDim(3);
  }
}

static void GetFilterInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DDxInputArgs> &conv3d_dx_args,
                   size_t filter_input_index) {
  auto filter_desc = context->GetInputDesc(filter_input_index);
  conv3d_dx_args->b_dtype = filter_desc->GetDataType();

  auto filter_ori_format = filter_desc->GetOriginFormat();
  conv3d_dx_args->b_format = filter_ori_format;

  auto filter_shape = context->GetInputShape(filter_input_index);
  auto &filter_ori_shape = filter_shape->GetOriginShape();

  if (filter_ori_format == ge::FORMAT_NCDHW) {
    // filter format is NCDHW; 0, 1, 2, 3, 4 is dim index for filter_n, filter_c, filter_d, filter_h, filter_w
    conv3d_dx_args->b_shape_n = filter_ori_shape.GetDim(0);
    conv3d_dx_args->b_shape_c = filter_ori_shape.GetDim(1);
    conv3d_dx_args->b_shape_d = filter_ori_shape.GetDim(2);
    conv3d_dx_args->b_shape_h = filter_ori_shape.GetDim(3);
    conv3d_dx_args->b_shape_w = filter_ori_shape.GetDim(4);
  } else if (filter_ori_format == ge::FORMAT_DHWCN) {
    // filter format is DHWCN
    conv3d_dx_args->b_shape_n = filter_ori_shape.GetDim(4);
    conv3d_dx_args->b_shape_c = filter_ori_shape.GetDim(3);
    conv3d_dx_args->b_shape_d = filter_ori_shape.GetDim(0);
    conv3d_dx_args->b_shape_h = filter_ori_shape.GetDim(1);
    conv3d_dx_args->b_shape_w = filter_ori_shape.GetDim(2);
  } else {
    // filter format is NDHWC
    conv3d_dx_args->b_shape_n = filter_ori_shape.GetDim(0);
    conv3d_dx_args->b_shape_c = filter_ori_shape.GetDim(4);
    conv3d_dx_args->b_shape_d = filter_ori_shape.GetDim(1);
    conv3d_dx_args->b_shape_h = filter_ori_shape.GetDim(2);
    conv3d_dx_args->b_shape_w = filter_ori_shape.GetDim(3);
  }
}

static std::string DisplayInfoDict(std::shared_ptr<void> &input_args, size_t size, std::string op_type) {
  auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
  auto iter = func.find(ge::AscendString(op_type.c_str()));
  const auto &parse_func = iter->second.GetBankKeyParseFuncV2();
  ge::AscendString info_dict_json_as;
  parse_func(input_args, size, info_dict_json_as);
  std::string info_dict_json_str(info_dict_json_as.GetString(), info_dict_json_as.GetLength());
  return info_dict_json_str;
}

static bool TilingForDynConv3DBackpropInput(const gert::TilingContext *context, std::shared_ptr<void> &input_args,
                                     size_t &size) {
  /*
    input_size     filter       dedy
          \          |           /
           \         |          /
            v        v         v
            Conv3DBackpropInput
                     |
                     |
                     v
                   dedx
  */
  RTKB_CHECK(context == nullptr, CANNKB_LOGE("context is nullptr."), return false);
  std::shared_ptr<Conv3DDxInputArgs> conv3d_dx_args = RuntimeKb::MakeShared<Conv3DDxInputArgs>();
  RTKB_CHECK(conv3d_dx_args == nullptr, CANNKB_LOGE("conv3d_dx_args is nullptr."), return false);
  size_t dedy_input_index = 2;
  size_t filter_input_index = 1;
  GetFilterInfo(context, conv3d_dx_args, filter_input_index);
  GetDedyInfo(context, conv3d_dx_args, dedy_input_index);
  GetDedxInfo(context, conv3d_dx_args);
  GetAttrsInfo(context, conv3d_dx_args);
  input_args = conv3d_dx_args;
  size = sizeof(Conv3DDxInputArgs);

  std::string op_type = "Conv3DBackpropInput";
  CANNKB_LOGD("Op Conv3DBackpropInput query repo by info dict: %s", DisplayInfoDict(input_args, size, op_type).c_str());

  return true;
}

static bool TilingForDynConv3DTranspose(const gert::TilingContext *context, std::shared_ptr<void> &input_args, size_t &size) {
  /*
    input_size     x       filter
          \        |         /
           \       |        /
            v      v       v
            Conv3DTranspose
                   |
                   |
                   v
                   y
    >>add bias by graph fusion
  */
  RTKB_CHECK(context == nullptr, CANNKB_LOGE("context is nullptr."), return false);
  std::shared_ptr<Conv3DDxInputArgs> conv3d_dx_args = RuntimeKb::MakeShared<Conv3DDxInputArgs>();
  RTKB_CHECK(conv3d_dx_args == nullptr, CANNKB_LOGE("conv3d_dx_args is nullptr."), return false);
  size_t x_input_index = 1;
  size_t filter_input_index = 2;
  GetFilterInfo(context, conv3d_dx_args, filter_input_index);
  GetDedyInfo(context, conv3d_dx_args, x_input_index);
  GetDedxInfo(context, conv3d_dx_args);
  GetAttrsInfo(context, conv3d_dx_args);
  input_args = conv3d_dx_args;
  size = sizeof(Conv3DDxInputArgs);

  std::string op_type = "Conv3DTranspose";
  CANNKB_LOGD("Op Conv3DTranspose query repo by info dict: %s", DisplayInfoDict(input_args, size, op_type).c_str());

  return true;
}

DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv3DBackpropInput, Conv3DDxInputArgs, a_dtype, b_dtype, c_dtype, a_shape_n, a_shape_d,
                              a_shape_h, a_shape_w, b_shape_n, b_shape_c, b_shape_d, b_shape_h, b_shape_w, c_shape_d,
                              c_shape_h, c_shape_w, a_format, b_format, c_format, groups, stride_expand_d,
                              stride_expand_h, stride_expand_w, dilation_d, dilation_h, dilation_w, pad_h, pad_t, pad_u,
                              pad_d, pad_l, pad_r, hf32_flag, cub_double_num, fused_double_operand_num,
                              reserved_params1, reserved_params2, reserved_params3, reserved_params4, reserved_params5);

REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(Conv3DBackpropInput, TilingForDynConv3DBackpropInput);
REGISTER_TUNING_TILING_CLASS(Conv3DBackpropInput, Conv3DDxTunnerTiling);

static auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
static auto iter = func.find(ge::AscendString("Conv3DBackpropInput"));
static const auto &dx_parse_func = iter->second.GetBankKeyParseFuncV2();
static const auto &dx_load_func = iter->second.GetBankKeyLoadFuncV2();

REGISTER_OP_BANK_KEY_PARSE_FUN_V2(Conv3DTranspose, dx_parse_func, dx_load_func);
REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(Conv3DTranspose, TilingForDynConv3DTranspose);
REGISTER_TUNING_TILING_CLASS(Conv3DTranspose, Conv3DDxTunnerTiling);
}  // namespace tuningtiling