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
 * \file instruction_param.cc
 * \brief
 */
#include "cube/platform/instruction_param.h"
#include <sstream>
#include "op_log.h"

namespace optiling {
namespace cachetiling {
InstructionParam &InstructionParam::Instance() {
   static InstructionParam inst;
   return inst;
}

bool Load3dInstrictionParam::IsValid(const Load3dParam &param) const {
   if (!IsPadValid(param)) {
      return false;
   }

   if (!IsStrideValid(param)) {
      return false;
   }

   if (!IsKernelValid(param)) {
      return false;
   }

   if (!IsDilationValid(param)) {
      return false;
   }

   return true;
};

bool Load3dInstrictionParam::IsPadValid(const Load3dParam &param) const {
   bool pad_u_valid = param.pad_u >= pad_min_ && param.pad_u <= pad_max_;
   bool pad_d_valid = param.pad_d >= pad_min_ && param.pad_d <= pad_max_;
   bool pad_l_valid = param.pad_l >= pad_min_ && param.pad_l <= pad_max_;
   bool pad_r_valid = param.pad_r >= pad_min_ && param.pad_r <= pad_max_;
   return pad_u_valid && pad_d_valid && pad_l_valid && pad_r_valid;
}

bool Load3dInstrictionParam::IsStrideValid(const Load3dParam &param) const {
   bool stride_h_valid = param.stride_h >= stride_min_ && param.stride_h <= stride_max_;
   bool stride_w_valid = param.stride_w >= stride_min_ && param.stride_w <= stride_max_;
   return stride_h_valid && stride_w_valid;
}

bool Load3dInstrictionParam::IsKernelValid(const Load3dParam &param) const {
   bool kernel_h_valid = param.kernel_h >= kernel_min_ && param.kernel_h <= kernel_max_;
   bool kernel_w_valid = param.kernel_w >= kernel_min_ && param.kernel_w <= kernel_max_;
   return kernel_h_valid && kernel_w_valid;
}

bool Load3dInstrictionParam::IsDilationValid(const Load3dParam &param) const {
   bool dilation_h_valid = param.dilation_h >= dilation_min_ && param.dilation_h <= dilation_max_;
   bool dilation_w_valid = param.dilation_w >= dilation_min_ && param.dilation_w <= dilation_max_;
   return dilation_h_valid && dilation_w_valid;
}

std::string Load3dInstrictionParam::ToString() const {
  std::stringstream ss;
  ss << "pad up/down/left/right support range is [" << pad_min_ << ", " << pad_max_ << "]"
     << " stride h/w support range is [" << stride_min_ << ", " << stride_max_ << "]"
     << " kernel h/w support range is [" << kernel_min_ << ", " << kernel_max_ << "]"
     << " dilation h/w support range is [" << dilation_min_ << ", " << dilation_max_ << "]";
  return ss.str();
}

void Load3dInstrictionParam::SetPadRange(int32_t min, int32_t max) {
   pad_min_ = min;
   pad_max_ = max;
}

void Load3dInstrictionParam::SetStrideRange(int32_t min, int32_t max) {
   stride_min_ = min;
   stride_max_ = max;
}

void Load3dInstrictionParam::SetKernelRange(int32_t min, int32_t max) {
   kernel_min_ = min;
   kernel_max_ = max;
}

void Load3dInstrictionParam::SetDilationRange(int32_t min, int32_t max) {
   dilation_min_ = min;
   dilation_max_ = max;
}
}
}