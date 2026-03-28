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
 * \file instruction_param.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_INSTRUCTION_PARAM_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_INSTRUCTION_PARAM_H_

#include <cstdint>
#include <string>
#include <atomic>

namespace optiling {
namespace cachetiling {
constexpr int32_t kPadMin = 0;
constexpr int32_t kPadMax = 255;
constexpr int32_t kKernelMin = 1;
constexpr int32_t kKernelMax = 255;
constexpr int32_t kKernelMaxV220 = 511;
constexpr int32_t kStrideMin = 1;
constexpr int32_t kStrideMax = 63;
constexpr int32_t kDilationMin = 1;
constexpr int32_t kDilationMax = 255;

struct Load3dParam {
 public:
    int32_t pad_u;
    int32_t pad_d;
    int32_t pad_l;
    int32_t pad_r;
    int32_t stride_h;
    int32_t stride_w;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t dilation_h;
    int32_t dilation_w;
};

class Load3dInstrictionParam {
 public:
  bool IsValid(const Load3dParam &load3d_param) const;
  bool IsPadValid(const Load3dParam &load3d_param) const;
  bool IsStrideValid(const Load3dParam &load3d_param) const;
  bool IsKernelValid(const Load3dParam &load3d_param) const;
  bool IsDilationValid(const Load3dParam &load3d_param) const;
  std::string ToString() const;

  void SetPadRange(int32_t min, int32_t max);
  void SetStrideRange(int32_t min, int32_t max);
  void SetKernelRange(int32_t min, int32_t max);
  void SetDilationRange(int32_t min, int32_t max);

 private:
  std::atomic<int32_t> pad_min_{kPadMin};
  std::atomic<int32_t> pad_max_{kPadMax};
  std::atomic<int32_t> stride_min_{kStrideMin};
  std::atomic<int32_t> stride_max_{kStrideMax};
  std::atomic<int32_t> kernel_min_{kKernelMin};
  std::atomic<int32_t> kernel_max_{kKernelMax};
  std::atomic<int32_t> dilation_min_{kDilationMin};
  std::atomic<int32_t> dilation_max_{kDilationMax};
};

class InstructionParam {
 public:
  static InstructionParam &Instance();
  Load3dInstrictionParam &get_load3d_inst_param() { return load3d_inst_param_; }

 private:
  InstructionParam() {};
  ~InstructionParam() {};

  Load3dInstrictionParam load3d_inst_param_;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_INSTRUCTION_PARAM_H_