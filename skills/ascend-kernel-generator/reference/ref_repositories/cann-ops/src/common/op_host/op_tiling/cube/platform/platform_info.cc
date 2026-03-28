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
 * \file platform_info.cc
 * \brief
 */
#include "cube/platform/platform_info.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {
namespace {
constexpr int32_t kCoreNum = 32;
constexpr int32_t kL2Size = 64 * 1024 * 1024;
constexpr int32_t kL1Size = 1024 * 1024;
constexpr int32_t kL0aSize = 64 * 1024;
constexpr int32_t kL0bSize = 64 * 1024;
constexpr int32_t kL0cSize = 256 * 1024;
constexpr int32_t kUbSize = 256 * 1024;
constexpr int32_t kBlockSize = 16;
}
PlatformInfo::PlatformInfo() {
  core_num_ = kCoreNum;
  l2_size_ = kL2Size;
  l1_size_ = kL1Size;
  l0a_size_ = kL0aSize;
  l0b_size_ = kL0bSize;
  l0c_size_ = kL0cSize;
  ub_size_ = kUbSize;
}

void PlatformInfo::SetRuntimePlatformInfo(const CubeCompileInfo &compile_info) {
  core_num_ = compile_info.core_num;
  l2_size_ = compile_info.l2_size;
  l1_size_ = compile_info.l1_size;
  l0a_size_ = compile_info.l0a_size;
  l0b_size_ = compile_info.l0b_size;
  l0c_size_ = compile_info.l0c_size;
  ub_size_ = compile_info.ub_size;
  bt_size_ = compile_info.bt_size;
  soc_version_ = compile_info.soc_version;
  load3d_constraints_ = compile_info.load3d_constraints;
  intrinsic_data_move_l12ub_ = compile_info.intrinsic_data_move_l12ub;
  intrinsic_data_move_l0c2ub_ = compile_info.intrinsic_data_move_l0c2ub;
  intrinsic_fix_pipe_l0c2out_ = compile_info.intrinsic_fix_pipe_l0c2out;
  intrinsic_fix_pipe_l0c2ub_ = compile_info.intrinsic_fix_pipe_l0c2ub;
  intrinsic_data_move_out2l1_nd2nz_ = compile_info.intrinsic_data_move_out2l1_nd2nz;
}

bool PlatformInfo::SetRuntimePlatformInfo(const nlohmann::json &compile_info) {
  if (!compile_info.contains("hardware_info") || !compile_info["hardware_info"].is_object()) return false;
  std::string hardware_info_keys[] = {"CORE_NUM",
                                      "L2_SIZE",
                                      "L1_SIZE",
                                      "L0A_SIZE",
                                      "L0B_SIZE",
                                      "L0C_SIZE",
                                      "UB_SIZE",
                                      "BT_SIZE",
                                      "load3d_constraints",
                                      "Intrinsic_data_move_l12ub",
                                      "Intrinsic_data_move_l0c2ub",
                                      "Intrinsic_fix_pipe_l0c2out",
                                      "Intrinsic_data_move_out2l1_nd2nz"};
  for (string key : hardware_info_keys) {
    if (!compile_info["hardware_info"].contains(key)) {
      OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "Exception: no %s in compile info", key.c_str());
      return false;
    }
  }

  try {
    core_num_ = compile_info["hardware_info"]["CORE_NUM"].get<uint32_t>();
    l2_size_ = compile_info["hardware_info"]["L2_SIZE"].get<uint64_t>();
    l1_size_ = compile_info["hardware_info"]["L1_SIZE"].get<uint64_t>();
    l0a_size_ = compile_info["hardware_info"]["L0A_SIZE"].get<uint64_t>();
    l0b_size_ = compile_info["hardware_info"]["L0B_SIZE"].get<uint64_t>();
    l0c_size_ = compile_info["hardware_info"]["L0C_SIZE"].get<uint64_t>();
    ub_size_ = compile_info["hardware_info"]["UB_SIZE"].get<uint64_t>();
    bt_size_ = compile_info["hardware_info"]["BT_SIZE"].get<uint64_t>();

    if ("1" == compile_info["hardware_info"]["load3d_constraints"].get<string>()) {
      load3d_constraints_ = true;
    } else {
      load3d_constraints_ = false;
    }

    intrinsic_data_move_l12ub_ = compile_info["hardware_info"]["Intrinsic_data_move_l12ub"].get<bool>();
    intrinsic_data_move_l0c2ub_ = compile_info["hardware_info"]["Intrinsic_data_move_l0c2ub"].get<bool>();
    intrinsic_fix_pipe_l0c2out_ = compile_info["hardware_info"]["Intrinsic_fix_pipe_l0c2out"].get<bool>();
    intrinsic_data_move_out2l1_nd2nz_ = compile_info["hardware_info"]["Intrinsic_data_move_out2l1_nd2nz"].get<bool>();
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO in runtime1.0: %s", ToString().c_str());
  } catch (const std::exception &e) {
    OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "Got exception for parsing compile info: %s", e.what());
    return false;
  }
  return true;
}

bool PlatformInfo::IsValid() const {
  OP_LOGE_IF(core_num_ == 0, false, "NO_OP_NAME", "core num is 0");
  OP_LOGE_IF(l2_size_ == 0, false, "NO_OP_NAME", "L2 size is 0");
  OP_LOGE_IF(l1_size_ == 0, false, "NO_OP_NAME", "L1 size is 0");
  OP_LOGE_IF(l0a_size_ == 0, false, "NO_OP_NAME", "L0A size is 0");
  OP_LOGE_IF(l0b_size_ == 0, false, "NO_OP_NAME", "L0B size is 0");
  OP_LOGE_IF(l0c_size_ == 0, false, "NO_OP_NAME", "L0C size is 0");
  OP_LOGE_IF(ub_size_ == 0, false, "NO_OP_NAME", "UB size is 0");
  return true;
}

bool PlatformInfo::IsValidL1Size(int32_t l1_size) const {
  return l1_size > 0 && static_cast<uint64_t>(l1_size) <= l1_size_;
}

bool PlatformInfo::IsValidL1Size(int64_t l1_size) const {
  return l1_size > 0 && static_cast<uint64_t>(l1_size) <= l1_size_;
}

bool PlatformInfo::IsValidL0ASize(int32_t l0a_size) const {
  return l0a_size > 0 && static_cast<uint64_t>(l0a_size) <= l0a_size_;
}

bool PlatformInfo::IsValidL0BSize(int32_t l0b_size) const {
  return l0b_size > 0 && static_cast<uint64_t>(l0b_size) <= l0b_size_;
}

bool PlatformInfo::IsValidL0CSize(int32_t l0c_size) const {
  return l0c_size > 0 && static_cast<uint64_t>(l0c_size) <= l0c_size_;
}

std::string PlatformInfo::ToString() const {
  std::stringstream ss;
  ss << " load3d_constraints: " << load3d_constraints_
     << " support_l0c2out: " << intrinsic_fix_pipe_l0c2out_
     << " support_data_move_l12ub: " << intrinsic_data_move_l12ub_
     << " support_data_move_l0c2ub: " << intrinsic_data_move_l0c2ub_
     << " support_data_move_out2l1_nd2nz: " << intrinsic_data_move_out2l1_nd2nz_
     << " core num: " << core_num_
     << " l2_size: " << l2_size_
     << " l1_size: " << l1_size_
     << " l0a_size: " << l0a_size_
     << " l0b_size: " << l0b_size_
     << " l0c_size: " << l0c_size_
     << " ub_size: " << ub_size_
     << " bt_size: " << bt_size_;
  return ss.str();
}
}  // namespace cachetiling
}  // namespace optiling