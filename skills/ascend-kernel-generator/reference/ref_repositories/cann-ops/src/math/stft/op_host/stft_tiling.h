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
 * \file stft_tiling.h
 * \brief
 */
#pragma once
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/tiling_base.h"

namespace optiling {

struct STFTCompileInfo {
  uint64_t coreNum;
  uint64_t aivCoreNum;
  uint64_t aicCoreNum;
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l0ASize;
  uint64_t l0BSize;
  uint64_t l0CSize;
  uint64_t sysWorkspaceSize;
};
}  // namespace optiling
