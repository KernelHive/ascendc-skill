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
 * \file cache_tiling.cc
 * \brief
 */
#include "cube/include/cache_tiling.h"

#include "cube/algorithm/cache_tiling_impl.h"
// #include "cube/util/timer.h"
#include "op_log.h"

namespace optiling {
namespace cachetiling {


static FactoryInst<std::shared_ptr<CacheTilingImpl>> inst_;

bool GenTiling(const CubeTilingParam &params, CubeTiling &tiling) {
  // CACHE_TILING_TIME_STAMP_START(GenTiling);
  // OP_LOG_FULL(DLOG_DEBUG, params.op_type, "[CubeTilingParam][%s]", params.ToString().c_str());
  if (!params.IsValid()) {
    OP_LOGE(params.op_type, "Invalid input param");
    return false;
  }

  auto impl = inst_.Get(params.type);
  if (impl == nullptr) {
    impl = CacheTilingFactory().Create(params.type, params);
    if (impl == nullptr) {
      OP_LOGE(params.op_type, "Creator TilingGenerator failed");
      return false;
    }
    inst_.Add(params.type, impl);
  }
  if (!impl->Init(params)) {
    OP_LOGE(params.op_type, "Failed to init TilingImpl!");
    return false;
  }
  bool res = impl->GenTiling(tiling);
  impl->Clear();
  // CACHE_TILING_TIME_STAMP_END(GenTiling, params.op_type);
  if (!tiling.IsValid()) {
    OP_LOGE(params.op_type, "Invalid output param");
    return false;
  }
  return res;
  // return false;
}

void DestoryTilingFactory() {
  // inst_.Clear();
}
}  // namespace cachetiling
}  // namespace optiling
