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
 * \file cache_tiling_impl.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_

#include "cube/algorithm/entity/status.h"
#include "cube/include/cache_tiling.h"
#include "cube/include/cube_tiling.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/util/registry.h"

namespace optiling {
namespace cachetiling {
class CacheTilingImpl {
 public:
  explicit CacheTilingImpl(const CubeTilingParam &params);
  virtual ~CacheTilingImpl() { params_ = nullptr; }
  virtual bool GenTiling(CubeTiling &tiling) = 0;
  virtual bool Init(const CubeTilingParam &params);
  virtual void Clear();

 protected:
  void UpdateTiling(CubeTiling &tiling) const;
  void UpdateTilingBlockDims(CubeTiling &tiling) const;
  void UpdateTilingL1Status(CubeTiling &tiling) const;
  void UpdateTilingL0Status(CubeTiling &tiling) const;
  void UpdateTilingUbStatus(CubeTiling &tiling) const;

  const CubeTilingParam *params_ = nullptr;
  SingleCoreStatus single_core_status_;
};

DECLARE_REGISTRY_TYPE(CacheTilingFactory, CacheTilingImpl, const CubeTilingParam &);
#define REGISTER_TILING_GENERATOR(op_type, derived_clazz) \
  REGISTER_TYPE_CLASS(CacheTilingFactory, op_type, derived_clazz)
}  // namespace cachetiling
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_CACHE_TILING_IMPL_H_