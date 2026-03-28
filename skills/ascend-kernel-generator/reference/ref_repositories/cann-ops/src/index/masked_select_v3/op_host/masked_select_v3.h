/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file masked_select_v3.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaskedSelectV3TilingData) 
  TILING_DATA_FIELD_DEF(uint64_t, formerNum);
  TILING_DATA_FIELD_DEF(uint64_t, formerLength);
  TILING_DATA_FIELD_DEF(uint64_t, formertileNum);
  TILING_DATA_FIELD_DEF(uint64_t, formertileLength);
  TILING_DATA_FIELD_DEF(uint64_t, formerlasttileLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailtileNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailtileLength);
  TILING_DATA_FIELD_DEF(uint64_t, taillasttileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedSelectV3, MaskedSelectV3TilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_MASKED_SELECT_V3_H