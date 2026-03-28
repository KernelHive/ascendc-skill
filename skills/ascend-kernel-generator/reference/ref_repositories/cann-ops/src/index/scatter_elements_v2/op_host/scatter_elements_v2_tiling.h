/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file scatter_elements_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsV2TilingData)
  TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, eachNum);
  TILING_DATA_FIELD_DEF(uint64_t, extraTaskCore);
  TILING_DATA_FIELD_DEF(uint64_t, eachPiece);
  TILING_DATA_FIELD_DEF(uint64_t, inputOnePiece);
  TILING_DATA_FIELD_DEF(uint64_t, inputCount);
  TILING_DATA_FIELD_DEF(uint64_t, indicesCount);
  TILING_DATA_FIELD_DEF(uint64_t, updatesCount);
  TILING_DATA_FIELD_DEF(uint64_t, inputOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, indicesOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, updatesOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, inputEach);
  TILING_DATA_FIELD_DEF(uint64_t, indicesEach);
  TILING_DATA_FIELD_DEF(uint64_t, inputLast);
  TILING_DATA_FIELD_DEF(uint64_t, indicesLast);
  TILING_DATA_FIELD_DEF(uint64_t, inputLoop);
  TILING_DATA_FIELD_DEF(uint64_t, indicesLoop);
  TILING_DATA_FIELD_DEF(uint64_t, inputAlign);
  TILING_DATA_FIELD_DEF(uint64_t, indicesAlign);
  TILING_DATA_FIELD_DEF(uint64_t, updatesAlign);
  TILING_DATA_FIELD_DEF(uint64_t, lastIndicesLoop);
  TILING_DATA_FIELD_DEF(uint64_t, lastIndicesEach);
  TILING_DATA_FIELD_DEF(uint64_t, lastIndicesLast);
  TILING_DATA_FIELD_DEF(uint64_t, oneTime);
  TILING_DATA_FIELD_DEF(uint64_t, lastOneTime);
  TILING_DATA_FIELD_DEF(uint64_t, modeFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElementsV2, ScatterElementsV2TilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_SCATTER_ELEMENTS_V2_H
