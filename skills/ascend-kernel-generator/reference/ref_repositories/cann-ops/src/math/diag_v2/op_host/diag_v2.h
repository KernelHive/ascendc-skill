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
 * \file diag_v2_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DiagV2TilingData)
  TILING_DATA_FIELD_DEF(int64_t, xWidth);
  TILING_DATA_FIELD_DEF(int64_t, xHeight);
  TILING_DATA_FIELD_DEF(int64_t, gmOffset);
  TILING_DATA_FIELD_DEF(int64_t, numOut);
  TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, numPerCore);
  TILING_DATA_FIELD_DEF(int64_t, tailNum);
  TILING_DATA_FIELD_DEF(int64_t, tilingKey);
  TILING_DATA_FIELD_DEF(int64_t, matrixRowLength);
  TILING_DATA_FIELD_DEF(int64_t, inputNum);  // form here for DiagFlat
  TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, normalCoreHandleNum);
  TILING_DATA_FIELD_DEF(int64_t, lastCoreHandleNum);
  TILING_DATA_FIELD_DEF(int64_t, ubInputSize);
  TILING_DATA_FIELD_DEF(int64_t, ubOutputSize);
  TILING_DATA_FIELD_DEF(int64_t, diagonal);
  TILING_DATA_FIELD_DEF(int64_t, workspaceSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DiagV2, DiagV2TilingData)
}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DIAG_V2_H_
