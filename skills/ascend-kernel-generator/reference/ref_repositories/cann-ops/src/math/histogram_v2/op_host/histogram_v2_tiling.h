/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file histogram_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_HISTOGRAM_V2_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_HISTOGRAM_V2_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HistogramV2TilingData)
  TILING_DATA_FIELD_DEF(int64_t, bins);
  TILING_DATA_FIELD_DEF(int64_t, ubBinsLength);

  TILING_DATA_FIELD_DEF(int64_t, formerNum);
  TILING_DATA_FIELD_DEF(int64_t, formerLength);
  TILING_DATA_FIELD_DEF(int64_t, formerLengthAligned);
  TILING_DATA_FIELD_DEF(int64_t, tailLength);
  TILING_DATA_FIELD_DEF(int64_t, tailLengthAligned);

  TILING_DATA_FIELD_DEF(int64_t, formerTileNum);
  TILING_DATA_FIELD_DEF(int64_t, formerTileDataLength);
  TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLength);
  TILING_DATA_FIELD_DEF(int64_t, formerTileLeftDataLengthAligned);

  TILING_DATA_FIELD_DEF(int64_t, tailTileNum);
  TILING_DATA_FIELD_DEF(int64_t, tailTileDataLength);
  TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLength);
  TILING_DATA_FIELD_DEF(int64_t, tailTileLeftDataLengthAligned);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(HistogramV2, HistogramV2TilingData)
}
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_HISTOGRAM_V2_H