/*
* 版权所有 (c) 华为技术有限公司 2024
*/
// Copyright 2024 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. 
#ifndef XLOGY_TILING_H 
#define XLOGY_TILING_H

#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(XlogyTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, x1_length);
  TILING_DATA_FIELD_DEF(uint32_t, x2_length);
  TILING_DATA_FIELD_DEF(int64_t, numshapes);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 128, shape);  
  TILING_DATA_FIELD_DEF_ARR(int64_t, 64, shapefull);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Xlogy, XlogyTilingData)
}

#endif