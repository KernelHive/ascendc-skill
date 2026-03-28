/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AOE_OP_TILING_TUNING_CONV2D_DW_TUNING_TILING_H_
#define AOE_OP_TILING_TUNING_CONV2D_DW_TUNING_TILING_H_
#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"
namespace tuningtiling {
#pragma pack(push, 1)
struct Conv2DDwInputArgs {
  int64_t a_shape_n;
  int64_t a_shape_h;
  int64_t a_shape_w;
  int64_t b_shape_h;
  int64_t b_shape_w;
  int64_t c_shape_n;
  int64_t c_shape_c;
  int64_t c_shape_h;
  int64_t c_shape_w;
  int64_t groups;
  int64_t stride_h;
  int64_t stride_w;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pad_u;
  int64_t pad_d;
  int64_t pad_l;
  int64_t pad_r;
  ge::DataType a_dtype;
  ge::DataType b_dtype;
  ge::DataType c_dtype;
  uint32_t binary_mode;
  int32_t hf32_flag;
  int64_t reserved_params1;
  int64_t reserved_params2;
  int64_t reserved_params3;
  int64_t reserved_params4;
  int64_t reserved_params5;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv2DDwTunnerTiling)
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_dim);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_dim);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_dim);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_dim);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, group_dim);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_al1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_al1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_bl1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_bl1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_l0);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_aub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_bub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_cub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_l0);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_l0);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_l0);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_cub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_aub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_aub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_bub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_bub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_al1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_bl1);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_l0a);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_l0b);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_l0c);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_aub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_bub);
  TUNING_TILING_DATA_FIELD_DEF(uint32_t, db_cub);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv2DDwTunnerTiling,
  FIELD(Conv2DDwTunnerTiling, batch_dim),
  FIELD(Conv2DDwTunnerTiling, n_dim),
  FIELD(Conv2DDwTunnerTiling, m_dim),
  FIELD(Conv2DDwTunnerTiling, k_dim),
  FIELD(Conv2DDwTunnerTiling, group_dim),
  FIELD(Conv2DDwTunnerTiling, m_al1),
  FIELD(Conv2DDwTunnerTiling, k_al1),
  FIELD(Conv2DDwTunnerTiling, k_bl1),
  FIELD(Conv2DDwTunnerTiling, n_bl1),
  FIELD(Conv2DDwTunnerTiling, batch_l0),
  FIELD(Conv2DDwTunnerTiling, batch_aub),
  FIELD(Conv2DDwTunnerTiling, batch_bub),
  FIELD(Conv2DDwTunnerTiling, batch_cub),
  FIELD(Conv2DDwTunnerTiling, m_l0),
  FIELD(Conv2DDwTunnerTiling, k_l0),
  FIELD(Conv2DDwTunnerTiling, n_l0),
  FIELD(Conv2DDwTunnerTiling, n_cub),
  FIELD(Conv2DDwTunnerTiling, m_aub),
  FIELD(Conv2DDwTunnerTiling, k_aub),
  FIELD(Conv2DDwTunnerTiling, k_bub),
  FIELD(Conv2DDwTunnerTiling, n_bub),
  FIELD(Conv2DDwTunnerTiling, db_al1),
  FIELD(Conv2DDwTunnerTiling, db_bl1),
  FIELD(Conv2DDwTunnerTiling, db_l0a),
  FIELD(Conv2DDwTunnerTiling, db_l0b),
  FIELD(Conv2DDwTunnerTiling, db_l0c),
  FIELD(Conv2DDwTunnerTiling, db_aub),
  FIELD(Conv2DDwTunnerTiling, db_bub),
  FIELD(Conv2DDwTunnerTiling, db_cub));
}  // namespace tuningtiling
#endif
