
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PerformerAttentionCustomTilingData)
    // Shapes: q_phi/k_phi [B,H,S,F], v [B,H,S,D], y [B,H,S,D]
    TILING_DATA_FIELD_DEF(uint32_t, b);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, s);
    TILING_DATA_FIELD_DEF(uint32_t, f);
    TILING_DATA_FIELD_DEF(uint32_t, d);

    // Numerics
    TILING_DATA_FIELD_DEF(float, eps);

    // Launch mapping
    TILING_DATA_FIELD_DEF(uint32_t, block_dim); // = B*H

    // Micro-tiling knobs
    TILING_DATA_FIELD_DEF(uint32_t, d_tile);    // used for vectorized mul-add loops
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PerformerAttentionCustom, PerformerAttentionCustomTilingData)
} // namespace optiling
