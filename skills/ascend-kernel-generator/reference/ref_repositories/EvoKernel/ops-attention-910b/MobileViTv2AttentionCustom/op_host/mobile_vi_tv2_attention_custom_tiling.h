
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MobileViTv2AttentionCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);
    TILING_DATA_FIELD_DEF(uint32_t, nq);
    TILING_DATA_FIELD_DEF(uint32_t, d);

    TILING_DATA_FIELD_DEF(uint32_t, qTile);   // prefer nq (single tile) when small
    TILING_DATA_FIELD_DEF(uint32_t, dTile);   // K/V tile
    TILING_DATA_FIELD_DEF(uint32_t, doTile);  // output-channel tile (per block)

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MobileViTv2AttentionCustom, MobileViTv2AttentionCustomTilingData)
} // namespace optiling
