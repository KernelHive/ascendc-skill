
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwiseSeparable2dCustomTilingData)
    // Task maps to (n, oh, owTile, coTile). coTile covers CO_TILE output channels.
    TILING_DATA_FIELD_DEF(uint32_t, tasks);

    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, w);
    TILING_DATA_FIELD_DEF(uint32_t, oh);
    TILING_DATA_FIELD_DEF(uint32_t, ow);

    TILING_DATA_FIELD_DEF(uint32_t, tile_ow);
    TILING_DATA_FIELD_DEF(uint32_t, ow_tiles);

    TILING_DATA_FIELD_DEF(uint32_t, co_tile);
    TILING_DATA_FIELD_DEF(uint32_t, co_tiles);

    // Interior region for ow0 where 3x3 window is fully in-bounds for all lanes in the tile.
    TILING_DATA_FIELD_DEF(uint32_t, ow_interior_start); // inclusive
    TILING_DATA_FIELD_DEF(uint32_t, ow_interior_end);   // exclusive

    // Interior region for oh where rows ih=oh-1..oh+1 are in-bounds (pad=1, kh=3).
    TILING_DATA_FIELD_DEF(uint32_t, oh_interior_start); // inclusive
    TILING_DATA_FIELD_DEF(uint32_t, oh_interior_end);   // exclusive
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwiseSeparable2dCustom,
                           ConvDepthwiseSeparable2dCustomTilingData)
} // namespace optiling
