
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData)
    // Task maps to (n, oh, coBlock2, owTile)
    TILING_DATA_FIELD_DEF(uint32_t, tasks);
    TILING_DATA_FIELD_DEF(uint32_t, ow);
    TILING_DATA_FIELD_DEF(uint32_t, oh);
    TILING_DATA_FIELD_DEF(uint32_t, w);
    TILING_DATA_FIELD_DEF(uint32_t, h);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);

    TILING_DATA_FIELD_DEF(uint32_t, tile_ow);
    TILING_DATA_FIELD_DEF(uint32_t, ow_tiles);

    TILING_DATA_FIELD_DEF(uint32_t, coblock);      // fixed 2
    TILING_DATA_FIELD_DEF(uint32_t, co_blocks);    // COUT/2

    // Interior tiles on OW dimension where all horizontal taps for all lanes are in-bounds
    TILING_DATA_FIELD_DEF(uint32_t, interior_tile_begin); // inclusive
    TILING_DATA_FIELD_DEF(uint32_t, interior_tile_end);   // exclusive
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustom,
                           ConvStandard2dSquareInputAsymmetricKernelDilatedPaddedCustomTilingData)
} // namespace optiling
