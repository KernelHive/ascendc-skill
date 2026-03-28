
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTiles); // N*COUT*HOUT*ceil(WOUT/TILE_WO)
    TILING_DATA_FIELD_DEF(uint32_t, tilesPerRow); // ceil(WOUT/TILE_WO)
    TILING_DATA_FIELD_DEF(uint32_t, wout);        // WOUT
    TILING_DATA_FIELD_DEF(uint32_t, tileWo);      // TILE_WO
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);    // launch blocks
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed2dAsymmetricInputAsymmetricKernelCustom,
                           ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData)
} // namespace optiling
