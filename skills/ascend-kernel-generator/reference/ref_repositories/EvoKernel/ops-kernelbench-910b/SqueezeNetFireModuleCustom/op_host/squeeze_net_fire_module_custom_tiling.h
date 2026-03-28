
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SqueezeNetFireModuleCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalX);
    TILING_DATA_FIELD_DEF(uint32_t, totalWs);
    TILING_DATA_FIELD_DEF(uint32_t, totalBs);
    TILING_DATA_FIELD_DEF(uint32_t, totalW1);
    TILING_DATA_FIELD_DEF(uint32_t, totalB1);
    TILING_DATA_FIELD_DEF(uint32_t, totalW3);
    TILING_DATA_FIELD_DEF(uint32_t, totalB3);
    TILING_DATA_FIELD_DEF(uint32_t, totalY);

    // 2D mapping flattened:
    // blockId -> (rowNH, tileIdx)
    // rowNH in [0, N*H), tileIdx in [0, OUTC/tileC)
    TILING_DATA_FIELD_DEF(uint32_t, rowsNH);      // N*H
    TILING_DATA_FIELD_DEF(uint32_t, outcTiles);   // OUTC/tileC
    TILING_DATA_FIELD_DEF(uint32_t, tileC);       // 8
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SqueezeNetFireModuleCustom,
                           SqueezeNetFireModuleCustomTilingData)
} // namespace optiling
