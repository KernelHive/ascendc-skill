
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard2dSquareInputAsymmetricKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);        // rows = N*COUT*HO
    TILING_DATA_FIELD_DEF(uint32_t, wo);          // WO
    TILING_DATA_FIELD_DEF(uint32_t, h);           // H
    TILING_DATA_FIELD_DEF(uint32_t, w);           // W
    TILING_DATA_FIELD_DEF(uint32_t, cin);         // CIN
    TILING_DATA_FIELD_DEF(uint32_t, cout);        // COUT
    TILING_DATA_FIELD_DEF(uint32_t, ho);          // HO
    TILING_DATA_FIELD_DEF(uint32_t, tile_wo);     // WO tile size
    TILING_DATA_FIELD_DEF(uint32_t, wo_tiles);    // ceil_div(WO, tile_wo)
    TILING_DATA_FIELD_DEF(uint32_t, tasks);       // tasks = rows * wo_tiles
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard2dSquareInputAsymmetricKernelCustom,
                           ConvStandard2dSquareInputAsymmetricKernelCustomTilingData)
} // namespace optiling
