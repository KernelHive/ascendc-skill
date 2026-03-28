
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard2dAsymmetricInputSquareKernelCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rows);        // rows = N*COUT*OH
    TILING_DATA_FIELD_DEF(uint32_t, ow);          // OW
    TILING_DATA_FIELD_DEF(uint32_t, h);           // H
    TILING_DATA_FIELD_DEF(uint32_t, w);           // W
    TILING_DATA_FIELD_DEF(uint32_t, cin);         // CIN
    TILING_DATA_FIELD_DEF(uint32_t, cout);        // COUT
    TILING_DATA_FIELD_DEF(uint32_t, oh);          // OH
    TILING_DATA_FIELD_DEF(uint32_t, tile_ow);     // OW tile size
    TILING_DATA_FIELD_DEF(uint32_t, ow_tiles);    // ceil_div(OW, tile_ow)
    TILING_DATA_FIELD_DEF(uint32_t, tasks);       // tasks = rows * ow_tiles
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard2dAsymmetricInputSquareKernelCustom,
                           ConvStandard2dAsymmetricInputSquareKernelCustomTilingData)
} // namespace optiling
