
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TripletMarginLossCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);      // dim0
    TILING_DATA_FIELD_DEF(uint32_t, featSize);       // product dims[1:]
    TILING_DATA_FIELD_DEF(uint32_t, featTile);       // tile length (aligned, elements)
    TILING_DATA_FIELD_DEF(uint32_t, featTileNum);    // number of tiles over featSize
    TILING_DATA_FIELD_DEF(uint32_t, featLast);       // last tile valid length (elements)
    TILING_DATA_FIELD_DEF(float, invBatch);          // 1.0f/batchSize (0 if batchSize==0)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TripletMarginLossCustom, TripletMarginLossCustomTilingData)
} // namespace optiling
