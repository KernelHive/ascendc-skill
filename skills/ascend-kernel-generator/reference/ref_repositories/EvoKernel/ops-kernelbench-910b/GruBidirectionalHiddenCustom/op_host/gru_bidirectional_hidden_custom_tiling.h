
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GruBidirectionalHiddenCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, T);
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, I);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, L);
    TILING_DATA_FIELD_DEF(uint32_t, D);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GruBidirectionalHiddenCustom, GruBidirectionalHiddenCustomTilingData)
} // namespace optiling
