
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GruCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, T);
    TILING_DATA_FIELD_DEF(uint32_t, B);
    TILING_DATA_FIELD_DEF(uint32_t, I);
    TILING_DATA_FIELD_DEF(uint32_t, H);
    TILING_DATA_FIELD_DEF(uint32_t, L);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GruCustom, GruCustomTilingData)
} // namespace optiling
