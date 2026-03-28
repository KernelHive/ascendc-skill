
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SinkhornKnoppCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // B*N*N
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, tmax);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, clampMin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SinkhornKnoppCustom, SinkhornKnoppCustomTilingData)
} // namespace optiling
