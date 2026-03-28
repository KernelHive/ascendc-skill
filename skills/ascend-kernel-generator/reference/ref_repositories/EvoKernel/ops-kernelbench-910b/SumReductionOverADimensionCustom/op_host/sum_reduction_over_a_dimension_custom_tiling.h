
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, outerCount); // B*S
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SumReductionOverADimensionCustom, TilingData)
} // namespace optiling
