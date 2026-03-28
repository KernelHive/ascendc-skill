
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, rank);
TILING_DATA_FIELD_DEF(int32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, outerSize);
TILING_DATA_FIELD_DEF(uint32_t, axisSize);
TILING_DATA_FIELD_DEF(uint32_t, totalElems);
TILING_DATA_FIELD_DEF(uint32_t, tileElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaskedCumsumCustom, TilingData)

} // namespace optiling
