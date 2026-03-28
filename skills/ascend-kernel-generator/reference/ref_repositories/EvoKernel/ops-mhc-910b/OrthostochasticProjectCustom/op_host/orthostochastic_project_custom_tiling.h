
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(OrthostochasticProjectCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, n0);
  TILING_DATA_FIELD_DEF(uint32_t, n1);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, mn);
  TILING_DATA_FIELD_DEF(uint32_t, transpose);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OrthostochasticProjectCustom, OrthostochasticProjectCustomTilingData)

} // namespace optiling
