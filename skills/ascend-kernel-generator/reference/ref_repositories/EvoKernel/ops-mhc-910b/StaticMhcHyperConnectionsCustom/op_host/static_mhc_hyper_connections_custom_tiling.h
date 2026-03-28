
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(StaticMhcHyperConnectionsCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, S);
  TILING_DATA_FIELD_DEF(uint32_t, D);
  TILING_DATA_FIELD_DEF(uint32_t, Dpad);   // ceil(D,8)
  TILING_DATA_FIELD_DEF(uint32_t, tileD);  // fixed tile in D
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StaticMhcHyperConnectionsCustom, StaticMhcHyperConnectionsCustomTilingData)

} // namespace optiling
