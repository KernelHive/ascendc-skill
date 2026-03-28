
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcBlock2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, H);
  TILING_DATA_FIELD_DEF(uint32_t, W);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcBlock2dCustom, MhcBlock2dCustomTilingData)

} // namespace optiling
