
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcBlockBottleneck2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, H);
  TILING_DATA_FIELD_DEF(uint32_t, W);
  TILING_DATA_FIELD_DEF(uint32_t, S);
  TILING_DATA_FIELD_DEF(uint32_t, Cps);     // C / S
  TILING_DATA_FIELD_DEF(uint32_t, K);       // Cps*H*W
  TILING_DATA_FIELD_DEF(uint32_t, tileK);   // tile over K (elements)
  TILING_DATA_FIELD_DEF(uint32_t, Kpad);    // align(tileK, 16) for fp16 vectors
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcBlockBottleneck2dCustom, MhcBlockBottleneck2dCustomTilingData)

} // namespace optiling
