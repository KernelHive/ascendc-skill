
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StreamWriteCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, T);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, BT);          // B*T
  TILING_DATA_FIELD_DEF(uint32_t, tileC);       // per-tile columns along C
  TILING_DATA_FIELD_DEF(uint32_t, cTiles);      // ceil(C/tileC)
  TILING_DATA_FIELD_DEF(uint32_t, totalTiles);  // BT * cTiles
  TILING_DATA_FIELD_DEF(uint32_t, Npad);        // ceil(N,8) for safe GM->UB copy
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StreamWriteCustom, StreamWriteCustomTilingData)
} // namespace optiling
