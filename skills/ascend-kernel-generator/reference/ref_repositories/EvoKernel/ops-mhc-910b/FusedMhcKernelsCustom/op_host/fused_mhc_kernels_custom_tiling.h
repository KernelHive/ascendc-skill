
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedMhcKernelsCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, L);
  TILING_DATA_FIELD_DEF(uint32_t, D);
  TILING_DATA_FIELD_DEF(uint32_t, BL);
  TILING_DATA_FIELD_DEF(uint32_t, outCols);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, nn);
  TILING_DATA_FIELD_DEF(uint32_t, Dpad);
  TILING_DATA_FIELD_DEF(uint32_t, outPad);
  TILING_DATA_FIELD_DEF(uint32_t, nPad);
  TILING_DATA_FIELD_DEF(uint32_t, nnPad);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedMhcKernelsCustom, FusedMhcKernelsCustomTilingData)

} // namespace optiling
