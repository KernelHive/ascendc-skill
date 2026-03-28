
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConvTranspose2dAddMinGeluMultiplyCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, HW);
  TILING_DATA_FIELD_DEF(uint32_t, total);     // N*C*HW (clamped to u32)
  TILING_DATA_FIELD_DEF(float, addv);         // baked constant
  TILING_DATA_FIELD_DEF(float, mulv);         // baked constant
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose2dAddMinGeluMultiplyCustom,
                           ConvTranspose2dAddMinGeluMultiplyCustomTilingData)

}  // namespace optiling
