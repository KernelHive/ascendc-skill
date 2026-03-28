
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalResidual);
TILING_DATA_FIELD_DEF(uint32_t, totalFn);
TILING_DATA_FIELD_DEF(uint32_t, totalScale);
TILING_DATA_FIELD_DEF(uint32_t, totalBase);

TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, hc);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, dFlat);       // hc*H
TILING_DATA_FIELD_DEF(uint32_t, hc2);         // hc*hc
TILING_DATA_FIELD_DEF(uint32_t, hc3);         // 2*hc + hc*hc
TILING_DATA_FIELD_DEF(uint32_t, blockTokens);

TILING_DATA_FIELD_DEF(float, invDFlat);
TILING_DATA_FIELD_DEF(float, rmsEps);
TILING_DATA_FIELD_DEF(float, hcPreEps);
TILING_DATA_FIELD_DEF(float, hcSinkhornEps);
TILING_DATA_FIELD_DEF(float, hcPostMultValue);
TILING_DATA_FIELD_DEF(uint32_t, sinkhornRepeat);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPreBlockCustom, TilingData)

} // namespace optiling
