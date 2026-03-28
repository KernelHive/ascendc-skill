
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalX);
TILING_DATA_FIELD_DEF(uint32_t, totalPhi);
TILING_DATA_FIELD_DEF(uint32_t, totalBias);
TILING_DATA_FIELD_DEF(uint32_t, totalRmsScale);
TILING_DATA_FIELD_DEF(uint32_t, totalAlphaPre);
TILING_DATA_FIELD_DEF(uint32_t, totalAlphaPost);
TILING_DATA_FIELD_DEF(uint32_t, totalAlphaRes);
TILING_DATA_FIELD_DEF(uint32_t, totalW);

TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, D);         // input_dim
TILING_DATA_FIELD_DEF(uint32_t, n);         // expansion_rate (specialize to 4)
TILING_DATA_FIELD_DEF(uint32_t, SD);        // n*D
TILING_DATA_FIELD_DEF(uint32_t, mapDim);    // n*n+2*n (24)
TILING_DATA_FIELD_DEF(uint32_t, tokens);    // B*S
TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);

TILING_DATA_FIELD_DEF(float, invSD);
TILING_DATA_FIELD_DEF(float, rmsEps);
TILING_DATA_FIELD_DEF(float, sinkEps);
TILING_DATA_FIELD_DEF(uint32_t, sinkIters);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OptimizedMHCLayerWithFusionCustom, TilingData)

} // namespace optiling
