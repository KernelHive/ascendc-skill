
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelRmsNormCustom {
public:
    __aicore__ inline KernelRmsNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t b, uint32_t c, uint32_t inner,
                               uint32_t innerTile, uint32_t tilesPerB,
                               float eps, float invC)
    {
        this->b = b;
        this->c = c;
        this->inner = inner;
        this->innerTile = innerTile;
        this->tilesPerB = tilesPerB;
        this->eps = eps;
        this->invC = invC;

        xBase = reinterpret_cast<__gm__ float*>(x);
        yBase = reinterpret_cast<__gm__ float*>(y);

        // UB vectors of size innerTile (max 256): sumsq, inv, tmp, xTile, yTile
        pipe.InitBuffer(qSumSq, BUFFER_NUM, this->innerTile * sizeof(float));
        pipe.InitBuffer(qInv,   BUFFER_NUM, this->innerTile * sizeof(float));
        pipe.InitBuffer(qTmp,   BUFFER_NUM, this->innerTile * sizeof(float));
        pipe.InitBuffer(qX,     BUFFER_NUM, this->innerTile * sizeof(float));
        pipe.InitBuffer(qY,     BUFFER_NUM, this->innerTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->b == 0 || this->c == 0 || this->inner == 0 || this->innerTile == 0 || this->tilesPerB == 0) {
            return;
        }

        // Avoid any casts from AscendC::GetBlockNum/GetBlockIdx to prevent forbidden float<->uint conversions.
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum == 0) blockNum = 1;

        const uint32_t totalTiles = this->b * this->tilesPerB;

        // Grid-stride loop over tiles for good load balance even when blockDim < totalTiles.
        for (uint32_t globalTile = blockIdx; globalTile < totalTiles; globalTile += blockNum) {
            const uint32_t bIdx = globalTile / this->tilesPerB;
            const uint32_t tileId = globalTile - bIdx * this->tilesPerB;
            const uint32_t innerStart = tileId * this->innerTile;
            uint32_t len = this->inner - innerStart;
            if (len > this->innerTile) len = this->innerTile;
            ComputeTile(bIdx, innerStart, len);
        }
    }

private:
    __aicore__ inline uint64_t BaseOffset(uint32_t bIdx, uint32_t cIdx, uint32_t innerStart) const
    {
        // x[b, c, inner]
        return static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(this->c) * static_cast<uint64_t>(this->inner)
             + static_cast<uint64_t>(cIdx) * static_cast<uint64_t>(this->inner)
             + static_cast<uint64_t>(innerStart);
    }

    __aicore__ inline void ComputeTile(uint32_t bIdx, uint32_t innerStart, uint32_t len)
    {
        // Allocate once per tile and reuse across channel loops (reduces queue churn).
        AscendC::LocalTensor<float> sumsq = qSumSq.AllocTensor<float>();
        AscendC::LocalTensor<float> inv   = qInv.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp   = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> xTile = qX.AllocTensor<float>();
        AscendC::LocalTensor<float> yTile = qY.AllocTensor<float>();

        AscendC::Duplicate<float>(sumsq, 0.0f, (int32_t)len);

        // 1) sumsq[i] = sum_c x^2 for i in tile (contiguous inner loads per channel)
        for (uint32_t ci = 0; ci < this->c; ++ci) {
            const uint64_t base = BaseOffset(bIdx, ci, innerStart);
            // Contiguous GM->UB scalar loads; then vector ops across len.
            for (uint32_t i = 0; i < len; ++i) {
                xTile.SetValue(i, xBase[base + i]);
            }
            AscendC::Mul<float>(tmp, xTile, xTile, (int32_t)len);
            AscendC::Add<float>(sumsq, sumsq, tmp, (int32_t)len);
        }

        // 2) inv = 1 / sqrt(sumsq * invC + eps)
        AscendC::Muls<float>(inv, sumsq, this->invC, (int32_t)len);
        AscendC::Adds<float>(inv, inv, this->eps, (int32_t)len);
        AscendC::Sqrt<float>(inv, inv, (int32_t)len);
        AscendC::Duplicate<float>(tmp, 1.0f, (int32_t)len);
        AscendC::Div<float>(inv, tmp, inv, (int32_t)len);

        // 3) y = x * inv (second pass; contiguous stores per channel)
        for (uint32_t ci = 0; ci < this->c; ++ci) {
            const uint64_t base = BaseOffset(bIdx, ci, innerStart);
            for (uint32_t i = 0; i < len; ++i) {
                xTile.SetValue(i, xBase[base + i]);
            }
            AscendC::Mul<float>(yTile, xTile, inv, (int32_t)len);
            for (uint32_t i = 0; i < len; ++i) {
                yBase[base + i] = yTile.GetValue(i);
            }
        }

        qY.FreeTensor(yTile);
        qX.FreeTensor(xTile);
        qTmp.FreeTensor(tmp);
        qInv.FreeTensor(inv);
        qSumSq.FreeTensor(sumsq);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSumSq;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qInv;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qTmp;
    AscendC::TQue<AscendC::TPosition::VECIN,   BUFFER_NUM> qX;
    AscendC::TQue<AscendC::TPosition::VECOUT,  BUFFER_NUM> qY;

    __gm__ float* xBase = nullptr;
    __gm__ float* yBase = nullptr;

    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t inner = 0;
    uint32_t innerTile = 0;
    uint32_t tilesPerB = 0;
    float eps = 1e-5f;
    float invC = 0.0f;
};

extern "C" __global__ __aicore__ void rms_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNormCustom op;
    op.Init(x, y,
            tiling_data.b, tiling_data.c, tiling_data.inner,
            tiling_data.innerTile, tiling_data.tilesPerB,
            tiling_data.eps, tiling_data.invC);
    op.Process();
}
