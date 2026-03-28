
#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TQue;
using AscendC::TPipe;

class KernelStreamWeightedSumCustom {
public:
    __aicore__ inline KernelStreamWeightedSumCustom() {}

    __aicore__ inline void Init(GM_ADDR x_stream, GM_ADDR weights, GM_ADDR out,
                               uint32_t B, uint32_t T, uint32_t N, uint32_t C,
                               uint32_t BT, uint32_t cTile,
                               uint32_t tilesPerRow, uint32_t numTiles)
    {
        this->B = B;
        this->T = T;
        this->N = N;
        this->C = C;
        this->BT = BT;
        this->cTile = cTile;
        this->tilesPerRow = tilesPerRow;
        this->numTiles = numTiles;

        const uint64_t xSize = (uint64_t)B * (uint64_t)T * (uint64_t)N * (uint64_t)C;
        const uint64_t wSize = (uint64_t)B * (uint64_t)T * (uint64_t)N;
        const uint64_t oSize = (uint64_t)B * (uint64_t)T * (uint64_t)C;

        xGm.SetGlobalBuffer((__gm__ float*)x_stream, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)weights,  wSize);
        outGm.SetGlobalBuffer((__gm__ float*)out,    oSize);

        // Double-buffer packed x and y to overlap DMA with compute/store.
        pipe.InitBuffer(qxPack, 2, (uint64_t)4 * (uint64_t)cTile * sizeof(float));
        pipe.InitBuffer(qy,     2, (uint64_t)cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core  = AscendC::GetBlockIdx();
        const uint32_t cores = (AscendC::GetBlockNum() == 0) ? 1 : AscendC::GetBlockNum();
        if (BT == 0 || N == 0 || C == 0) return;

        // Hot path: N==4 and perfectly tiled C.
        if (N == 4 && (C % 8 == 0) && (cTile % 8 == 0) && (tilesPerRow > 0) && (C == tilesPerRow * cTile)) {
            ProcessN4Tiled(core, cores);
            return;
        }

        // Correct fallback (scalar) for unusual shapes/N.
        ProcessScalarFallback(core, cores);
    }

private:
    __aicore__ inline void ProcessN4Tiled(uint32_t core, uint32_t cores)
    {
        // Each core iterates over linear tiles. Tile id maps to (row=(b,t), tileC).
        for (uint32_t tileId = core; tileId < numTiles; tileId += cores) {
            const uint32_t row = tileId / tilesPerRow;          // 0..BT-1
            const uint32_t tileC = tileId - row * tilesPerRow;  // 0..tilesPerRow-1
            const uint32_t c0 = tileC * cTile;

            const uint32_t b = row / T;
            const uint32_t t = row - b * T;

            const uint64_t wBase = ((uint64_t)b * (uint64_t)T + (uint64_t)t) * 4ull; // (b,t,0)
            const uint64_t xBase = (((uint64_t)b * (uint64_t)T + (uint64_t)t) * 4ull) * (uint64_t)C; // (b,t,0,0)
            const uint64_t oBase = ((uint64_t)b * (uint64_t)T + (uint64_t)t) * (uint64_t)C; // (b,t,0)

            // Load weights once per row; scalar loads are only 4 values and amortized across tiles.
            // (We could cache per-row across tiles, but that would require grouping tiles by row and adds scalar control.)
            const float w0 = wGm.GetValue(wBase + 0);
            const float w1 = wGm.GetValue(wBase + 1);
            const float w2 = wGm.GetValue(wBase + 2);
            const float w3 = wGm.GetValue(wBase + 3);

            const uint64_t xPackOff = xBase + (uint64_t)c0;  // packed [4,cTile] contiguous
            const uint64_t oOff     = oBase + (uint64_t)c0;

            // Stage 1: GM->UB packed load (producer)
            LocalTensor<float> xPack = qxPack.AllocTensor<float>();
            AscendC::DataCopy(xPack, xGm[xPackOff], (uint32_t)(4u * cTile));
            qxPack.EnQue(xPack);

            // Stage 2: compute + UB->GM store (consumer)
            LocalTensor<float> vxPack = qxPack.DeQue<float>();
            LocalTensor<float> vx0 = vxPack;
            LocalTensor<float> vx1 = vxPack[(uint32_t)(1u * cTile)];
            LocalTensor<float> vx2 = vxPack[(uint32_t)(2u * cTile)];
            LocalTensor<float> vx3 = vxPack[(uint32_t)(3u * cTile)];

            LocalTensor<float> y = qy.AllocTensor<float>();
            AscendC::Muls(y, vx0, w0, cTile);
            AscendC::Axpy(y, vx1, w1, cTile);
            AscendC::Axpy(y, vx2, w2, cTile);
            AscendC::Axpy(y, vx3, w3, cTile);
            qy.EnQue(y);

            // Free input pack once compute is issued.
            qxPack.FreeTensor(vxPack);

            LocalTensor<float> vy = qy.DeQue<float>();
            AscendC::DataCopy(outGm[oOff], vy, cTile);
            qy.FreeTensor(vy);
        }
    }

    __aicore__ inline void ProcessScalarFallback(uint32_t core, uint32_t cores)
    {
        for (uint32_t row = core; row < BT; row += cores) {
            const uint32_t b = row / T;
            const uint32_t t = row - b * T;

            const uint64_t wBase = ((uint64_t)b * (uint64_t)T + (uint64_t)t) * (uint64_t)N;
            const uint64_t xBase = (((uint64_t)b * (uint64_t)T + (uint64_t)t) * (uint64_t)N) * (uint64_t)C;
            const uint64_t oBase = ((uint64_t)b * (uint64_t)T + (uint64_t)t) * (uint64_t)C;

            for (uint32_t c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (uint32_t n = 0; n < N; ++n) {
                    const float wn = wGm.GetValue(wBase + (uint64_t)n);
                    const uint64_t xOff = xBase + (uint64_t)n * (uint64_t)C + (uint64_t)c;
                    acc += wn * xGm.GetValue(xOff);
                }
                outGm.SetValue(oBase + (uint64_t)c, acc);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> wGm;
    GlobalTensor<float> outGm;

    uint32_t B = 0, T = 0, N = 0, C = 0, BT = 0;
    uint32_t cTile = 256;
    uint32_t tilesPerRow = 0;
    uint32_t numTiles = 0;

    TPipe pipe;
    TQue<AscendC::QuePosition::VECIN,  2> qxPack;
    TQue<AscendC::QuePosition::VECOUT, 2> qy;
};

extern "C" __global__ __aicore__ void stream_weighted_sum_custom(GM_ADDR x_stream, GM_ADDR weights, GM_ADDR out,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelStreamWeightedSumCustom op;
    op.Init(x_stream, weights, out,
            tiling_data.B, tiling_data.T, tiling_data.N, tiling_data.C,
            tiling_data.BT, tiling_data.cTile,
            tiling_data.tilesPerRow, tiling_data.numTiles);
    op.Process();
}
