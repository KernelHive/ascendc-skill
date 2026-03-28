
#include "kernel_operator.h"

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

class KernelGroupedGEMMCustom {
public:
    __aicore__ inline KernelGroupedGEMMCustom() {}

    __aicore__ inline void Init(GM_ADDR lhs,
                               GM_ADDR rhs,
                               GM_ADDR m_indices,
                               GM_ADDR out,
                               uint32_t m, uint32_t k, uint32_t n, uint32_t g,
                               uint32_t nTile,
                               uint32_t kTile,
                               uint32_t tilesPerRow,
                               uint32_t totalTiles,
                               AscendC::TPipe* pipe)
    {
        this->m = m;
        this->k = k;
        this->n = n;
        this->g = g;
        this->nTile = nTile;
        this->kTile = kTile;
        this->tilesPerRow = tilesPerRow;
        this->totalTiles = totalTiles;
        this->pipe = pipe;

        lhsGm.SetGlobalBuffer((__gm__ uint16_t*)lhs, (uint64_t)m * (uint64_t)k);
        rhsGm.SetGlobalBuffer((__gm__ uint16_t*)rhs, (uint64_t)g * (uint64_t)n * (uint64_t)k);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)m_indices, (uint64_t)m);
        outGm.SetGlobalBuffer((__gm__ uint16_t*)out, (uint64_t)m * (uint64_t)n);

        // UB buffers:
        // - lhsTile: kTile bf16
        // - rhsPack: nTile*kTile bf16 packed as [nTile, kTile] with fixed stride kTile (safe for tails)
        // - acc: nTile fp32
        // - out: nTile bf16
        pipe->InitBuffer(lhsTileBuf, this->kTile * (uint32_t)sizeof(uint16_t));
        pipe->InitBuffer(rhsPackBuf, this->nTile * this->kTile * (uint32_t)sizeof(uint16_t));
        pipe->InitBuffer(accBuf, this->nTile * (uint32_t)sizeof(float));
        pipe->InitBuffer(outBuf, this->nTile * (uint32_t)sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t cores = (coreNum == 0 ? 1u : coreNum);

        AscendC::LocalTensor<uint16_t> lhsTile = lhsTileBuf.Get<uint16_t>();
        AscendC::LocalTensor<uint16_t> rhsPack = rhsPackBuf.Get<uint16_t>();
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::LocalTensor<uint16_t> outTile = outBuf.Get<uint16_t>();

        for (uint32_t tileId = coreIdx; tileId < totalTiles; tileId += cores) {
            const uint32_t mi = tileId / tilesPerRow;
            const uint32_t tileInRow = tileId - mi * tilesPerRow;
            const uint32_t n0 = tileInRow * nTile;
            if (mi >= m || n0 >= n) continue;

            const uint32_t curN = (n0 + nTile <= n) ? nTile : (n - n0);

            const int32_t gi32 = idxGm.GetValue((uint64_t)mi);
            const bool valid = (gi32 >= 0) && ((uint32_t)gi32 < g);
            const uint32_t gi = valid ? (uint32_t)gi32 : 0u;

            // Initialize acc for this tile.
            AscendC::Duplicate<float>(acc, 0.0f, (int32_t)curN);
            AscendC::PipeBarrier<PIPE_V>();

            if (valid) {
                const uint64_t lhsBase = (uint64_t)mi * (uint64_t)k;
                const uint64_t rhsGroupBase = (uint64_t)gi * (uint64_t)n * (uint64_t)k;

                for (uint32_t k0 = 0; k0 < k; k0 += kTile) {
                    const uint32_t curK = (k0 + kTile <= k) ? kTile : (k - k0);

                    // 1) Load lhs slice.
                    AscendC::DataCopy(lhsTile, lhsGm[lhsBase + (uint64_t)k0], curK);

                    // 2) Pack RHS tile row-by-row into rhsPack with fixed stride kTile.
                    // This is correct for any curK < K because each row copy is contiguous and independent.
                    // We only read first curK elements per row; the rest of the row in rhsPack is unused.
                    for (uint32_t j = 0; j < curN; ++j) {
                        const uint64_t rhsRowBase = rhsGroupBase + (uint64_t)(n0 + j) * (uint64_t)k + (uint64_t)k0;
                        AscendC::DataCopy(rhsPack[(uint64_t)j * (uint64_t)kTile], rhsGm[rhsRowBase], curK);
                    }
                    AscendC::PipeBarrier<PIPE_MTE2>();

                    // Accumulate. Unroll over output rows by 2 to cut loop overhead.
                    uint32_t j = 0;
                    for (; j + 1 < curN; j += 2) {
                        float sum0 = acc(j);
                        float sum1 = acc(j + 1);
                        const uint32_t off0 = j * kTile;
                        const uint32_t off1 = (j + 1) * kTile;

                        #pragma unroll 8
                        for (uint32_t kk = 0; kk < curK; ++kk) {
                            const float a = Bf16ToF32(lhsTile(kk));
                            sum0 += a * Bf16ToF32(rhsPack(off0 + kk));
                            sum1 += a * Bf16ToF32(rhsPack(off1 + kk));
                        }
                        acc(j) = sum0;
                        acc(j + 1) = sum1;
                    }
                    if (j < curN) {
                        float sum = acc(j);
                        const uint32_t off = j * kTile;
                        #pragma unroll 8
                        for (uint32_t kk = 0; kk < curK; ++kk) {
                            sum += Bf16ToF32(lhsTile(kk)) * Bf16ToF32(rhsPack(off + kk));
                        }
                        acc(j) = sum;
                    }
                }
            }

            // Convert and store.
            for (uint32_t j = 0; j < curN; ++j) {
                outTile(j) = F32ToBf16(acc(j));
            }
            AscendC::PipeBarrier<PIPE_V>();

            const uint64_t outBase = (uint64_t)mi * (uint64_t)n + (uint64_t)n0;
            AscendC::DataCopy(outGm[outBase], outTile, curN);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ inline float Bf16ToF32(uint16_t v) const
    {
        union { uint32_t u; float f; } x;
        x.u = ((uint32_t)v) << 16;
        return x.f;
    }

    __aicore__ inline uint16_t F32ToBf16(float f) const
    {
        union { uint32_t u; float f; } x;
        x.f = f;
        const uint32_t lsb = (x.u >> 16) & 1u;
        const uint32_t bias = 0x7FFFu + lsb;
        const uint32_t rounded = x.u + bias;
        return (uint16_t)(rounded >> 16);
    }

private:
    AscendC::TPipe* pipe = nullptr;

    AscendC::TBuf<AscendC::TPosition::VECCALC> lhsTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rhsPackBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;

    AscendC::GlobalTensor<uint16_t> lhsGm;
    AscendC::GlobalTensor<uint16_t> rhsGm;
    AscendC::GlobalTensor<int32_t>  idxGm;
    AscendC::GlobalTensor<uint16_t> outGm;

    uint32_t m = 0, k = 0, n = 0, g = 0;
    uint32_t nTile = 0, kTile = 0;
    uint32_t tilesPerRow = 0, totalTiles = 0;
};

extern "C" __global__ __aicore__ void grouped_gemm_custom(GM_ADDR lhs,
                                                          GM_ADDR rhs,
                                                          GM_ADDR m_indices,
                                                          GM_ADDR out,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    AscendC::TPipe pipe;
    KernelGroupedGEMMCustom op;
    op.Init(lhs, rhs, m_indices, out,
            td.m, td.k, td.n, td.g,
            td.nTile, td.kTile,
            td.tilesPerRow, td.totalTiles,
            &pipe);
    op.Process();
}
