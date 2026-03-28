
#include "kernel_operator.h"

class KernelPAMCustom {
public:
    __aicore__ inline KernelPAMCustom() {}

    __aicore__ inline void Init(GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR y,
                               uint32_t N, uint32_t S, uint32_t C, uint32_t S_pad,
                               uint32_t totalRows, uint32_t rowsPerCore, uint32_t cTile)
    {
        this->N = N;
        this->S = S;
        this->C = C;
        this->S_pad = S_pad; // expected 64
        this->totalRows = totalRows;
        this->rowsPerCore = rowsPerCore;
        this->cTile = cTile;

        uint64_t bSize = static_cast<uint64_t>(N) * S * C; // [N,S,C]
        uint64_t cSize = static_cast<uint64_t>(N) * C * S; // [N,C,S]
        uint64_t dSize = static_cast<uint64_t>(N) * S * C; // [N,S,C]
        uint64_t ySize = static_cast<uint64_t>(N) * C * S; // [N,C,S]

        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        cGm.SetGlobalBuffer((__gm__ float*)c, cSize);
        dGm.SetGlobalBuffer((__gm__ float*)d, dSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers:
        // bFull: [C]
        // scoreRow/attnRow/expRow: [S_pad=64]
        // cTileBuf: [cTile, S_pad] (we only fill first S columns; rest 0)
        // dVecTile: [cTile]
        // outTile: [cTile]
        pipe.InitBuffer(bFullBuf, static_cast<uint64_t>(C) * sizeof(float));
        pipe.InitBuffer(scoreRowBuf, static_cast<uint64_t>(S_pad) * sizeof(float));
        pipe.InitBuffer(attnRowBuf, static_cast<uint64_t>(S_pad) * sizeof(float));
        pipe.InitBuffer(expRowBuf, static_cast<uint64_t>(S_pad) * sizeof(float));

        pipe.InitBuffer(cTile2dBuf, static_cast<uint64_t>(cTile) * static_cast<uint64_t>(S_pad) * sizeof(float));

        pipe.InitBuffer(dVecTileBuf, static_cast<uint64_t>(cTile) * sizeof(float));
        pipe.InitBuffer(outTileBuf, static_cast<uint64_t>(cTile) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t rowStart = core * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) rowEnd = totalRows;
        if (rowStart >= rowEnd) return;

        AscendC::LocalTensor<float> bFull = bFullBuf.Get<float>();
        AscendC::LocalTensor<float> scoreRow = scoreRowBuf.Get<float>();
        AscendC::LocalTensor<float> attnRow = attnRowBuf.Get<float>();
        AscendC::LocalTensor<float> expRow = expRowBuf.Get<float>();
        AscendC::LocalTensor<float> cTile2d = cTile2dBuf.Get<float>();
        AscendC::LocalTensor<float> dVecTile = dVecTileBuf.Get<float>();
        AscendC::LocalTensor<float> outTile = outTileBuf.Get<float>();

        const uint64_t bBatchStride = static_cast<uint64_t>(S) * C; // [S,C]
        const uint64_t cBatchStride = static_cast<uint64_t>(C) * S; // [C,S]
        const uint64_t yBatchStride = static_cast<uint64_t>(C) * S; // [C,S]

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            const uint32_t n = r / S;
            const uint32_t i = r - n * S;

            const uint64_t bBase = static_cast<uint64_t>(n) * bBatchStride + static_cast<uint64_t>(i) * C;
            const uint64_t cBase = static_cast<uint64_t>(n) * cBatchStride;
            const uint64_t dBase = static_cast<uint64_t>(n) * bBatchStride;
            const uint64_t yBase = static_cast<uint64_t>(n) * yBatchStride;

            // Load B row once
            AscendC::DataCopy(bFull, bGm[bBase], C);
            AscendC::PipeBarrier<PIPE_V>();

            // Init scoreRow padded
            for (uint32_t k = 0; k < S_pad; ++k) {
                scoreRow.SetValue(k, 0.0f);
            }

            // Accumulate scores over channel tiles:
            // score[k] += sum_j b[j]*c[j,k]
            for (uint32_t j0 = 0; j0 < C; j0 += cTile) {
                const uint32_t cLen = (j0 + cTile <= C) ? cTile : (C - j0);

                // Build cTile2d as [cLen, S_pad] in UB. We fill only first S columns.
                // This converts repeated strided GM reads into UB reads reused across k accumulation.
                // Initialize to 0 for safety (only need S_pad*cLen <= 128*64=8192 floats typical).
                AscendC::Duplicate(cTile2d, 0.0f, static_cast<int32_t>(cLen * S_pad));
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t jj = 0; jj < cLen; ++jj) {
                    const uint32_t j = j0 + jj;
                    const uint64_t cRowBase = cBase + static_cast<uint64_t>(j) * S;
                    const uint64_t ubRow = static_cast<uint64_t>(jj) * S_pad;
                    // Fill first S entries from GM (strided overall, but contiguous within the row of C: [S])
                    for (uint32_t k = 0; k < S; ++k) {
                        float cv = cGm.GetValue(cRowBase + k);
                        cTile2d.SetValue(ubRow + k, cv);
                    }
                }

                // Accumulate
                for (uint32_t k = 0; k < S; ++k) {
                    float acc = scoreRow.GetValue(k);
                    for (uint32_t jj = 0; jj < cLen; ++jj) {
                        float bv = bFull.GetValue(j0 + jj);
                        float cv = cTile2d.GetValue(static_cast<uint64_t>(jj) * S_pad + k);
                        acc += bv * cv;
                    }
                    scoreRow.SetValue(k, acc);
                }
            }

            // Softmax over S (pad tail to -inf then exp 64-wide)
            float maxv = scoreRow.GetValue(0);
            for (uint32_t k = 1; k < S; ++k) {
                float v = scoreRow.GetValue(k);
                if (v > maxv) maxv = v;
            }
            for (uint32_t k = 0; k < S; ++k) {
                expRow.SetValue(k, scoreRow.GetValue(k) - maxv);
            }
            for (uint32_t k = S; k < S_pad; ++k) {
                expRow.SetValue(k, -3.402823466e+38f); // -FLT_MAX -> exp ~ 0
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(expRow, expRow, static_cast<int32_t>(S_pad));
            AscendC::PipeBarrier<PIPE_V>();

            float sum = 0.0f;
            for (uint32_t k = 0; k < S; ++k) sum += expRow.GetValue(k);
            float invSum = 1.0f / sum;
            for (uint32_t k = 0; k < S; ++k) attnRow.SetValue(k, expRow.GetValue(k) * invSum);
            for (uint32_t k = S; k < S_pad; ++k) attnRow.SetValue(k, 0.0f);

            // Compute output y[:,i] in channel tiles, using contiguous DataCopy from D (row-major [S,C])
            for (uint32_t j0 = 0; j0 < C; j0 += cTile) {
                const uint32_t cLen = (j0 + cTile <= C) ? cTile : (C - j0);

                AscendC::Duplicate(outTile, 0.0f, static_cast<int32_t>(cLen));
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t k = 0; k < S; ++k) {
                    float a = attnRow.GetValue(k);
                    const uint64_t dRowBase = dBase + static_cast<uint64_t>(k) * C + j0;
                    AscendC::DataCopy(dVecTile, dGm[dRowBase], cLen);
                    AscendC::PipeBarrier<PIPE_V>();
                    // outTile += a * dVecTile
                    for (uint32_t jj = 0; jj < cLen; ++jj) {
                        float ov = outTile.GetValue(jj);
                        float dv = dVecTile.GetValue(jj);
                        outTile.SetValue(jj, ov + a * dv);
                    }
                }

                // Store y[n, j0+jj, i] (layout [N,C,S])
                const uint64_t yColBase = yBase + static_cast<uint64_t>(j0) * S + i;
                for (uint32_t jj = 0; jj < cLen; ++jj) {
                    yGm.SetValue(yColBase + static_cast<uint64_t>(jj) * S, outTile.GetValue(jj));
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> bFullBuf;
    AscendC::TBuf<> scoreRowBuf;
    AscendC::TBuf<> attnRowBuf;
    AscendC::TBuf<> expRowBuf;

    AscendC::TBuf<> cTile2dBuf;

    AscendC::TBuf<> dVecTileBuf;
    AscendC::TBuf<> outTileBuf;

    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::GlobalTensor<float> dGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t N {0}, S {0}, C {0}, S_pad {0};
    uint32_t totalRows {0}, rowsPerCore {0}, cTile {0};
};

extern "C" __global__ __aicore__ void pam_custom(GM_ADDR b, GM_ADDR c, GM_ADDR d, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelPAMCustom op;
    op.Init(b, c, d, y,
            tiling_data.N, tiling_data.S, tiling_data.C, tiling_data.S_pad,
            tiling_data.totalRows, tiling_data.rowsPerCore, tiling_data.cTile);
    op.Process();
}
