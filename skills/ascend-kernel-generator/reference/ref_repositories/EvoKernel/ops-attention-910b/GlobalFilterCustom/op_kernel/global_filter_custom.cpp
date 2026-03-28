
#include "kernel_operator.h"

// Fused rfft2(7x7, ortho) -> complex mul -> irfft2(7x7, ortho)
// Specialized for H=W=7, FW=4, C=512, N=49, float32.
//
// This round focuses on reducing scalar-bound GM traffic:
// - Load activation tile [49, cCount] contiguously into UB via one DataCopy per block tile.
// - Store output tile [49, cCount] back to GM via one DataCopy.
// - Keep weights as UB slab, and avoid UB GetValue/SetValue for spectra by using register arrays.

class KernelGlobalFilterCustom {
public:
    __aicore__ inline KernelGlobalFilterCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t B, uint32_t N, uint32_t C,
                               uint32_t H, uint32_t W, uint32_t FW,
                               uint32_t xTotal, uint32_t wTotal,
                               float invSqrtHW, uint32_t cTile)
    {
        this->B = (int32_t)B;
        this->N = (int32_t)N;
        this->C = (int32_t)C;
        this->H = (int32_t)H;
        this->W = (int32_t)W;
        this->FW = (int32_t)FW;
        this->invSqrtHW = invSqrtHW;
        this->cTile = (int32_t)cTile;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)xTotal);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)xTotal);
        wGm.SetGlobalBuffer((__gm__ float*)w, (uint64_t)wTotal);

        // UB buffers:
        // wSlab: [7*4*cTile*2]
        // xTile: [49*cTile] contiguous gather for activations
        // yTile: [49*cTile] contiguous scatter for outputs
        pipe.InitBuffer(wSlabBuf, (uint32_t)(7U * 4U) * (uint32_t)cTile * 2U * sizeof(float));
        pipe.InitBuffer(xTileBuf, (uint32_t)49U * (uint32_t)cTile * sizeof(float));
        pipe.InitBuffer(yTileBuf, (uint32_t)49U * (uint32_t)cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (H != 7 || W != 7 || FW != 4 || C != 512 || N != 49) return;
        if (cTile <= 0) return;

        const int32_t blk = (int32_t)AscendC::GetBlockIdx();
        const int32_t blkNum = (int32_t)AscendC::GetBlockNum();
        if (blkNum <= 0) return;

        const int32_t tilesPerB = (C + cTile - 1) / cTile;
        const int32_t totalTiles = B * tilesPerB;

        for (int32_t t = blk; t < totalTiles; t += blkNum) {
            const int32_t b = t / tilesPerB;
            const int32_t tileId = t - b * tilesPerB;
            const int32_t cStart = tileId * cTile;
            int32_t cCount = C - cStart;
            if (cCount > cTile) cCount = cTile;
            if (cCount <= 0) continue;

            ProcessOneBatchTile(b, cStart, cCount);
        }
    }

private:
    __aicore__ inline void ProcessOneBatchTile(int32_t b, int32_t cStart, int32_t cCount)
    {
        AscendC::LocalTensor<float> wSlab = wSlabBuf.Get<float>(); // [28*cTile*2]
        AscendC::LocalTensor<float> xTile = xTileBuf.Get<float>(); // [49*cTile]
        AscendC::LocalTensor<float> yTile = yTileBuf.Get<float>(); // [49*cTile]

        // twiddle tables for size-7 (constants)
        const float COS7[7] = {
            1.0f,
            0.623489802f,
            -0.222520934f,
            -0.900968868f,
            -0.900968868f,
            -0.222520934f,
            0.623489802f
        };
        const float SIN7[7] = {
            0.0f,
            0.781831482f,
            0.974927912f,
            0.433883739f,
            -0.433883739f,
            -0.974927912f,
            -0.781831482f
        };

        // 1) Contiguous activation tile load: x[b, :, cStart:cStart+cCount] -> xTile[49, cCount]
        // GM layout is [B, N, C] contiguous in C, so the slice over C is contiguous.
        // For each n, copy cCount floats from GM into UB at offset n*cTile.
        uint64_t xBase = (uint64_t)b * (uint64_t)N * (uint64_t)C + (uint64_t)cStart;
        for (int32_t n = 0; n < 49; ++n) {
            uint32_t dstOff = (uint32_t)(n * cTile);
            uint64_t srcOff = xBase + (uint64_t)n * (uint64_t)C;
            AscendC::DataCopy(xTile[dstOff], xGm[srcOff], (uint32_t)cCount);
        }

        // 2) Load weight slab [7,4,cStart:cStart+cCount,2] into UB once.
        for (int32_t k = 0; k < 7; ++k) {
            for (int32_t l = 0; l < 4; ++l) {
                uint64_t wBase = ((uint64_t)k * 4ULL + (uint64_t)l) * (uint64_t)C * 2ULL + (uint64_t)cStart * 2ULL;
                uint32_t dstOff = (uint32_t)(((k * 4 + l) * cTile) * 2);
                AscendC::DataCopy(wSlab[dstOff], wGm[wBase], (uint32_t)(cCount * 2));
            }
        }

        // 3) Compute per channel; write results into yTile[49, cCount] (contiguous in cc)
        for (int32_t cc = 0; cc < cCount; ++cc) {
            float xreg[49];
            #pragma unroll
            for (int32_t n = 0; n < 49; ++n) {
                xreg[n] = xTile.GetValue((uint32_t)(n * cTile + cc));
            }

            float Xre[28];
            float Xim[28];
            ForwardRFFT2_7x7_FromRegs(xreg, COS7, SIN7, Xre, Xim);

            float Yre[28];
            float Yim[28];
            MultiplyWeight_FromSlab(cc, wSlab, Xre, Xim, Yre, Yim);

            // Inverse and write into yTile
            InverseIRFFT2_7x7_ToUB(Yre, Yim, COS7, SIN7, yTile, cc);
        }

        // 4) Contiguous output tile store: yTile[49,cCount] -> y[b,:,cStart:cStart+cCount]
        uint64_t yBase = (uint64_t)b * (uint64_t)N * (uint64_t)C + (uint64_t)cStart;
        for (int32_t n = 0; n < 49; ++n) {
            uint32_t srcOff = (uint32_t)(n * cTile);
            uint64_t dstOff = yBase + (uint64_t)n * (uint64_t)C;
            AscendC::DataCopy(yGm[dstOff], yTile[srcOff], (uint32_t)cCount);
        }
    }

    __aicore__ inline void ForwardRFFT2_7x7_FromRegs(const float xreg[49],
                                                     const float* COS7,
                                                     const float* SIN7,
                                                     float Xre[28],
                                                     float Xim[28])
    {
        for (int32_t k = 0; k < 7; ++k) {
            for (int32_t l = 0; l < 4; ++l) {
                float re = 0.0f;
                float im = 0.0f;
                for (int32_t n = 0; n < 7; ++n) {
                    int32_t tn = (k * n) % 7;
                    float ca = COS7[tn];
                    float sa = SIN7[tn];
                    const int32_t rowOff = n * 7;
                    if (l == 0) {
                        for (int32_t m = 0; m < 7; ++m) {
                            float xr = xreg[rowOff + m];
                            re += xr * ca;
                            im -= xr * sa;
                        }
                    } else {
                        for (int32_t m = 0; m < 7; ++m) {
                            int32_t tm = (l * m) % 7;
                            float cb = COS7[tm];
                            float sb = SIN7[tm];
                            float cosab = ca * cb - sa * sb;
                            float sinab = sa * cb + ca * sb;
                            float xr = xreg[rowOff + m];
                            re += xr * cosab;
                            im -= xr * sinab;
                        }
                    }
                }
                re *= invSqrtHW;
                im *= invSqrtHW;
                const int32_t q = k * 4 + l;
                Xre[q] = re;
                Xim[q] = im;
            }
        }
    }

    __aicore__ inline void MultiplyWeight_FromSlab(int32_t cc,
                                                   const AscendC::LocalTensor<float>& wSlab,
                                                   const float Xre[28],
                                                   const float Xim[28],
                                                   float Yre[28],
                                                   float Yim[28])
    {
        for (int32_t q = 0; q < 28; ++q) {
            uint32_t wOff = (uint32_t)(q * cTile * 2 + cc * 2);
            float Wre = wSlab.GetValue(wOff + 0U);
            float Wim = wSlab.GetValue(wOff + 1U);
            float xr = Xre[q];
            float xi = Xim[q];
            Yre[q] = xr * Wre - xi * Wim;
            Yim[q] = xr * Wim + xi * Wre;
        }
    }

    __aicore__ inline void InverseIRFFT2_7x7_ToUB(const float Yre[28],
                                                 const float Yim[28],
                                                 const float* COS7,
                                                 const float* SIN7,
                                                 const AscendC::LocalTensor<float>& yTile,
                                                 int32_t cc)
    {
        // y[n0,m0] = invSqrtHW * ( sum_k Re( Y[k,0] * e^{j2pi k n0/7} )
        //                        + sum_k sum_{l=1..3} 2*Re( Y[k,l] * e^{j2pi (k n0 + l m0)/7} ) )
        for (int32_t n0 = 0; n0 < 7; ++n0) {
            for (int32_t m0 = 0; m0 < 7; ++m0) {
                float acc = 0.0f;
                for (int32_t k = 0; k < 7; ++k) {
                    int32_t tno = (k * n0) % 7;
                    float ca0 = COS7[tno];
                    float sa0 = SIN7[tno];

                    // l = 0
                    {
                        const int32_t q0 = k * 4 + 0;
                        acc += (Yre[q0] * ca0 - Yim[q0] * sa0);
                    }

                    // l = 1..3
                    for (int32_t l = 1; l < 4; ++l) {
                        int32_t tmo = (l * m0) % 7;
                        float cb0 = COS7[tmo];
                        float sb0 = SIN7[tmo];

                        float cosab0 = ca0 * cb0 - sa0 * sb0;
                        float sinab0 = sa0 * cb0 + ca0 * sb0;

                        const int32_t q = k * 4 + l;
                        float reProj = (Yre[q] * cosab0 - Yim[q] * sinab0);
                        acc += 2.0f * reProj;
                    }
                }
                acc *= invSqrtHW;
                const int32_t n = n0 * 7 + m0;
                yTile.SetValue((uint32_t)(n * cTile + cc), acc);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> wSlabBuf;
    AscendC::TBuf<> xTileBuf;
    AscendC::TBuf<> yTileBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> wGm;

    int32_t B{0}, N{0}, C{0}, H{0}, W{0}, FW{0};
    float invSqrtHW{0.0f};
    int32_t cTile{128};
};

extern "C" __global__ __aicore__ void global_filter_custom(
    GM_ADDR x, GM_ADDR complex_weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGlobalFilterCustom op;
    op.Init(x, complex_weight, y,
            tiling_data.B, tiling_data.N, tiling_data.C,
            tiling_data.H, tiling_data.W, tiling_data.FW,
            tiling_data.xTotal, tiling_data.wTotal,
            tiling_data.invSqrtHW, tiling_data.cTile);
    op.Process();
}
