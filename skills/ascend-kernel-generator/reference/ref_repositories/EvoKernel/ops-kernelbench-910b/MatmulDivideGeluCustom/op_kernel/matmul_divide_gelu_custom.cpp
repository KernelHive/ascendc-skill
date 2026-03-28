
#include "kernel_operator.h"

class KernelMatmulDivideGeluCustom {
public:
    __aicore__ inline KernelMatmulDivideGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR divisor, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t nTile, uint32_t vecN, uint32_t geluTile)
    {
        M_ = M; K_ = K; N_ = N;
        nTile_ = (nTile == 0 ? 1 : nTile);
        vecN_  = (vecN == 0 ? 1 : vecN);
        geluTile_ = (geluTile == 0 ? 1 : geluTile);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        dGm_.SetGlobalBuffer((__gm__ float*)divisor);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        // UB for vector GELU (keep small and fixed).
        pipe_.InitBuffer(inQ_, 1, geluTile_ * sizeof(float));
        pipe_.InitBuffer(outQ_, 1, geluTile_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0) return;

        const uint32_t nTiles = (N_ + nTile_ - 1U) / nTile_;
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();

        const uint32_t m = blockIdx / nTiles;
        const uint32_t t = blockIdx - m * nTiles;
        if (m >= M_) return;

        const uint32_t nStart = t * nTile_;
        uint32_t nEnd = nStart + nTile_;
        if (nStart >= N_) return;
        if (nEnd > N_) nEnd = N_;

        const float d = dGm_.GetValue(0);
        const float invD = (d == 0.0f) ? 0.0f : (1.0f / d);

        const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
        const uint64_t outBase = (uint64_t)m * (uint64_t)N_;

        // Produce output in small chunks for vector GELU.
        uint32_t n = nStart;
        while (n < nEnd) {
            uint32_t tile = geluTile_;
            if (n + tile > nEnd) tile = nEnd - n;

            AscendC::LocalTensor<float> inLocal = inQ_.AllocTensor<float>();
            AscendC::LocalTensor<float> outLocal = outQ_.AllocTensor<float>();

            // Compute tile outputs into inLocal.
            if (vecN_ == 4) {
                uint32_t j = 0;
                for (; j + 3U < tile; j += 4U) {
                    float v0, v1, v2, v3;
                    ComputeFour(xBase, (uint32_t)(n + j), invD, v0, v1, v2, v3);
                    inLocal.SetValue(j + 0U, v0);
                    inLocal.SetValue(j + 1U, v1);
                    inLocal.SetValue(j + 2U, v2);
                    inLocal.SetValue(j + 3U, v3);
                }
                for (; j < tile; ++j) {
                    float v;
                    ComputeOne(xBase, (uint32_t)(n + j), invD, v);
                    inLocal.SetValue(j, v);
                }
            } else if (vecN_ == 2) {
                uint32_t j = 0;
                for (; j + 1U < tile; j += 2U) {
                    float v0, v1;
                    ComputeTwo(xBase, (uint32_t)(n + j), invD, v0, v1);
                    inLocal.SetValue(j + 0U, v0);
                    inLocal.SetValue(j + 1U, v1);
                }
                for (; j < tile; ++j) {
                    float v;
                    ComputeOne(xBase, (uint32_t)(n + j), invD, v);
                    inLocal.SetValue(j, v);
                }
            } else {
                for (uint32_t j = 0; j < tile; ++j) {
                    float v;
                    ComputeOne(xBase, (uint32_t)(n + j), invD, v);
                    inLocal.SetValue(j, v);
                }
            }

            // GELU vector op on UB
            AscendC::Gelu(outLocal, inLocal, tile);

            // Store back to GM
            for (uint32_t j = 0; j < tile; ++j) {
                yGm_.SetValue(outBase + (uint64_t)(n + j), outLocal.GetValue(j));
            }

            inQ_.FreeTensor(inLocal);
            outQ_.FreeTensor(outLocal);

            n += tile;
        }
    }

private:
    __aicore__ inline void ComputeOne(uint64_t xBase, uint32_t n, float invD, float &out)
    {
        const uint64_t wBase = (uint64_t)n * (uint64_t)K_;
        float acc = bGm_.GetValue((uint64_t)n);

        uint32_t k = 0;
        for (; k + 3U < K_; k += 4U) {
            const float x0 = xGm_.GetValue(xBase + (uint64_t)(k + 0U));
            const float x1 = xGm_.GetValue(xBase + (uint64_t)(k + 1U));
            const float x2 = xGm_.GetValue(xBase + (uint64_t)(k + 2U));
            const float x3 = xGm_.GetValue(xBase + (uint64_t)(k + 3U));

            const float w0 = wGm_.GetValue(wBase + (uint64_t)(k + 0U));
            const float w1 = wGm_.GetValue(wBase + (uint64_t)(k + 1U));
            const float w2 = wGm_.GetValue(wBase + (uint64_t)(k + 2U));
            const float w3 = wGm_.GetValue(wBase + (uint64_t)(k + 3U));

            acc += x0 * w0;
            acc += x1 * w1;
            acc += x2 * w2;
            acc += x3 * w3;
        }
        for (; k < K_; ++k) {
            acc += xGm_.GetValue(xBase + (uint64_t)k) * wGm_.GetValue(wBase + (uint64_t)k);
        }

        out = acc * invD;
    }

    __aicore__ inline void ComputeTwo(uint64_t xBase, uint32_t n, float invD, float &o0, float &o1)
    {
        const uint64_t wBase0 = (uint64_t)n * (uint64_t)K_;
        const uint64_t wBase1 = (uint64_t)(n + 1U) * (uint64_t)K_;

        float acc0 = bGm_.GetValue((uint64_t)n);
        float acc1 = bGm_.GetValue((uint64_t)(n + 1U));

        uint32_t k = 0;
        for (; k + 3U < K_; k += 4U) {
            const float x0 = xGm_.GetValue(xBase + (uint64_t)(k + 0U));
            const float x1 = xGm_.GetValue(xBase + (uint64_t)(k + 1U));
            const float x2 = xGm_.GetValue(xBase + (uint64_t)(k + 2U));
            const float x3 = xGm_.GetValue(xBase + (uint64_t)(k + 3U));

            const float w00 = wGm_.GetValue(wBase0 + (uint64_t)(k + 0U));
            const float w01 = wGm_.GetValue(wBase0 + (uint64_t)(k + 1U));
            const float w02 = wGm_.GetValue(wBase0 + (uint64_t)(k + 2U));
            const float w03 = wGm_.GetValue(wBase0 + (uint64_t)(k + 3U));

            const float w10 = wGm_.GetValue(wBase1 + (uint64_t)(k + 0U));
            const float w11 = wGm_.GetValue(wBase1 + (uint64_t)(k + 1U));
            const float w12 = wGm_.GetValue(wBase1 + (uint64_t)(k + 2U));
            const float w13 = wGm_.GetValue(wBase1 + (uint64_t)(k + 3U));

            acc0 += x0 * w00; acc1 += x0 * w10;
            acc0 += x1 * w01; acc1 += x1 * w11;
            acc0 += x2 * w02; acc1 += x2 * w12;
            acc0 += x3 * w03; acc1 += x3 * w13;
        }
        for (; k < K_; ++k) {
            const float xv = xGm_.GetValue(xBase + (uint64_t)k);
            acc0 += xv * wGm_.GetValue(wBase0 + (uint64_t)k);
            acc1 += xv * wGm_.GetValue(wBase1 + (uint64_t)k);
        }

        o0 = acc0 * invD;
        o1 = acc1 * invD;
    }

    __aicore__ inline void ComputeFour(uint64_t xBase, uint32_t n, float invD,
                                       float &o0, float &o1, float &o2, float &o3)
    {
        const uint64_t wBase0 = (uint64_t)(n + 0U) * (uint64_t)K_;
        const uint64_t wBase1 = (uint64_t)(n + 1U) * (uint64_t)K_;
        const uint64_t wBase2 = (uint64_t)(n + 2U) * (uint64_t)K_;
        const uint64_t wBase3 = (uint64_t)(n + 3U) * (uint64_t)K_;

        float acc0 = bGm_.GetValue((uint64_t)(n + 0U));
        float acc1 = bGm_.GetValue((uint64_t)(n + 1U));
        float acc2 = bGm_.GetValue((uint64_t)(n + 2U));
        float acc3 = bGm_.GetValue((uint64_t)(n + 3U));

        uint32_t k = 0;
        for (; k + 3U < K_; k += 4U) {
            const float x0 = xGm_.GetValue(xBase + (uint64_t)(k + 0U));
            const float x1 = xGm_.GetValue(xBase + (uint64_t)(k + 1U));
            const float x2 = xGm_.GetValue(xBase + (uint64_t)(k + 2U));
            const float x3 = xGm_.GetValue(xBase + (uint64_t)(k + 3U));

            const float w00 = wGm_.GetValue(wBase0 + (uint64_t)(k + 0U));
            const float w01 = wGm_.GetValue(wBase0 + (uint64_t)(k + 1U));
            const float w02 = wGm_.GetValue(wBase0 + (uint64_t)(k + 2U));
            const float w03 = wGm_.GetValue(wBase0 + (uint64_t)(k + 3U));

            const float w10 = wGm_.GetValue(wBase1 + (uint64_t)(k + 0U));
            const float w11 = wGm_.GetValue(wBase1 + (uint64_t)(k + 1U));
            const float w12 = wGm_.GetValue(wBase1 + (uint64_t)(k + 2U));
            const float w13 = wGm_.GetValue(wBase1 + (uint64_t)(k + 3U));

            const float w20 = wGm_.GetValue(wBase2 + (uint64_t)(k + 0U));
            const float w21 = wGm_.GetValue(wBase2 + (uint64_t)(k + 1U));
            const float w22 = wGm_.GetValue(wBase2 + (uint64_t)(k + 2U));
            const float w23 = wGm_.GetValue(wBase2 + (uint64_t)(k + 3U));

            const float w30 = wGm_.GetValue(wBase3 + (uint64_t)(k + 0U));
            const float w31 = wGm_.GetValue(wBase3 + (uint64_t)(k + 1U));
            const float w32 = wGm_.GetValue(wBase3 + (uint64_t)(k + 2U));
            const float w33 = wGm_.GetValue(wBase3 + (uint64_t)(k + 3U));

            acc0 += x0 * w00; acc1 += x0 * w10; acc2 += x0 * w20; acc3 += x0 * w30;
            acc0 += x1 * w01; acc1 += x1 * w11; acc2 += x1 * w21; acc3 += x1 * w31;
            acc0 += x2 * w02; acc1 += x2 * w12; acc2 += x2 * w22; acc3 += x2 * w32;
            acc0 += x3 * w03; acc1 += x3 * w13; acc2 += x3 * w23; acc3 += x3 * w33;
        }
        for (; k < K_; ++k) {
            const float xv = xGm_.GetValue(xBase + (uint64_t)k);
            acc0 += xv * wGm_.GetValue(wBase0 + (uint64_t)k);
            acc1 += xv * wGm_.GetValue(wBase1 + (uint64_t)k);
            acc2 += xv * wGm_.GetValue(wBase2 + (uint64_t)k);
            acc3 += xv * wGm_.GetValue(wBase3 + (uint64_t)k);
        }

        o0 = acc0 * invD;
        o1 = acc1 * invD;
        o2 = acc2 * invD;
        o3 = acc3 * invD;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQ_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQ_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> dGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t nTile_{256};
    uint32_t vecN_{4};
    uint32_t geluTile_{8};
};

extern "C" __global__ __aicore__ void matmul_divide_gelu_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR divisor,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulDivideGeluCustom op;
    op.Init(x, w, b, divisor, y,
            td.M, td.K, td.N,
            td.nTile, td.vecN, td.geluTile);
    op.Process();
}
