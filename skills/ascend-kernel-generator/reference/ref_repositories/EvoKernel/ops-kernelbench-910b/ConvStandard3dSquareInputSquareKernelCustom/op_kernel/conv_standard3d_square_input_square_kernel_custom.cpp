
#include "kernel_operator.h"

// Fixed specialization (benchmark contract):
// x: [16, 3, 64, 64, 64] float32 (NCDHW)
// w: [64, 3, 3, 3, 3] float32
// y: [16, 64, 62, 62, 62] float32
// stride=1, pad=0, dilation=1, groups=1, bias=false
//
// This round:
// - Pass blockDim via tiling to avoid GetBlockNum() overhead and improve determinism.
// - Increase launch parallelism (host) to reduce pipeline gaps.
// - Keep weight-per-oc caching and ow tiling (TOW=4).

class KernelConvStandard3dSquare {
public:
    __aicore__ inline KernelConvStandard3dSquare() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        blockDim_ = blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 16;
        constexpr int CIN = 3;
        constexpr int D = 64;
        constexpr int H = 64;
        constexpr int W = 64;

        constexpr int COUT = 64;
        constexpr int DO = 62;
        constexpr int HO = 62;
        constexpr int WO = 62;

        constexpr int K = 3;
        constexpr int TOW = 4;

        constexpr int64_t HW = (int64_t)H * W;
        constexpr int64_t DHW = (int64_t)D * H * W;

        // Rows are (n, oc, od, oh)
        constexpr int64_t ROWS = (int64_t)N * COUT * DO * HO;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bdim = blockDim_;
        if (bdim == 0) { bdim = 1; }

        const int64_t chunk = (ROWS + (int64_t)bdim - 1) / (int64_t)bdim;
        int64_t rowStart = (int64_t)bid * chunk;
        int64_t rowEnd = rowStart + chunk;
        if (rowEnd > ROWS) rowEnd = ROWS;

        for (int64_t row = rowStart; row < rowEnd; ++row) {
            int64_t t = row;
            int oh = (int)(t % HO); t /= HO;
            int od = (int)(t % DO); t /= DO;
            int oc = (int)(t % COUT); t /= COUT;
            int n  = (int)t;

            const int64_t yRowBase = (((((int64_t)n * COUT + oc) * DO + od) * HO + oh) * WO);

            const int id0 = od;
            const int ih0 = oh;

            const int64_t xNBase = (int64_t)n * (int64_t)CIN * DHW;

            int64_t pBase[27];
#pragma unroll
            for (int ic = 0; ic < CIN; ++ic) {
                const int64_t xNcBase = xNBase + (int64_t)ic * DHW;
#pragma unroll
                for (int kz = 0; kz < 3; ++kz) {
                    const int64_t xZBase = xNcBase + (int64_t)(id0 + kz) * HW;
#pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        const int p = ic * 9 + kz * 3 + ky;
                        pBase[p] = xZBase + (int64_t)(ih0 + ky) * W;
                    }
                }
            }

            float w0[27], w1[27], w2[27];
            const int64_t wBase = (int64_t)oc * (CIN * K * K * K);
#pragma unroll
            for (int p = 0; p < 27; ++p) {
                const int64_t wb = wBase + (int64_t)p * 3;
                w0[p] = wGm.GetValue((int)(wb + 0));
                w1[p] = wGm.GetValue((int)(wb + 1));
                w2[p] = wGm.GetValue((int)(wb + 2));
            }

            int ow = 0;
            for (; ow + TOW - 1 < WO; ow += TOW) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#pragma unroll
                for (int p = 0; p < 27; ++p) {
                    const int64_t xb = pBase[p] + (int64_t)ow;

                    float x0 = xGm.GetValue((int)(xb + 0));
                    float x1 = xGm.GetValue((int)(xb + 1));
                    float x2 = xGm.GetValue((int)(xb + 2));
                    float x3 = xGm.GetValue((int)(xb + 3));
                    float x4 = xGm.GetValue((int)(xb + 4));
                    float x5 = xGm.GetValue((int)(xb + 5));

                    const float ww0 = w0[p];
                    const float ww1 = w1[p];
                    const float ww2 = w2[p];

                    acc0 += x0 * ww0; acc0 += x1 * ww1; acc0 += x2 * ww2;
                    acc1 += x1 * ww0; acc1 += x2 * ww1; acc1 += x3 * ww2;
                    acc2 += x2 * ww0; acc2 += x3 * ww1; acc2 += x4 * ww2;
                    acc3 += x3 * ww0; acc3 += x4 * ww1; acc3 += x5 * ww2;
                }

                const int64_t yb = yRowBase + (int64_t)ow;
                yGm.SetValue((int)(yb + 0), acc0);
                yGm.SetValue((int)(yb + 1), acc1);
                yGm.SetValue((int)(yb + 2), acc2);
                yGm.SetValue((int)(yb + 3), acc3);
            }

            for (; ow < WO; ++ow) {
                float acc = 0.0f;
#pragma unroll
                for (int p = 0; p < 27; ++p) {
                    const int64_t xb = pBase[p] + (int64_t)ow;
                    float x0v = xGm.GetValue((int)(xb + 0));
                    float x1v = xGm.GetValue((int)(xb + 1));
                    float x2v = xGm.GetValue((int)(xb + 2));
                    acc += x0v * w0[p];
                    acc += x1v * w1[p];
                    acc += x2v * w2[p];
                }
                yGm.SetValue((int)(yRowBase + (int64_t)ow), acc);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_standard3d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard3dSquare op;
    op.Init(x, weight, y, tiling_data.blockDim);
    op.Process();
}
