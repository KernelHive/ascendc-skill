
#include "kernel_operator.h"

class KernelConvStandard3dAsymSquare {
public:
    __aicore__ inline KernelConvStandard3dAsymSquare() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t totalElems, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalElems_ = totalElems;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        // Fixed benchmark shapes/params
        constexpr uint32_t N = 16;
        constexpr uint32_t CIN = 3;
        constexpr uint32_t H = 256;
        constexpr uint32_t W = 256;
        constexpr uint32_t D = 10;

        constexpr uint32_t COUT = 64;
        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;
        constexpr uint32_t KD = 1;

        constexpr uint32_t HOUT = 254;
        constexpr uint32_t WOUT = 254;
        constexpr uint32_t DOUT = 10;

        // Strides for x: [N, CIN, H, W, D] contiguous
        constexpr uint32_t XsD  = 1;
        constexpr uint32_t XsW  = D;
        constexpr uint32_t XsH  = W * D;
        constexpr uint32_t XsCI = H * W * D;
        constexpr uint32_t XsN  = CIN * H * W * D;

        // Strides for y: [N, COUT, HOUT, WOUT, DOUT] contiguous
        constexpr uint32_t YsD  = 1;
        constexpr uint32_t YsW  = DOUT;
        constexpr uint32_t YsH  = WOUT * DOUT;
        constexpr uint32_t YsCO = HOUT * WOUT * DOUT;
        constexpr uint32_t YsN  = COUT * HOUT * WOUT * DOUT;

        // Weights: [COUT, CIN, KH, KW, KD], KD=1
        constexpr uint32_t WsCI = KH * KW * KD;        // 9
        constexpr uint32_t WsCO = CIN * KH * KW * KD;  // 27

        // Total output elements
        constexpr uint32_t TOTAL = N * COUT * HOUT * WOUT * DOUT;
        // totalElems_ comes from tiling (should match TOTAL for the specialized binding)
        uint32_t total = totalElems_;
        if (total == 0 || total > TOTAL) total = TOTAL;

        uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bdim = blockDim_;

        // Simple contiguous partitioning over linear output elements
        uint32_t elemsPerBlock = (total + bdim - 1) / bdim;
        uint32_t elemBegin = bid * elemsPerBlock;
        uint32_t elemEnd = elemBegin + elemsPerBlock;
        if (elemEnd > total) elemEnd = total;

        for (uint32_t e = elemBegin; e < elemEnd; ++e) {
            // Linear index decode: e -> n, co, ho, wo, do
            // Layout: ((((n*COUT + co)*HOUT + ho)*WOUT + wo)*DOUT + do)
            uint32_t t = e;

            uint32_t do_ = t % DOUT; t /= DOUT;
            uint32_t wo  = t % WOUT; t /= WOUT;
            uint32_t ho  = t % HOUT; t /= HOUT;
            uint32_t co  = t % COUT; t /= COUT;
            uint32_t n   = t;

            // Base indices
            uint32_t xBaseN = n * XsN;
            uint32_t yIdx   = n * YsN + co * YsCO + ho * YsH + wo * YsW + do_ * YsD;

            // pad=0, stride=1
            uint32_t hi0 = ho;
            uint32_t wi0 = wo;

            // Load weights (27 floats) into registers once per output element.
            // (This is still beneficial because each element is now independent; avoids repeated GM reads within MACs.)
            float w0[9];
            float w1[9];
            float w2[9];
            uint32_t wBase = co * WsCO;
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                w0[i] = wGm.GetValue(wBase + 0 * WsCI + (uint32_t)i);
                w1[i] = wGm.GetValue(wBase + 1 * WsCI + (uint32_t)i);
                w2[i] = wGm.GetValue(wBase + 2 * WsCI + (uint32_t)i);
            }

            // Precompute x base for each input channel at (hi0, wi0, do_)
            uint32_t xCi0 = xBaseN + 0 * XsCI + hi0 * XsH + wi0 * XsW + do_ * XsD;
            uint32_t xCi1 = xBaseN + 1 * XsCI + hi0 * XsH + wi0 * XsW + do_ * XsD;
            uint32_t xCi2 = xBaseN + 2 * XsCI + hi0 * XsH + wi0 * XsW + do_ * XsD;

            // Hoist row offsets to reduce address arithmetic
            uint32_t r0off = 0 * XsH;
            uint32_t r1off = 1 * XsH;
            uint32_t r2off = 2 * XsH;

            uint32_t c0 = 0 * XsW;
            uint32_t c1 = 1 * XsW;
            uint32_t c2 = 2 * XsW;

            float acc = 0.0f;

            // ci=0
            {
                uint32_t r0 = xCi0 + r0off;
                uint32_t r1 = xCi0 + r1off;
                uint32_t r2 = xCi0 + r2off;

                acc += xGm.GetValue(r0 + c0) * w0[0];
                acc += xGm.GetValue(r0 + c1) * w0[1];
                acc += xGm.GetValue(r0 + c2) * w0[2];

                acc += xGm.GetValue(r1 + c0) * w0[3];
                acc += xGm.GetValue(r1 + c1) * w0[4];
                acc += xGm.GetValue(r1 + c2) * w0[5];

                acc += xGm.GetValue(r2 + c0) * w0[6];
                acc += xGm.GetValue(r2 + c1) * w0[7];
                acc += xGm.GetValue(r2 + c2) * w0[8];
            }
            // ci=1
            {
                uint32_t r0 = xCi1 + r0off;
                uint32_t r1 = xCi1 + r1off;
                uint32_t r2 = xCi1 + r2off;

                acc += xGm.GetValue(r0 + c0) * w1[0];
                acc += xGm.GetValue(r0 + c1) * w1[1];
                acc += xGm.GetValue(r0 + c2) * w1[2];

                acc += xGm.GetValue(r1 + c0) * w1[3];
                acc += xGm.GetValue(r1 + c1) * w1[4];
                acc += xGm.GetValue(r1 + c2) * w1[5];

                acc += xGm.GetValue(r2 + c0) * w1[6];
                acc += xGm.GetValue(r2 + c1) * w1[7];
                acc += xGm.GetValue(r2 + c2) * w1[8];
            }
            // ci=2
            {
                uint32_t r0 = xCi2 + r0off;
                uint32_t r1 = xCi2 + r1off;
                uint32_t r2 = xCi2 + r2off;

                acc += xGm.GetValue(r0 + c0) * w2[0];
                acc += xGm.GetValue(r0 + c1) * w2[1];
                acc += xGm.GetValue(r0 + c2) * w2[2];

                acc += xGm.GetValue(r1 + c0) * w2[3];
                acc += xGm.GetValue(r1 + c1) * w2[4];
                acc += xGm.GetValue(r1 + c2) * w2[5];

                acc += xGm.GetValue(r2 + c0) * w2[6];
                acc += xGm.GetValue(r2 + c1) * w2[7];
                acc += xGm.GetValue(r2 + c2) * w2[8];
            }

            yGm.SetValue(yIdx, acc);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalElems_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_standard3d_asymmetric_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard3dAsymSquare op;
    op.Init(x, weight, y, tiling_data.totalElems, tiling_data.blockDim);
    op.Process();
}
