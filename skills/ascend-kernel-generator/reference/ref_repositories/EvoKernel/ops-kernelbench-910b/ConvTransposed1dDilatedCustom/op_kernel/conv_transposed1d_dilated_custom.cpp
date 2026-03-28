
#include "kernel_operator.h"

// Specialized ConvTranspose1d (PyTorch NCL):
// x:      [N=32, Cin=32, Lin=131072]
// weight: [Cin=32, Cout=64, K=5]   (groups=1)
// stride=1, padding=0, dilation=3, output_padding=0, bias=False
// Lout = (Lin - 1)*1 - 2*0 + 3*(5-1) + 0 + 1 = 131084
// y:      [N=32, Cout=64, Lout=131084]
//
// Implementation notes (this round):
// - GM-only scalar path: avoid version-fragile UB APIs.
// - No atomic add, no in-kernel memset: each y element computed exactly once.
// - Parallelize by sharding linearized y indices across blockDim cores.

class KernelConvTranspose1dDilatedS1P0D3K5_Fp32_GmScalar {
public:
    __aicore__ inline KernelConvTranspose1dDilatedS1P0D3K5_Fp32_GmScalar() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalY, uint32_t lout, uint32_t blockDim)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        totalY_ = totalY;
        lout_ = lout;
        blockDim_ = (blockDim == 0) ? 1 : blockDim;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t N = 32;
        constexpr int32_t CIN = 32;
        constexpr int32_t LIN = 131072;

        constexpr int32_t COUT = 64;
        constexpr int32_t K = 5;

        constexpr int32_t STRIDE = 1;
        constexpr int32_t PAD = 0;
        constexpr int32_t DIL = 3;
        constexpr int32_t OUTPAD = 0;

        constexpr int32_t LOUT = (LIN - 1) * STRIDE - 2 * PAD + DIL * (K - 1) + OUTPAD + 1; // 131084

        constexpr int64_t TOTALY_CONST = (int64_t)N * COUT * (int64_t)LOUT;

        uint32_t totalY = totalY_;
        if (totalY == 0 || (int64_t)totalY > TOTALY_CONST) totalY = (uint32_t)TOTALY_CONST;
        if ((int32_t)lout_ != LOUT) return;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = blockDim_;

        const uint32_t elemsPerBlock = (totalY + bdim - 1) / bdim;
        uint32_t begin = bid * elemsPerBlock;
        uint32_t end = begin + elemsPerBlock;
        if (begin > totalY) begin = totalY;
        if (end > totalY) end = totalY;
        if (begin >= end) return;

        // Strides for NCL contiguous
        constexpr int64_t X_STRIDE_C = (int64_t)LIN;
        constexpr int64_t X_STRIDE_N = (int64_t)CIN * X_STRIDE_C;

        constexpr int64_t Y_STRIDE_C = (int64_t)LOUT;
        constexpr int64_t Y_STRIDE_N = (int64_t)COUT * Y_STRIDE_C;

        // Weight layout [Cin, Cout, K]
        constexpr int64_t W_STRIDE_COUT = (int64_t)K;
        constexpr int64_t W_STRIDE_CIN = (int64_t)COUT * W_STRIDE_COUT;

        // For transposed conv (stride=1,pad=0,outpad=0):
        // outPos = inPos + k*DIL => inPos = outPos - k*DIL
        for (uint32_t yIdx = begin; yIdx < end; ++yIdx) {
            uint32_t t = yIdx;
            const int32_t outPos = (int32_t)(t % (uint32_t)LOUT); t /= (uint32_t)LOUT;
            const int32_t co = (int32_t)(t % (uint32_t)COUT);   t /= (uint32_t)COUT;
            const int32_t n  = (int32_t)t;

            const int64_t xBaseN = (int64_t)n * X_STRIDE_N;
            const int64_t yBase = (int64_t)n * Y_STRIDE_N + (int64_t)co * Y_STRIDE_C + (int64_t)outPos;

            float acc = 0.0f;

            // small K=5: unroll k and only check inPos range.
            for (int32_t ci = 0; ci < CIN; ++ci) {
                const int64_t xBaseCi = xBaseN + (int64_t)ci * X_STRIDE_C;
                const int64_t wBaseCiCo = (int64_t)ci * W_STRIDE_CIN + (int64_t)co * W_STRIDE_COUT;

                // k=0..4
                int32_t inPos;
                inPos = outPos - 0 * DIL;
                if ((uint32_t)inPos < (uint32_t)LIN) {
                    const float xf = xGm.GetValue((uint64_t)(xBaseCi + (int64_t)inPos));
                    const float wf = wGm.GetValue((uint64_t)(wBaseCiCo + 0));
                    acc += xf * wf;
                }
                inPos = outPos - 1 * DIL;
                if ((uint32_t)inPos < (uint32_t)LIN) {
                    const float xf = xGm.GetValue((uint64_t)(xBaseCi + (int64_t)inPos));
                    const float wf = wGm.GetValue((uint64_t)(wBaseCiCo + 1));
                    acc += xf * wf;
                }
                inPos = outPos - 2 * DIL;
                if ((uint32_t)inPos < (uint32_t)LIN) {
                    const float xf = xGm.GetValue((uint64_t)(xBaseCi + (int64_t)inPos));
                    const float wf = wGm.GetValue((uint64_t)(wBaseCiCo + 2));
                    acc += xf * wf;
                }
                inPos = outPos - 3 * DIL;
                if ((uint32_t)inPos < (uint32_t)LIN) {
                    const float xf = xGm.GetValue((uint64_t)(xBaseCi + (int64_t)inPos));
                    const float wf = wGm.GetValue((uint64_t)(wBaseCiCo + 3));
                    acc += xf * wf;
                }
                inPos = outPos - 4 * DIL;
                if ((uint32_t)inPos < (uint32_t)LIN) {
                    const float xf = xGm.GetValue((uint64_t)(xBaseCi + (int64_t)inPos));
                    const float wf = wGm.GetValue((uint64_t)(wBaseCiCo + 4));
                    acc += xf * wf;
                }
            }

            yGm.SetValue((uint64_t)yBase, acc);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalY_{0};
    uint32_t lout_{0};
    uint32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void conv_transposed1d_dilated_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvTranspose1dDilatedS1P0D3K5_Fp32_GmScalar op;
    op.Init(x, weight, y, tiling_data.totalY, tiling_data.lout, tiling_data.blockDim);
    op.Process();
}
