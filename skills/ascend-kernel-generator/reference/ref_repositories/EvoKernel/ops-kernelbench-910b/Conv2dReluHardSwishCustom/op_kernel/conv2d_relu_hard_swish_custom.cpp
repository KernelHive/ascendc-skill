
#include "kernel_operator.h"

__aicore__ inline uint32_t U32Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

class KernelConv2dReluHardSwishCustom {
public:
    __aicore__ inline KernelConv2dReluHardSwishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               uint32_t totalY, uint32_t elemsPerCore)
    {
        n_ = n; cin_ = cin; hin_ = hin; win_ = win;
        cout_ = cout; kh_ = kh; kw_ = kw;
        hout_ = hout; wout_ = wout;
        totalY_ = totalY; elemsPerCore_ = elemsPerCore;

        const uint64_t xSize = static_cast<uint64_t>(n_) * cin_ * hin_ * win_;
        const uint64_t wSize = static_cast<uint64_t>(cout_) * cin_ * kh_ * kw_;
        const uint64_t bSize = static_cast<uint64_t>(cout_);
        const uint64_t ySize = static_cast<uint64_t>(n_) * cout_ * hout_ * wout_;

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm_.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint64_t start = static_cast<uint64_t>(bid) * static_cast<uint64_t>(elemsPerCore_);
        if (start >= totalY_) return;

        const uint64_t end = U32Min(static_cast<uint32_t>(start + elemsPerCore_), totalY_);

        // Specialization constants (match host checks).
        constexpr uint32_t CIN = 8;
        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;

        const uint32_t hwOut = hout_ * wout_;
        const uint32_t chwOut = cout_ * hwOut;

        for (uint64_t linear = start; linear < end; ++linear) {
            uint32_t tt = static_cast<uint32_t>(linear);

            const uint32_t ni = tt / chwOut;
            tt -= ni * chwOut;
            const uint32_t co = tt / hwOut;
            tt -= co * hwOut;
            const uint32_t ho = tt / wout_;
            const uint32_t wo = tt - ho * wout_;

            const uint32_t hi0 = ho;  // stride=1, pad=0
            const uint32_t wi0 = wo;

            float acc = bGm_.GetValue(static_cast<uint64_t>(co));

            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (static_cast<uint64_t>(CIN) * KH * KW);

            #pragma unroll
            for (uint32_t ci = 0; ci < CIN; ++ci) {
                const uint64_t xBase =
                    ((static_cast<uint64_t>(ni) * CIN + ci) * static_cast<uint64_t>(hin_) + hi0) *
                        static_cast<uint64_t>(win_) +
                    wi0;

                const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                #pragma unroll
                for (uint32_t kH = 0; kH < KH; ++kH) {
                    const uint64_t xRow = xBase + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win_);
                    const uint64_t wRow = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);

                    #pragma unroll
                    for (uint32_t kW = 0; kW < KW; ++kW) {
                        const float xv = xGm_.GetValue(xRow + kW);
                        const float wv = wGm_.GetValue(wRow + kW);
                        acc += xv * wv;
                    }
                }
            }

            // ReLU
            if (acc < 0.0f) acc = 0.0f;

            // HardSwish: x * clamp((x+3)/6, 0, 1)
            float t = (acc + 3.0f) * (1.0f / 6.0f);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            const float out = acc * t;

            yGm_.SetValue(linear, out);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t n_{0}, cin_{0}, hin_{0}, win_{0};
    uint32_t cout_{0}, kh_{0}, kw_{0};
    uint32_t hout_{0}, wout_{0};
    uint32_t totalY_{0}, elemsPerCore_{0};
};

extern "C" __global__ __aicore__ void conv2d_relu_hard_swish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dReluHardSwishCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.totalY, t.elemsPerCore);
    op.Process();
}
