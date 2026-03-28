
#include "kernel_operator.h"

// Specialized fused operator (float32 only):
// x: [N=128,Cin=8,H=W=128]
// weight: [Cout=64,Cin=8,Kh=3,Kw=3], stride=1,pad=0 => y [128,64,126,126]
// fused post-ops: HardSwish then ReLU
// HardSwish: x * clamp((x+3)/6, 0, 1)
// ReLU: max(0,x)
//
// Correctness-first scalar GM access, parallelized by flattened output elements.

class KernelConv2dHardSwishReluCustom {
public:
    __aicore__ inline KernelConv2dHardSwishReluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->totalY = total_y;
        this->elemsPerBlock = elems_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (end > totalY) end = totalY;
        if (start >= end) return;

        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;

        const uint32_t hwOut = hout * wout;
        const uint32_t chwOut = cout * hwOut;

        for (uint32_t linear = start; linear < end; ++linear) {
            // y: [N,Cout,Hout,Wout]
            uint32_t tt = linear;
            const uint32_t ni = tt / chwOut;
            tt -= ni * chwOut;
            const uint32_t co = tt / hwOut;
            tt -= co * hwOut;
            const uint32_t ho = tt / wout;
            const uint32_t wo = tt - ho * wout;

            // conv top-left input index (stride=1,pad=0)
            const uint32_t hi0 = ho;
            const uint32_t wi0 = wo;

            float acc = bGm.GetValue(static_cast<uint64_t>(co));

            const uint64_t wBaseCo = static_cast<uint64_t>(co) * (8ull * KH * KW);

            #pragma unroll
            for (uint32_t ci = 0; ci < 8; ++ci) {
                const uint64_t xBase =
                    ((static_cast<uint64_t>(ni) * 8ull + static_cast<uint64_t>(ci)) *
                     static_cast<uint64_t>(hin) + static_cast<uint64_t>(hi0)) *
                     static_cast<uint64_t>(win) + static_cast<uint64_t>(wi0);

                const uint64_t wBaseC = wBaseCo + static_cast<uint64_t>(ci) * (KH * KW);

                #pragma unroll
                for (uint32_t kH = 0; kH < KH; ++kH) {
                    const uint64_t xRow = xBase + static_cast<uint64_t>(kH) * static_cast<uint64_t>(win);
                    const uint64_t wRow = wBaseC + static_cast<uint64_t>(kH) * static_cast<uint64_t>(KW);
                    #pragma unroll
                    for (uint32_t kW = 0; kW < KW; ++kW) {
                        const float xv = xGm.GetValue(xRow + static_cast<uint64_t>(kW));
                        const float wv = wGm.GetValue(wRow + static_cast<uint64_t>(kW));
                        acc += xv * wv;
                    }
                }
            }

            // HardSwish: x * clamp((x+3)/6, 0, 1)
            float t = (acc + 3.0f) * (1.0f / 6.0f);
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
            float out = acc * t;

            // ReLU
            if (out < 0.0f) out = 0.0f;

            yGm.SetValue(static_cast<uint64_t>(linear), out);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv2d_hard_swish_relu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dHardSwishReluCustom op;
    op.Init(x, weight, conv_bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.total_y, t.elems_per_block);
    op.Process();
}
