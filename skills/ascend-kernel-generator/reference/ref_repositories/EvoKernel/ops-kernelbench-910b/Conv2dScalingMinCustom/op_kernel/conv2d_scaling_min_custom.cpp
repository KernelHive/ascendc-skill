
#include "kernel_operator.h"

// Fused conv2d + scale + reduce_min over Cout, specialized to:
// N=64, Cin=64, H=W=256, Cout=128, Kh=Kw=3, stride=1, pad=0
// Hout=Wout=254, scale=2.0
//
// Output: y [N,1,254,254] where
//   y[n,0,oh,ow] = min_{co} ( (conv(x,w,bias)[n,co,oh,ow]) * scale )
//
// Launch: one block per N.
// UB tiling: compute one output row (oh) in width stripes (tile_w=64), keep a running min buffer in UB.
//
// Optimizations vs baseline:
// 1) For each output channel, compute two output pixels per iteration (ow, ow+1) to reuse a 3x4 input patch.
// 2) Unroll Cin by 2 to reduce loop/control and amortize address setup.
// 3) Load 3x3 weights once per (co,ci) and reuse across the entire width stripe (moved out of the width loop).
// 4) Bias loaded once per co.

class KernelConv2dScalingMinCustom {
public:
    __aicore__ inline KernelConv2dScalingMinCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR /*scale*/, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               float scalev,
                               uint32_t blocks,
                               uint32_t tile_w)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->scalev = scalev;
        this->blocks = blocks;
        this->tile_w = tile_w;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * cin * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * 1ULL * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe.InitBuffer(bufMin, tile_w * sizeof(float));
        pipe.InitBuffer(bufTmp, tile_w * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;
        const uint32_t ni = bid;

        const uint32_t Cin  = cin;   // 64
        const uint32_t Hin  = hin;   // 256
        const uint32_t Win  = win;   // 256
        const uint32_t Cout = cout;  // 128
        const uint32_t Hout = hout;  // 254
        const uint32_t Wout = wout;  // 254

        const uint64_t HW_in  = static_cast<uint64_t>(Hin) * static_cast<uint64_t>(Win);
        const uint64_t HW_out = static_cast<uint64_t>(Hout) * static_cast<uint64_t>(Wout);

        const uint64_t xNBase = static_cast<uint64_t>(ni) * static_cast<uint64_t>(Cin) * HW_in;
        const uint64_t yNBase = static_cast<uint64_t>(ni) * HW_out; // output has C=1

        for (uint32_t oh = 0; oh < Hout; ++oh) {
            const uint64_t yRowBase = yNBase + static_cast<uint64_t>(oh) * static_cast<uint64_t>(Wout);

            const uint64_t inRow0Off = static_cast<uint64_t>(oh) * static_cast<uint64_t>(Win);
            const uint64_t inRow1Off = inRow0Off + static_cast<uint64_t>(Win);
            const uint64_t inRow2Off = inRow1Off + static_cast<uint64_t>(Win);

            for (uint32_t ow0 = 0; ow0 < Wout; ow0 += tile_w) {
                const uint32_t curW = (ow0 + tile_w <= Wout) ? tile_w : (Wout - ow0);

                AscendC::LocalTensor<float> minLocal = bufMin.Get<float>();
                AscendC::Duplicate(minLocal, 3.402823466e+38f, static_cast<int32_t>(curW));

                for (uint32_t co = 0; co < Cout; ++co) {
                    AscendC::LocalTensor<float> tmpLocal = bufTmp.Get<float>();
                    const float biasv = bGm.GetValue(static_cast<uint64_t>(co));
                    AscendC::Duplicate(tmpLocal, biasv, static_cast<int32_t>(curW));

                    const uint64_t wCoBase = static_cast<uint64_t>(co) * static_cast<uint64_t>(Cin) * 9ULL;

                    // Cin unroll by 2 (Cin=64 even)
                    for (uint32_t ci = 0; ci < Cin; ci += 2U) {
                        // ---------- ci0 weights loaded once ----------
                        const uint64_t wC0 = wCoBase + static_cast<uint64_t>(ci) * 9ULL;
                        const float w00 = wGm.GetValue(wC0 + 0ULL);
                        const float w01 = wGm.GetValue(wC0 + 1ULL);
                        const float w02 = wGm.GetValue(wC0 + 2ULL);
                        const float w10 = wGm.GetValue(wC0 + 3ULL);
                        const float w11 = wGm.GetValue(wC0 + 4ULL);
                        const float w12 = wGm.GetValue(wC0 + 5ULL);
                        const float w20 = wGm.GetValue(wC0 + 6ULL);
                        const float w21 = wGm.GetValue(wC0 + 7ULL);
                        const float w22 = wGm.GetValue(wC0 + 8ULL);

                        const uint64_t xC0 = xNBase + static_cast<uint64_t>(ci) * HW_in;

                        // process width in pairs; keep a safe tail for odd curW
                        uint32_t t = 0;
                        for (; t + 1U < curW; t += 2U) {
                            const uint32_t ow = ow0 + t;
                            const uint64_t r00 = xC0 + inRow0Off + static_cast<uint64_t>(ow);
                            const uint64_t r10 = xC0 + inRow1Off + static_cast<uint64_t>(ow);
                            const uint64_t r20 = xC0 + inRow2Off + static_cast<uint64_t>(ow);

                            // 3x4 patch for two outputs
                            const float x00 = xGm.GetValue(r00 + 0ULL);
                            const float x01 = xGm.GetValue(r00 + 1ULL);
                            const float x02 = xGm.GetValue(r00 + 2ULL);
                            const float x03 = xGm.GetValue(r00 + 3ULL);

                            const float x10 = xGm.GetValue(r10 + 0ULL);
                            const float x11 = xGm.GetValue(r10 + 1ULL);
                            const float x12 = xGm.GetValue(r10 + 2ULL);
                            const float x13 = xGm.GetValue(r10 + 3ULL);

                            const float x20 = xGm.GetValue(r20 + 0ULL);
                            const float x21 = xGm.GetValue(r20 + 1ULL);
                            const float x22 = xGm.GetValue(r20 + 2ULL);
                            const float x23 = xGm.GetValue(r20 + 3ULL);

                            float a0 = tmpLocal.GetValue(static_cast<int32_t>(t));
                            float a1 = tmpLocal.GetValue(static_cast<int32_t>(t + 1U));

                            // ow
                            a0 += x00*w00 + x01*w01 + x02*w02
                               +  x10*w10 + x11*w11 + x12*w12
                               +  x20*w20 + x21*w21 + x22*w22;
                            // ow+1
                            a1 += x01*w00 + x02*w01 + x03*w02
                               +  x11*w10 + x12*w11 + x13*w12
                               +  x21*w20 + x22*w21 + x23*w22;

                            tmpLocal.SetValue(static_cast<int32_t>(t), a0);
                            tmpLocal.SetValue(static_cast<int32_t>(t + 1U), a1);
                        }
                        if (t < curW) {
                            const uint32_t ow = ow0 + t;
                            const uint64_t r00 = xC0 + inRow0Off + static_cast<uint64_t>(ow);
                            const uint64_t r10 = xC0 + inRow1Off + static_cast<uint64_t>(ow);
                            const uint64_t r20 = xC0 + inRow2Off + static_cast<uint64_t>(ow);

                            const float x00 = xGm.GetValue(r00 + 0ULL);
                            const float x01 = xGm.GetValue(r00 + 1ULL);
                            const float x02 = xGm.GetValue(r00 + 2ULL);

                            const float x10 = xGm.GetValue(r10 + 0ULL);
                            const float x11 = xGm.GetValue(r10 + 1ULL);
                            const float x12 = xGm.GetValue(r10 + 2ULL);

                            const float x20 = xGm.GetValue(r20 + 0ULL);
                            const float x21 = xGm.GetValue(r20 + 1ULL);
                            const float x22 = xGm.GetValue(r20 + 2ULL);

                            float a0 = tmpLocal.GetValue(static_cast<int32_t>(t));
                            a0 += x00*w00 + x01*w01 + x02*w02
                               +  x10*w10 + x11*w11 + x12*w12
                               +  x20*w20 + x21*w21 + x22*w22;
                            tmpLocal.SetValue(static_cast<int32_t>(t), a0);
                        }

                        // ---------- ci1 weights loaded once ----------
                        const uint64_t ci1 = static_cast<uint64_t>(ci + 1U);
                        const uint64_t wC1 = wCoBase + ci1 * 9ULL;
                        const float v00 = wGm.GetValue(wC1 + 0ULL);
                        const float v01 = wGm.GetValue(wC1 + 1ULL);
                        const float v02 = wGm.GetValue(wC1 + 2ULL);
                        const float v10 = wGm.GetValue(wC1 + 3ULL);
                        const float v11 = wGm.GetValue(wC1 + 4ULL);
                        const float v12 = wGm.GetValue(wC1 + 5ULL);
                        const float v20 = wGm.GetValue(wC1 + 6ULL);
                        const float v21 = wGm.GetValue(wC1 + 7ULL);
                        const float v22 = wGm.GetValue(wC1 + 8ULL);

                        const uint64_t xC1 = xNBase + ci1 * HW_in;

                        t = 0;
                        for (; t + 1U < curW; t += 2U) {
                            const uint32_t ow = ow0 + t;
                            const uint64_t r00 = xC1 + inRow0Off + static_cast<uint64_t>(ow);
                            const uint64_t r10 = xC1 + inRow1Off + static_cast<uint64_t>(ow);
                            const uint64_t r20 = xC1 + inRow2Off + static_cast<uint64_t>(ow);

                            const float x00 = xGm.GetValue(r00 + 0ULL);
                            const float x01 = xGm.GetValue(r00 + 1ULL);
                            const float x02 = xGm.GetValue(r00 + 2ULL);
                            const float x03 = xGm.GetValue(r00 + 3ULL);

                            const float x10 = xGm.GetValue(r10 + 0ULL);
                            const float x11 = xGm.GetValue(r10 + 1ULL);
                            const float x12 = xGm.GetValue(r10 + 2ULL);
                            const float x13 = xGm.GetValue(r10 + 3ULL);

                            const float x20 = xGm.GetValue(r20 + 0ULL);
                            const float x21 = xGm.GetValue(r20 + 1ULL);
                            const float x22 = xGm.GetValue(r20 + 2ULL);
                            const float x23 = xGm.GetValue(r20 + 3ULL);

                            float a0 = tmpLocal.GetValue(static_cast<int32_t>(t));
                            float a1 = tmpLocal.GetValue(static_cast<int32_t>(t + 1U));

                            a0 += x00*v00 + x01*v01 + x02*v02
                               +  x10*v10 + x11*v11 + x12*v12
                               +  x20*v20 + x21*v21 + x22*v22;

                            a1 += x01*v00 + x02*v01 + x03*v02
                               +  x11*v10 + x12*v11 + x13*v12
                               +  x21*v20 + x22*v21 + x23*v22;

                            tmpLocal.SetValue(static_cast<int32_t>(t), a0);
                            tmpLocal.SetValue(static_cast<int32_t>(t + 1U), a1);
                        }
                        if (t < curW) {
                            const uint32_t ow = ow0 + t;
                            const uint64_t r00 = xC1 + inRow0Off + static_cast<uint64_t>(ow);
                            const uint64_t r10 = xC1 + inRow1Off + static_cast<uint64_t>(ow);
                            const uint64_t r20 = xC1 + inRow2Off + static_cast<uint64_t>(ow);

                            const float x00 = xGm.GetValue(r00 + 0ULL);
                            const float x01 = xGm.GetValue(r00 + 1ULL);
                            const float x02 = xGm.GetValue(r00 + 2ULL);

                            const float x10 = xGm.GetValue(r10 + 0ULL);
                            const float x11 = xGm.GetValue(r10 + 1ULL);
                            const float x12 = xGm.GetValue(r10 + 2ULL);

                            const float x20 = xGm.GetValue(r20 + 0ULL);
                            const float x21 = xGm.GetValue(r20 + 1ULL);
                            const float x22 = xGm.GetValue(r20 + 2ULL);

                            float a0 = tmpLocal.GetValue(static_cast<int32_t>(t));
                            a0 += x00*v00 + x01*v01 + x02*v02
                               +  x10*v10 + x11*v11 + x12*v12
                               +  x20*v20 + x21*v21 + x22*v22;
                            tmpLocal.SetValue(static_cast<int32_t>(t), a0);
                        }
                    } // ci

                    // scale and update min
                    for (uint32_t t = 0; t < curW; ++t) {
                        const float v = tmpLocal.GetValue(static_cast<int32_t>(t)) * scalev;
                        const float m = minLocal.GetValue(static_cast<int32_t>(t));
                        minLocal.SetValue(static_cast<int32_t>(t), (v < m) ? v : m);
                    }
                } // co

                const uint64_t yOff = yRowBase + static_cast<uint64_t>(ow0);
                for (uint32_t t = 0; t < curW; ++t) {
                    yGm.SetValue(yOff + static_cast<uint64_t>(t), minLocal.GetValue(static_cast<int32_t>(t)));
                }
            } // ow0
        } // oh
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMin;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp;

    AscendC::GlobalTensor<float> xGm, wGm, bGm, yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    float scalev;
    uint32_t blocks;
    uint32_t tile_w;
};

extern "C" __global__ __aicore__ void conv2d_scaling_min_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR bias,
    GM_ADDR scale,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    (void)scale;

    GET_TILING_DATA(t, tiling);

    KernelConv2dScalingMinCustom op;
    op.Init(x, weight, bias, scale, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.scale_value,
            t.blocks,
            t.tile_w);
    op.Process();
}
