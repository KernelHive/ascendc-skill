
#include "kernel_operator.h"

// Specialized fused conv2d + min + add + mul for:
// N=128, Cin=64, H=W=128, Cout=128, Kh=Kw=3, stride=1, pad=0
// Hout=Wout=126, constv=0.5, scalev=2.0
//
// Launch: one block per N.
// Optimization: cache all weights for one (co,oh) in registers (per ci) by hoisting weight loads
// out of the ow-loop; compute 4 output pixels per iteration along width using shared 3x6 patch.

class KernelConv2dMinAddMultiplyCustom {
public:
    __aicore__ inline KernelConv2dMinAddMultiplyCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR cb, GM_ADDR pb,
                               GM_ADDR /*cst*/, GM_ADDR /*scl*/, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t /*kh*/, uint32_t /*kw*/,
                               uint32_t hout, uint32_t wout,
                               float constv, float scalev,
                               uint32_t blocks)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout;
        this->hout = hout; this->wout = wout;
        this->constv = constv;
        this->scalev = scalev;
        this->blocks = blocks;

        const uint64_t xSize  = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize  = static_cast<uint64_t>(cout) * cin * 3u * 3u;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t pbSize = static_cast<uint64_t>(cout);
        const uint64_t ySize  = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        cbGm.SetGlobalBuffer((__gm__ float*)cb, cbSize);
        pbGm.SetGlobalBuffer((__gm__ float*)pb, pbSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;

        const uint32_t ni = bid;

        const uint32_t Cin  = cin;   // 64
        const uint32_t Hin  = hin;   // 128
        const uint32_t Win  = win;   // 128
        const uint32_t Cout = cout;  // 128
        const uint32_t Hout = hout;  // 126
        const uint32_t Wout = wout;  // 126

        const uint64_t HW_in  = static_cast<uint64_t>(Hin) * static_cast<uint64_t>(Win);
        const uint64_t HW_out = static_cast<uint64_t>(Hout) * static_cast<uint64_t>(Wout);

        const uint64_t xNBase = static_cast<uint64_t>(ni) * static_cast<uint64_t>(Cin) * HW_in;
        const uint64_t yNBase = static_cast<uint64_t>(ni) * static_cast<uint64_t>(Cout) * HW_out;

        // Store weights for all Cin in local arrays per (co,oh).
        // Layout: wv[ci][0..8] but flattened to 9 arrays to help compiler.
        float w0[64], w1[64], w2[64], w3[64], w4[64], w5[64], w6[64], w7[64], w8[64];

        for (uint32_t co = 0; co < Cout; ++co) {
            const float cbv = cbGm.GetValue(static_cast<uint64_t>(co));
            const float pbv = pbGm.GetValue(static_cast<uint64_t>(co));

            const uint64_t wCoBase = static_cast<uint64_t>(co) * static_cast<uint64_t>(Cin) * 9ULL;
            const uint64_t yCoBase = yNBase + static_cast<uint64_t>(co) * HW_out;

            for (uint32_t oh = 0; oh < Hout; ++oh) {
                // Preload all weights for this co once per output row oh
                // (weights are independent of oh; hoisting here to keep register pressure manageable).
                for (uint32_t ci = 0; ci < Cin; ++ci) {
                    const uint64_t wCBase = wCoBase + static_cast<uint64_t>(ci) * 9ULL;
                    w0[ci] = wGm.GetValue(wCBase + 0ULL);
                    w1[ci] = wGm.GetValue(wCBase + 1ULL);
                    w2[ci] = wGm.GetValue(wCBase + 2ULL);
                    w3[ci] = wGm.GetValue(wCBase + 3ULL);
                    w4[ci] = wGm.GetValue(wCBase + 4ULL);
                    w5[ci] = wGm.GetValue(wCBase + 5ULL);
                    w6[ci] = wGm.GetValue(wCBase + 6ULL);
                    w7[ci] = wGm.GetValue(wCBase + 7ULL);
                    w8[ci] = wGm.GetValue(wCBase + 8ULL);
                }

                const uint64_t yRowBase = yCoBase + static_cast<uint64_t>(oh) * static_cast<uint64_t>(Wout);
                const uint64_t inRow0Off = static_cast<uint64_t>(oh) * static_cast<uint64_t>(Win);
                const uint64_t inRow1Off = inRow0Off + static_cast<uint64_t>(Win);
                const uint64_t inRow2Off = inRow1Off + static_cast<uint64_t>(Win);

                // Compute 4 outputs per loop. Wout=126 => 31*4=124 plus tail 2.
                uint32_t ow = 0;
                for (; ow + 3U < Wout; ow += 4U) {
                    float acc0 = cbv, acc1 = cbv, acc2 = cbv, acc3 = cbv;

                    const uint64_t ow64 = static_cast<uint64_t>(ow);

                    for (uint32_t ci = 0; ci < Cin; ++ci) {
                        const uint64_t xCBase = xNBase + static_cast<uint64_t>(ci) * HW_in;

                        const uint64_t r0 = xCBase + inRow0Off + ow64;
                        const uint64_t r1 = xCBase + inRow1Off + ow64;
                        const uint64_t r2 = xCBase + inRow2Off + ow64;

                        // Load 6 columns to cover 4 outputs (3x6 patch)
                        const float x00 = xGm.GetValue(r0 + 0ULL);
                        const float x01 = xGm.GetValue(r0 + 1ULL);
                        const float x02 = xGm.GetValue(r0 + 2ULL);
                        const float x03 = xGm.GetValue(r0 + 3ULL);
                        const float x04 = xGm.GetValue(r0 + 4ULL);
                        const float x05 = xGm.GetValue(r0 + 5ULL);

                        const float x10 = xGm.GetValue(r1 + 0ULL);
                        const float x11 = xGm.GetValue(r1 + 1ULL);
                        const float x12 = xGm.GetValue(r1 + 2ULL);
                        const float x13 = xGm.GetValue(r1 + 3ULL);
                        const float x14 = xGm.GetValue(r1 + 4ULL);
                        const float x15 = xGm.GetValue(r1 + 5ULL);

                        const float x20 = xGm.GetValue(r2 + 0ULL);
                        const float x21 = xGm.GetValue(r2 + 1ULL);
                        const float x22 = xGm.GetValue(r2 + 2ULL);
                        const float x23 = xGm.GetValue(r2 + 3ULL);
                        const float x24 = xGm.GetValue(r2 + 4ULL);
                        const float x25 = xGm.GetValue(r2 + 5ULL);

                        const float ww0 = w0[ci], ww1 = w1[ci], ww2 = w2[ci];
                        const float ww3 = w3[ci], ww4 = w4[ci], ww5 = w5[ci];
                        const float ww6 = w6[ci], ww7 = w7[ci], ww8 = w8[ci];

                        // ow
                        acc0 += x00*ww0 + x01*ww1 + x02*ww2 + x10*ww3 + x11*ww4 + x12*ww5 + x20*ww6 + x21*ww7 + x22*ww8;
                        // ow+1
                        acc1 += x01*ww0 + x02*ww1 + x03*ww2 + x11*ww3 + x12*ww4 + x13*ww5 + x21*ww6 + x22*ww7 + x23*ww8;
                        // ow+2
                        acc2 += x02*ww0 + x03*ww1 + x04*ww2 + x12*ww3 + x13*ww4 + x14*ww5 + x22*ww6 + x23*ww7 + x24*ww8;
                        // ow+3
                        acc3 += x03*ww0 + x04*ww1 + x05*ww2 + x13*ww3 + x14*ww4 + x15*ww5 + x23*ww6 + x24*ww7 + x25*ww8;
                    }

                    const float min0 = (acc0 < constv) ? acc0 : constv;
                    const float min1 = (acc1 < constv) ? acc1 : constv;
                    const float min2 = (acc2 < constv) ? acc2 : constv;
                    const float min3 = (acc3 < constv) ? acc3 : constv;

                    const float out0 = (min0 + pbv) * scalev;
                    const float out1 = (min1 + pbv) * scalev;
                    const float out2 = (min2 + pbv) * scalev;
                    const float out3 = (min3 + pbv) * scalev;

                    const uint64_t yOff = yRowBase + ow64;
                    yGm.SetValue(yOff + 0ULL, out0);
                    yGm.SetValue(yOff + 1ULL, out1);
                    yGm.SetValue(yOff + 2ULL, out2);
                    yGm.SetValue(yOff + 3ULL, out3);
                }

                // Tail: remaining 2 pixels (Wout=126 => ow==124)
                for (; ow < Wout; ++ow) {
                    float acc = cbv;
                    const uint64_t ow64 = static_cast<uint64_t>(ow);

                    for (uint32_t ci = 0; ci < Cin; ++ci) {
                        const uint64_t xCBase = xNBase + static_cast<uint64_t>(ci) * HW_in;

                        const uint64_t r0 = xCBase + inRow0Off + ow64;
                        const uint64_t r1 = xCBase + inRow1Off + ow64;
                        const uint64_t r2 = xCBase + inRow2Off + ow64;

                        const float ww0 = w0[ci], ww1 = w1[ci], ww2 = w2[ci];
                        const float ww3 = w3[ci], ww4 = w4[ci], ww5 = w5[ci];
                        const float ww6 = w6[ci], ww7 = w7[ci], ww8 = w8[ci];

                        acc += xGm.GetValue(r0 + 0ULL) * ww0;
                        acc += xGm.GetValue(r0 + 1ULL) * ww1;
                        acc += xGm.GetValue(r0 + 2ULL) * ww2;

                        acc += xGm.GetValue(r1 + 0ULL) * ww3;
                        acc += xGm.GetValue(r1 + 1ULL) * ww4;
                        acc += xGm.GetValue(r1 + 2ULL) * ww5;

                        acc += xGm.GetValue(r2 + 0ULL) * ww6;
                        acc += xGm.GetValue(r2 + 1ULL) * ww7;
                        acc += xGm.GetValue(r2 + 2ULL) * ww8;
                    }

                    const float minv = (acc < constv) ? acc : constv;
                    const float outv = (minv + pbv) * scalev;
                    yGm.SetValue(yRowBase + ow64, outv);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm, wGm, cbGm, pbGm, yGm;
    uint32_t n, cin, hin, win;
    uint32_t cout;
    uint32_t hout, wout;
    float constv, scalev;
    uint32_t blocks;
};

extern "C" __global__ __aicore__ void conv2d_min_add_multiply_custom(
    GM_ADDR x,
    GM_ADDR weight,
    GM_ADDR conv_bias,
    GM_ADDR post_bias,
    GM_ADDR constant_value,
    GM_ADDR scaling_factor,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    (void)constant_value;
    (void)scaling_factor;

    GET_TILING_DATA(t, tiling);

    KernelConv2dMinAddMultiplyCustom op;
    op.Init(x, weight, conv_bias, post_bias, constant_value, scaling_factor, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.constant_value, t.scaling_factor,
            t.blocks);
    op.Process();
}
