
#include "kernel_operator.h"

// Optimized mapping: one block computes one (n, co) plane of size DHW.
// This removes per-element div/mod and enables 2048 blocks total for the benchmark.
// Also removes TQue ping-pong and caches per-channel params once per block.
//
// Specialized to:
// x: [128,3,16,64,64], w: [16,3,3,3,3], conv_bias: [16], scaling_factor: [16], bias: [16]
// y: [128,16,14,62,62], stride=1,pad=0,dil=1,groups=1,k=3

class KernelConv3dScalingTanhMultiplySigmoidCustom {
public:
    __aicore__ inline KernelConv3dScalingTanhMultiplySigmoidCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR sf, GM_ADDR bias, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t dout, uint32_t hout, uint32_t wout,
                               uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t groups,
                               uint32_t blocks, uint32_t dhw, uint32_t tile)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->dout = dout; this->hout = hout; this->wout = wout;
        this->stride = stride; this->pad = pad; this->dilation = dilation; this->groups = groups;
        this->blocks = blocks;
        this->dhw = dhw;
        this->tile = tile;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * (cin / groups) * kd * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t sSize  = static_cast<uint64_t>(cout);
        const uint64_t bSize  = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * dout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBiasGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        scaleGm.SetGlobalBuffer((__gm__ float*)sf, sSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers for vector math
        pipe.InitBuffer(xBuf, tile * sizeof(float));
        pipe.InitBuffer(ePosBuf, tile * sizeof(float));
        pipe.InitBuffer(eNegBuf, tile * sizeof(float));
        pipe.InitBuffer(tmpBuf,  tile * sizeof(float));
        pipe.InitBuffer(oneBuf,  tile * sizeof(float));
    }

    __aicore__ inline float ConvAt(uint32_t ni, uint32_t co, uint32_t od, uint32_t oh, uint32_t ow, float cb) const
    {
        float acc = cb;

        const uint64_t cin64 = static_cast<uint64_t>(cin);
        const uint64_t din64 = static_cast<uint64_t>(din);
        const uint64_t hin64 = static_cast<uint64_t>(hin);
        const uint64_t win64 = static_cast<uint64_t>(win);

        // groups=1, stride=1, pad=0, dilation=1, K=3
        for (uint32_t ci = 0; ci < cin; ++ci) {
            for (uint32_t kD = 0; kD < 3; ++kD) {
                const uint32_t id = od + kD;
                for (uint32_t kH = 0; kH < 3; ++kH) {
                    const uint32_t ih = oh + kH;

                    const uint64_t xBase =
                        (((static_cast<uint64_t>(ni) * cin64 + static_cast<uint64_t>(ci)) * din64 +
                           static_cast<uint64_t>(id)) * hin64 + static_cast<uint64_t>(ih)) * win64;

                    const uint64_t wBase =
                        (((static_cast<uint64_t>(co) * cin64 + static_cast<uint64_t>(ci)) * 3ull +
                           static_cast<uint64_t>(kD)) * 3ull + static_cast<uint64_t>(kH)) * 3ull;

                    // unroll kw=3
                    acc += xGm.GetValue(xBase + static_cast<uint64_t>(ow + 0)) * wGm.GetValue(wBase + 0);
                    acc += xGm.GetValue(xBase + static_cast<uint64_t>(ow + 1)) * wGm.GetValue(wBase + 1);
                    acc += xGm.GetValue(xBase + static_cast<uint64_t>(ow + 2)) * wGm.GetValue(wBase + 2);
                }
            }
        }
        return acc;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= blocks) return;

        const uint32_t co = bid % cout;
        const uint32_t ni = bid / cout;

        const float cb = convBiasGm.GetValue(static_cast<uint64_t>(co));
        const float sf = scaleGm.GetValue(static_cast<uint64_t>(co));
        const float bb = biasGm.GetValue(static_cast<uint64_t>(co));

        AscendC::LocalTensor<float> xLocal  = xBuf.Get<float>();
        AscendC::LocalTensor<float> ePos    = ePosBuf.Get<float>();
        AscendC::LocalTensor<float> eNeg    = eNegBuf.Get<float>();
        AscendC::LocalTensor<float> tmp     = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> one     = oneBuf.Get<float>();

        AscendC::Duplicate(one, 1.0f, tile);

        // Walk contiguous DHW for this (n,co) plane
        uint32_t baseY = (ni * cout + co) * dhw;
        for (uint32_t off = 0; off < dhw; off += tile) {
            const uint32_t cnt = (dhw - off) > tile ? tile : (dhw - off);

            // 1) conv + scale into xLocal (pad remaining lanes with 0 to keep vector ops safe)
            for (uint32_t i = 0; i < cnt; ++i) {
                const uint32_t idx = off + i;

                const uint32_t HW = hout * wout;
                const uint32_t od = idx / HW;
                const uint32_t rem = idx - od * HW;
                const uint32_t oh = rem / wout;
                const uint32_t ow = rem - oh * wout;

                float v = ConvAt(ni, co, od, oh, ow, cb);
                v *= sf;
                xLocal.SetValue(i, v);
            }
            for (uint32_t i = cnt; i < tile; ++i) {
                xLocal.SetValue(i, 0.0f);
            }

            // 2) tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
            AscendC::Exp(ePos, xLocal, tile);           // ePos = exp(x)
            AscendC::Muls(eNeg, xLocal, -1.0f, tile);   // eNeg = -x
            AscendC::Exp(eNeg, eNeg, tile);             // eNeg = exp(-x)
            AscendC::Sub(tmp, ePos, eNeg, tile);        // tmp = ePos - eNeg
            AscendC::Add(ePos, ePos, eNeg, tile);       // ePos = ePos + eNeg (den)
            AscendC::Div(xLocal, tmp, ePos, tile);      // xLocal = tanh(x)

            // 3) * bias (scalar) and sigmoid: 1/(1+exp(-u))
            AscendC::Muls(xLocal, xLocal, bb, tile);    // u = tanh(x)*bias
            AscendC::Muls(tmp, xLocal, -1.0f, tile);    // tmp = -u
            AscendC::Exp(tmp, tmp, tile);               // tmp = exp(-u)
            AscendC::Add(tmp, tmp, one, tile);          // tmp = 1 + exp(-u)
            AscendC::Reciprocal(xLocal, tmp, tile);     // xLocal = sigmoid(u)

            // 4) store only valid lanes
            const uint64_t yBase = static_cast<uint64_t>(baseY + off);
            for (uint32_t i = 0; i < cnt; ++i) {
                yGm.SetValue(yBase + static_cast<uint64_t>(i), xLocal.GetValue(i));
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ePosBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> eNegBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> convBiasGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n{}, cin{}, din{}, hin{}, win{};
    uint32_t cout{}, kd{}, kh{}, kw{};
    uint32_t dout{}, hout{}, wout{};
    uint32_t stride{}, pad{}, dilation{}, groups{};
    uint32_t blocks{}, dhw{}, tile{};
};

extern "C" __global__ __aicore__ void conv3d_scaling_tanh_multiply_sigmoid_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR scaling_factor, GM_ADDR bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv3dScalingTanhMultiplySigmoidCustom op;
    op.Init(x, weight, conv_bias, scaling_factor, bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dout, t.hout, t.wout,
            t.stride, t.pad, t.dilation, t.groups,
            t.blocks, t.dhw, t.tile);
    op.Process();
}
