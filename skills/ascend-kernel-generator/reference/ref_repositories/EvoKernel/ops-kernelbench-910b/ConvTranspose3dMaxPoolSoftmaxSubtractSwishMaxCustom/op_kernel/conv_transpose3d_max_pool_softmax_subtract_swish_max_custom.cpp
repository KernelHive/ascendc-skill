
#include "kernel_operator.h"

// Specialized fused operator for benchmark configuration (float32 only):
// N=128, Cin=3, Cout=16, Din=16, Hin=32, Win=32
// ConvTranspose3d: k=3, stride=2, pad=1, output_pad=1 => [N,16,32,64,64]
// MaxPool3d: k=2, s=2 => [N,16,16,32,32]
// Softmax(dim=1) -> subtract(channel) -> swish -> reduceMax(channel)
// Output y: [N,16,32,32]

class KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom {
public:
    __aicore__ inline KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR cb, GM_ADDR sub, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t dp, uint32_t hp, uint32_t wp,
                               uint32_t total_y, uint32_t elems_per_block)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->dp = dp; this->hp = hp; this->wp = wp;
        this->totalY = total_y;
        this->elemsPerBlock = elems_per_block;

        const uint64_t xSize  = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize  = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t sSize  = static_cast<uint64_t>(cout);
        const uint64_t ySize  = static_cast<uint64_t>(n) * dp * hp * wp;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        cbGm.SetGlobalBuffer((__gm__ float*)cb, cbSize);
        subGm.SetGlobalBuffer((__gm__ float*)sub, sSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers reused for all elements handled by this block:
        // param: 32 floats (bias[16], sub[16])
        // v/tmp/ex: 16 floats each
        pipe.InitBuffer(qParam, 1, 32u * sizeof(float));
        pipe.InitBuffer(qV,     1, 16u * sizeof(float));
        pipe.InitBuffer(qTmp,   1, 16u * sizeof(float));
        pipe.InitBuffer(qEx,    1, 16u * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t STR = 2;
        constexpr int32_t PAD = 1;

        constexpr int32_t POOL_K = 2;
        constexpr int32_t POOL_S = 2;

        const int32_t Din = static_cast<int32_t>(din);
        const int32_t Hin = static_cast<int32_t>(hin);
        const int32_t Win = static_cast<int32_t>(win);

        const int32_t Kd = static_cast<int32_t>(kd);
        const int32_t Kh = static_cast<int32_t>(kh);
        const int32_t Kw = static_cast<int32_t>(kw);

        const float neg_inf = -3.402823466e+38f;
        const float eps = 1e-20f;

        const uint32_t bid = AscendC::GetBlockIdx();
        uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (end > totalY) end = totalY;
        if (start >= end) return;

        // Cache conv_bias and subtract into UB once per block
        AscendC::LocalTensor<float> param = qParam.AllocTensor<float>();
        #pragma unroll
        for (uint32_t co = 0; co < 16; ++co) {
            param.SetValue(co, cbGm.GetValue(static_cast<uint64_t>(co)));
            param.SetValue(co + 16u, subGm.GetValue(static_cast<uint64_t>(co)));
        }

        AscendC::LocalTensor<float> v   = qV.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> ex  = qEx.AllocTensor<float>();

        for (uint32_t linear = start; linear < end; ++linear) {
            // Unflatten linear -> (n,pd,ph,pw) for y: [n,dp,hp,wp]
            uint32_t t = linear;
            const uint32_t pw = t % wp; t /= wp;
            const uint32_t ph = t % hp; t /= hp;
            const uint32_t pd = t % dp; t /= dp;
            const uint32_t ni = t;

            // Initialize pooled activations (per Cout)
            #pragma unroll
            for (uint32_t co = 0; co < 16; ++co) {
                v.SetValue(co, neg_inf);
            }

            const int32_t od0 = static_cast<int32_t>(pd) * POOL_S;
            const int32_t oh0 = static_cast<int32_t>(ph) * POOL_S;
            const int32_t ow0 = static_cast<int32_t>(pw) * POOL_S;

            // MaxPool window 2x2x2 over convT output
            #pragma unroll
            for (int32_t md = 0; md < POOL_K; ++md) {
                const int32_t od = od0 + md;
                #pragma unroll
                for (int32_t mh = 0; mh < POOL_K; ++mh) {
                    const int32_t oh = oh0 + mh;
                    #pragma unroll
                    for (int32_t mw = 0; mw < POOL_K; ++mw) {
                        const int32_t ow = ow0 + mw;

                        // For each output channel compute ConvTranspose at (ni,co,od,oh,ow)
                        #pragma unroll
                        for (uint32_t co = 0; co < 16; ++co) {
                            float acc = param.GetValue(co); // bias

                            #pragma unroll
                            for (uint32_t ci = 0; ci < 3; ++ci) {
                                #pragma unroll
                                for (int32_t kD = 0; kD < 3; ++kD) {
                                    const int32_t numD = od + PAD - kD;
                                    if (numD < 0 || (numD & 1) != 0) continue;
                                    const int32_t id = numD >> 1;
                                    if (id < 0 || id >= Din) continue;

                                    #pragma unroll
                                    for (int32_t kH = 0; kH < 3; ++kH) {
                                        const int32_t numH = oh + PAD - kH;
                                        if (numH < 0 || (numH & 1) != 0) continue;
                                        const int32_t ih = numH >> 1;
                                        if (ih < 0 || ih >= Hin) continue;

                                        const uint64_t xBase =
                                            (((static_cast<uint64_t>(ni) * 3u + static_cast<uint64_t>(ci)) * static_cast<uint64_t>(din) + static_cast<uint64_t>(id))
                                              * static_cast<uint64_t>(hin) + static_cast<uint64_t>(ih)) * static_cast<uint64_t>(win);

                                        const uint64_t wBase =
                                            ((((static_cast<uint64_t>(ci) * 16u + static_cast<uint64_t>(co)) * 3u + static_cast<uint64_t>(kD))
                                               * 3u + static_cast<uint64_t>(kH)) * 3u);

                                        #pragma unroll
                                        for (int32_t kW = 0; kW < 3; ++kW) {
                                            const int32_t numW = ow + PAD - kW;
                                            if (numW < 0 || (numW & 1) != 0) continue;
                                            const int32_t iw = numW >> 1;
                                            if (iw < 0 || iw >= Win) continue;

                                            const float xv = xGm.GetValue(xBase + static_cast<uint64_t>(iw));
                                            const float wv = wGm.GetValue(wBase + static_cast<uint64_t>(kW));
                                            acc += xv * wv;
                                        }
                                    }
                                }
                            }

                            float prev = v.GetValue(co);
                            v.SetValue(co, (acc > prev) ? acc : prev);
                        }
                    }
                }
            }

            // Softmax over 16 channels (vector Exp), then subtract, swish, reduce-max.
            float rowMax = v.GetValue(0);
            #pragma unroll
            for (uint32_t co = 1; co < 16; ++co) {
                float vv = v.GetValue(co);
                rowMax = (vv > rowMax) ? vv : rowMax;
            }

            AscendC::DataCopy(tmp, v, 16);
            AscendC::Adds(tmp, tmp, -rowMax, 16);
            AscendC::Exp(ex, tmp, 16);

            float sumExp = 0.0f;
            #pragma unroll
            for (uint32_t co = 0; co < 16; ++co) sumExp += ex.GetValue(co);
            const float invSum = 1.0f / (sumExp + eps);

            // tmp := softmax - subtract
            #pragma unroll
            for (uint32_t co = 0; co < 16; ++co) {
                float sm = ex.GetValue(co) * invSum;
                float u = sm - param.GetValue(co + 16u);
                tmp.SetValue(co, u);
            }

            // sigmoid(tmp) using vector ops: ex = exp(-tmp), tmp = 1/(1+ex), ex = tmp*u (swish)
            AscendC::Muls(v, tmp, -1.0f, 16); // reuse v as -u
            AscendC::Exp(ex, v, 16);          // ex = exp(-u)
            #pragma unroll
            for (uint32_t co = 0; co < 16; ++co) {
                float e = ex.GetValue(co);
                float sig = 1.0f / (1.0f + e);
                float u = tmp.GetValue(co);
                ex.SetValue(co, u * sig); // swish
            }

            float maxAfter = ex.GetValue(0);
            #pragma unroll
            for (uint32_t co = 1; co < 16; ++co) {
                float sw = ex.GetValue(co);
                maxAfter = (sw > maxAfter) ? sw : maxAfter;
            }

            yGm.SetValue(static_cast<uint64_t>(linear), maxAfter);
        }

        qEx.FreeTensor(ex);
        qTmp.FreeTensor(tmp);
        qV.FreeTensor(v);
        qParam.FreeTensor(param);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qParam;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qV;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qEx;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> cbGm;
    AscendC::GlobalTensor<float> subGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    uint32_t dp, hp, wp;
    uint32_t totalY;
    uint32_t elemsPerBlock;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_pool_softmax_subtract_swish_max_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR subtract,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom op;
    op.Init(x, weight, conv_bias, subtract, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dp, t.hp, t.wp,
            t.total_y, t.elems_per_block);
    op.Process();
}
