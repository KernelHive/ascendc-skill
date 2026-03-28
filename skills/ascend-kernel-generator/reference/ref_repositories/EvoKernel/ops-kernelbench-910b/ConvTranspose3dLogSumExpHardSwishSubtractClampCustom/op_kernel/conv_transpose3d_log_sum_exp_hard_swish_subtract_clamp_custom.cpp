
#include "kernel_operator.h"

// Fused operator for benchmark configuration:
// x: [N=128, Cin=3, D=16, H=32, W=32] fp32
// w: [Cin=3, Cout=16, Kd=3, Kh=3, Kw=3] fp32 (PyTorch ConvTranspose3d layout)
// bias: [Cout=16] fp32
// sub_bias: [1,1,1,1] fp32 scalar
//
// ConvTranspose3d stride=2 pad=1 output_padding=0 dilation=1 groups=1
// IMPORTANT: match PyTorch ConvTranspose3d semantics by flipping spatial kernel indices.
//
// Output after convT: [N,16,31,63,63]
// logsumexp over C => [N,1,31,63,63]
// hard-swish, subtract, clamp[-1,1].

class KernelConvTranspose3dLogSumExpHardSwishSubtractClampCustom {
public:
    __aicore__ inline KernelConvTranspose3dLogSumExpHardSwishSubtractClampCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR conv_b, GM_ADDR sub_b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t dout, uint32_t hout, uint32_t wout,
                               float clamp_min, float clamp_max)
    {
        this->n = n;
        this->cin = cin;
        this->din = din;
        this->hin = hin;
        this->win = win;
        this->cout = cout;
        this->kd = kd;
        this->kh = kh;
        this->kw = kw;
        this->dout = dout;
        this->hout = hout;
        this->wout = wout;
        this->clamp_min = clamp_min;
        this->clamp_max = clamp_max;

        const uint64_t xSize  = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize  = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t cbSize = static_cast<uint64_t>(cout);
        const uint64_t sbSize = 1;
        const uint64_t ySize  = static_cast<uint64_t>(n) * dout * hout * wout; // [N,1,dout,hout,wout]

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        convBGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        subBGm.SetGlobalBuffer((__gm__ float*)sub_b, sbSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB layout: tmp16 | exp16 | cb16 | scalar1 => 49 floats
        pipe.InitBuffer(ubBuf, 49 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Fixed hyperparams
        constexpr int32_t STR = 2;
        constexpr int32_t PAD = 1;
        constexpr int32_t DIL = 1;

        // Specialization expected by binding
        const int32_t Cin  = static_cast<int32_t>(cin);   // 3
        const int32_t Cout = static_cast<int32_t>(cout);  // 16
        const int32_t Din  = static_cast<int32_t>(din);
        const int32_t Hin  = static_cast<int32_t>(hin);
        const int32_t Win  = static_cast<int32_t>(win);
        const int32_t Dout = static_cast<int32_t>(dout);
        const int32_t Hout = static_cast<int32_t>(hout);
        const int32_t Wout = static_cast<int32_t>(wout);

        AscendC::LocalTensor<float> ub = ubBuf.Get<float>();
        AscendC::LocalTensor<float> tmp16   = ub;        // 0..15
        AscendC::LocalTensor<float> exp16   = ub[16];    // 16..31
        AscendC::LocalTensor<float> cb16    = ub[32];    // 32..47
        AscendC::LocalTensor<float> scalar1 = ub[48];    // 48

        // Cache conv bias once
        #pragma unroll
        for (int32_t co = 0; co < 16; ++co) {
            cb16.SetValue(co, convBGm.GetValue(static_cast<uint64_t>(co)));
        }
        const float sub_bias_scalar = subBGm.GetValue(0);
        const float inv6 = 1.0f / 6.0f;

        const uint64_t in_plane = static_cast<uint64_t>(din) * hin * win;
        const uint64_t in_dhw   = static_cast<uint64_t>(hin) * win;

        for (uint32_t ni = 0; ni < n; ++ni) {
            const uint64_t x_n_base = static_cast<uint64_t>(ni) * (static_cast<uint64_t>(cin) * in_plane);
            const uint64_t y_n_base = static_cast<uint64_t>(ni) * (static_cast<uint64_t>(dout) * hout * wout);

            for (int32_t od = 0; od < Dout; ++od) {
                for (int32_t oh = 0; oh < Hout; ++oh) {
                    for (int32_t ow = 0; ow < Wout; ++ow) {

                        // Accumulate convT output across Cout into registers (avoid writing v16 to UB).
                        float acc0  = cb16.GetValue(0);
                        float acc1  = cb16.GetValue(1);
                        float acc2  = cb16.GetValue(2);
                        float acc3  = cb16.GetValue(3);
                        float acc4  = cb16.GetValue(4);
                        float acc5  = cb16.GetValue(5);
                        float acc6  = cb16.GetValue(6);
                        float acc7  = cb16.GetValue(7);
                        float acc8  = cb16.GetValue(8);
                        float acc9  = cb16.GetValue(9);
                        float acc10 = cb16.GetValue(10);
                        float acc11 = cb16.GetValue(11);
                        float acc12 = cb16.GetValue(12);
                        float acc13 = cb16.GetValue(13);
                        float acc14 = cb16.GetValue(14);
                        float acc15 = cb16.GetValue(15);

                        // Iterate over input positions contributing to this output position.
                        // For transposed conv: id = (od + pad - kD) / stride with stride divisibility check,
                        // but weight uses flipped k indices to match PyTorch conv_transpose semantics.
                        #pragma unroll
                        for (int32_t ci = 0; ci < 3; ++ci) {
                            const uint64_t x_c_base = x_n_base + static_cast<uint64_t>(ci) * in_plane;

                            #pragma unroll
                            for (int32_t kD = 0; kD < 3; ++kD) {
                                const int32_t numD = od + PAD - kD * DIL;
                                if (numD < 0 || (numD % STR) != 0) continue;
                                const int32_t id = numD / STR;
                                if (id < 0 || id >= Din) continue;

                                #pragma unroll
                                for (int32_t kH = 0; kH < 3; ++kH) {
                                    const int32_t numH = oh + PAD - kH * DIL;
                                    if (numH < 0 || (numH % STR) != 0) continue;
                                    const int32_t ih = numH / STR;
                                    if (ih < 0 || ih >= Hin) continue;

                                    #pragma unroll
                                    for (int32_t kW = 0; kW < 3; ++kW) {
                                        const int32_t numW = ow + PAD - kW * DIL;
                                        if (numW < 0 || (numW % STR) != 0) continue;
                                        const int32_t iw = numW / STR;
                                        if (iw < 0 || iw >= Win) continue;

                                        const uint64_t xIdx = x_c_base
                                            + static_cast<uint64_t>(id) * in_dhw
                                            + static_cast<uint64_t>(ih) * win
                                            + static_cast<uint64_t>(iw);
                                        const float xv = xGm.GetValue(xIdx);

                                        // Flip kernel indices for conv_transpose semantics
                                        const uint64_t fkD = static_cast<uint64_t>(2 - kD);
                                        const uint64_t fkH = static_cast<uint64_t>(2 - kH);
                                        const uint64_t fkW = static_cast<uint64_t>(2 - kW);

                                        // Base for weights for this (ci,fkD,fkH,fkW), varying co
                                        // w layout: [ci, co, kD, kH, kW]
                                        const uint64_t w_base =
                                            ((((static_cast<uint64_t>(ci) * static_cast<uint64_t>(cout)) * 3u + fkD) * 3u + fkH) * 3u + fkW);
                                        // wIdx = w_base + co*(3*3*3)
                                        constexpr uint64_t CO_STRIDE = 27u;

                                        acc0  += xv * wGm.GetValue(w_base + 0u  * CO_STRIDE);
                                        acc1  += xv * wGm.GetValue(w_base + 1u  * CO_STRIDE);
                                        acc2  += xv * wGm.GetValue(w_base + 2u  * CO_STRIDE);
                                        acc3  += xv * wGm.GetValue(w_base + 3u  * CO_STRIDE);
                                        acc4  += xv * wGm.GetValue(w_base + 4u  * CO_STRIDE);
                                        acc5  += xv * wGm.GetValue(w_base + 5u  * CO_STRIDE);
                                        acc6  += xv * wGm.GetValue(w_base + 6u  * CO_STRIDE);
                                        acc7  += xv * wGm.GetValue(w_base + 7u  * CO_STRIDE);
                                        acc8  += xv * wGm.GetValue(w_base + 8u  * CO_STRIDE);
                                        acc9  += xv * wGm.GetValue(w_base + 9u  * CO_STRIDE);
                                        acc10 += xv * wGm.GetValue(w_base + 10u * CO_STRIDE);
                                        acc11 += xv * wGm.GetValue(w_base + 11u * CO_STRIDE);
                                        acc12 += xv * wGm.GetValue(w_base + 12u * CO_STRIDE);
                                        acc13 += xv * wGm.GetValue(w_base + 13u * CO_STRIDE);
                                        acc14 += xv * wGm.GetValue(w_base + 14u * CO_STRIDE);
                                        acc15 += xv * wGm.GetValue(w_base + 15u * CO_STRIDE);
                                    }
                                }
                            }
                        }

                        // logsumexp across 16 channels:
                        float maxv = acc0;
                        maxv = acc1  > maxv ? acc1  : maxv;
                        maxv = acc2  > maxv ? acc2  : maxv;
                        maxv = acc3  > maxv ? acc3  : maxv;
                        maxv = acc4  > maxv ? acc4  : maxv;
                        maxv = acc5  > maxv ? acc5  : maxv;
                        maxv = acc6  > maxv ? acc6  : maxv;
                        maxv = acc7  > maxv ? acc7  : maxv;
                        maxv = acc8  > maxv ? acc8  : maxv;
                        maxv = acc9  > maxv ? acc9  : maxv;
                        maxv = acc10 > maxv ? acc10 : maxv;
                        maxv = acc11 > maxv ? acc11 : maxv;
                        maxv = acc12 > maxv ? acc12 : maxv;
                        maxv = acc13 > maxv ? acc13 : maxv;
                        maxv = acc14 > maxv ? acc14 : maxv;
                        maxv = acc15 > maxv ? acc15 : maxv;

                        tmp16.SetValue(0,  acc0  - maxv);
                        tmp16.SetValue(1,  acc1  - maxv);
                        tmp16.SetValue(2,  acc2  - maxv);
                        tmp16.SetValue(3,  acc3  - maxv);
                        tmp16.SetValue(4,  acc4  - maxv);
                        tmp16.SetValue(5,  acc5  - maxv);
                        tmp16.SetValue(6,  acc6  - maxv);
                        tmp16.SetValue(7,  acc7  - maxv);
                        tmp16.SetValue(8,  acc8  - maxv);
                        tmp16.SetValue(9,  acc9  - maxv);
                        tmp16.SetValue(10, acc10 - maxv);
                        tmp16.SetValue(11, acc11 - maxv);
                        tmp16.SetValue(12, acc12 - maxv);
                        tmp16.SetValue(13, acc13 - maxv);
                        tmp16.SetValue(14, acc14 - maxv);
                        tmp16.SetValue(15, acc15 - maxv);

                        // vector exp over 16
                        AscendC::Exp(exp16, tmp16, 16);

                        float sumExp = 0.0f;
                        #pragma unroll
                        for (int32_t i = 0; i < 16; ++i) {
                            sumExp += exp16.GetValue(i);
                        }

                        scalar1.SetValue(0, sumExp);
                        AscendC::Log(scalar1, scalar1, 1);
                        const float lse = maxv + scalar1.GetValue(0);

                        // hard-swish
                        const float z = lse + 3.0f;
                        scalar1.SetValue(0, -z);
                        AscendC::Exp(scalar1, scalar1, 1);
                        const float e = scalar1.GetValue(0);
                        const float sig = 1.0f / (1.0f + e);
                        float outv = (lse * sig * inv6) - sub_bias_scalar;

                        if (outv < clamp_min) outv = clamp_min;
                        if (outv > clamp_max) outv = clamp_max;

                        const uint64_t yIdx = y_n_base
                            + (static_cast<uint64_t>(od) * static_cast<uint64_t>(hout) + static_cast<uint64_t>(oh)) * static_cast<uint64_t>(wout)
                            + static_cast<uint64_t>(ow);
                        yGm.SetValue(yIdx, outv);
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> convBGm;
    AscendC::GlobalTensor<float> subBGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    uint32_t dout, hout, wout;
    float clamp_min, clamp_max;
};

extern "C" __global__ __aicore__ void conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR conv_bias, GM_ADDR sub_bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dLogSumExpHardSwishSubtractClampCustom op;
    op.Init(x, weight, conv_bias, sub_bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dout, t.hout, t.wout,
            t.clamp_min, t.clamp_max);
    op.Process();
}
