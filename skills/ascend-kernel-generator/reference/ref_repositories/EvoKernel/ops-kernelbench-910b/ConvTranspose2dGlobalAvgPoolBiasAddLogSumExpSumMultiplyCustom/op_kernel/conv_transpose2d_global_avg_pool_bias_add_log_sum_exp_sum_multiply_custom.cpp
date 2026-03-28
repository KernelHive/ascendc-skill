
#include "kernel_operator.h"

// Fused operator specialized for benchmark/model, using precomputed sumw[Cin,Cout]:
//   sumw[ci,co] = sum_{kh,kw} weight[ci,co,kh,kw] for 3x3 convT.
//
// Then:
//   SumOut[co] = sum_ci (SumX[ci] * sumw[ci,co]) + (Hout*Wout)*conv_bias[co]
//   mean = SumOut / (Hout*Wout)
//   vals[co] = mean + extra_bias[co]
//   output[n] = logsumexp(vals, dim=co) * 10

class KernelConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom {
public:
    __aicore__ inline KernelConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sumw, GM_ADDR conv_b, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t hout, uint32_t wout,
                               float mul, float inv_hw_out)
    {
        this->n = n;
        this->cin = cin;
        this->hin = hin;
        this->win = win;
        this->cout = cout;
        this->hout = hout;
        this->wout = wout;
        this->mul = mul;
        this->inv_hw_out = inv_hw_out;

        const uint64_t xSize   = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t swSize  = static_cast<uint64_t>(cin) * cout;
        const uint64_t cbSize  = static_cast<uint64_t>(cout);
        const uint64_t bSize   = static_cast<uint64_t>(cout); // (Cout,1,1) contiguous => Cout elements
        const uint64_t ySize   = static_cast<uint64_t>(n);    // [N,1] => N elements

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        sumWGm.SetGlobalBuffer((__gm__ float*)sumw, swSize);
        convBGm.SetGlobalBuffer((__gm__ float*)conv_b, cbSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // Fixed small UB footprint (avoid the previous over-allocation/overlap failure pattern).
        pipe.InitBuffer(qVals,   1, cout * sizeof(float));
        pipe.InitBuffer(qTmp,    1, cout * sizeof(float));
        pipe.InitBuffer(qWork,   1, cout * sizeof(float));
        pipe.InitBuffer(qCBias,  1, cout * sizeof(float));
        pipe.InitBuffer(qEBias,  1, cout * sizeof(float));
        pipe.InitBuffer(qSumX,   1, cin  * sizeof(float));
        pipe.InitBuffer(qScalar, 1, 32);
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> vals = qVals.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp  = qTmp.AllocTensor<float>();
        AscendC::LocalTensor<float> work = qWork.AllocTensor<float>();
        AscendC::LocalTensor<float> cb   = qCBias.AllocTensor<float>();
        AscendC::LocalTensor<float> eb   = qEBias.AllocTensor<float>();
        AscendC::LocalTensor<float> sumX = qSumX.AllocTensor<float>();
        AscendC::LocalTensor<float> sc   = qScalar.AllocTensor<float>();

        // Prefetch biases once (reused across all n).
        AscendC::DataCopy(cb, convBGm, cout);
        AscendC::DataCopy(eb, bGm, cout);

        const uint64_t hwIn = static_cast<uint64_t>(hin) * static_cast<uint64_t>(win);

        for (uint32_t ni = 0; ni < n; ++ni) {
            // sumX[ci] = sum over H*W (scalar reduction; stable and avoids UB MTE hazards).
            for (uint32_t ci = 0; ci < cin; ++ci) {
                const uint64_t baseX = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(cin) + static_cast<uint64_t>(ci)) * hwIn;
                float acc = 0.0f;
                for (uint64_t idx = 0; idx < hwIn; ++idx) {
                    acc += xGm.GetValue(baseX + idx);
                }
                sumX.SetValue(ci, acc);
            }

            // vals = conv_bias + extra_bias (vector op)
            AscendC::Add(vals, cb, eb, static_cast<int32_t>(cout));

            // vals += inv_hw_out * sum_ci(sumX[ci] * sumW[ci,co])
            for (uint32_t ci = 0; ci < cin; ++ci) {
                const float sx = sumX.GetValue(ci) * inv_hw_out;
                const uint64_t baseSW = static_cast<uint64_t>(ci) * static_cast<uint64_t>(cout);
                for (uint32_t co = 0; co < cout; ++co) {
                    float v = vals.GetValue(co);
                    v += sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co));
                    vals.SetValue(co, v);
                }
            }

            // Add conv_bias contribution averaged over spatial: + conv_bias (because (HWout*conv_bias)/HWout)
            // Already included in vals via cb.

            // logsumexp over cout => scalar
            float maxv = -3.402823466e+38f;
            for (uint32_t co = 0; co < cout; ++co) {
                float v = vals.GetValue(co);
                maxv = (v > maxv) ? v : maxv;
            }

            AscendC::Adds(tmp, vals, -maxv, static_cast<int32_t>(cout));
            AscendC::Exp(work, tmp, static_cast<int32_t>(cout));

            float sumExp = 0.0f;
            for (uint32_t co = 0; co < cout; ++co) {
                sumExp += work.GetValue(co);
            }

            const float eps = 1e-20f;
            sc.SetValue(0, sumExp + eps);
            AscendC::Log(sc, sc, 1);
            const float lse = maxv + sc.GetValue(0);

            yGm.SetValue(static_cast<uint64_t>(ni), lse * mul);
        }

        qScalar.FreeTensor(sc);
        qSumX.FreeTensor(sumX);
        qEBias.FreeTensor(eb);
        qCBias.FreeTensor(cb);
        qWork.FreeTensor(work);
        qTmp.FreeTensor(tmp);
        qVals.FreeTensor(vals);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qVals;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qWork;
    AscendC::TQue<AscendC::TPosition::VECIN,   1> qCBias;
    AscendC::TQue<AscendC::TPosition::VECIN,   1> qEBias;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qSumX;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qScalar;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> sumWGm;
    AscendC::GlobalTensor<float> convBGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, hin, win;
    uint32_t cout;
    uint32_t hout, wout;
    float mul;
    float inv_hw_out;
};

extern "C" __global__ __aicore__ void conv_transpose2d_global_avg_pool_bias_add_log_sum_exp_sum_multiply_custom(
    GM_ADDR x, GM_ADDR sumw, GM_ADDR conv_bias, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dGlobalAvgPoolBiasAddLogSumExpSumMultiplyCustom op;
    op.Init(x, sumw, conv_bias, bias, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.hout, t.wout,
            t.mul, t.inv_hw_out);
    op.Process();
}
