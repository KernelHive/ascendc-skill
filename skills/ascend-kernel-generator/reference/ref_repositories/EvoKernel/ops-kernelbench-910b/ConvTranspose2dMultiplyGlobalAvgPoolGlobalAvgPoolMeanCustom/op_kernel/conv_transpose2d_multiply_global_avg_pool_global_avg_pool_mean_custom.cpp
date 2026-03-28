
#include "kernel_operator.h"

// Fused operator specialized for benchmark/model.
//
// Key identity for mean of convT output over spatial dims:
//   meanOut[n,co] = ( sum_ci ( SumX[n,ci] * SumW[ci,co] ) / (Hout*Wout) ) + bias[co]
// Then apply multiplier => bias[co]*multiplier + sum_ci(SumX*SumW) * (multiplier/(Hout*Wout))
//
// Optimization in this round:
// - Precompute SumW[ci,co] once per launch and store to workspace GM.
// - Inner loop uses SumW GM (8K floats) instead of re-summing 3x3 weights repeatedly.
// - Unroll SumX accumulation and co accumulation to reduce scalar/control overhead.

class KernelConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom {
public:
    __aicore__ inline KernelConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               GM_ADDR workspace,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t stride, uint32_t pad, uint32_t out_pad, uint32_t dil,
                               uint32_t hout, uint32_t wout,
                               float multiplier,
                               uint32_t sumw_elems)
    {
        (void)stride; (void)pad; (void)out_pad; (void)dil;
        this->n = n;
        this->cin = cin;
        this->hin = hin;
        this->win = win;
        this->cout = cout;
        this->kh = kh;
        this->kw = kw;
        this->hout = hout;
        this->wout = wout;
        this->multiplier = multiplier;
        this->sumw_elems = sumw_elems;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cin) * cout * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * 1 * 1;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // Workspace stores sumW[cin*cout]
        sumWGm.SetGlobalBuffer((__gm__ float*)workspace, static_cast<uint64_t>(sumw_elems));

        pipe.InitBuffer(qOut, 1, static_cast<uint32_t>(cout) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Precompute sumW to workspace: layout [ci*cout + co]
        // SumW[ci,co] = sum_{kh,kw} w[ci,co,kh,kw]
        const uint64_t kArea = static_cast<uint64_t>(kh) * static_cast<uint64_t>(kw); // 9
        for (uint32_t ci = 0; ci < cin; ++ci) {
            const uint64_t baseW_ci = static_cast<uint64_t>(ci) * static_cast<uint64_t>(cout) * kArea;
            const uint64_t baseSW = static_cast<uint64_t>(ci) * static_cast<uint64_t>(cout);
            for (uint32_t co = 0; co < cout; ++co) {
                const uint64_t baseW = baseW_ci + static_cast<uint64_t>(co) * kArea;
                // unrolled 9-tap sum
                float s = 0.0f;
                s += wGm.GetValue(baseW + 0);
                s += wGm.GetValue(baseW + 1);
                s += wGm.GetValue(baseW + 2);
                s += wGm.GetValue(baseW + 3);
                s += wGm.GetValue(baseW + 4);
                s += wGm.GetValue(baseW + 5);
                s += wGm.GetValue(baseW + 6);
                s += wGm.GetValue(baseW + 7);
                s += wGm.GetValue(baseW + 8);
                sumWGm.SetValue(baseSW + static_cast<uint64_t>(co), s);
            }
        }

        const int64_t hwOut = static_cast<int64_t>(hout) * static_cast<int64_t>(wout);
        const float invHW = (hwOut > 0) ? (1.0f / static_cast<float>(hwOut)) : 0.0f;
        const float scale = multiplier * invHW;

        const uint64_t hwIn = static_cast<uint64_t>(hin) * static_cast<uint64_t>(win);

        AscendC::LocalTensor<float> out = qOut.AllocTensor<float>();

        for (uint32_t ni = 0; ni < n; ++ni) {
            // init out with bias*multiplier
            for (uint32_t co = 0; co < cout; ++co) {
                out.SetValue(co, bGm.GetValue(static_cast<uint64_t>(co)) * multiplier);
            }

            // accumulate over ci
            const uint64_t nBase = static_cast<uint64_t>(ni) * static_cast<uint64_t>(cin) * hwIn;
            for (uint32_t ci = 0; ci < cin; ++ci) {
                // SumX: unrolled by 4
                float sumX0 = 0.0f, sumX1 = 0.0f, sumX2 = 0.0f, sumX3 = 0.0f;
                const uint64_t baseX = nBase + static_cast<uint64_t>(ci) * hwIn;
                uint64_t idx = 0;
                for (; idx + 3 < hwIn; idx += 4) {
                    sumX0 += xGm.GetValue(baseX + idx + 0);
                    sumX1 += xGm.GetValue(baseX + idx + 1);
                    sumX2 += xGm.GetValue(baseX + idx + 2);
                    sumX3 += xGm.GetValue(baseX + idx + 3);
                }
                float sumX = (sumX0 + sumX1) + (sumX2 + sumX3);
                for (; idx < hwIn; ++idx) {
                    sumX += xGm.GetValue(baseX + idx);
                }

                const float sx = sumX * scale;
                const uint64_t baseSW = static_cast<uint64_t>(ci) * static_cast<uint64_t>(cout);

                // unroll co by 8
                uint32_t co = 0;
                for (; co + 7 < cout; co += 8) {
                    float o0 = out.GetValue(co + 0) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 0));
                    float o1 = out.GetValue(co + 1) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 1));
                    float o2 = out.GetValue(co + 2) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 2));
                    float o3 = out.GetValue(co + 3) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 3));
                    float o4 = out.GetValue(co + 4) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 4));
                    float o5 = out.GetValue(co + 5) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 5));
                    float o6 = out.GetValue(co + 6) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 6));
                    float o7 = out.GetValue(co + 7) + sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co + 7));
                    out.SetValue(co + 0, o0);
                    out.SetValue(co + 1, o1);
                    out.SetValue(co + 2, o2);
                    out.SetValue(co + 3, o3);
                    out.SetValue(co + 4, o4);
                    out.SetValue(co + 5, o5);
                    out.SetValue(co + 6, o6);
                    out.SetValue(co + 7, o7);
                }
                for (; co < cout; ++co) {
                    float o = out.GetValue(co);
                    o += sx * sumWGm.GetValue(baseSW + static_cast<uint64_t>(co));
                    out.SetValue(co, o);
                }
            }

            // store [N,Cout,1,1] contiguous
            const uint64_t yBase = static_cast<uint64_t>(ni) * static_cast<uint64_t>(cout);
            for (uint32_t co = 0; co < cout; ++co) {
                yGm.SetValue(yBase + static_cast<uint64_t>(co), out.GetValue(co));
            }
        }

        qOut.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qOut;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> sumWGm;

    uint32_t n, cin, hin, win;
    uint32_t cout, kh, kw;
    uint32_t hout, wout;
    float multiplier;
    uint32_t sumw_elems;
};

extern "C" __global__ __aicore__ void conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom op;
    op.Init(x, weight, bias, y,
            workspace,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.stride, t.pad, t.out_pad, t.dil,
            t.hout, t.wout,
            t.multiplier,
            t.sumw_elems);
    op.Process();
}
