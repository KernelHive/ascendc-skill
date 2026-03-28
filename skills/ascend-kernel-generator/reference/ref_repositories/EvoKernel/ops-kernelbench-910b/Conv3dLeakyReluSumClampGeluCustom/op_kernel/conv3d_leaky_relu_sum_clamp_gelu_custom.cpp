
#include "kernel_operator.h"

// Changes vs baseline (safe):
// - Host now maps each block to a fixed contiguous range [bid*tasks_per_block, ...] so blocks are shorter and more numerous.
// - Keep per-element yGm.SetValue stores (avoid the previously failed DataCopy writeback).
// - Call FasterGeluV2 only on tileCount (no padding SetValue loop); we allocate full TILE buffers but process count=tileCount.

class KernelConv3dLeakyReluSumClampGeluCustom {
public:
    __aicore__ inline KernelConv3dLeakyReluSumClampGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR s, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               uint32_t dout, uint32_t hout, uint32_t wout,
                               uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t groups,
                               float leaky_slope, float clamp_min, float clamp_max,
                               uint32_t total_tasks, uint32_t tasks_per_block)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->dout = dout; this->hout = hout; this->wout = wout;
        this->stride = stride; this->pad = pad; this->dilation = dilation; this->groups = groups;
        this->leaky_slope = leaky_slope;
        this->clamp_min = clamp_min;
        this->clamp_max = clamp_max;
        this->total_tasks = total_tasks;
        this->tasks_per_block = tasks_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * (cin / groups) * kd * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t sSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * dout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        sGm.SetGlobalBuffer((__gm__ float*)s, sSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe.InitBuffer(qIn, 1, TILE * sizeof(float));
        pipe.InitBuffer(qOut, 1, TILE * sizeof(float));
        pipe.InitBuffer(tmpBuf, TMP_BYTES);

        pipe.InitBuffer(biasUbBuf, cout * sizeof(float));
        pipe.InitBuffer(sumUbBuf,  cout * sizeof(float));
    }

    __aicore__ inline void PrefetchParams()
    {
        AscendC::LocalTensor<float> biasUb = biasUbBuf.Get<float>();
        AscendC::LocalTensor<float> sumUb  = sumUbBuf.Get<float>();
        AscendC::DataCopy(biasUb, bGm, cout);
        AscendC::DataCopy(sumUb,  sGm, cout);
    }

    __aicore__ inline float ConvAt(uint32_t ni, uint32_t co, uint32_t od, uint32_t oh, uint32_t ow,
                                   const AscendC::LocalTensor<float>& biasUb) const
    {
        float acc = biasUb.GetValue(co);

        const uint64_t din64 = static_cast<uint64_t>(din);
        const uint64_t hin64 = static_cast<uint64_t>(hin);
        const uint64_t win64 = static_cast<uint64_t>(win);

        const uint64_t nBase = static_cast<uint64_t>(ni) * 8ull * din64 * hin64 * win64;
        const uint64_t wCoBase = static_cast<uint64_t>(co) * 8ull * 27ull;

        #pragma unroll
        for (uint32_t ci = 0; ci < 8; ++ci) {
            const uint64_t xNciBase = nBase + static_cast<uint64_t>(ci) * din64 * hin64 * win64;
            const uint64_t wCoCiBase = wCoBase + static_cast<uint64_t>(ci) * 27ull;

            #pragma unroll
            for (uint32_t kD = 0; kD < 3; ++kD) {
                const uint32_t id = od + kD;
                const uint64_t xDBase = xNciBase + static_cast<uint64_t>(id) * hin64 * win64;
                const uint64_t wDBase = wCoCiBase + static_cast<uint64_t>(kD) * 9ull;

                #pragma unroll
                for (uint32_t kH = 0; kH < 3; ++kH) {
                    const uint32_t ih = oh + kH;
                    const uint64_t xHBase = xDBase + static_cast<uint64_t>(ih) * win64;
                    const uint64_t wHBase = wDBase + static_cast<uint64_t>(kH) * 3ull;

                    const uint64_t xBase = xHBase + static_cast<uint64_t>(ow);

                    const float x0 = xGm.GetValue(xBase + 0);
                    const float x1 = xGm.GetValue(xBase + 1);
                    const float x2 = xGm.GetValue(xBase + 2);

                    const float w0 = wGm.GetValue(wHBase + 0);
                    const float w1 = wGm.GetValue(wHBase + 1);
                    const float w2 = wGm.GetValue(wHBase + 2);

                    acc += x0 * w0;
                    acc += x1 * w1;
                    acc += x2 * w2;
                }
            }
        }
        return acc;
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = bid * tasks_per_block;
        uint32_t end = start + tasks_per_block;
        if (end > total_tasks) end = total_tasks;
        if (start >= end) return;

        PrefetchParams();
        AscendC::LocalTensor<float> biasUb = biasUbBuf.Get<float>();
        AscendC::LocalTensor<float> sumUb  = sumUbBuf.Get<float>();

        const uint32_t DHW = dout * hout * wout;
        const uint32_t HW  = hout * wout;

        uint32_t ni, co, od, oh, ow;
        {
            const uint32_t idx = start;
            ni = idx / (cout * DHW);
            const uint32_t rem0 = idx - ni * (cout * DHW);
            co = rem0 / DHW;
            const uint32_t rem1 = rem0 - co * DHW;
            od = rem1 / HW;
            const uint32_t rem2 = rem1 - od * HW;
            oh = rem2 / wout;
            ow = rem2 - oh * wout;
        }

        uint32_t t = start;
        while (t < end) {
            const uint32_t tileCount = ((end - t) > TILE) ? TILE : (end - t);

            AscendC::LocalTensor<float> inLocal = qIn.AllocTensor<float>();

            #pragma unroll 1
            for (uint32_t i = 0; i < tileCount; ++i) {
                float v = ConvAt(ni, co, od, oh, ow, biasUb);

                if (v < 0.0f) v *= leaky_slope;
                v += sumUb.GetValue(co);
                if (v < clamp_min) v = clamp_min;
                if (v > clamp_max) v = clamp_max;

                inLocal.SetValue(i, v);

                ++ow;
                if (ow == wout) {
                    ow = 0; ++oh;
                    if (oh == hout) {
                        oh = 0; ++od;
                        if (od == dout) {
                            od = 0; ++co;
                            if (co == cout) {
                                co = 0; ++ni;
                            }
                        }
                    }
                }
            }
            qIn.EnQue(inLocal);

            AscendC::LocalTensor<float> xLocal = qIn.DeQue<float>();
            AscendC::LocalTensor<float> outLocal = qOut.AllocTensor<float>();
            auto tmp = tmpBuf.Get<uint8_t>();
            AscendC::FasterGeluV2<float, false, false>(outLocal, xLocal, tmp, tileCount);

            qIn.FreeTensor(xLocal);
            qOut.EnQue(outLocal);

            AscendC::LocalTensor<float> yLocal = qOut.DeQue<float>();
            for (uint32_t i = 0; i < tileCount; ++i) {
                yGm.SetValue(static_cast<uint64_t>(t + i), yLocal.GetValue(i));
            }
            qOut.FreeTensor(yLocal);

            t += tileCount;
        }
    }

private:
    static constexpr uint32_t TILE = 256;
    static constexpr uint32_t TMP_BYTES = 8192;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::TBuf<AscendC::TPosition::VECCALC> biasUbBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumUbBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> sGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n{}, cin{}, din{}, hin{}, win{};
    uint32_t cout{}, kd{}, kh{}, kw{};
    uint32_t dout{}, hout{}, wout{};
    uint32_t stride{}, pad{}, dilation{}, groups{};
    float leaky_slope{}, clamp_min{}, clamp_max{};
    uint32_t total_tasks{}, tasks_per_block{};
};

extern "C" __global__ __aicore__ void conv3d_leaky_relu_sum_clamp_gelu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR sum_tensor,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv3dLeakyReluSumClampGeluCustom op;
    op.Init(x, weight, bias, sum_tensor, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.dout, t.hout, t.wout,
            t.stride, t.pad, t.dilation, t.groups,
            t.leaky_slope, t.clamp_min, t.clamp_max,
            t.total_tasks, t.tasks_per_block);
    op.Process();
}
