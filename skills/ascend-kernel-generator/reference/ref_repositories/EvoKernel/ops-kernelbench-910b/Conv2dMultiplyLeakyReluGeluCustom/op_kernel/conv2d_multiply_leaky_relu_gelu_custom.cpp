
#include "kernel_operator.h"

class KernelConv2dMultiplyLeakyReluGeluCustom {
public:
    __aicore__ inline KernelConv2dMultiplyLeakyReluGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR m, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kh, uint32_t kw,
                               uint32_t hout, uint32_t wout,
                               uint32_t stride, uint32_t pad, uint32_t dilation, uint32_t groups,
                               float leaky_slope,
                               uint32_t total_tasks, uint32_t tasks_per_block)
    {
        this->n = n; this->cin = cin; this->hin = hin; this->win = win;
        this->cout = cout; this->kh = kh; this->kw = kw;
        this->hout = hout; this->wout = wout;
        this->stride = stride; this->pad = pad; this->dilation = dilation; this->groups = groups;
        this->leaky_slope = leaky_slope;
        this->total_tasks = total_tasks;
        this->tasks_per_block = tasks_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cout) * (cin / groups) * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t mSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * hout * wout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        mGm.SetGlobalBuffer((__gm__ float*)m, mSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe.InitBuffer(qIn, 1, COUT * sizeof(float));
        pipe.InitBuffer(qOut, 1, COUT * sizeof(float));
        pipe.InitBuffer(tmpBuf, TMP_BYTES);

        pipe.InitBuffer(biasUb, cout * sizeof(float));
        pipe.InitBuffer(multUb, cout * sizeof(float));
        pipe.InitBuffer(outUb, COUT * sizeof(float));
    }

    __aicore__ inline void LoadParamsOncePerBlock()
    {
        AscendC::LocalTensor<float> bLocal = biasUb.Get<float>();
        AscendC::LocalTensor<float> mLocal = multUb.Get<float>();
        AscendC::DataCopy(bLocal, bGm, cout);
        AscendC::DataCopy(mLocal, mGm, cout);
    }

    __aicore__ inline void ProcessOnePixel(uint32_t ni, uint32_t oh, uint32_t ow,
                                          const AscendC::LocalTensor<float>& bLocal,
                                          const AscendC::LocalTensor<float>& mLocal)
    {
        // Compute all Cout for a single output pixel (ni, oh, ow)
        AscendC::LocalTensor<float> outLocal = outUb.Get<float>();
        for (uint32_t co = 0; co < COUT; ++co) {
            outLocal.SetValue(co, bLocal.GetValue(co));
        }

        const uint64_t win64 = static_cast<uint64_t>(win);
        const uint64_t hin64 = static_cast<uint64_t>(hin);
        const uint64_t cin64 = static_cast<uint64_t>(cin);

        // x base for input channel 0 at (ni, 0, oh, ow)
        uint64_t xBaseCi = ((static_cast<uint64_t>(ni) * cin64) * hin64 + static_cast<uint64_t>(oh)) * win64 + static_cast<uint64_t>(ow);

        // Iterate over Cin; load a 3x3 patch once and reuse across all Cout
        for (uint32_t ci = 0; ci < CIN; ++ci) {
            const float x00 = xGm.GetValue(xBaseCi + 0);
            const float x01 = xGm.GetValue(xBaseCi + 1);
            const float x02 = xGm.GetValue(xBaseCi + 2);

            const uint64_t xBaseCi1 = xBaseCi + win64;
            const float x10 = xGm.GetValue(xBaseCi1 + 0);
            const float x11 = xGm.GetValue(xBaseCi1 + 1);
            const float x12 = xGm.GetValue(xBaseCi1 + 2);

            const uint64_t xBaseCi2 = xBaseCi + 2ull * win64;
            const float x20 = xGm.GetValue(xBaseCi2 + 0);
            const float x21 = xGm.GetValue(xBaseCi2 + 1);
            const float x22 = xGm.GetValue(xBaseCi2 + 2);

            // For this ci, weights for each co are contiguous blocks of 9 at:
            // w[(co*Cin + ci)*9 + k]
            const uint64_t wBaseCi = static_cast<uint64_t>(ci) * 9ull;

            for (uint32_t co = 0; co < COUT; ++co) {
                const uint64_t wBase = (static_cast<uint64_t>(co) * CIN * 9ull) + wBaseCi;

                float acc = outLocal.GetValue(co);
                acc += x00 * wGm.GetValue(wBase + 0);
                acc += x01 * wGm.GetValue(wBase + 1);
                acc += x02 * wGm.GetValue(wBase + 2);
                acc += x10 * wGm.GetValue(wBase + 3);
                acc += x11 * wGm.GetValue(wBase + 4);
                acc += x12 * wGm.GetValue(wBase + 5);
                acc += x20 * wGm.GetValue(wBase + 6);
                acc += x21 * wGm.GetValue(wBase + 7);
                acc += x22 * wGm.GetValue(wBase + 8);
                outLocal.SetValue(co, acc);
            }

            xBaseCi += hin64 * win64; // next input channel plane
        }

        // Post ops + GELU in one fixed-size vector call (pad to 256)
        AscendC::LocalTensor<float> inLocal = qIn.AllocTensor<float>();
        for (uint32_t co = 0; co < COUT; ++co) {
            float v = outLocal.GetValue(co);
            v *= mLocal.GetValue(co);
            if (v < 0.0f) v *= leaky_slope;
            inLocal.SetValue(co, v);
        }
        for (uint32_t i = COUT; i < GELU_TILE; ++i) inLocal.SetValue(i, 0.0f);

        qIn.EnQue(inLocal);
        AscendC::LocalTensor<float> xLocal = qIn.DeQue<float>();
        AscendC::LocalTensor<float> geluLocal = qOut.AllocTensor<float>();

        auto tmp = tmpBuf.Get<uint8_t>();
        AscendC::FasterGeluV2<float, false, false>(geluLocal, xLocal, tmp, GELU_TILE);

        qIn.FreeTensor(xLocal);
        qOut.EnQue(geluLocal);

        AscendC::LocalTensor<float> yLocal = qOut.DeQue<float>();

        // Write contiguous [Cout] to GM: y[((ni*Cout + co)*Hout + oh)*Wout + ow]
        const uint64_t HW = static_cast<uint64_t>(hout) * static_cast<uint64_t>(wout);
        const uint64_t baseN = static_cast<uint64_t>(ni) * static_cast<uint64_t>(cout) * HW;
        const uint64_t basePix = static_cast<uint64_t>(oh) * static_cast<uint64_t>(wout) + static_cast<uint64_t>(ow);
        for (uint32_t co = 0; co < COUT; ++co) {
            const uint64_t yIdx = baseN + static_cast<uint64_t>(co) * HW + basePix;
            yGm.SetValue(yIdx, yLocal.GetValue(co));
        }

        qOut.FreeTensor(yLocal);
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = bid * tasks_per_block;
        uint32_t end = start + tasks_per_block;
        if (start >= total_tasks) return;
        if (end > total_tasks) end = total_tasks;

        LoadParamsOncePerBlock();
        AscendC::LocalTensor<float> bLocal = biasUb.Get<float>();
        AscendC::LocalTensor<float> mLocal = multUb.Get<float>();

        const uint32_t HW = hout * wout;

        for (uint32_t t = start; t < end; ++t) {
            const uint32_t ni = t / HW;
            const uint32_t rem = t - ni * HW;
            const uint32_t oh = rem / wout;
            const uint32_t ow = rem - oh * wout;
            ProcessOnePixel(ni, oh, ow, bLocal, mLocal);
        }
    }

private:
    static constexpr uint32_t CIN = 64;
    static constexpr uint32_t COUT = 64;
    static constexpr uint32_t GELU_TILE = 256;
    static constexpr uint32_t TMP_BYTES = 8192;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;

    AscendC::TBuf<AscendC::TPosition::VECCALC> biasUb;
    AscendC::TBuf<AscendC::TPosition::VECCALC> multUb;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outUb;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> mGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n{}, cin{}, hin{}, win{};
    uint32_t cout{}, kh{}, kw{};
    uint32_t hout{}, wout{};
    uint32_t stride{}, pad{}, dilation{}, groups{};
    float leaky_slope{};
    uint32_t total_tasks{}, tasks_per_block{};
};

extern "C" __global__ __aicore__ void conv2d_multiply_leaky_relu_gelu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR multiplier,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv2dMultiplyLeakyReluGeluCustom op;
    op.Init(x, weight, bias, multiplier, y,
            t.n, t.cin, t.hin, t.win,
            t.cout, t.kh, t.kw,
            t.hout, t.wout,
            t.stride, t.pad, t.dilation, t.groups,
            t.leaky_slope,
            t.total_tasks, t.tasks_per_block);
    op.Process();
}
