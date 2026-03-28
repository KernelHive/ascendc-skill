
#include "kernel_operator.h"

// Fused operator for benchmark configuration:
// x:          [N=16, Cin=16, Din=16, Hin=32, Win=32] float32
// weight:     [Cin=16, Cout=32, Kd=3,Kh=3,Kw=3] float32 (PyTorch ConvTranspose3d layout)
// bias:       [Cout=32] float32
// multiplier: [Cout=32,1,1,1] float32 (broadcast by channel)
// ConvTranspose3d params: stride=2, pad=1, out_pad=1, dilation=1
// convT out:  [16,32, Dout=32, Hout=64, Wout=64]
// LeakyReLU(0.2) -> *multiplier[co] -> LeakyReLU(0.2) -> MaxPool3d(k=2,s=2)
// y:          [16,32, Dp=16, Hp=32, Wp=32]

class KernelConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom {
public:
    __aicore__ inline KernelConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR m, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               float neg_slope,
                               uint32_t total_tasks, uint32_t tasks_per_block)
    {
        this->n = n; this->cin = cin; this->din = din; this->hin = hin; this->win = win;
        this->cout = cout; this->kd = kd; this->kh = kh; this->kw = kw;
        this->neg_slope = neg_slope;
        this->total_tasks = total_tasks;
        this->tasks_per_block = tasks_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t mSize = static_cast<uint64_t>(cout); // read multiplier as 1D by co
        const uint64_t ySize = static_cast<uint64_t>(n) * cout * 16ull * 32ull * 32ull;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        mGm.SetGlobalBuffer((__gm__ float*)m, mSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // x strides [N,C,D,H,W]
        xStrideW = 1ull;
        xStrideH = static_cast<uint64_t>(win);
        xStrideD = static_cast<uint64_t>(hin) * xStrideH;
        xStrideC = static_cast<uint64_t>(din) * xStrideD;
        xStrideN = static_cast<uint64_t>(cin) * xStrideC;

        // weight strides [Cin,Cout,Kd,Kh,Kw] contiguous
        wStrideKw = 1ull;
        wStrideKh = static_cast<uint64_t>(kw);
        wStrideKd = static_cast<uint64_t>(kh) * wStrideKh;
        wStrideCo = static_cast<uint64_t>(kd) * wStrideKd;
        wStrideCi = static_cast<uint64_t>(cout) * wStrideCo;
    }

    __aicore__ inline float Leaky(float v) const
    {
        return (v >= 0.0f) ? v : (v * neg_slope);
    }

    // stride=2, pad=1, k=3, out_pad=1 -> output sizes match benchmark.
    // Contrib condition: i = (o + pad - k) / stride must be integer and within bounds.
    __aicore__ inline int32_t BuildContrib1D(int32_t o, int32_t inBound, int32_t* iList, int32_t* kList) const
    {
        int32_t cnt = 0;
        const int32_t t = o + 1; // pad=1

        // k=0..2, stride=2
        int32_t num = t - 0;
        if ((num & 1) == 0) { int32_t i = num >> 1; if ((uint32_t)i < (uint32_t)inBound) { iList[cnt] = i; kList[cnt] = 0; ++cnt; } }
        num = t - 1;
        if ((num & 1) == 0) { int32_t i = num >> 1; if ((uint32_t)i < (uint32_t)inBound) { iList[cnt] = i; kList[cnt] = 1; ++cnt; } }
        num = t - 2;
        if ((num & 1) == 0) { int32_t i = num >> 1; if ((uint32_t)i < (uint32_t)inBound) { iList[cnt] = i; kList[cnt] = 2; ++cnt; } }

        return cnt;
    }

    __aicore__ inline float ConvTPoint(uint32_t ni, uint32_t co, int32_t od, int32_t oh, int32_t ow, float biasCo) const
    {
        int32_t idList[2], ihList[2], iwList[2];
        int32_t kDList[2], kHList[2], kWList[2];

        const int32_t cntD = BuildContrib1D(od, static_cast<int32_t>(din), idList, kDList);
        const int32_t cntH = BuildContrib1D(oh, static_cast<int32_t>(hin), ihList, kHList);
        const int32_t cntW = BuildContrib1D(ow, static_cast<int32_t>(win), iwList, kWList);

        float acc = biasCo;
        if (cntD == 0 || cntH == 0 || cntW == 0) {
            return acc;
        }

        const uint64_t nBaseX = static_cast<uint64_t>(ni) * xStrideN;
        const uint64_t coBaseW = static_cast<uint64_t>(co) * wStrideCo;

        for (uint32_t ci = 0; ci < cin; ++ci) {
            const uint64_t ciBaseX = nBaseX + static_cast<uint64_t>(ci) * xStrideC;
            const uint64_t ciBaseW = static_cast<uint64_t>(ci) * wStrideCi + coBaseW;

            for (int32_t a = 0; a < cntD; ++a) {
                const uint64_t idBaseX = ciBaseX + static_cast<uint64_t>(idList[a]) * xStrideD;
                const uint64_t kdBaseW = ciBaseW + static_cast<uint64_t>(kDList[a]) * wStrideKd;

                for (int32_t b = 0; b < cntH; ++b) {
                    const uint64_t ihBaseX = idBaseX + static_cast<uint64_t>(ihList[b]) * xStrideH;
                    const uint64_t khBaseW = kdBaseW + static_cast<uint64_t>(kHList[b]) * wStrideKh;

                    for (int32_t c = 0; c < cntW; ++c) {
                        const uint64_t xIdx = ihBaseX + static_cast<uint64_t>(iwList[c]) * xStrideW;
                        const uint64_t wIdx = khBaseW + static_cast<uint64_t>(kWList[c]) * wStrideKw;
                        acc += xGm.GetValue(xIdx) * wGm.GetValue(wIdx);
                    }
                }
            }
        }
        return acc;
    }

    __aicore__ inline void Process()
    {
        // convT out: D=32,H=64,W=64. MaxPool(k=2,s=2): Dp=16,Hp=32,Wp=32.
        constexpr int32_t Dp = 16;
        constexpr int32_t Hp = 32;
        constexpr int32_t Wp = 32;
        constexpr int32_t POOL_S = 2;
        constexpr float NEG_INF = -3.402823466e+38f;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t start = bid * tasks_per_block;
        uint32_t end = start + tasks_per_block;
        if (end > total_tasks) end = total_tasks;

        for (uint32_t task = start; task < end; ++task) {
            const uint32_t ni = task / cout;
            const uint32_t co = task - ni * cout;

            const float biasCo = bGm.GetValue(static_cast<uint64_t>(co));
            const float mulc = mGm.GetValue(static_cast<uint64_t>(co));

            const uint64_t yBaseNCo = (static_cast<uint64_t>(ni) * cout + co) * (static_cast<uint64_t>(Dp) * Hp * Wp);

            for (int32_t pd = 0; pd < Dp; ++pd) {
                const int32_t od0 = pd * POOL_S;
                const uint64_t yBaseD = yBaseNCo + static_cast<uint64_t>(pd) * (static_cast<uint64_t>(Hp) * Wp);

                for (int32_t ph = 0; ph < Hp; ++ph) {
                    const int32_t oh0 = ph * POOL_S;
                    const uint64_t yBaseH = yBaseD + static_cast<uint64_t>(ph) * Wp;

                    for (int32_t pw = 0; pw < Wp; ++pw) {
                        const int32_t ow0 = pw * POOL_S;

                        float vmax = NEG_INF;
                        float v;

                        // 2x2x2 pool window over convT output, fused activations and mul
                        v = ConvTPoint(ni, co, od0 + 0, oh0 + 0, ow0 + 0, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 0, oh0 + 0, ow0 + 1, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 0, oh0 + 1, ow0 + 0, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 0, oh0 + 1, ow0 + 1, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;

                        v = ConvTPoint(ni, co, od0 + 1, oh0 + 0, ow0 + 0, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 1, oh0 + 0, ow0 + 1, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 1, oh0 + 1, ow0 + 0, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;
                        v = ConvTPoint(ni, co, od0 + 1, oh0 + 1, ow0 + 1, biasCo); v = Leaky(v); v *= mulc; v = Leaky(v); vmax = (v > vmax) ? v : vmax;

                        yGm.SetValue(yBaseH + static_cast<uint64_t>(pw), vmax);
                    }
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> mGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    float neg_slope;

    uint32_t total_tasks;
    uint32_t tasks_per_block;

    uint64_t xStrideW, xStrideH, xStrideD, xStrideC, xStrideN;
    uint64_t wStrideKw, wStrideKh, wStrideKd, wStrideCo, wStrideCi;
};

extern "C" __global__ __aicore__ void conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR multiplier, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom op;
    op.Init(x, weight, bias, multiplier, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.negative_slope,
            t.total_tasks, t.tasks_per_block);
    op.Process();
}
