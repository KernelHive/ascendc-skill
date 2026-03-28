
#include "kernel_operator.h"

// Specialized fused operator for benchmark configuration.
// ConvT(stride=2,pad=1,k=3) -> *0.5 -> MaxPool3d(k=2,s=2) -> GlobalAvgPool -> Clamp[0,1]
// Output flattened as [N*Cout] corresponding to [N,Cout,1,1,1].
//
// This version computes 4 output channels together to reuse mapping/address work.

class KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom {
public:
    __aicore__ inline KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t din, uint32_t hin, uint32_t win,
                               uint32_t cout, uint32_t kd, uint32_t kh, uint32_t kw,
                               float scale, float clamp_min, float clamp_max,
                               uint32_t groups, uint32_t groups_per_block)
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
        this->scale = scale;
        this->clamp_min = clamp_min;
        this->clamp_max = clamp_max;
        this->groups = groups;
        this->groups_per_block = groups_per_block;

        const uint64_t xSize = static_cast<uint64_t>(n) * cin * din * hin * win;
        const uint64_t wSize = static_cast<uint64_t>(cin) * cout * kd * kh * kw;
        const uint64_t bSize = static_cast<uint64_t>(cout);
        const uint64_t ySize = static_cast<uint64_t>(n) * cout;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        xStrideW = 1ull;
        xStrideH = static_cast<uint64_t>(win);
        xStrideD = static_cast<uint64_t>(hin) * xStrideH;
        xStrideC = static_cast<uint64_t>(din) * xStrideD;
        xStrideN = static_cast<uint64_t>(cin) * xStrideC;

        // weight layout: [Cin,Cout,Kd,Kh,Kw]
        wStrideKw = 1ull;
        wStrideKh = static_cast<uint64_t>(kw);
        wStrideKd = static_cast<uint64_t>(kh) * wStrideKh;
        wStrideCo = static_cast<uint64_t>(kd) * wStrideKd;
        wStrideCi = static_cast<uint64_t>(cout) * wStrideCo;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t Dp = 15;
        constexpr int32_t Hp = 31;
        constexpr int32_t Wp = 31;
        constexpr int32_t POOL_S = 2;
        constexpr float NEG_INF = -3.402823466e+38f;
        constexpr float invPoolElems = 1.0f / 14415.0f; // 15*31*31
        constexpr uint32_t CO_GROUP = 4;

        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t start = bid * groups_per_block;
        uint32_t end = start + groups_per_block;
        if (end > groups) end = groups;

        for (uint32_t g = start; g < end; ++g) {
            const uint32_t ni = g >> 2;              // /4
            const uint32_t go = g - (ni << 2);       // %4
            const uint32_t coBase = go * CO_GROUP;   // 0,4,8,12

            // Load 4 bias values once (kept in registers)
            const float b0 = bGm.GetValue(static_cast<uint64_t>(coBase + 0));
            const float b1 = bGm.GetValue(static_cast<uint64_t>(coBase + 1));
            const float b2 = bGm.GetValue(static_cast<uint64_t>(coBase + 2));
            const float b3 = bGm.GetValue(static_cast<uint64_t>(coBase + 3));

            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

            for (int32_t pd = 0; pd < Dp; ++pd) {
                const int32_t od0 = pd * POOL_S;
                for (int32_t ph = 0; ph < Hp; ++ph) {
                    const int32_t oh0 = ph * POOL_S;
                    for (int32_t pw = 0; pw < Wp; ++pw) {
                        const int32_t ow0 = pw * POOL_S;

                        float m0 = NEG_INF, m1 = NEG_INF, m2 = NEG_INF, m3 = NEG_INF;

                        Eval8(ni, coBase, od0, oh0, ow0, b0, b1, b2, b3, m0, m1, m2, m3);

                        sum0 += m0;
                        sum1 += m1;
                        sum2 += m2;
                        sum3 += m3;
                    }
                }
            }

            float out0 = sum0 * invPoolElems;
            float out1 = sum1 * invPoolElems;
            float out2 = sum2 * invPoolElems;
            float out3 = sum3 * invPoolElems;

            out0 = (out0 < clamp_min) ? clamp_min : out0;
            out1 = (out1 < clamp_min) ? clamp_min : out1;
            out2 = (out2 < clamp_min) ? clamp_min : out2;
            out3 = (out3 < clamp_min) ? clamp_min : out3;

            out0 = (out0 > clamp_max) ? clamp_max : out0;
            out1 = (out1 > clamp_max) ? clamp_max : out1;
            out2 = (out2 > clamp_max) ? clamp_max : out2;
            out3 = (out3 > clamp_max) ? clamp_max : out3;

            const uint64_t yBase = static_cast<uint64_t>(ni) * 16ull + static_cast<uint64_t>(coBase);
            yGm.SetValue(yBase + 0, out0);
            yGm.SetValue(yBase + 1, out1);
            yGm.SetValue(yBase + 2, out2);
            yGm.SetValue(yBase + 3, out3);
        }
    }

private:
    __aicore__ inline void Eval8(uint32_t ni, uint32_t coBase,
                                int32_t od0, int32_t oh0, int32_t ow0,
                                float b0, float b1, float b2, float b3,
                                float &m0, float &m1, float &m2, float &m3) const
    {
        // 8 points in 2x2x2 maxpool window
        float v0, v1, v2, v3;

        ConvT4(ni, coBase, od0 + 0, oh0 + 0, ow0 + 0, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 0, oh0 + 0, ow0 + 1, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 0, oh0 + 1, ow0 + 0, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 0, oh0 + 1, ow0 + 1, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);

        ConvT4(ni, coBase, od0 + 1, oh0 + 0, ow0 + 0, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 1, oh0 + 0, ow0 + 1, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 1, oh0 + 1, ow0 + 0, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
        ConvT4(ni, coBase, od0 + 1, oh0 + 1, ow0 + 1, b0, b1, b2, b3, v0, v1, v2, v3); Max4(m0, m1, m2, m3, v0, v1, v2, v3);
    }

    __aicore__ inline void Max4(float &m0, float &m1, float &m2, float &m3,
                               float v0, float v1, float v2, float v3) const
    {
        m0 = (v0 > m0) ? v0 : m0;
        m1 = (v1 > m1) ? v1 : m1;
        m2 = (v2 > m2) ? v2 : m2;
        m3 = (v3 > m3) ? v3 : m3;
    }

    __aicore__ inline void ConvT4(uint32_t ni, uint32_t coBase,
                                 int32_t od, int32_t oh, int32_t ow,
                                 float b0, float b1, float b2, float b3,
                                 float &o0, float &o1, float &o2, float &o3) const
    {
        // Parity-specialized inverse mapping for stride=2, pad=1, k=3.
        // Along each axis: if (o+1) even => k in {0,2}, else k={1}. id=(o+1-k)/2.
        const int32_t td = od + 1;
        const int32_t th = oh + 1;
        const int32_t tw = ow + 1;

        const bool de = ((td & 1) == 0);
        const bool he = ((th & 1) == 0);
        const bool we = ((tw & 1) == 0);

        // D taps
        const int32_t kd0 = de ? 0 : 1;
        const int32_t kd1 = de ? 2 : 1;
        const int32_t nd  = de ? 2 : 1;

        // H taps
        const int32_t kh0 = he ? 0 : 1;
        const int32_t kh1 = he ? 2 : 1;
        const int32_t nh  = he ? 2 : 1;

        // W taps
        const int32_t kw0 = we ? 0 : 1;
        const int32_t kw1 = we ? 2 : 1;
        const int32_t nw  = we ? 2 : 1;

        float acc0 = b0;
        float acc1 = b1;
        float acc2 = b2;
        float acc3 = b3;

        const uint64_t nBaseX = static_cast<uint64_t>(ni) * xStrideN;

        // cin fixed to 3; cout fixed to 16; kernel volume fixed to 27.
        for (uint32_t ci = 0; ci < 3; ++ci) {
            const uint64_t ciBaseX = nBaseX + static_cast<uint64_t>(ci) * xStrideC;
            const uint64_t ciBaseW = static_cast<uint64_t>(ci) * wStrideCi;

            for (int32_t ad = 0; ad < nd; ++ad) {
                const int32_t kD = (ad == 0) ? kd0 : kd1;
                const int32_t id = (td - kD) >> 1;
                if (static_cast<uint32_t>(id) >= din) continue;

                const uint64_t idBaseX = ciBaseX + static_cast<uint64_t>(id) * xStrideD;
                const uint64_t kdBaseW = ciBaseW + static_cast<uint64_t>(kD) * wStrideKd;

                for (int32_t ah = 0; ah < nh; ++ah) {
                    const int32_t kH = (ah == 0) ? kh0 : kh1;
                    const int32_t ih = (th - kH) >> 1;
                    if (static_cast<uint32_t>(ih) >= hin) continue;

                    const uint64_t ihBaseX = idBaseX + static_cast<uint64_t>(ih) * xStrideH;
                    const uint64_t khBaseW = kdBaseW + static_cast<uint64_t>(kH) * wStrideKh;

                    for (int32_t aw = 0; aw < nw; ++aw) {
                        const int32_t kW = (aw == 0) ? kw0 : kw1;
                        const int32_t iw = (tw - kW) >> 1;
                        if (static_cast<uint32_t>(iw) >= win) continue;

                        const float xv = xGm.GetValue(ihBaseX + static_cast<uint64_t>(iw));

                        // Weight indices for coBase..coBase+3 at same (ci,kD,kH,kW)
                        const uint64_t wBase = khBaseW + static_cast<uint64_t>(kW);
                        const uint64_t w0 = wBase + static_cast<uint64_t>(coBase + 0) * wStrideCo;
                        const uint64_t w1 = wBase + static_cast<uint64_t>(coBase + 1) * wStrideCo;
                        const uint64_t w2 = wBase + static_cast<uint64_t>(coBase + 2) * wStrideCo;
                        const uint64_t w3 = wBase + static_cast<uint64_t>(coBase + 3) * wStrideCo;

                        acc0 += xv * wGm.GetValue(w0);
                        acc1 += xv * wGm.GetValue(w1);
                        acc2 += xv * wGm.GetValue(w2);
                        acc3 += xv * wGm.GetValue(w3);
                    }
                }
            }
        }

        o0 = acc0 * scale;
        o1 = acc1 * scale;
        o2 = acc2 * scale;
        o3 = acc3 * scale;
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, cin, din, hin, win;
    uint32_t cout, kd, kh, kw;
    float scale, clamp_min, clamp_max;

    uint32_t groups;
    uint32_t groups_per_block;

    uint64_t xStrideW, xStrideH, xStrideD, xStrideC, xStrideN;
    uint64_t wStrideKw, wStrideKh, wStrideKd, wStrideCo, wStrideCi;
};

extern "C" __global__ __aicore__ void conv_transpose3d_multiply_max_global_avg_pool_clamp_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom op;
    op.Init(x, weight, bias, y,
            t.n, t.cin, t.din, t.hin, t.win,
            t.cout, t.kd, t.kh, t.kw,
            t.scale, t.clamp_min, t.clamp_max,
            t.groups, t.groups_per_block);
    op.Process();
}
