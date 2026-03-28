
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelConv2dGroupNormScaleMaxPoolClampCustom {
public:
    __aicore__ inline KernelConv2dGroupNormScaleMaxPoolClampCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                               GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                               GM_ADDR y,
                               uint32_t Cin, uint32_t Hin, uint32_t Win,
                               uint32_t Cout, uint32_t K,
                               uint32_t Hc, uint32_t Wc,
                               uint32_t Ho, uint32_t Wo,
                               uint32_t G, uint32_t CperG,
                               uint32_t poolK, uint32_t poolS,
                               uint32_t elemsPerG, float invElemsPerG,
                               float eps, float clampMin, float clampMax,
                               uint32_t N)
    {
        Cin_ = Cin; Hin_ = Hin; Win_ = Win;
        Cout_ = Cout; K_ = K;
        Hc_ = Hc; Wc_ = Wc;
        Ho_ = Ho; Wo_ = Wo;
        G_ = G; CperG_ = CperG;
        poolK_ = poolK; poolS_ = poolS;
        elemsPerG_ = elemsPerG;
        invElemsPerG_ = invElemsPerG;
        eps_ = eps;
        clampMin_ = clampMin;
        clampMax_ = clampMax;
        N_ = N;

        xBase_ = reinterpret_cast<__gm__ float*>(x);
        wBase_ = reinterpret_cast<__gm__ float*>(w);
        bBase_ = reinterpret_cast<__gm__ float*>(b);
        gammaBase_ = reinterpret_cast<__gm__ float*>(gamma);
        betaBase_ = reinterpret_cast<__gm__ float*>(beta);
        scaleBase_ = reinterpret_cast<__gm__ float*>(scale);
        yBase_ = reinterpret_cast<__gm__ float*>(y);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t perCore = (N_ + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * perCore;

        uint32_t count = 0;
        if (start < N_) {
            const uint32_t remain = N_ - start;
            count = (remain < perCore) ? remain : perCore;
        }
        nStart_ = start;
        nCount_ = count;

        // UB buffers:
        // - sc_: rsqrt temp (at least 1)
        // - param caches: bias/gamma/beta/scale for 4 channels in group
        pipe_.InitBuffer(qUb_, BUFFER_NUM, 32 * sizeof(float));
        ub_ = qUb_.AllocTensor<float>();
    }

    __aicore__ inline void Process()
    {
        if (nCount_ == 0) { qUb_.FreeTensor(ub_); return; }
        for (uint32_t i = 0; i < nCount_; ++i) {
            ComputeOneSample(nStart_ + i);
        }
        qUb_.FreeTensor(ub_);
    }

private:
    __aicore__ inline float Rsqrt1(float v)
    {
        // Use ub_[0] safely as 1-element tensor base (LocalTensor supports SetValue/GetValue)
        ub_.SetValue(0, v);
        AscendC::Rsqrt<float>(ub_, ub_, 1);
        return ub_.GetValue(0);
    }

    __aicore__ inline float Conv2dPoint(uint64_t xBaseN, uint64_t wBaseCo,
                                        uint32_t oh, uint32_t ow,
                                        float biasVal,
                                        uint64_t HW_in, uint64_t W_in) const
    {
        float acc = biasVal;

        // Fixed Cin=8,K=3
        #pragma unroll
        for (uint32_t ci = 0; ci < 8u; ++ci) {
            const uint64_t xBaseNC = xBaseN + static_cast<uint64_t>(ci) * HW_in;
            const uint64_t wBaseCoCi = wBaseCo + static_cast<uint64_t>(ci) * 9ull;

            const uint64_t xY0 = xBaseNC + static_cast<uint64_t>(oh) * W_in + static_cast<uint64_t>(ow);
            const uint64_t xY1 = xY0 + W_in;
            const uint64_t xY2 = xY1 + W_in;

            acc += xBase_[xY0 + 0] * wBase_[wBaseCoCi + 0];
            acc += xBase_[xY0 + 1] * wBase_[wBaseCoCi + 1];
            acc += xBase_[xY0 + 2] * wBase_[wBaseCoCi + 2];

            acc += xBase_[xY1 + 0] * wBase_[wBaseCoCi + 3];
            acc += xBase_[xY1 + 1] * wBase_[wBaseCoCi + 4];
            acc += xBase_[xY1 + 2] * wBase_[wBaseCoCi + 5];

            acc += xBase_[xY2 + 0] * wBase_[wBaseCoCi + 6];
            acc += xBase_[xY2 + 1] * wBase_[wBaseCoCi + 7];
            acc += xBase_[xY2 + 2] * wBase_[wBaseCoCi + 8];
        }
        return acc;
    }

    __aicore__ inline void ComputeOneSample(uint32_t n)
    {
        if (n >= N_) return;

        const uint64_t HW_in = static_cast<uint64_t>(Hin_) * static_cast<uint64_t>(Win_);
        const uint64_t CHW_in = static_cast<uint64_t>(Cin_) * HW_in;
        const uint64_t xBaseN = static_cast<uint64_t>(n) * CHW_in;

        const uint64_t yStrideH = static_cast<uint64_t>(Wo_);
        const uint64_t yStrideC = static_cast<uint64_t>(Ho_) * yStrideH;
        const uint64_t yStrideN = static_cast<uint64_t>(Cout_) * yStrideC;
        const uint64_t yBaseN = static_cast<uint64_t>(n) * yStrideN;

        // UB param-cache layout inside ub_:
        // ub_[0] used for rsqrt
        // bias[4] at 4..7, gamma[4] at 8..11, beta[4] at 12..15, scale[4] at 16..19
        constexpr uint32_t OFF_BIAS  = 4;
        constexpr uint32_t OFF_GAMMA = 8;
        constexpr uint32_t OFF_BETA  = 12;
        constexpr uint32_t OFF_SCALE = 16;

        for (uint32_t gg = 0; gg < G_; ++gg) {
            const uint32_t cStart = gg * CperG_; // CperG == 4

            // cache params for 4 channels once per group
            #pragma unroll
            for (uint32_t ci = 0; ci < 4u; ++ci) {
                const uint32_t co = cStart + ci;
                ub_.SetValue(OFF_BIAS  + ci, bBase_[co]);
                ub_.SetValue(OFF_GAMMA + ci, gammaBase_[co]);
                ub_.SetValue(OFF_BETA  + ci, betaBase_[co]);
                ub_.SetValue(OFF_SCALE + ci, scaleBase_[co]);
            }

            float sum = 0.0f;
            float sumsq = 0.0f;

            // Pass1: stats over conv outputs for group (recomputing conv, but with cached bias)
            for (uint32_t ci = 0; ci < 4u; ++ci) {
                const uint32_t co = cStart + ci;
                const uint64_t wBaseCo = static_cast<uint64_t>(co) * static_cast<uint64_t>(Cin_) * 9ull;
                const float biasVal = ub_.GetValue(OFF_BIAS + ci);

                for (uint32_t oh = 0; oh < Hc_; ++oh) {
                    for (uint32_t ow = 0; ow < Wc_; ++ow) {
                        const float v = Conv2dPoint(xBaseN, wBaseCo, oh, ow, biasVal, HW_in, static_cast<uint64_t>(Win_));
                        sum += v;
                        sumsq += v * v;
                    }
                }
            }

            const float mean = sum * invElemsPerG_;
            float var = sumsq * invElemsPerG_ - mean * mean;
            if (var < 0.0f) var = 0.0f;
            const float invStd = Rsqrt1(var + eps_);

            // Pass2: produce pooled+clamped output for channels in this group
            #pragma unroll
            for (uint32_t ci = 0; ci < 4u; ++ci) {
                const uint32_t co = cStart + ci;
                const uint64_t wBaseCo = static_cast<uint64_t>(co) * static_cast<uint64_t>(Cin_) * 9ull;

                const float biasVal = ub_.GetValue(OFF_BIAS + ci);
                const float gam = ub_.GetValue(OFF_GAMMA + ci);
                const float bet = ub_.GetValue(OFF_BETA + ci);
                const float sc  = ub_.GetValue(OFF_SCALE + ci);

                const uint64_t yBaseNC = yBaseN + static_cast<uint64_t>(co) * yStrideC;

                for (uint32_t ph = 0; ph < Ho_; ++ph) {
                    const uint32_t h0 = ph * poolS_;
                    const uint64_t yRowBase = yBaseNC + static_cast<uint64_t>(ph) * yStrideH;

                    for (uint32_t pw = 0; pw < Wo_; ++pw) {
                        const uint32_t w0 = pw * poolS_;

                        float maxv = -3.402823466e+38f;

                        // Fully unrolled 4x4 pool window with fixed offsets
                        #pragma unroll
                        for (uint32_t rh = 0; rh < 4u; ++rh) {
                            const uint32_t oh = h0 + rh;
                            const uint32_t owBase = w0;

                            float v0 = Conv2dPoint(xBaseN, wBaseCo, oh, owBase + 0u, biasVal, HW_in, static_cast<uint64_t>(Win_));
                            float v1 = Conv2dPoint(xBaseN, wBaseCo, oh, owBase + 1u, biasVal, HW_in, static_cast<uint64_t>(Win_));
                            float v2 = Conv2dPoint(xBaseN, wBaseCo, oh, owBase + 2u, biasVal, HW_in, static_cast<uint64_t>(Win_));
                            float v3 = Conv2dPoint(xBaseN, wBaseCo, oh, owBase + 3u, biasVal, HW_in, static_cast<uint64_t>(Win_));

                            v0 = ((v0 - mean) * invStd) * gam + bet; v0 *= sc;
                            v1 = ((v1 - mean) * invStd) * gam + bet; v1 *= sc;
                            v2 = ((v2 - mean) * invStd) * gam + bet; v2 *= sc;
                            v3 = ((v3 - mean) * invStd) * gam + bet; v3 *= sc;

                            if (v0 > maxv) maxv = v0;
                            if (v1 > maxv) maxv = v1;
                            if (v2 > maxv) maxv = v2;
                            if (v3 > maxv) maxv = v3;
                        }

                        if (maxv < clampMin_) maxv = clampMin_;
                        if (maxv > clampMax_) maxv = clampMax_;
                        yBase_[yRowBase + static_cast<uint64_t>(pw)] = maxv;
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qUb_;
    AscendC::LocalTensor<float> ub_;

    __gm__ float* xBase_ = nullptr;
    __gm__ float* wBase_ = nullptr;
    __gm__ float* bBase_ = nullptr;
    __gm__ float* gammaBase_ = nullptr;
    __gm__ float* betaBase_ = nullptr;
    __gm__ float* scaleBase_ = nullptr;
    __gm__ float* yBase_ = nullptr;

    uint32_t Cin_ = 0, Hin_ = 0, Win_ = 0;
    uint32_t Cout_ = 0, K_ = 0;
    uint32_t Hc_ = 0, Wc_ = 0;
    uint32_t Ho_ = 0, Wo_ = 0;
    uint32_t G_ = 0, CperG_ = 0;
    uint32_t poolK_ = 0, poolS_ = 0;
    uint32_t elemsPerG_ = 0;
    float invElemsPerG_ = 0.0f, eps_ = 1e-5f;
    float clampMin_ = 0.0f, clampMax_ = 1.0f;

    uint32_t N_ = 0;
    uint32_t nStart_ = 0, nCount_ = 0;
};

extern "C" __global__ __aicore__ void conv2d_group_norm_scale_max_pool_clamp_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR gn_gamma, GM_ADDR gn_beta, GM_ADDR scale,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConv2dGroupNormScaleMaxPoolClampCustom op;
    op.Init(x, weight, bias, gn_gamma, gn_beta, scale, y,
            tiling_data.Cin, tiling_data.Hin, tiling_data.Win,
            tiling_data.Cout, tiling_data.K,
            tiling_data.Hc, tiling_data.Wc,
            tiling_data.Ho, tiling_data.Wo,
            tiling_data.G, tiling_data.CperG,
            tiling_data.poolK, tiling_data.poolS,
            tiling_data.elemsPerG, tiling_data.invElemsPerG,
            tiling_data.eps, tiling_data.clampMin, tiling_data.clampMax,
            tiling_data.N);
    op.Process();
}
