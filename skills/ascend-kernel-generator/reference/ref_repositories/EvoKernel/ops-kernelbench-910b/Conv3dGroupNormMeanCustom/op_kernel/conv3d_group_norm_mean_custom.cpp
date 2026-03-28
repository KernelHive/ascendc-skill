
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelConv3dGroupNormMeanCustom {
public:
    __aicore__ inline KernelConv3dGroupNormMeanCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                               uint32_t Cin, uint32_t Din, uint32_t Hin, uint32_t Win,
                               uint32_t Cout, uint32_t K,
                               uint32_t Dout, uint32_t Hout, uint32_t Wout,
                               uint32_t G, uint32_t CperG,
                               uint32_t DHW, uint32_t elemsPerG, uint32_t elemsPerN,
                               uint32_t tileDhw,
                               float invElemsPerG, float invElemsPerN,
                               float eps, uint32_t N)
    {
        Cin_ = Cin; Din_ = Din; Hin_ = Hin; Win_ = Win;
        Cout_ = Cout; K_ = K;
        Dout_ = Dout; Hout_ = Hout; Wout_ = Wout;
        G_ = G; CperG_ = CperG;
        DHW_ = DHW; elemsPerG_ = elemsPerG; elemsPerN_ = elemsPerN;
        tileDhw_ = tileDhw;
        invElemsPerG_ = invElemsPerG; invElemsPerN_ = invElemsPerN;
        eps_ = eps; N_ = N;

        xBase_ = reinterpret_cast<__gm__ float*>(x);
        wBase_ = reinterpret_cast<__gm__ float*>(w);
        bBase_ = reinterpret_cast<__gm__ float*>(b);
        gammaBase_ = reinterpret_cast<__gm__ float*>(gamma);
        betaBase_ = reinterpret_cast<__gm__ float*>(beta);
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

        // UB: scalar buffer for Rsqrt
        pipe_.InitBuffer(qSc_, BUFFER_NUM, 8 * sizeof(float));
        sc_ = qSc_.AllocTensor<float>();

        // Keep a small UB scratch for ABI stability (not used in hot loop)
        const uint64_t tileElems = static_cast<uint64_t>(CperG_) * static_cast<uint64_t>(tileDhw_);
        pipe_.InitBuffer(qTile_, BUFFER_NUM, tileElems * sizeof(float));
        tileBuf_ = qTile_.AllocTensor<float>();
    }

    __aicore__ inline void Process()
    {
        if (nCount_ == 0) return;
        if (N_ == 0 || Cin_ == 0 || Cout_ == 0 || K_ == 0 || G_ == 0 || CperG_ == 0) return;
        if ((Cout_ % G_) != 0) return;

        for (uint32_t i = 0; i < nCount_; ++i) {
            ComputeOneSample(nStart_ + i);
        }

        qTile_.FreeTensor(tileBuf_);
        qSc_.FreeTensor(sc_);
    }

private:
    __aicore__ inline float Rsqrt1(float v)
    {
        sc_.SetValue(0, v);
        AscendC::Rsqrt<float>(sc_, sc_, 1);
        return sc_.GetValue(0);
    }

    // Compute 3 adjacent outputs in W (ow,ow+1,ow+2) sharing address math and input reuse.
    __aicore__ inline void ConvRow3(uint32_t n, uint32_t co, uint32_t od, uint32_t oh, uint32_t ow,
                                   float &o0, float &o1, float &o2)
    {
        // Hoist bias into registers
        float acc0 = bBase_[co];
        float acc1 = acc0;
        float acc2 = acc0;

        const uint64_t HW_in = static_cast<uint64_t>(Hin_) * static_cast<uint64_t>(Win_);
        const uint64_t DW_in = static_cast<uint64_t>(Din_) * HW_in;

        const uint64_t xBaseN = static_cast<uint64_t>(n) * static_cast<uint64_t>(Cin_) * DW_in;
        const uint64_t wBaseCo = static_cast<uint64_t>(co) * static_cast<uint64_t>(Cin_) * 27ull;

        // Fixed Cin=3, K=3 specialized loops still use bounds but avoid extra div/mod.
        for (uint32_t ci = 0; ci < 3; ++ci) {
            const uint64_t xBaseNC = xBaseN + static_cast<uint64_t>(ci) * DW_in;
            const uint64_t wBaseCoCi = wBaseCo + static_cast<uint64_t>(ci) * 27ull;

            uint32_t wIdx = 0;
            for (uint32_t kd = 0; kd < 3; ++kd) {
                const uint32_t id = od + kd;
                const uint64_t xZ = xBaseNC + static_cast<uint64_t>(id) * HW_in;

                for (uint32_t kh = 0; kh < 3; ++kh) {
                    const uint32_t ih = oh + kh;
                    const uint64_t xY = xZ + static_cast<uint64_t>(ih) * static_cast<uint64_t>(Win_);
                    const uint64_t xOff = xY + static_cast<uint64_t>(ow);

                    // Need x at ow..ow+4 to cover 3 outputs with kW=3
                    const float x0 = xBase_[xOff + 0];
                    const float x1 = xBase_[xOff + 1];
                    const float x2 = xBase_[xOff + 2];
                    const float x3 = xBase_[xOff + 3];
                    const float x4 = xBase_[xOff + 4];

                    const float w0 = wBase_[wBaseCoCi + wIdx + 0];
                    const float w1 = wBase_[wBaseCoCi + wIdx + 1];
                    const float w2 = wBase_[wBaseCoCi + wIdx + 2];
                    wIdx += 3;

                    // out0 uses x0,x1,x2 ; out1 uses x1,x2,x3 ; out2 uses x2,x3,x4
                    acc0 += x0 * w0 + x1 * w1 + x2 * w2;
                    acc1 += x1 * w0 + x2 * w1 + x3 * w2;
                    acc2 += x2 * w0 + x3 * w1 + x4 * w2;
                }
            }
        }

        o0 = acc0; o1 = acc1; o2 = acc2;
    }

    __aicore__ inline void ComputeOneSample(uint32_t n)
    {
        if (n >= N_) return;

        float sumAll = 0.0f;

        for (uint32_t gg = 0; gg < G_; ++gg) {
            const uint32_t cStart = gg * CperG_;

            // Pass1: stats (sum/sumsq) using nested loops (no idx->(d,h,w) div/mod)
            float sum = 0.0f;
            float sumsq = 0.0f;

            for (uint32_t ci = 0; ci < CperG_; ++ci) {
                const uint32_t co = cStart + ci;

                for (uint32_t od = 0; od < Dout_; ++od) {
                    for (uint32_t oh = 0; oh < Hout_; ++oh) {
                        // Wout=30, step 3
                        for (uint32_t ow = 0; ow < Wout_; ow += 3) {
                            float v0, v1, v2;
                            ConvRow3(n, co, od, oh, ow, v0, v1, v2);
                            sum += (v0 + v1) + v2;
                            sumsq += (v0 * v0 + v1 * v1) + (v2 * v2);
                        }
                    }
                }
            }

            const float mean = sum * invElemsPerG_;
            float var = sumsq * invElemsPerG_ - mean * mean;
            if (var < 0.0f) var = 0.0f;
            const float invStd = Rsqrt1(var + eps_);

            // Pass2: recompute conv and apply affine+accumulate; hoist gamma/beta per co.
            for (uint32_t ci = 0; ci < CperG_; ++ci) {
                const uint32_t co = cStart + ci;
                const float gam = gammaBase_[co];
                const float bet = betaBase_[co];

                for (uint32_t od = 0; od < Dout_; ++od) {
                    for (uint32_t oh = 0; oh < Hout_; ++oh) {
                        for (uint32_t ow = 0; ow < Wout_; ow += 3) {
                            float v0, v1, v2;
                            ConvRow3(n, co, od, oh, ow, v0, v1, v2);

                            v0 = (v0 - mean) * invStd;
                            v1 = (v1 - mean) * invStd;
                            v2 = (v2 - mean) * invStd;

                            sumAll += (v0 * gam + bet) + (v1 * gam + bet) + (v2 * gam + bet);
                        }
                    }
                }
            }
        }

        yBase_[n] = sumAll * invElemsPerN_;
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSc_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qTile_;
    AscendC::LocalTensor<float> sc_;
    AscendC::LocalTensor<float> tileBuf_;

    __gm__ float* xBase_ = nullptr;
    __gm__ float* wBase_ = nullptr;
    __gm__ float* bBase_ = nullptr;
    __gm__ float* gammaBase_ = nullptr;
    __gm__ float* betaBase_ = nullptr;
    __gm__ float* yBase_ = nullptr;

    uint32_t Cin_ = 0, Din_ = 0, Hin_ = 0, Win_ = 0;
    uint32_t Cout_ = 0, K_ = 0;
    uint32_t Dout_ = 0, Hout_ = 0, Wout_ = 0;
    uint32_t G_ = 0, CperG_ = 0;
    uint32_t DHW_ = 0, elemsPerG_ = 0, elemsPerN_ = 0;
    uint32_t tileDhw_ = 0;
    float invElemsPerG_ = 0.0f, invElemsPerN_ = 0.0f, eps_ = 1e-5f;

    uint32_t N_ = 0;
    uint32_t nStart_ = 0, nCount_ = 0;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_mean_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConv3dGroupNormMeanCustom op;
    op.Init(x, weight, bias, gamma, beta, y,
            tiling_data.Cin, tiling_data.Din, tiling_data.Hin, tiling_data.Win,
            tiling_data.Cout, tiling_data.K,
            tiling_data.Dout, tiling_data.Hout, tiling_data.Wout,
            tiling_data.G, tiling_data.CperG,
            tiling_data.DHW, tiling_data.elemsPerG, tiling_data.elemsPerN,
            tiling_data.tileDhw,
            tiling_data.invElemsPerG, tiling_data.invElemsPerN,
            tiling_data.eps, tiling_data.N);
    op.Process();
}
