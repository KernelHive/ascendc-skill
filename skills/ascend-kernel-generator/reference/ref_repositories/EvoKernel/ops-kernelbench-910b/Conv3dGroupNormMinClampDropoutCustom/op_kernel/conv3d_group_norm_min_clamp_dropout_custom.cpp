
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelConv3dGroupNormMinClampDropoutCustom {
public:
    __aicore__ inline KernelConv3dGroupNormMinClampDropoutCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                               GM_ADDR gn_gamma, GM_ADDR gn_beta,
                               GM_ADDR y,
                               uint32_t Cin, uint32_t Din, uint32_t Hin, uint32_t Win,
                               uint32_t Cout, uint32_t K,
                               uint32_t Dout, uint32_t Hout, uint32_t Wout,
                               uint32_t G, uint32_t CperG,
                               uint32_t DHW, uint32_t elemsPerG,
                               float invElemsPerG, float eps,
                               float minValue, float clampMin, float clampMax,
                               uint32_t dropThresholdU32,
                               float invKeepProb,
                               uint32_t N,
                               uint32_t tasksPerSample)
    {
        Cin_ = Cin; Din_ = Din; Hin_ = Hin; Win_ = Win;
        Cout_ = Cout; K_ = K;
        Dout_ = Dout; Hout_ = Hout; Wout_ = Wout;
        G_ = G; CperG_ = CperG;
        DHW_ = DHW; elemsPerG_ = elemsPerG;
        invElemsPerG_ = invElemsPerG; eps_ = eps;

        minValue_ = minValue;
        clampMin_ = clampMin;
        clampMax_ = clampMax;

        dropThresholdU32_ = dropThresholdU32;
        invKeepProb_ = invKeepProb;

        N_ = N;
        tasksPerSample_ = tasksPerSample;

        xBase_ = reinterpret_cast<__gm__ float*>(x);
        wBase_ = reinterpret_cast<__gm__ float*>(w);
        bBase_ = reinterpret_cast<__gm__ float*>(b);
        gammaBase_ = reinterpret_cast<__gm__ float*>(gn_gamma);
        betaBase_  = reinterpret_cast<__gm__ float*>(gn_beta);
        yBase_ = reinterpret_cast<__gm__ float*>(y);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t tasks = N_ * tasksPerSample_;
        const uint32_t perCore = (tasks + blockNum - 1u) / blockNum;
        const uint32_t startTask = blockIdx * perCore;

        uint32_t count = 0;
        if (startTask < tasks) {
            const uint32_t remain = tasks - startTask;
            count = (remain < perCore) ? remain : perCore;
        }
        taskStart_ = startTask;
        taskCount_ = count;

        // scalar scratch for Rsqrt(1 element)
        pipe_.InitBuffer(qSc_, BUFFER_NUM, 64 * sizeof(float));
        sc_ = qSc_.AllocTensor<float>();

        // cache per-channel params into UB once per block: bias/gamma/beta for 16 channels
        pipe_.InitBuffer(qParam_, BUFFER_NUM, 3 * 16 * sizeof(float));
        param_ = qParam_.AllocTensor<float>();
        #pragma unroll
        for (uint32_t i = 0; i < 16u; ++i) {
            param_.SetValue(i + 0u,  bBase_[i]);
            param_.SetValue(i + 16u, gammaBase_[i]);
            param_.SetValue(i + 32u, betaBase_[i]);
        }
    }

    __aicore__ inline void Process()
    {
        if (taskCount_ == 0) {
            qParam_.FreeTensor(param_);
            qSc_.FreeTensor(sc_);
            return;
        }
        for (uint32_t t = 0; t < taskCount_; ++t) {
            ComputeOneTask(taskStart_ + t);
        }
        qParam_.FreeTensor(param_);
        qSc_.FreeTensor(sc_);
    }

private:
    __aicore__ inline float Rsqrt1(float v)
    {
        sc_.SetValue(0, v);
        AscendC::Rsqrt<float>(sc_, sc_, 1);
        return sc_.GetValue(0);
    }

    __aicore__ inline float ApplyMinClamp(float v) const
    {
        // torch.min(v, 0.0): cap values ABOVE 0 down to 0
        if (v > minValue_) v = minValue_;
        // clamp [0,1]
        if (v < clampMin_) v = clampMin_;
        if (v > clampMax_) v = clampMax_;
        return v;
    }

    // Cheaper RNG: per-(n,co) seed and simple LCG per element
    __aicore__ inline uint32_t Seed(uint32_t n, uint32_t co) const
    {
        uint32_t s = 0xD1B54A35u;
        s ^= (n + 1u) * 0x9E3779B9u;
        s ^= (co + 1u) * 0x85EBCA6Bu;
        return s;
    }
    __aicore__ inline uint32_t LcgNext(uint32_t s) const
    {
        return s * 1664525u + 1013904223u;
    }

    __aicore__ inline float ConvAt(const uint64_t xBaseN,
                                   const uint32_t co, const uint32_t od, const uint32_t oh, const uint32_t ow,
                                   const float biasVal,
                                   const uint64_t strideC, const uint64_t strideD, const uint64_t strideH) const
    {
        // Specialized: Cin=3, K=3, stride=1, pad=0, dil=1
        float acc = biasVal;
        const uint64_t wBaseCo = static_cast<uint64_t>(co) * 81ull;

        #pragma unroll
        for (uint32_t ci = 0; ci < 3u; ++ci) {
            const uint64_t xBaseNC = xBaseN + static_cast<uint64_t>(ci) * strideC;
            const uint64_t wBaseCoCi = wBaseCo + static_cast<uint64_t>(ci) * 27ull;

            uint32_t wIdx = 0;
            #pragma unroll
            for (uint32_t kd = 0; kd < 3u; ++kd) {
                const uint32_t id = od + kd;
                const uint64_t xZ = xBaseNC + static_cast<uint64_t>(id) * strideD;

                #pragma unroll
                for (uint32_t kh = 0; kh < 3u; ++kh) {
                    const uint32_t ih = oh + kh;
                    const uint64_t xY = xZ + static_cast<uint64_t>(ih) * strideH;
                    const uint64_t xOff = xY + static_cast<uint64_t>(ow);

                    // contiguous in W
                    const float x0 = xBase_[xOff + 0];
                    const float x1 = xBase_[xOff + 1];
                    const float x2 = xBase_[xOff + 2];

                    const float w0 = wBase_[wBaseCoCi + static_cast<uint64_t>(wIdx + 0)];
                    const float w1 = wBase_[wBaseCoCi + static_cast<uint64_t>(wIdx + 1)];
                    const float w2 = wBase_[wBaseCoCi + static_cast<uint64_t>(wIdx + 2)];
                    wIdx += 3;

                    acc += x0 * w0 + x1 * w1 + x2 * w2;
                }
            }
        }
        return acc;
    }

    __aicore__ inline void ComputeOneTask(uint32_t taskId)
    {
        // taskId maps to (n, group)
        const uint32_t n = taskId / tasksPerSample_;
        const uint32_t gg = taskId - n * tasksPerSample_;
        if (n >= N_ || gg >= G_) return;

        const uint64_t strideH = static_cast<uint64_t>(Win_);
        const uint64_t strideD = static_cast<uint64_t>(Hin_) * strideH;
        const uint64_t strideC = static_cast<uint64_t>(Din_) * strideD;
        const uint64_t xBaseN = static_cast<uint64_t>(n) * static_cast<uint64_t>(Cin_) * strideC;

        const uint64_t yBaseN = static_cast<uint64_t>(n) * static_cast<uint64_t>(Cout_) * static_cast<uint64_t>(DHW_);
        const uint32_t cStart = gg * CperG_;

        float sum = 0.0f;
        float sumsq = 0.0f;

        // Pass1: stats over conv outputs in group (pre-affine)
        for (uint32_t ci = 0; ci < CperG_; ++ci) {
            const uint32_t co = cStart + ci;
            const float biasVal = param_.GetValue(co);

            for (uint32_t od = 0; od < Dout_; ++od) {
                for (uint32_t oh = 0; oh < Hout_; ++oh) {
                    uint32_t ow = 0;
                    for (; ow < Wout_; ++ow) {
                        const float v = ConvAt(xBaseN, co, od, oh, ow, biasVal, strideC, strideD, strideH);
                        sum += v;
                        sumsq += v * v;
                    }
                }
            }
        }

        const float mean = sum * invElemsPerG_;
        float var = sumsq * invElemsPerG_ - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const float invStd = Rsqrt1(var + eps_);

        // Pass2: write outputs with gn affine + min+clamp + dropout
        for (uint32_t ci = 0; ci < CperG_; ++ci) {
            const uint32_t co = cStart + ci;

            // hoist per-channel params into registers
            const float biasVal = param_.GetValue(co);
            const float gam = param_.GetValue(16u + co);
            const float bet = param_.GetValue(32u + co);

            const uint64_t yBaseCo = yBaseN + static_cast<uint64_t>(co) * static_cast<uint64_t>(DHW_);

            uint32_t rng = Seed(n, co);
            uint32_t idx = 0;
            for (uint32_t od = 0; od < Dout_; ++od) {
                for (uint32_t oh = 0; oh < Hout_; ++oh) {
                    for (uint32_t ow = 0; ow < Wout_; ++ow, ++idx) {
                        float v = ConvAt(xBaseN, co, od, oh, ow, biasVal, strideC, strideD, strideH);
                        v = (v - mean) * invStd;
                        v = v * gam + bet;

                        v = ApplyMinClamp(v);

                        rng = LcgNext(rng);
                        if (rng < dropThresholdU32_) {
                            v = 0.0f;
                        } else {
                            v = v * invKeepProb_;
                        }

                        yBase_[yBaseCo + static_cast<uint64_t>(idx)] = v;
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSc_;
    AscendC::LocalTensor<float> sc_;

    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qParam_;
    AscendC::LocalTensor<float> param_;

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
    uint32_t DHW_ = 0, elemsPerG_ = 0;

    float invElemsPerG_ = 0.0f, eps_ = 1e-5f;
    float minValue_ = 0.0f, clampMin_ = 0.0f, clampMax_ = 1.0f;

    uint32_t dropThresholdU32_ = 0;
    float invKeepProb_ = 1.0f;

    uint32_t N_ = 0;
    uint32_t tasksPerSample_ = 0;
    uint32_t taskStart_ = 0, taskCount_ = 0;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_min_clamp_dropout_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR gn_gamma, GM_ADDR gn_beta,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConv3dGroupNormMinClampDropoutCustom op;
    op.Init(x, weight, bias, gn_gamma, gn_beta, y,
            t.Cin, t.Din, t.Hin, t.Win,
            t.Cout, t.K,
            t.Dout, t.Hout, t.Wout,
            t.G, t.CperG,
            t.DHW, t.elemsPerG,
            t.invElemsPerG, t.eps,
            t.minValue, t.clampMin, t.clampMax,
            t.dropThresholdU32,
            t.invKeepProb,
            t.N,
            t.tasksPerSample);
    op.Process();
}
