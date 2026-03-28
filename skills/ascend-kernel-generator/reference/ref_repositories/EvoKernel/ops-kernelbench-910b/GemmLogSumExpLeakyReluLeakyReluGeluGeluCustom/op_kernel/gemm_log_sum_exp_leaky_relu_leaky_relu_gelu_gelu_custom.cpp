
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [1024,8192] float32 contiguous
// w: [N,K] = [8192,8192] float32 contiguous (Linear weight [out,in])
// b: [N]   = [8192] float32 contiguous
// y: [M,1] = [1024,1] float32 contiguous
//
// Computes per row m:
//   z[n] = sum_k x[m,k] * w[n,k] + b[n]
//   lse  = logsumexp(z) (stable): max + log(sum(exp(z - max)))
//   u    = leaky_relu(leaky_relu(lse, 0.01), 0.01)
//   u    = gelu(gelu(u))  (tanh-approx GELU)
//   y[m,0] = u
//
// Conservative implementation: scalar GM loads/stores only; parallelize by rows.

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}
__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

class KernelGemmLogSumExpLeakyReluLeakyReluGeluGeluCustom {
public:
    __aicore__ inline KernelGemmLogSumExpLeakyReluLeakyReluGeluGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N)
    {
        M_ = M; K_ = K; N_ = N;
        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0) return;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();

        const uint32_t rowsPerBlock = CeilDivU32(M_, blockNum);
        const uint32_t mStart = blockIdx * rowsPerBlock;
        const uint32_t mEnd = MinU32(M_, mStart + rowsPerBlock);
        if (mStart >= mEnd) return;

        for (uint32_t m = mStart; m < mEnd; ++m) {
            float lse = ComputeLogSumExpRow_(m);
            float out = PostOps_(lse);
            // y is [M,1] contiguous, store at linear index m
            yGm_.SetValue((uint64_t)m, out);
        }
    }

private:
    __aicore__ inline float Leaky_(float v) const {
        constexpr float slope = 0.01f;
        return (v >= 0.0f) ? v : (v * slope);
    }

    // exp(v) using AscendC vector op on 1-element UB tensor
    __aicore__ inline float ExpScalar_(float v) const {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> q0;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> q1;
        pipe.InitBuffer(q0, 1, 1 * sizeof(float));
        pipe.InitBuffer(q1, 1, 1 * sizeof(float));

        AscendC::LocalTensor<float> t0 = q0.AllocTensor<float>();
        AscendC::LocalTensor<float> t1 = q1.AllocTensor<float>();
        t0(0) = v;
        AscendC::Exp(t1, t0, 1);
        float out = t1(0);
        q1.FreeTensor(t1);
        q0.FreeTensor(t0);
        return out;
    }

    __aicore__ inline float LogScalar_(float v) const {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> q0;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> q1;
        pipe.InitBuffer(q0, 1, 1 * sizeof(float));
        pipe.InitBuffer(q1, 1, 1 * sizeof(float));

        AscendC::LocalTensor<float> t0 = q0.AllocTensor<float>();
        AscendC::LocalTensor<float> t1 = q1.AllocTensor<float>();
        t0(0) = v;
        AscendC::Log(t1, t0, 1);
        float out = t1(0);
        q1.FreeTensor(t1);
        q0.FreeTensor(t0);
        return out;
    }

    // Tanh-based GELU approximation:
    // gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    __aicore__ inline float GeluTanhApprox_(float x) const {
        constexpr float kHalf = 0.5f;
        constexpr float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
        constexpr float kC = 0.044715f;

        float x3 = x * x * x;
        float inner = kSqrt2OverPi * (x + kC * x3);

        // tanh(inner) = (2/(1+exp(-2*inner))) - 1
        float e = ExpScalar_(-2.0f * inner);
        float t = (2.0f / (1.0f + e)) - 1.0f;
        return kHalf * x * (1.0f + t);
    }

    __aicore__ inline float ComputeLogSumExpRow_(uint32_t m)
    {
        // Pass 1: compute max(z)
        float maxVal = -3.402823466e+38f; // -FLT_MAX
        const uint64_t xRowBase = (uint64_t)m * (uint64_t)K_;

        for (uint32_t n = 0; n < N_; ++n) {
            float acc = bGm_.GetValue((uint64_t)n);
            const uint64_t wRowBase = (uint64_t)n * (uint64_t)K_;
            for (uint32_t k = 0; k < K_; ++k) {
                const float xv = xGm_.GetValue(xRowBase + (uint64_t)k);
                const float wv = wGm_.GetValue(wRowBase + (uint64_t)k);
                acc += xv * wv;
            }
            if (acc > maxVal) maxVal = acc;
        }

        // Pass 2: compute sum(exp(z - max))
        float sumExp = 0.0f;
        for (uint32_t n = 0; n < N_; ++n) {
            float acc = bGm_.GetValue((uint64_t)n);
            const uint64_t wRowBase = (uint64_t)n * (uint64_t)K_;
            for (uint32_t k = 0; k < K_; ++k) {
                const float xv = xGm_.GetValue(xRowBase + (uint64_t)k);
                const float wv = wGm_.GetValue(wRowBase + (uint64_t)k);
                acc += xv * wv;
            }
            sumExp += ExpScalar_(acc - maxVal);
        }

        // lse = log(sumExp) + maxVal
        return LogScalar_(sumExp) + maxVal;
    }

    __aicore__ inline float PostOps_(float v) const
    {
        // LeakyReLU twice
        v = Leaky_(v);
        v = Leaky_(v);

        // GELU twice
        v = GeluTanhApprox_(v);
        v = GeluTanhApprox_(v);
        return v;
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t M_{0}, K_{0}, N_{0};
};

extern "C" __global__ __aicore__ void gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelGemmLogSumExpLeakyReluLeakyReluGeluGeluCustom op;
    op.Init(x, w, b, y, td.M, td.K, td.N);
    op.Process();
}
