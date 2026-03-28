
#include "kernel_operator.h"

using namespace AscendC;

// Specialized contract (fixed shapes/types):
// x: [M,K] = [32768,1024] float32
// w: [N,K] = [4096,1024] float32 (Linear weight [out,in])
// linear_bias: [N] float32
// add_bias:    [N] float32
// gamma/beta:  [N] float32
// num_groups: scalar int32 (expected 64)
// eps: scalar float32
// y: [M,N] float32
//
// Computes per row m:
//   z = x @ W^T + linear_bias
//   z = swish(z)
//   z = z + add_bias
//   GroupNorm over channels with G groups (group size 64):
//     z = (z - mean_g) / sqrt(var_g + eps) * gamma + beta
//
// Notes:
// - Correctness/compile-stability first: GEMM uses scalar loops.
// - Swish/rsqrt uses vector APIs Exp/Sqrt/Reciprocal.
// - Process group-by-group (64 elems) to keep UB small.

class KernelMatmulSwishSumGroupNormCustom {
public:
    __aicore__ inline KernelMatmulSwishSumGroupNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR linear_bias, GM_ADDR add_bias,
                               GM_ADDR gamma, GM_ADDR beta, GM_ADDR num_groups, GM_ADDR eps,
                               GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t G, uint32_t groupSize)
    {
        M_ = (int32_t)M;
        K_ = (int32_t)K;
        N_ = (int32_t)N;
        G_ = (int32_t)G;
        groupSize_ = (int32_t)groupSize;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        linBiasGm_.SetGlobalBuffer((__gm__ float*)linear_bias);
        addBiasGm_.SetGlobalBuffer((__gm__ float*)add_bias);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ float*)beta);
        numGroupsGm_.SetGlobalBuffer((__gm__ int32_t*)num_groups);
        epsGm_.SetGlobalBuffer((__gm__ float*)eps);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        pipe_.InitBuffer(vecBuf_, 1, VEC_UB_FLOATS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (M_ <= 0 || N_ <= 0 || K_ <= 0) return;

        // Read scalars once per core (binding validates specialization; kernel trusts tiling)
        (void)numGroupsGm_.GetValue(0);
        float eps = epsGm_.GetValue(0);
        if (!(eps > 0.0f)) eps = 1e-5f;

        const int64_t blockNum = (int64_t)GetBlockNum();
        const int64_t blockIdx = (int64_t)GetBlockIdx();

        const int64_t rowsPerCore = ((int64_t)M_ + blockNum - 1) / blockNum;
        int64_t mStart = blockIdx * rowsPerCore;
        int64_t mEnd = mStart + rowsPerCore;
        if (mEnd > (int64_t)M_) mEnd = (int64_t)M_;
        if (mStart >= mEnd) return;

        for (int64_t m = mStart; m < mEnd; ++m) {
            const int64_t xRowBase = m * (int64_t)K_;
            const int64_t yRowBase = m * (int64_t)N_;

            for (int32_t g = 0; g < G_; ++g) {
                const int32_t cStart = g * groupSize_;
                const int32_t cEnd = cStart + groupSize_; // 64
                const int32_t len = (cEnd <= N_) ? groupSize_ : (N_ - cStart);
                if (len <= 0) continue;

                // UB layout: v[64], t0[64], t1[64], t2[64], work[64] => 320 floats
                LocalTensor<float> ub = vecBuf_.AllocTensor<float>();
                LocalTensor<float> v    = ub;                              // [len]
                LocalTensor<float> t0   = ub[UB_STRIDE_1];                 // [len]
                LocalTensor<float> t1   = ub[UB_STRIDE_2];                 // [len]
                LocalTensor<float> t2   = ub[UB_STRIDE_3];                 // [len]
                LocalTensor<float> work = ub[UB_STRIDE_4];                 // [len] (reserved)

                (void)work;

                // Compute z = x @ W^T + linear_bias, only for channels in this group.
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    float acc = 0.0f;
                    const int64_t wRowBase = (int64_t)n * (int64_t)K_;
                    for (int32_t k = 0; k < K_; ++k) {
                        acc += xGm_.GetValue((uint64_t)(xRowBase + k)) *
                               wGm_.GetValue((uint64_t)(wRowBase + k));
                    }
                    acc += linBiasGm_.GetValue((uint64_t)n);
                    v.SetValue((uint32_t)i, acc);
                }

                // Swish: v = v * sigmoid(v), sigmoid(v)=1/(1+exp(-v))
                Muls(t0, v, -1.0f, (uint32_t)len);
                Exp(t1, t0, (uint32_t)len);
                Adds(t1, t1, 1.0f, (uint32_t)len);
                Reciprocal(t2, t1, (uint32_t)len);
                Mul(v, v, t2, (uint32_t)len);

                // Add add_bias
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    float vv = v.GetValue((uint32_t)i) + addBiasGm_.GetValue((uint64_t)n);
                    v.SetValue((uint32_t)i, vv);
                }

                // Compute mean/var over this group (len==64 in specialization)
                float sum = 0.0f;
                float sqsum = 0.0f;
                for (int32_t i = 0; i < len; ++i) {
                    float vv = v.GetValue((uint32_t)i);
                    sum += vv;
                    sqsum += vv * vv;
                }
                const float invLen = 1.0f / (float)len;
                const float mean = sum * invLen;
                float var = sqsum * invLen - mean * mean;
                if (var < 0.0f) var = 0.0f;

                // invStd = 1/sqrt(var+eps)
                Duplicate(t0, var + eps, (uint32_t)len);
                Sqrt(t1, t0, (uint32_t)len);
                Reciprocal(t2, t1, (uint32_t)len);

                // Normalize: (v-mean)*invStd
                Adds(t0, v, -mean, (uint32_t)len);
                Mul(v, t0, t2, (uint32_t)len);

                // Affine + store
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    float out = v.GetValue((uint32_t)i);
                    out = out * gammaGm_.GetValue((uint64_t)n) + betaGm_.GetValue((uint64_t)n);
                    yGm_.SetValue((uint64_t)(yRowBase + n), out);
                }

                vecBuf_.FreeTensor(ub);
            }
        }
    }

private:
    static constexpr uint32_t UB_STRIDE_1 = 64;
    static constexpr uint32_t UB_STRIDE_2 = 128;
    static constexpr uint32_t UB_STRIDE_3 = 192;
    static constexpr uint32_t UB_STRIDE_4 = 256;
    static constexpr uint32_t VEC_UB_FLOATS = 320;

    TPipe pipe_;
    TQue<TPosition::VECCALC, 1> vecBuf_;

    GlobalTensor<float> xGm_;
    GlobalTensor<float> wGm_;
    GlobalTensor<float> linBiasGm_;
    GlobalTensor<float> addBiasGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<int32_t> numGroupsGm_;
    GlobalTensor<float> epsGm_;
    GlobalTensor<float> yGm_;

    int32_t M_{0}, K_{0}, N_{0};
    int32_t G_{1}, groupSize_{1};
};

extern "C" __global__ __aicore__ void matmul_swish_sum_group_norm_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR linear_bias, GM_ADDR add_bias,
    GM_ADDR gamma, GM_ADDR beta, GM_ADDR num_groups, GM_ADDR eps,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulSwishSumGroupNormCustom op;
    op.Init(x, w, linear_bias, add_bias, gamma, beta, num_groups, eps, y,
            td.M, td.K, td.N, td.G, td.groupSize);
    op.Process();
}
