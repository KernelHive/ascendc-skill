
#include "kernel_operator.h"

// Specialized contract (fixed shapes/types):
// x: [M,K] = [1024,8192] float32
// w: [N,K] = [8192,8192] float32 (Linear weight [out,in])
// b: [N]   = [8192] float32
// gamma/beta: [N] float32 (GroupNorm affine)
// mul_w: [N] float32 (channelwise multiply)
// num_groups: scalar int32 (expected 256)
// eps: scalar float32
// y: [M,N] float32
//
// Computes per row m:
//   z = x @ W^T + b
//   GroupNorm over channels with G groups (group size 32):
//     z = (z - mean_g) / sqrt(var_g + eps) * gamma + beta
//   swish(z) = z * sigmoid(z)
//   z *= mul_w
//   swish(z)
//
// Implementation notes (to avoid toolchain issues):
// - No <math.h> and no expf/sqrtf; use AscendC vector APIs Exp/Sqrt/Reciprocal.
// - Keep all intermediate vectors in UB. For each row: process group by group (32 elems).

class KernelGemmGroupNormSwishMultiplySwishCustom {
public:
    __aicore__ inline KernelGemmGroupNormSwishMultiplySwishCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                               GM_ADDR gamma, GM_ADDR beta, GM_ADDR mul_w,
                               GM_ADDR num_groups, GM_ADDR eps,
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
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ float*)beta);
        mulWGm_.SetGlobalBuffer((__gm__ float*)mul_w);
        numGroupsGm_.SetGlobalBuffer((__gm__ int32_t*)num_groups);
        epsGm_.SetGlobalBuffer((__gm__ float*)eps);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        pipe_.InitBuffer(vecBuf_, 1, VEC_UB_FLOATS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (M_ <= 0 || N_ <= 0 || K_ <= 0) return;

        // Read scalars once per core
        int32_t ng = numGroupsGm_.GetValue(0);
        if (ng <= 0) ng = 1;
        // Specialization: trust tiling
        ng = G_;
        float eps = epsGm_.GetValue(0);
        // Accept eps<=0 but clamp to default
        if (!(eps > 0.0f)) eps = 1e-5f;

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t rowsPerCore = ((int64_t)M_ + blockNum - 1) / blockNum;
        int64_t mStart = blockIdx * rowsPerCore;
        int64_t mEnd = mStart + rowsPerCore;
        if (mEnd > (int64_t)M_) mEnd = (int64_t)M_;

        // UB layout:
        // z[32], tmp0[32], tmp1[32], tmp2[32], work[32]
        for (int64_t m = mStart; m < mEnd; ++m) {
            const int64_t xRowBase = m * (int64_t)K_;
            const int64_t yRowBase = m * (int64_t)N_;

            for (int32_t g = 0; g < ng; ++g) {
                const int32_t cStart = g * groupSize_;
                const int32_t cEnd = cStart + groupSize_; // 32
                // Safety for tail (not expected in this specialization)
                const int32_t len = (cEnd <= N_) ? groupSize_ : (N_ - cStart);
                if (len <= 0) continue;

                // Allocate UB slices
                AscendC::LocalTensor<float> ub = vecBuf_.AllocTensor<float>();
                AscendC::LocalTensor<float> z = ub;                              // [len]
                AscendC::LocalTensor<float> tmp0 = ub[UB_STRIDE_1];              // [len]
                AscendC::LocalTensor<float> tmp1 = ub[UB_STRIDE_2];              // [len]
                AscendC::LocalTensor<float> tmp2 = ub[UB_STRIDE_3];              // [len]
                AscendC::LocalTensor<float> work = ub[UB_STRIDE_4];              // [len]

                // Compute GEMM for the group channels into z (scalar inner-loop over K).
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    float acc = 0.0f;
                    const int64_t wRowBase = (int64_t)n * (int64_t)K_;
                    for (int32_t k = 0; k < K_; ++k) {
                        acc += xGm_.GetValue((uint64_t)(xRowBase + k)) *
                               wGm_.GetValue((uint64_t)(wRowBase + k));
                    }
                    acc += bGm_.GetValue((uint64_t)n);
                    z.SetValue((uint32_t)i, acc);
                }

                // mean = sum(z) / len
                float sum = 0.0f;
                float sqsum = 0.0f;
                for (int32_t i = 0; i < len; ++i) {
                    float v = z.GetValue((uint32_t)i);
                    sum += v;
                    sqsum += v * v;
                }
                const float invLen = 1.0f / (float)len;
                const float mean = sum * invLen;
                float var = sqsum * invLen - mean * mean;
                if (var < 0.0f) var = 0.0f;

                // invStd = 1/sqrt(var+eps) using vector ops on a duplicated scalar.
                AscendC::Duplicate(tmp0, var + eps, (uint32_t)len);
                AscendC::Sqrt(tmp1, tmp0, (uint32_t)len);
                AscendC::Reciprocal(tmp2, tmp1, (uint32_t)len); // tmp2 = invStd

                // Normalize: (z - mean) * invStd
                AscendC::Adds(tmp0, z, -mean, (uint32_t)len); // tmp0 = z - mean
                AscendC::Mul(z, tmp0, tmp2, (uint32_t)len);  // z = (z-mean)*invStd

                // Affine: z = z*gamma + beta (gamma/beta are GM, load to UB)
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    const float gv = gammaGm_.GetValue((uint64_t)n);
                    const float bv = betaGm_.GetValue((uint64_t)n);
                    float v = z.GetValue((uint32_t)i);
                    z.SetValue((uint32_t)i, v * gv + bv);
                }

                // Swish: z * sigmoid(z) where sigmoid(z)=1/(1+exp(-z))
                AscendC::Muls(tmp0, z, -1.0f, (uint32_t)len);  // tmp0 = -z
                AscendC::Exp(tmp1, tmp0, (uint32_t)len);       // tmp1 = exp(-z)
                AscendC::Adds(tmp1, tmp1, 1.0f, (uint32_t)len);// tmp1 = 1+exp(-z)
                AscendC::Reciprocal(tmp2, tmp1, (uint32_t)len);// tmp2 = sigmoid(z)
                AscendC::Mul(z, z, tmp2, (uint32_t)len);       // z = z*sigmoid(z)

                // Multiply by mul_w (GM scalar per channel)
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    float v = z.GetValue((uint32_t)i);
                    v *= mulWGm_.GetValue((uint64_t)n);
                    z.SetValue((uint32_t)i, v);
                }

                // Swish again
                AscendC::Muls(tmp0, z, -1.0f, (uint32_t)len);
                AscendC::Exp(tmp1, tmp0, (uint32_t)len);
                AscendC::Adds(tmp1, tmp1, 1.0f, (uint32_t)len);
                AscendC::Reciprocal(tmp2, tmp1, (uint32_t)len);
                AscendC::Mul(z, z, tmp2, (uint32_t)len);

                // Store back to GM
                for (int32_t i = 0; i < len; ++i) {
                    const int32_t n = cStart + i;
                    yGm_.SetValue((uint64_t)(yRowBase + n), z.GetValue((uint32_t)i));
                }

                vecBuf_.FreeTensor(ub);
            }
        }
    }

private:
    // UB sizing: 5 vectors * 32 floats = 160 floats
    static constexpr uint32_t UB_STRIDE_1 = 32;
    static constexpr uint32_t UB_STRIDE_2 = 64;
    static constexpr uint32_t UB_STRIDE_3 = 96;
    static constexpr uint32_t UB_STRIDE_4 = 128;
    static constexpr uint32_t VEC_UB_FLOATS = 160;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> vecBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> gammaGm_;
    AscendC::GlobalTensor<float> betaGm_;
    AscendC::GlobalTensor<float> mulWGm_;
    AscendC::GlobalTensor<int32_t> numGroupsGm_;
    AscendC::GlobalTensor<float> epsGm_;
    AscendC::GlobalTensor<float> yGm_;

    int32_t M_{0}, K_{0}, N_{0};
    int32_t G_{0}, groupSize_{0};
};

extern "C" __global__ __aicore__ void gemm_group_norm_swish_multiply_swish_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b,
    GM_ADDR gamma, GM_ADDR beta, GM_ADDR mul_w,
    GM_ADDR num_groups, GM_ADDR eps,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelGemmGroupNormSwishMultiplySwishCustom op;
    op.Init(x, w, b, gamma, beta, mul_w, num_groups, eps, y,
            td.M, td.K, td.N, td.G, td.groupSize);
    op.Process();
}
