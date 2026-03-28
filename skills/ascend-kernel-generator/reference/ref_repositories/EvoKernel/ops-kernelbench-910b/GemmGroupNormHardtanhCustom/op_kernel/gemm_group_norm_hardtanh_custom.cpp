
#include "kernel_operator.h"

// Optimized fused operator (float32 only), specialized:
// x:        [1024,8192]
// weight:   [8192,8192] stored as [N,K] (PyTorch Linear weight [out,in])
// lin_bias: [8192]
// gn_gamma: [8192]
// gn_beta:  [8192]
// y:        [1024,8192]
//
// Key optimization vs baseline:
//   For each (row, group), compute GEMM+bias ONCE for the whole group (512 channels),
//   store the 512 intermediate values in UB, reduce mean/var from UB, then normalize+affine
//   +hardtanh and write using the UB cache. This eliminates the baseline's second full GEMM pass.

class KernelGemmGroupNormHardtanhCustom {
public:
    __aicore__ inline KernelGemmGroupNormHardtanhCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR lin_bias,
                               GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t num_groups, uint32_t group_size,
                               uint32_t total_tasks, uint32_t block_dim)
    {
        M_ = (int32_t)M;
        K_ = (int32_t)K;
        N_ = (int32_t)N;
        G_ = (int32_t)num_groups;
        groupSize_ = (int32_t)group_size; // 512
        totalTasks_ = (int32_t)total_tasks;
        blockDim_ = (int32_t)block_dim;

        const uint64_t xSize = (uint64_t)M * (uint64_t)K;
        const uint64_t wSize = (uint64_t)N * (uint64_t)K;
        const uint64_t vSize = (uint64_t)N;
        const uint64_t ySize = (uint64_t)M * (uint64_t)N;

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        linBiasGm_.SetGlobalBuffer((__gm__ float*)lin_bias, vSize);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma, vSize);
        betaGm_.SetGlobalBuffer((__gm__ float*)beta, vSize);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB: cache group activations (512 floats) + scratch (512 floats) + small 8-lane helpers
        pipe_.InitBuffer(qAct_,  1, 512u * sizeof(float));
        pipe_.InitBuffer(qTmp_,  1, 512u * sizeof(float));
        pipe_.InitBuffer(qOne8_, 1, 8u * sizeof(float));
        pipe_.InitBuffer(qS8_,   1, 8u * sizeof(float));
    }

    __aicore__ inline float HardTanh(float v)
    {
        if (v < -2.0f) v = -2.0f;
        if (v >  2.0f) v =  2.0f;
        return v;
    }

    __aicore__ inline float InvStdFromVar(float var, float eps,
                                         AscendC::LocalTensor<float>& one8,
                                         AscendC::LocalTensor<float>& s8)
    {
        float denom = var + eps;
        if (denom < eps) denom = eps;

        s8.SetValue(0, denom);
        #pragma unroll
        for (int32_t i = 1; i < 8; ++i) s8.SetValue((uint32_t)i, denom);

        AscendC::Sqrt(s8, s8, 8);
        AscendC::Div(s8, one8, s8, 8);
        return s8.GetValue(0);
    }

    __aicore__ inline void ComputeGroupToUb(int32_t m, int32_t c0,
                                           AscendC::LocalTensor<float>& act)
    {
        const int64_t xBase = (int64_t)m * (int64_t)K_;
        // Compute 512 outputs: dot(x[m,:], w[n,:]) + bias[n]
        for (int32_t i = 0; i < groupSize_; ++i) {
            const int32_t n = c0 + i;
            const int64_t wBase = (int64_t)n * (int64_t)K_;
            float acc = 0.0f;
            #pragma unroll 1
            for (int32_t k = 0; k < K_; ++k) {
                acc += xGm_.GetValue((uint64_t)(xBase + (int64_t)k)) *
                       wGm_.GetValue((uint64_t)(wBase + (int64_t)k));
            }
            acc += linBiasGm_.GetValue((uint64_t)(int64_t)n);
            act.SetValue((uint32_t)i, acc);
        }
    }

    __aicore__ inline void Process()
    {
        const int32_t bid = (int32_t)AscendC::GetBlockIdx();

        AscendC::LocalTensor<float> one8 = qOne8_.AllocTensor<float>();
        AscendC::LocalTensor<float> s8   = qS8_.AllocTensor<float>();
        AscendC::Duplicate(one8, 1.0f, 8);

        const float eps = 1e-5f;
        const float invCount = 1.0f / 512.0f;

        // Grid-stride over tasks to tolerate blockDim cap.
        for (int32_t task = bid; task < totalTasks_; task += blockDim_) {
            const int32_t m = task / G_;
            const int32_t g = task - m * G_;
            const int32_t c0 = g * groupSize_;
            const int64_t outRowBase = (int64_t)m * (int64_t)N_;

            AscendC::LocalTensor<float> act = qAct_.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
            (void)tmp;

            // 1) GEMM+bias once -> UB cache
            ComputeGroupToUb(m, c0, act);

            // 2) Reduce mean/var from UB cache
            float sum = 0.0f;
            float sumsq = 0.0f;
            #pragma unroll 1
            for (int32_t i = 0; i < 512; ++i) {
                const float v = act.GetValue((uint32_t)i);
                sum += v;
                sumsq += v * v;
            }
            const float mean = sum * invCount;
            float var = sumsq * invCount - mean * mean;
            if (var < 0.0f) var = 0.0f;

            const float invStd = InvStdFromVar(var, eps, one8, s8);

            // 3) Normalize + affine + hardtanh and write
            for (int32_t i = 0; i < 512; ++i) {
                const int32_t n = c0 + i;
                float yn = (act.GetValue((uint32_t)i) - mean) * invStd;
                const uint64_t no = (uint64_t)(int64_t)n;
                yn = yn * gammaGm_.GetValue(no) + betaGm_.GetValue(no);
                yn = HardTanh(yn);
                yGm_.SetValue((uint64_t)(outRowBase + (int64_t)n), yn);
            }

            qTmp_.FreeTensor(tmp);
            qAct_.FreeTensor(act);
        }

        qS8_.FreeTensor(s8);
        qOne8_.FreeTensor(one8);
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qAct_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qOne8_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qS8_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> linBiasGm_;
    AscendC::GlobalTensor<float> gammaGm_;
    AscendC::GlobalTensor<float> betaGm_;
    AscendC::GlobalTensor<float> yGm_;

    int32_t M_{0}, K_{0}, N_{0};
    int32_t G_{0}, groupSize_{0};
    int32_t totalTasks_{0};
    int32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void gemm_group_norm_hardtanh_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR lin_bias, GM_ADDR gn_gamma, GM_ADDR gn_beta,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    KernelGemmGroupNormHardtanhCustom op;
    op.Init(x, weight, lin_bias, gn_gamma, gn_beta, y,
            td.M, td.K, td.N, td.num_groups, td.group_size, td.total_tasks, td.block_dim);
    op.Process();
}
