
#include "kernel_operator.h"

// Optimized: compute each (row, group) once, cache 32 activated values in UB,
// reduce mean/var from UB, then normalize+affine and write. This removes the
// baseline's full recomputation of GEMM+activation in pass2.

class KernelGemmBiasAddHardtanhMishGroupNormCustom {
public:
    __aicore__ inline KernelGemmBiasAddHardtanhMishGroupNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR lin_bias, GM_ADDR bias,
                               GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t num_groups, uint32_t group_size,
                               uint32_t total_tasks, uint32_t block_dim)
    {
        M_ = (int32_t)M; K_ = (int32_t)K; N_ = (int32_t)N;
        G_ = (int32_t)num_groups; groupSize_ = (int32_t)group_size;
        totalTasks_ = (int32_t)total_tasks; blockDim_ = (int32_t)block_dim;

        const uint64_t xSize = (uint64_t)M * (uint64_t)K;
        const uint64_t wSize = (uint64_t)N * (uint64_t)K;
        const uint64_t vSize = (uint64_t)N;
        const uint64_t ySize = (uint64_t)M * (uint64_t)N;

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        linBiasGm_.SetGlobalBuffer((__gm__ float*)lin_bias, vSize);
        extraBiasGm_.SetGlobalBuffer((__gm__ float*)bias, vSize);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma, vSize);
        betaGm_.SetGlobalBuffer((__gm__ float*)beta, vSize);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB: group activations (32) + vector scratch (at least 32)
        pipe_.InitBuffer(qAct_,   1, 32u * sizeof(float));
        pipe_.InitBuffer(qTmp0_,  1, 32u * sizeof(float));
        pipe_.InitBuffer(qTmp1_,  1, 32u * sizeof(float));
        pipe_.InitBuffer(qTmp2_,  1, 32u * sizeof(float));
        pipe_.InitBuffer(qTmp3_,  1, 32u * sizeof(float));
    }

    __aicore__ inline void MishInplace32(AscendC::LocalTensor<float> &x,
                                        AscendC::LocalTensor<float> &tmp0,
                                        AscendC::LocalTensor<float> &tmp1,
                                        AscendC::LocalTensor<float> &tmp2,
                                        AscendC::LocalTensor<float> &tmp3)
    {
        // mish(x)=x*tanh(softplus(x))
        // softplus(x)=max(x,0)+log(1+exp(-abs(x)))
        // Use 32-lane vector math to reduce scalar pressure.
        AscendC::Abs(tmp0, x, 32);               // tmp0=abs(x)
        AscendC::Muls(tmp1, tmp0, -1.0f, 32);    // tmp1=-abs(x)
        AscendC::Exp(tmp2, tmp1, 32);            // tmp2=exp(-abs(x))
        AscendC::Duplicate(tmp3, 1.0f, 32);      // tmp3=1
        AscendC::Add(tmp2, tmp2, tmp3, 32);      // tmp2=1+exp(-abs(x))
        AscendC::Log(tmp2, tmp2, 32);            // tmp2=log(1+exp(-abs(x)))

        // tmp1 = max(x,0)
        AscendC::Duplicate(tmp3, 0.0f, 32);
        AscendC::Max(tmp1, x, tmp3, 32);
        AscendC::Add(tmp1, tmp1, tmp2, 32);      // tmp1=softplus(x)

        // tanh(t)= (1-exp(-2t))/(1+exp(-2t))
        AscendC::Muls(tmp2, tmp1, -2.0f, 32);
        AscendC::Exp(tmp2, tmp2, 32);            // tmp2=exp(-2t)
        AscendC::Duplicate(tmp3, 1.0f, 32);
        AscendC::Sub(tmp0, tmp3, tmp2, 32);      // tmp0=1-exp(-2t)
        AscendC::Add(tmp1, tmp3, tmp2, 32);      // tmp1=1+exp(-2t)
        AscendC::Div(tmp0, tmp0, tmp1, 32);      // tmp0=tanh(t)

        AscendC::Mul(x, x, tmp0, 32);
    }

    __aicore__ inline void ComputeGroupActivationsToUb(int32_t m, int32_t c0,
                                                      AscendC::LocalTensor<float> &act,
                                                      AscendC::LocalTensor<float> &tmp0,
                                                      AscendC::LocalTensor<float> &tmp1,
                                                      AscendC::LocalTensor<float> &tmp2,
                                                      AscendC::LocalTensor<float> &tmp3)
    {
        const int64_t xBase = (int64_t)m * (int64_t)K_;
        // For n in [c0, c0+32): compute dot and store into act[i]
        // Keep scalar K-loop (cannot fit x/weight into UB here) but only do it once.
        for (int32_t i = 0; i < groupSize_; ++i) {
            const int32_t n = c0 + i;
            const int64_t wBase = (int64_t)n * (int64_t)K_;

            float acc = 0.0f;
            #pragma unroll 1
            for (int32_t k = 0; k < K_; ++k) {
                acc += xGm_.GetValue((uint64_t)(xBase + (int64_t)k)) *
                       wGm_.GetValue((uint64_t)(wBase + (int64_t)k));
            }

            const uint64_t no = (uint64_t)(int64_t)n;
            acc += linBiasGm_.GetValue(no);
            acc += extraBiasGm_.GetValue(no);

            // hardtanh clamp [-1,1]
            if (acc < -1.0f) acc = -1.0f;
            if (acc >  1.0f) acc =  1.0f;
            act.SetValue((uint32_t)i, acc);
        }

        // mish in vector (32 lanes)
        MishInplace32(act, tmp0, tmp1, tmp2, tmp3);
    }

    __aicore__ inline void Process()
    {
        const int32_t bid = (int32_t)AscendC::GetBlockIdx();
        // Grid-stride over tasks to tolerate blockDim cap.
        for (int32_t task = bid; task < totalTasks_; task += blockDim_) {
            const int32_t m = task / G_;
            const int32_t g = task - m * G_;
            const int32_t c0 = g * groupSize_;

            AscendC::LocalTensor<float> act  = qAct_.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp0 = qTmp0_.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp1 = qTmp1_.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp2 = qTmp2_.AllocTensor<float>();
            AscendC::LocalTensor<float> tmp3 = qTmp3_.AllocTensor<float>();

            ComputeGroupActivationsToUb(m, c0, act, tmp0, tmp1, tmp2, tmp3);

            // Reduce mean/var (groupSize_=32), from UB.
            float sum = 0.0f;
            float sumsq = 0.0f;
            #pragma unroll
            for (int32_t i = 0; i < 32; ++i) {
                const float v = act.GetValue((uint32_t)i);
                sum += v;
                sumsq += v * v;
            }
            const float invCount = 1.0f / 32.0f;
            const float mean = sum * invCount;
            float var = sumsq * invCount - mean * mean;
            if (var < 0.0f) var = 0.0f;

            // invStd via vector sqrt/div
            AscendC::Duplicate(tmp0, var + 1e-5f, 32);
            AscendC::Sqrt(tmp0, tmp0, 32);
            AscendC::Duplicate(tmp1, 1.0f, 32);
            AscendC::Div(tmp0, tmp1, tmp0, 32); // tmp0 = invStd
            const float invStd = tmp0.GetValue(0);

            // Normalize + affine, write to GM; reuse UB act.
            const int64_t outRowBase = (int64_t)m * (int64_t)N_;
            for (int32_t i = 0; i < 32; ++i) {
                const int32_t n = c0 + i;
                float yn = (act.GetValue((uint32_t)i) - mean) * invStd;
                const uint64_t no = (uint64_t)(int64_t)n;
                yn = yn * gammaGm_.GetValue(no) + betaGm_.GetValue(no);
                yGm_.SetValue((uint64_t)(outRowBase + (int64_t)n), yn);
            }

            qTmp3_.FreeTensor(tmp3);
            qTmp2_.FreeTensor(tmp2);
            qTmp1_.FreeTensor(tmp1);
            qTmp0_.FreeTensor(tmp0);
            qAct_.FreeTensor(act);
        }
    }

private:
    AscendC::TPipe pipe_;

    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qAct_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp0_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp1_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp2_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qTmp3_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> linBiasGm_;
    AscendC::GlobalTensor<float> extraBiasGm_;
    AscendC::GlobalTensor<float> gammaGm_;
    AscendC::GlobalTensor<float> betaGm_;
    AscendC::GlobalTensor<float> yGm_;

    int32_t M_{0}, K_{0}, N_{0};
    int32_t G_{0}, groupSize_{0};
    int32_t totalTasks_{0};
    int32_t blockDim_{1};
};

extern "C" __global__ __aicore__ void gemm_bias_add_hardtanh_mish_group_norm_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR lin_bias, GM_ADDR bias,
    GM_ADDR gn_gamma, GM_ADDR gn_beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    KernelGemmBiasAddHardtanhMishGroupNormCustom op;
    op.Init(x, weight, lin_bias, bias, gn_gamma, gn_beta, y,
            td.M, td.K, td.N, td.num_groups, td.group_size, td.total_tasks, td.block_dim);
    op.Process();
}
