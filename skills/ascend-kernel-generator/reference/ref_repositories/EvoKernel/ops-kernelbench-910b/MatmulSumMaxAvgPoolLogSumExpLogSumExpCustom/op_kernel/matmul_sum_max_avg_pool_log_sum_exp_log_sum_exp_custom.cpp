
#include "kernel_operator.h"

// Fused operator for the model:
//
// x: [B=1024, K=8192] float32
// w: [O=8192, K=8192] float32  (PyTorch Linear weight layout [out,in])
// b: [O=8192] float32
//
// Original graph:
//   u = x @ w^T + b               => [B,O]
//   v = sum(u, dim=1, keepdim=1)  => [B,1]
//   max/mean/logsumexp/logsumexp over dim=1 on [B,1] are identities
//
// So output:
//   y[b,0] = sum_o ( sum_k x[b,k]*w[o,k] + b[o] )
//
// Note: This is correctness-first; blockDim=1.

class KernelMatmulSumMaxAvgPoolLogSumExpLogSumExpCustom {
public:
    __aicore__ inline KernelMatmulSumMaxAvgPoolLogSumExpLogSumExpCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t B, uint32_t K, uint32_t O)
    {
        B_ = B; K_ = K; O_ = O;

        const uint64_t xSize = static_cast<uint64_t>(B_) * static_cast<uint64_t>(K_);
        const uint64_t wSize = static_cast<uint64_t>(O_) * static_cast<uint64_t>(K_);
        const uint64_t bSize = static_cast<uint64_t>(O_);
        const uint64_t ySize = static_cast<uint64_t>(B_) * 1ULL;

        xGm_.SetGlobalBuffer((__gm__ float*)x, xSize);
        wGm_.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm_.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm_.SetGlobalBuffer((__gm__ float*)y, ySize);
    }

    __aicore__ inline void Process()
    {
        if (B_ == 0 || K_ == 0 || O_ == 0) return;

        // Precompute sumB = sum_o b[o]
        float sumB = 0.0f;
        for (uint32_t o = 0; o < O_; ++o) {
            sumB += bGm_.GetValue(static_cast<uint64_t>(o));
        }

        // Precompute sumWcol[k] = sum_o w[o,k] into UB (size K=8192 floats => 32KB).
        pipe_.InitBuffer(qWSum_, 1, K_ * sizeof(float));
        AscendC::LocalTensor<float> wSum = qWSum_.AllocTensor<float>();
        AscendC::Duplicate(wSum, 0.0f, static_cast<int32_t>(K_));

        for (uint32_t o = 0; o < O_; ++o) {
            const uint64_t wBase = static_cast<uint64_t>(o) * static_cast<uint64_t>(K_);
            for (uint32_t k = 0; k < K_; ++k) {
                float cur = wSum.GetValue(k);
                cur += wGm_.GetValue(wBase + static_cast<uint64_t>(k));
                wSum.SetValue(k, cur);
            }
        }

        // For each row b: dot(x[b,:], sumWcol[:]) + sumB
        for (uint32_t br = 0; br < B_; ++br) {
            const uint64_t xBase = static_cast<uint64_t>(br) * static_cast<uint64_t>(K_);
            float acc = sumB;
            for (uint32_t k = 0; k < K_; ++k) {
                acc += xGm_.GetValue(xBase + static_cast<uint64_t>(k)) * wSum.GetValue(k);
            }
            yGm_.SetValue(static_cast<uint64_t>(br), acc); // flattened [B,1]
        }

        qWSum_.FreeTensor(wSum);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> qWSum_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t B_{0}, K_{0}, O_{0};
};

extern "C" __global__ __aicore__ void matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelMatmulSumMaxAvgPoolLogSumExpLogSumExpCustom op;
    op.Init(x, w, b, y, t.B, t.K, t.O);
    op.Process();
}
