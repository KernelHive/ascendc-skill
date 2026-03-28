
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [16384,4096] float32
// w: [N,K] = [4096,4096] float32 (Linear weight [out,in])
// b: [N]   = [4096] float32
// scaling: [1] float32 (scalar in tensor)
// y: [M,N] = [16384,4096] float32
//
// PyTorch reference:
//
//   z = x @ w^T + b
//   orig = z.clone().detach()   // numerically identical to z
//   y = z * scaling + orig
//
// => y = z * (1 + scaling)
//
// Implementation: scalar GM loads/stores, parallel over flattened output elements.

class KernelMatmulScalingResidualAddCustom {
public:
    __aicore__ inline KernelMatmulScalingResidualAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
                               GM_ADDR y, uint32_t M, uint32_t K, uint32_t N,
                               uint32_t totalElems)
    {
        M_ = M; K_ = K; N_ = N;
        totalElems_ = totalElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        sGm_.SetGlobalBuffer((__gm__ float*)scaling);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        // Load scaling once per core.
        scale_ = sGm_.GetValue(0);
        factor_ = 1.0f + scale_;
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0 || totalElems_ == 0) return;

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t total = (int64_t)totalElems_;
        const int64_t chunk = (total + blockNum - 1) / blockNum;
        int64_t start = blockIdx * chunk;
        int64_t end = start + chunk;
        if (end > total) end = total;
        if (start >= end) return;

        for (int64_t outIdx = start; outIdx < end; ++outIdx) {
            const uint32_t m = (uint32_t)(outIdx / (int64_t)N_);
            const uint32_t n = (uint32_t)(outIdx - (int64_t)m * (int64_t)N_);

            float acc = 0.0f;
            const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
            const uint64_t wBase = (uint64_t)n * (uint64_t)K_;

            for (uint32_t k = 0; k < K_; ++k) {
                const float xv = xGm_.GetValue(xBase + (uint64_t)k);
                const float wv = wGm_.GetValue(wBase + (uint64_t)k);
                acc += xv * wv;
            }
            acc += bGm_.GetValue((uint64_t)n);

            yGm_.SetValue((uint64_t)outIdx, acc * factor_);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    float scale_{0.0f};
    float factor_{1.0f};
};

extern "C" __global__ __aicore__ void matmul_scaling_residual_add_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulScalingResidualAddCustom op;
    op.Init(x, w, b, scaling, y, td.M, td.K, td.N, td.totalElems);
    op.Process();
}
