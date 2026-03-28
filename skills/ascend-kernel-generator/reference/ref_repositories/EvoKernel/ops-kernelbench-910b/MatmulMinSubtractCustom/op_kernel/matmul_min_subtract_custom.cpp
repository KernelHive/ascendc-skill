
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [128,16384] float32
// w: [N,K] = [16384,16384] float32 (Linear weight [out,in])
// b: [N]   = [16384] float32
// c: scalar float32 (0D or [1])
// y: [M,N] = [128,16384] float32
//
// Computes:
//   z[m,n] = sum_k x[m,k] * w[n,k] + b[n]
//   y[m,n] = min(z[m,n], c) - c
//
// Conservative implementation: scalar GM loads/stores only.

class KernelMatmulMinSubtractCustom {
public:
    __aicore__ inline KernelMatmulMinSubtractCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR c, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N, uint32_t totalElems)
    {
        M_ = M;
        K_ = K;
        N_ = N;
        totalElems_ = totalElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        cGm_.SetGlobalBuffer((__gm__ float*)c);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0 || totalElems_ == 0) {
            return;
        }

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t total = (int64_t)totalElems_;
        const int64_t chunk = (total + blockNum - 1) / blockNum;
        int64_t start = blockIdx * chunk;
        int64_t end = start + chunk;
        if (end > total) end = total;

        const float cVal = cGm_.GetValue(0);

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

            if (acc > cVal) acc = cVal;
            acc -= cVal;

            yGm_.SetValue((uint64_t)outIdx, acc);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> cGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t M_{0};
    uint32_t K_{0};
    uint32_t N_{0};
    uint32_t totalElems_{0};
};

extern "C" __global__ __aicore__ void matmul_min_subtract_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR c,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulMinSubtractCustom op;
    op.Init(x, w, b, c, y, td.M, td.K, td.N, td.totalElems);
    op.Process();
}
