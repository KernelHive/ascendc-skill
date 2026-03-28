
#include "kernel_operator.h"

class KernelSumReductionDim1KeepdimILP8 {
public:
    __aicore__ inline KernelSumReductionDim1KeepdimILP8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t B, uint32_t N, uint32_t S)
    {
        this->B_ = B;
        this->N_ = N;
        this->S_ = S;
        this->outerCount_ = static_cast<uint64_t>(B) * static_cast<uint64_t>(S);

        const uint64_t xNumel = static_cast<uint64_t>(B) * static_cast<uint64_t>(N) * static_cast<uint64_t>(S);
        const uint64_t yNumel = static_cast<uint64_t>(B) * static_cast<uint64_t>(S);

        xGm_.SetGlobalBuffer((__gm__ float *)x, xNumel);
        yGm_.SetGlobalBuffer((__gm__ float *)y, yNumel);
    }

    __aicore__ inline void Process()
    {
        if (outerCount_ == 0 || N_ == 0) return;

        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;

        const uint64_t chunk = (outerCount_ + blockNum - 1) / blockNum;
        const uint64_t start = blockIdx * chunk;
        uint64_t end = start + chunk;
        if (end > outerCount_) end = outerCount_;

        const uint64_t strideS = static_cast<uint64_t>(S_);

        for (uint64_t outIdx = start; outIdx < end; ++outIdx) {
            // outIdx -> (b, s)
            const uint32_t b = static_cast<uint32_t>(outIdx / static_cast<uint64_t>(S_));
            const uint32_t s = static_cast<uint32_t>(outIdx - static_cast<uint64_t>(b) * static_cast<uint64_t>(S_));

            // base points to x[b,0,s]
            uint64_t off = (static_cast<uint64_t>(b) * static_cast<uint64_t>(N_) * strideS) + static_cast<uint64_t>(s);

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

            uint32_t n = 0;
            const uint32_t nUnroll = (N_ / 8u) * 8u;
            for (; n < nUnroll; n += 8u) {
                // Load 8 rows at same s, spaced by S
                const float v0 = xGm_.GetValue(off);
                const float v1 = xGm_.GetValue(off + strideS);
                const float v2 = xGm_.GetValue(off + strideS * 2u);
                const float v3 = xGm_.GetValue(off + strideS * 3u);
                const float v4 = xGm_.GetValue(off + strideS * 4u);
                const float v5 = xGm_.GetValue(off + strideS * 5u);
                const float v6 = xGm_.GetValue(off + strideS * 6u);
                const float v7 = xGm_.GetValue(off + strideS * 7u);

                acc0 += v0; acc1 += v1; acc2 += v2; acc3 += v3;
                acc4 += v4; acc5 += v5; acc6 += v6; acc7 += v7;

                off += strideS * 8u;
            }

            float acc = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

            // Tail
            for (; n < N_; ++n) {
                acc += xGm_.GetValue(off);
                off += strideS;
            }

            yGm_.SetValue(outIdx, acc);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t B_ = 0, N_ = 0, S_ = 0;
    uint64_t outerCount_ = 0;
};

extern "C" __global__ __aicore__ void sum_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSumReductionDim1KeepdimILP8 op;
    op.Init(x, y, tiling_data.B, tiling_data.N, tiling_data.S);
    op.Process();
}
