
#include "kernel_operator.h"

// Specialized contract:
// x: [128,4096,4095] float32 contiguous
// reduce along dim=1 (size 4096) => y: [128,4095]
//
// Key optimizations in this round:
// - Remap work over the flattened output space (outerCount = B*I) to increase block-level parallelism.
// - For each output element, compute one base offset and pointer-walk by constant stride (innerDim) in the hot loop.
// - 8-way ILP unroll to reduce dependency chain and help overlap GM loads with arithmetic.
// - Keep scalar GM GetValue/SetValue (no UB/DataCopy) for robustness.

class KernelMeanReductionOverADimensionCustomILP8 {
public:
    static constexpr uint32_t UNROLL = 8;

    __aicore__ inline KernelMeanReductionOverADimensionCustomILP8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t batch, uint32_t reduceDim, uint32_t innerDim,
                               uint32_t outerCount, float invReduce)
    {
        batch_ = batch;
        reduceDim_ = reduceDim;
        innerDim_ = innerDim;
        outerCount_ = outerCount;
        invReduce_ = invReduce;

        const uint64_t xNumel = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(reduceDim_) * static_cast<uint64_t>(innerDim_);
        const uint64_t yNumel = static_cast<uint64_t>(batch_) * static_cast<uint64_t>(innerDim_);
        xGm_.SetGlobalBuffer((__gm__ float*)x, xNumel);
        yGm_.SetGlobalBuffer((__gm__ float*)y, yNumel);
    }

    __aicore__ inline void Process()
    {
        if (outerCount_ == 0 || reduceDim_ == 0 || innerDim_ == 0) return;

        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        if (blockNum == 0) blockNum = 1;

        const uint64_t total = static_cast<uint64_t>(outerCount_);
        const uint64_t chunk = (total + blockNum - 1) / blockNum;
        const uint64_t start = blockIdx * chunk;
        uint64_t end = start + chunk;
        if (end > total) end = total;

        const uint64_t stride = static_cast<uint64_t>(innerDim_);

        for (uint64_t outIdx = start; outIdx < end; ++outIdx) {
            // outIdx maps to y[b, i] with i in [0, innerDim)
            // b = outIdx / innerDim, i = outIdx % innerDim
            const uint32_t b = static_cast<uint32_t>(outIdx / static_cast<uint64_t>(innerDim_));
            const uint32_t i = static_cast<uint32_t>(outIdx - static_cast<uint64_t>(b) * static_cast<uint64_t>(innerDim_));

            // x offset for x[b, 0, i]
            uint64_t off = (static_cast<uint64_t>(b) * static_cast<uint64_t>(reduceDim_) * stride) + static_cast<uint64_t>(i);

            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;

            uint32_t r = 0;
            const uint32_t rUnroll = (reduceDim_ / UNROLL) * UNROLL;
            for (; r < rUnroll; r += UNROLL) {
                const float v0 = xGm_.GetValue(off);                 off += stride;
                const float v1 = xGm_.GetValue(off);                 off += stride;
                const float v2 = xGm_.GetValue(off);                 off += stride;
                const float v3 = xGm_.GetValue(off);                 off += stride;
                const float v4 = xGm_.GetValue(off);                 off += stride;
                const float v5 = xGm_.GetValue(off);                 off += stride;
                const float v6 = xGm_.GetValue(off);                 off += stride;
                const float v7 = xGm_.GetValue(off);                 off += stride;

                a0 += v0; a1 += v1; a2 += v2; a3 += v3;
                a4 += v4; a5 += v5; a6 += v6; a7 += v7;
            }

            float sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));

            for (; r < reduceDim_; ++r) {
                sum += xGm_.GetValue(off);
                off += stride;
            }

            yGm_.SetValue(outIdx, sum * invReduce_);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t batch_{0};
    uint32_t reduceDim_{0};
    uint32_t innerDim_{0};
    uint32_t outerCount_{0};
    float invReduce_{0.f};
};

extern "C" __global__ __aicore__ void mean_reduction_over_a_dimension_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMeanReductionOverADimensionCustomILP8 op;
    op.Init(x, y,
            tiling_data.batch, tiling_data.reduceDim, tiling_data.innerDim,
            tiling_data.outerCount, tiling_data.invReduce);
    op.Process();
}
