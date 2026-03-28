
#include "kernel_operator.h"

// Specialized contract:
// x: [128,4096,4095] float32 contiguous (row-major)
// reduce along dim=1 => y: [128,4095]
//
// Optimization in this round:
// - Compute 8 output columns (i..i+7) per loop (ILP along inner dimension).
// - Reduce r with fixed compile-time unroll (8×), branch-free in hot path.
// - No UB allocations (avoids prior failure patterns), keep launch stable: one block per batch.

class KernelMaxReductionOverADimensionCustom {
public:
    __aicore__ inline KernelMaxReductionOverADimensionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t /*totalX*/, uint32_t /*totalY*/,
                               uint32_t batch, uint32_t reduceDim, uint32_t innerDim)
    {
        batch_ = batch;
        reduceDim_ = reduceDim;
        innerDim_ = innerDim;
        xGm_.SetGlobalBuffer((__gm__ float*)x);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0 || reduceDim_ == 0 || innerDim_ == 0) return;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        if (blockNum == 0) return;

        constexpr float NEG_INF = -3.402823466e+38f;
        constexpr uint32_t I_UNROLL = 8;  // columns per iteration
        constexpr uint32_t R_UNROLL = 8;  // reduction unroll

        for (uint32_t b = blockIdx; b < batch_; b += blockNum) {
            const uint64_t inner = (uint64_t)innerDim_;
            const uint64_t red   = (uint64_t)reduceDim_;
            const uint64_t batchBase = (uint64_t)b * red * inner;
            const uint64_t yBase = (uint64_t)b * inner;

            // Main body: groups of 8 columns.
            const uint32_t iMain = (innerDim_ / I_UNROLL) * I_UNROLL;
            for (uint32_t i = 0; i < iMain; i += I_UNROLL) {
                float m0 = NEG_INF, m1 = NEG_INF, m2 = NEG_INF, m3 = NEG_INF;
                float m4 = NEG_INF, m5 = NEG_INF, m6 = NEG_INF, m7 = NEG_INF;

                const uint64_t base = batchBase + (uint64_t)i;

                uint32_t r = 0;
                // For this specialized shape REDUCE_DIM=4096, divisible by 8.
#pragma unroll
                for (; r + (R_UNROLL - 1) < reduceDim_; r += R_UNROLL) {
                    // Unrolled r-chunks, each loads 8 columns.
#pragma unroll
                    for (uint32_t u = 0; u < R_UNROLL; ++u) {
                        const uint64_t off = base + (uint64_t)(r + u) * inner;
                        float v0 = xGm_.GetValue(off + 0);
                        float v1 = xGm_.GetValue(off + 1);
                        float v2 = xGm_.GetValue(off + 2);
                        float v3 = xGm_.GetValue(off + 3);
                        float v4 = xGm_.GetValue(off + 4);
                        float v5 = xGm_.GetValue(off + 5);
                        float v6 = xGm_.GetValue(off + 6);
                        float v7 = xGm_.GetValue(off + 7);

                        m0 = (v0 > m0) ? v0 : m0;
                        m1 = (v1 > m1) ? v1 : m1;
                        m2 = (v2 > m2) ? v2 : m2;
                        m3 = (v3 > m3) ? v3 : m3;
                        m4 = (v4 > m4) ? v4 : m4;
                        m5 = (v5 > m5) ? v5 : m5;
                        m6 = (v6 > m6) ? v6 : m6;
                        m7 = (v7 > m7) ? v7 : m7;
                    }
                }
                // Remainder (kept for correctness; should not run for 4096).
                for (; r < reduceDim_; ++r) {
                    const uint64_t off = base + (uint64_t)r * inner;
                    float v0 = xGm_.GetValue(off + 0);
                    float v1 = xGm_.GetValue(off + 1);
                    float v2 = xGm_.GetValue(off + 2);
                    float v3 = xGm_.GetValue(off + 3);
                    float v4 = xGm_.GetValue(off + 4);
                    float v5 = xGm_.GetValue(off + 5);
                    float v6 = xGm_.GetValue(off + 6);
                    float v7 = xGm_.GetValue(off + 7);

                    m0 = (v0 > m0) ? v0 : m0;
                    m1 = (v1 > m1) ? v1 : m1;
                    m2 = (v2 > m2) ? v2 : m2;
                    m3 = (v3 > m3) ? v3 : m3;
                    m4 = (v4 > m4) ? v4 : m4;
                    m5 = (v5 > m5) ? v5 : m5;
                    m6 = (v6 > m6) ? v6 : m6;
                    m7 = (v7 > m7) ? v7 : m7;
                }

                const uint64_t yo = yBase + (uint64_t)i;
                yGm_.SetValue(yo + 0, m0);
                yGm_.SetValue(yo + 1, m1);
                yGm_.SetValue(yo + 2, m2);
                yGm_.SetValue(yo + 3, m3);
                yGm_.SetValue(yo + 4, m4);
                yGm_.SetValue(yo + 5, m5);
                yGm_.SetValue(yo + 6, m6);
                yGm_.SetValue(yo + 7, m7);
            }

            // Tail columns (innerDim=4095 => 7 tail elems).
            for (uint32_t i = iMain; i < innerDim_; ++i) {
                const uint64_t base = batchBase + (uint64_t)i;
                float curMax = NEG_INF;

                uint32_t r = 0;
#pragma unroll
                for (; r + 7 < reduceDim_; r += 8) {
                    float v0 = xGm_.GetValue(base + (uint64_t)(r + 0) * inner);
                    float v1 = xGm_.GetValue(base + (uint64_t)(r + 1) * inner);
                    float v2 = xGm_.GetValue(base + (uint64_t)(r + 2) * inner);
                    float v3 = xGm_.GetValue(base + (uint64_t)(r + 3) * inner);
                    float v4 = xGm_.GetValue(base + (uint64_t)(r + 4) * inner);
                    float v5 = xGm_.GetValue(base + (uint64_t)(r + 5) * inner);
                    float v6 = xGm_.GetValue(base + (uint64_t)(r + 6) * inner);
                    float v7 = xGm_.GetValue(base + (uint64_t)(r + 7) * inner);

                    curMax = (v0 > curMax) ? v0 : curMax;
                    curMax = (v1 > curMax) ? v1 : curMax;
                    curMax = (v2 > curMax) ? v2 : curMax;
                    curMax = (v3 > curMax) ? v3 : curMax;
                    curMax = (v4 > curMax) ? v4 : curMax;
                    curMax = (v5 > curMax) ? v5 : curMax;
                    curMax = (v6 > curMax) ? v6 : curMax;
                    curMax = (v7 > curMax) ? v7 : curMax;
                }
                for (; r < reduceDim_; ++r) {
                    float v = xGm_.GetValue(base + (uint64_t)r * inner);
                    curMax = (v > curMax) ? v : curMax;
                }
                yGm_.SetValue(yBase + (uint64_t)i, curMax);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t batch_{0};
    uint32_t reduceDim_{0};
    uint32_t innerDim_{0};
};

extern "C" __global__ __aicore__ void max_reduction_over_a_dimension_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxReductionOverADimensionCustom op;
    op.Init(x, y,
            tiling_data.totalX, tiling_data.totalY,
            tiling_data.batch, tiling_data.reduceDim, tiling_data.innerDim);
    op.Process();
}
