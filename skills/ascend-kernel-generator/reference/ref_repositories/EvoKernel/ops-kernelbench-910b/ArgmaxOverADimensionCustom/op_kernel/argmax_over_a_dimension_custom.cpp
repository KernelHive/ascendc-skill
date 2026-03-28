
#include "kernel_operator.h"
#include <cstdint>

// Specialized contract:
// x: [128,4096,4095] float32 contiguous, reduce dim=1 => y: [128,4095] int64 indices
//
// Optimization: block-level tiling over inner dimension to increase parallelism,
// plus reduction unroll and direct __gm__ pointer loads to reduce scalar/control overhead.

class KernelArgmaxOverADimensionCustom {
public:
    __aicore__ inline KernelArgmaxOverADimensionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t batch, uint32_t reduceDim, uint32_t innerDim,
                               uint32_t tileInner, uint32_t tilesPerBatch)
    {
        batch_ = batch;
        reduceDim_ = reduceDim;
        innerDim_ = innerDim;
        tileInner_ = tileInner;
        tilesPerBatch_ = tilesPerBatch;

        xPtr_ = (__gm__ const float*)x;
        yPtr_ = (__gm__ int64_t*)y;
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0 || reduceDim_ == 0 || innerDim_ == 0 || tileInner_ == 0 || tilesPerBatch_ == 0) {
            return;
        }

        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t b = blockIdx / tilesPerBatch_;
        const uint32_t tileId = blockIdx - b * tilesPerBatch_;
        if (b >= batch_) {
            return;
        }

        const uint32_t i0 = tileId * tileInner_;
        if (i0 >= innerDim_) {
            return;
        }

        uint32_t tileLen = tileInner_;
        if (i0 + tileLen > innerDim_) {
            tileLen = innerDim_ - i0;
        }

        const uint64_t batchBase = (uint64_t)b * (uint64_t)reduceDim_ * (uint64_t)innerDim_;
        const uint32_t K = reduceDim_;
        const uint32_t I = innerDim_;

        // Process each output i in the tile in registers.
        for (uint32_t j = 0; j < tileLen; ++j) {
            const uint32_t i = i0 + j;
            const __gm__ float* p = xPtr_ + batchBase + (uint64_t)i;

            float curMax = -3.402823466e+38f; // -FLT_MAX
            uint32_t curArg = 0;

            uint32_t r = 0;
            // Unroll by 4 to reduce scalar loop overhead.
            for (; r + 3 < K; r += 4) {
                float v0 = p[(uint64_t)(r + 0) * (uint64_t)I];
                if (v0 > curMax) { curMax = v0; curArg = r + 0; }

                float v1 = p[(uint64_t)(r + 1) * (uint64_t)I];
                if (v1 > curMax) { curMax = v1; curArg = r + 1; }

                float v2 = p[(uint64_t)(r + 2) * (uint64_t)I];
                if (v2 > curMax) { curMax = v2; curArg = r + 2; }

                float v3 = p[(uint64_t)(r + 3) * (uint64_t)I];
                if (v3 > curMax) { curMax = v3; curArg = r + 3; }
            }
            for (; r < K; ++r) {
                float v = p[(uint64_t)r * (uint64_t)I];
                if (v > curMax) { curMax = v; curArg = r; }
            }

            const uint64_t outOffset = (uint64_t)b * (uint64_t)innerDim_ + (uint64_t)i;
            yPtr_[outOffset] = (int64_t)curArg;
        }
    }

private:
    __gm__ const float* xPtr_{nullptr};
    __gm__ int64_t* yPtr_{nullptr};
    uint32_t batch_{0};
    uint32_t reduceDim_{0};
    uint32_t innerDim_{0};
    uint32_t tileInner_{0};
    uint32_t tilesPerBatch_{0};
};

extern "C" __global__ __aicore__ void argmax_over_a_dimension_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgmaxOverADimensionCustom op;
    op.Init(x, y,
            tiling_data.batch, tiling_data.reduceDim, tiling_data.innerDim,
            tiling_data.tileInner, tiling_data.tilesPerBatch);
    op.Process();
}
