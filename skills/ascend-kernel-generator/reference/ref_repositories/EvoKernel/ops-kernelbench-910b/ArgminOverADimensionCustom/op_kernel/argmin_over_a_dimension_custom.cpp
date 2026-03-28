
#include "kernel_operator.h"
#include <cstdint>

// Specialized contract:
// x: [128,4096,4095] float32 contiguous, reduce dim=1 => y: [128,4095] int64 indices
//
// Mapping:
// Flatten output (b,i) -> out = b*I + i, outElems = B*I.
// Shard contiguous ranges across cores with GetBlockNum/GetBlockIdx.
//
// Tie-breaking:
// strict less-than => first occurrence wins (matches PyTorch default behavior).

class KernelArgminOverADimensionCustom {
public:
    __aicore__ inline KernelArgminOverADimensionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t /*totalX*/, uint32_t /*totalY*/,
                               uint32_t batch, uint32_t reduceDim, uint32_t innerDim,
                               uint32_t unrollR)
    {
        batch_ = batch;
        reduceDim_ = reduceDim;
        innerDim_ = innerDim;
        unrollR_ = (unrollR == 0 ? 1u : unrollR);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        yGm_.SetGlobalBuffer((__gm__ int64_t*)y);
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0 || reduceDim_ == 0 || innerDim_ == 0) return;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        if (blockNum == 0) return;

        const uint64_t I = (uint64_t)innerDim_;
        const uint64_t K = (uint64_t)reduceDim_;
        const uint64_t outElems = (uint64_t)batch_ * I;

        // Robust default tiling: ceil-div outputs across cores.
        const uint64_t chunk = (outElems + (uint64_t)blockNum - 1ULL) / (uint64_t)blockNum;
        const uint64_t start = (uint64_t)blockIdx * chunk;
        uint64_t end = start + chunk;
        if (end > outElems) end = outElems;

        for (uint64_t out = start; out < end; ++out) {
            const uint32_t b = (uint32_t)(out / I);
            const uint32_t i = (uint32_t)(out - (uint64_t)b * I);

            const uint64_t base = (uint64_t)b * K * I + (uint64_t)i; // x[b,0,i]
            const uint64_t strideR = I;

            float bestV = xGm_.GetValue(base);
            uint32_t bestIdx = 0;

            uint32_t r = 1;
            uint64_t p = base + strideR;

            if (unrollR_ == 8 && (reduceDim_ % 8 == 0)) {
#pragma unroll
                for (; r + 7 < reduceDim_; r += 8) {
                    const float v0 = xGm_.GetValue(p);
                    const float v1 = xGm_.GetValue(p + strideR);
                    const float v2 = xGm_.GetValue(p + strideR * 2);
                    const float v3 = xGm_.GetValue(p + strideR * 3);
                    const float v4 = xGm_.GetValue(p + strideR * 4);
                    const float v5 = xGm_.GetValue(p + strideR * 5);
                    const float v6 = xGm_.GetValue(p + strideR * 6);
                    const float v7 = xGm_.GetValue(p + strideR * 7);

                    if (v0 < bestV) { bestV = v0; bestIdx = r + 0; }
                    if (v1 < bestV) { bestV = v1; bestIdx = r + 1; }
                    if (v2 < bestV) { bestV = v2; bestIdx = r + 2; }
                    if (v3 < bestV) { bestV = v3; bestIdx = r + 3; }
                    if (v4 < bestV) { bestV = v4; bestIdx = r + 4; }
                    if (v5 < bestV) { bestV = v5; bestIdx = r + 5; }
                    if (v6 < bestV) { bestV = v6; bestIdx = r + 6; }
                    if (v7 < bestV) { bestV = v7; bestIdx = r + 7; }

                    p += strideR * 8;
                }
            }

            for (; r < reduceDim_; ++r) {
                const float v = xGm_.GetValue(p);
                if (v < bestV) { bestV = v; bestIdx = r; }
                p += strideR;
            }

            yGm_.SetValue(out, (int64_t)bestIdx);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<int64_t> yGm_;
    uint32_t batch_{0};
    uint32_t reduceDim_{0};
    uint32_t innerDim_{0};
    uint32_t unrollR_{1};
};

extern "C" __global__ __aicore__ void argmin_over_a_dimension_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgminOverADimensionCustom op;
    op.Init(x, y,
            tiling_data.totalX, tiling_data.totalY,
            tiling_data.batch, tiling_data.reduceDim, tiling_data.innerDim,
            tiling_data.unrollR);
    op.Process();
}
