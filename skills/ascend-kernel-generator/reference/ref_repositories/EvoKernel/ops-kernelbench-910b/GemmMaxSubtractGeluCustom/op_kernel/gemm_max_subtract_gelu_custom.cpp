
#include "kernel_operator.h"

// Specialized: output y is identically zero.
// Optimization: vectorized Duplicate + DataCopy using a single reusable UB buffer per block.

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelGemmMaxSubtractGeluCustom {
public:
    __aicore__ inline KernelGemmMaxSubtractGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR y, uint32_t totalOutElems, uint32_t tileElems)
    {
        totalOutElems_ = totalOutElems;
        tileElems_ = (tileElems == 0) ? 256 : tileElems;

        yGm_.SetGlobalBuffer((__gm__ float*)y);

        // Allocate one reusable UB buffer per block (no queueing to avoid pipeline gaps).
        pipe_.InitBuffer(outBuf_, tileElems_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t total = totalOutElems_;
        if (total == 0) return;

        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();

        const uint32_t chunk = (total + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * chunk;
        uint32_t end = start + chunk;
        if (end > total) end = total;
        if (start >= end) return;

        const uint32_t myLen = end - start;
        const uint32_t tiles = CeilDivU32(myLen, tileElems_);

        AscendC::LocalTensor<float> out = outBuf_.Get<float>();

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t off = t * tileElems_;
            const uint32_t len = (off + tileElems_ <= myLen) ? tileElems_ : (myLen - off);
            const uint32_t gmBase = start + off;

            AscendC::Duplicate(out, 0.0f, (int32_t)len);
            AscendC::DataCopy(yGm_[gmBase], out, len);
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf_;
    AscendC::GlobalTensor<float> yGm_;
    uint32_t totalOutElems_{0};
    uint32_t tileElems_{4096};
};

extern "C" __global__ __aicore__ void gemm_max_subtract_gelu_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)x;
    (void)w;
    (void)b;
    GET_TILING_DATA(td, tiling);

    KernelGemmMaxSubtractGeluCustom op;
    op.Init(y, td.totalOutElems, td.tileElems);
    op.Process();
}
