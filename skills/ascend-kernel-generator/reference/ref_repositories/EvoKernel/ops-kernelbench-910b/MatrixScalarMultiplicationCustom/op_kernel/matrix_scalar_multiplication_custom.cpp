
#include <cstdint>
#include "kernel_operator.h"

using AscendC::TPosition;

__aicore__ inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1ULL) / b; }

class KernelMatrixScalarMul {
public:
    __aicore__ inline KernelMatrixScalarMul() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR s, GM_ADDR c,
                               uint64_t total, uint32_t tileLen,
                               AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        total_ = total;
        tileLen_ = (tileLen == 0U) ? 8192U : tileLen;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), total_);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), total_);

        // Load scalar once per core from a 1-element GM tensor.
        AscendC::GlobalTensor<float> sGm;
        sGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(s), 1ULL);
        scalar_ = sGm.GetValue(0ULL);

        // Depth=1 queues: stable and low-UB footprint.
        pipe_->InitBuffer(qIn_,  1, static_cast<uint64_t>(tileLen_) * sizeof(float));
        pipe_->InitBuffer(qOut_, 1, static_cast<uint64_t>(tileLen_) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (total_ == 0ULL) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t bnum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (bnum == 0U) bnum = 1U;

        const uint64_t tileNum = CeilDivU64(total_, static_cast<uint64_t>(tileLen_));
        for (uint64_t tileId = static_cast<uint64_t>(bid); tileId < tileNum; tileId += static_cast<uint64_t>(bnum)) {
            const uint64_t start = tileId * static_cast<uint64_t>(tileLen_);
            if (start >= total_) continue;

            uint32_t len = tileLen_;
            const uint64_t end = start + static_cast<uint64_t>(len);
            if (end > total_) {
                len = static_cast<uint32_t>(total_ - start);
            }
            if (len == 0U) continue;

            // GM -> UB
            AscendC::LocalTensor<float> in = qIn_.AllocTensor<float>();
            AscendC::DataCopy(in, aGm_[start], len);
            qIn_.EnQue(in);

            // UB compute
            AscendC::LocalTensor<float> x = qIn_.DeQue<float>();
            AscendC::LocalTensor<float> y = qOut_.AllocTensor<float>();
            AscendC::Muls(y, x, scalar_, len);
            qOut_.EnQue(y);
            qIn_.FreeTensor(x);

            // UB -> GM
            AscendC::LocalTensor<float> out = qOut_.DeQue<float>();
            AscendC::DataCopy(cGm_[start], out, len);
            qOut_.FreeTensor(out);
        }
    }

private:
    AscendC::TPipe *pipe_ {nullptr};
    AscendC::TQue<TPosition::VECIN,  1> qIn_;
    AscendC::TQue<TPosition::VECOUT, 1> qOut_;
    AscendC::GlobalTensor<float> aGm_;
    AscendC::GlobalTensor<float> cGm_;
    uint64_t total_ {0ULL};
    uint32_t tileLen_ {0U};
    float scalar_ {0.0f};
};

extern "C" __global__ __aicore__ void matrix_scalar_multiplication_custom(
    GM_ADDR a, GM_ADDR s, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    KernelMatrixScalarMul op;
    op.Init(a, s, c,
            tilingData.totalLength,
            tilingData.tileLength,
            &pipe);
    op.Process();
}
