
#include "kernel_operator.h"

class KernelCumsumReverse {
public:
    // UB footprint moderate and deterministic.
    static constexpr uint32_t TILE_ELEMS = 4096;

    __aicore__ inline KernelCumsumReverse() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols,
                               uint32_t totalElems)
    {
        rows_ = rows;
        cols_ = cols;
        totalElems_ = totalElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x, totalElems_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalElems_);

        // Two reusable UB buffers allocated once per core.
        pipe_.InitBuffer(bufX_, TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(bufY_, TILE_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0 || cols_ == 0) return;

        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blockDim = static_cast<uint32_t>(AscendC::GetBlockNum());

        // Cyclic row assignment improves load balance and occupancy.
        for (uint32_t r = core; r < rows_; r += blockDim) {
            ProcessRow(r);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint32_t base = row * cols_;

        AscendC::LocalTensor<float> xUb = bufX_.Get<float>();
        AscendC::LocalTensor<float> yUb = bufY_.Get<float>();

        float carry = 0.0f;
        uint32_t remaining = cols_;

        while (remaining > 0) {
            uint32_t tileStart;
            uint32_t n;
            if (remaining > TILE_ELEMS) {
                n = TILE_ELEMS;
                tileStart = remaining - TILE_ELEMS;
            } else {
                n = remaining;
                tileStart = 0;
            }

            // GM -> UB (input tile)
            AscendC::DataCopy(xUb, xGm_[base + tileStart], n);
            AscendC::PipeBarrier<PIPE_V>(); // ensure xUb is ready

            // Scalar UB compute: minimize extra barriers inside compute loop.
            int32_t i = static_cast<int32_t>(n) - 1;

            // Unroll by 8 to reduce loop overhead.
            for (; i >= 7; i -= 8) {
                float v0 = xUb.GetValue(i);
                float v1 = xUb.GetValue(i - 1);
                float v2 = xUb.GetValue(i - 2);
                float v3 = xUb.GetValue(i - 3);
                float v4 = xUb.GetValue(i - 4);
                float v5 = xUb.GetValue(i - 5);
                float v6 = xUb.GetValue(i - 6);
                float v7 = xUb.GetValue(i - 7);

                carry += v0; yUb.SetValue(i, carry);
                carry += v1; yUb.SetValue(i - 1, carry);
                carry += v2; yUb.SetValue(i - 2, carry);
                carry += v3; yUb.SetValue(i - 3, carry);
                carry += v4; yUb.SetValue(i - 4, carry);
                carry += v5; yUb.SetValue(i - 5, carry);
                carry += v6; yUb.SetValue(i - 6, carry);
                carry += v7; yUb.SetValue(i - 7, carry);
            }
            for (; i >= 0; --i) {
                float v = xUb.GetValue(i);
                carry += v;
                yUb.SetValue(i, carry);
            }

            AscendC::PipeBarrier<PIPE_V>(); // ensure yUb writes visible to MTE
            // UB -> GM (output tile)
            AscendC::DataCopy(yGm_[base + tileStart], yUb, n);

            remaining = tileStart;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufY_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t totalElems_ = 0;
};

extern "C" __global__ __aicore__ void cumsum_reverse_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumsumReverse op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols, tiling_data.totalElems);
    op.Process();
}
