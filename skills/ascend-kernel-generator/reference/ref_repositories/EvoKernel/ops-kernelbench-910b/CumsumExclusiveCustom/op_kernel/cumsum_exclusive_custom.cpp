
#include "kernel_operator.h"

class KernelCumsumExclusive {
public:
    // UB footprint: 2 * 4096 * 4B = 32KB per core.
    static constexpr uint32_t TILE_ELEMS = 4096;

    __aicore__ inline KernelCumsumExclusive() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols,
                               uint32_t totalElems, uint32_t rowsPerCore)
    {
        rows_ = rows;
        cols_ = cols;
        totalElems_ = totalElems;
        rowsPerCore_ = rowsPerCore;

        xGm_.SetGlobalBuffer((__gm__ float*)x, totalElems_);
        yGm_.SetGlobalBuffer((__gm__ float*)y, totalElems_);

        // Reusable UB buffers allocated once per core; avoids queue overhead.
        pipe_.InitBuffer(bufX_, TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(bufY_, TILE_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0 || cols_ == 0) return;

        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t startRow = core * rowsPerCore_;
        uint32_t endRow = startRow + rowsPerCore_;
        if (endRow > rows_) endRow = rows_;
        if (startRow >= rows_) return;

        for (uint32_t r = startRow; r < endRow; ++r) {
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
        uint32_t c = 0;

        while (c < cols_) {
            uint32_t n = cols_ - c;
            if (n > TILE_ELEMS) n = TILE_ELEMS;

            // GM -> UB
            AscendC::DataCopy(xUb, xGm_[base + c], n);
            AscendC::PipeBarrier<PIPE_V>();

            // Exclusive scan: y[i] = carry; carry += x[i]
            uint32_t i = 0;

            // Unroll by 8 to reduce scalar loop overhead.
            for (; i + 7 < n; i += 8) {
                float v0 = xUb.GetValue(i + 0);
                float v1 = xUb.GetValue(i + 1);
                float v2 = xUb.GetValue(i + 2);
                float v3 = xUb.GetValue(i + 3);
                float v4 = xUb.GetValue(i + 4);
                float v5 = xUb.GetValue(i + 5);
                float v6 = xUb.GetValue(i + 6);
                float v7 = xUb.GetValue(i + 7);

                yUb.SetValue(i + 0, carry); carry += v0;
                yUb.SetValue(i + 1, carry); carry += v1;
                yUb.SetValue(i + 2, carry); carry += v2;
                yUb.SetValue(i + 3, carry); carry += v3;
                yUb.SetValue(i + 4, carry); carry += v4;
                yUb.SetValue(i + 5, carry); carry += v5;
                yUb.SetValue(i + 6, carry); carry += v6;
                yUb.SetValue(i + 7, carry); carry += v7;
            }
            for (; i < n; ++i) {
                float v = xUb.GetValue(i);
                yUb.SetValue(i, carry);
                carry += v;
            }

            AscendC::PipeBarrier<PIPE_V>();
            // UB -> GM
            AscendC::DataCopy(yGm_[base + c], yUb, n);

            c += n;
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
    uint32_t rowsPerCore_ = 0;
};

extern "C" __global__ __aicore__ void cumsum_exclusive_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumsumExclusive op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols, tiling_data.totalElems, tiling_data.rowsPerCore);
    op.Process();
}
