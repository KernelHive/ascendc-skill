
#include "kernel_operator.h"

class KernelCumprod {
public:
    static constexpr uint32_t TILE_ELEMS = 8192;
    static constexpr uint32_t QUEUE_DEPTH = 2;

    __aicore__ inline KernelCumprod() {}

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

        pipe_.InitBuffer(inQueueX_, QUEUE_DEPTH, TILE_ELEMS * sizeof(float));
        pipe_.InitBuffer(outQueueY_, QUEUE_DEPTH, TILE_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t rows = rows_;
        const uint32_t cols = cols_;
        if (rows == 0U || cols == 0U) return;

        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t rowsPerCore = rowsPerCore_;

        const uint32_t startRow = core * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > rows) endRow = rows;
        if (startRow >= rows) return;

        for (uint32_t r = startRow; r < endRow; ++r) {
            ProcessRow(r, cols);
        }
    }

private:
    __aicore__ inline void PrefetchTile(uint32_t gmOffset, uint32_t n)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm_[gmOffset], n);
        inQueueX_.EnQue<float>(xLocal);
    }

    __aicore__ inline void ProcessRow(uint32_t row, uint32_t cols)
    {
        const uint32_t base = row * cols;

        float carry = 1.0f;
        uint32_t c = 0;

        // Prime pipeline
        if (c < cols) {
            uint32_t n0 = cols - c;
            if (n0 > TILE_ELEMS) n0 = TILE_ELEMS;
            PrefetchTile(base + c, n0);
        }

        while (c < cols) {
            uint32_t n = cols - c;
            if (n > TILE_ELEMS) n = TILE_ELEMS;

            const uint32_t cNext = c + n;

            // Prefetch next tile
            if (cNext < cols) {
                uint32_t nNext = cols - cNext;
                if (nNext > TILE_ELEMS) nNext = TILE_ELEMS;
                PrefetchTile(base + cNext, nNext);
            }

            // Current tile
            AscendC::LocalTensor<float> xTile = inQueueX_.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY_.AllocTensor<float>();

            // Single handoff: MTE writes yLocal from xTile; avoids in-place hazards.
            AscendC::DataCopy(yLocal, xTile, n);

            // Order MTE->scalar reads/writes for yLocal.
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t i = 0;
            // Keep unroll (scalar dependency) but avoid extra barriers.
            for (; i + 16U <= n; i += 16U) {
                float v0 = yLocal(i + 0);  carry *= v0;  yLocal(i + 0)  = carry;
                float v1 = yLocal(i + 1);  carry *= v1;  yLocal(i + 1)  = carry;
                float v2 = yLocal(i + 2);  carry *= v2;  yLocal(i + 2)  = carry;
                float v3 = yLocal(i + 3);  carry *= v3;  yLocal(i + 3)  = carry;
                float v4 = yLocal(i + 4);  carry *= v4;  yLocal(i + 4)  = carry;
                float v5 = yLocal(i + 5);  carry *= v5;  yLocal(i + 5)  = carry;
                float v6 = yLocal(i + 6);  carry *= v6;  yLocal(i + 6)  = carry;
                float v7 = yLocal(i + 7);  carry *= v7;  yLocal(i + 7)  = carry;
                float v8 = yLocal(i + 8);  carry *= v8;  yLocal(i + 8)  = carry;
                float v9 = yLocal(i + 9);  carry *= v9;  yLocal(i + 9)  = carry;
                float v10 = yLocal(i + 10); carry *= v10; yLocal(i + 10) = carry;
                float v11 = yLocal(i + 11); carry *= v11; yLocal(i + 11) = carry;
                float v12 = yLocal(i + 12); carry *= v12; yLocal(i + 12) = carry;
                float v13 = yLocal(i + 13); carry *= v13; yLocal(i + 13) = carry;
                float v14 = yLocal(i + 14); carry *= v14; yLocal(i + 14) = carry;
                float v15 = yLocal(i + 15); carry *= v15; yLocal(i + 15) = carry;
            }
            for (; i < n; ++i) {
                float v = yLocal(i);
                carry *= v;
                yLocal(i) = carry;
            }

            outQueueY_.EnQue<float>(yLocal);
            inQueueX_.FreeTensor(xTile);

            // Store
            AscendC::LocalTensor<float> yTile = outQueueY_.DeQue<float>();
            AscendC::DataCopy(yGm_[base + c], yTile, n);
            outQueueY_.FreeTensor(yTile);

            c = cNext;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueY_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t rows_;
    uint32_t cols_;
    uint32_t totalElems_;
    uint32_t rowsPerCore_;
};

extern "C" __global__ __aicore__ void cumprod_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumprod op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols, tiling_data.totalElems, tiling_data.rowsPerCore);
    op.Process();
}
