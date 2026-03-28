
#include "kernel_operator.h"

class KernelCumsumInclusive {
public:
    // Keep UB usage modest and stable: 2 * 4096 floats = 32KB total.
    static constexpr uint32_t TILE_ELEMS = 4096;

    __aicore__ inline KernelCumsumInclusive() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t rows, uint32_t cols,
                               uint32_t totalElems, uint32_t rowsPerCore)
    {
        this->rows = rows;
        this->cols = cols;
        this->totalElems = totalElems;
        this->rowsPerCore = rowsPerCore;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);

        // Fixed UB buffers reused for all tiles; avoids queue overhead.
        pipe.InitBuffer(xUbBuf, TILE_ELEMS * sizeof(float));
        pipe.InitBuffer(yUbBuf, TILE_ELEMS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rows == 0 || cols == 0) return;

        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t startRow = core * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > rows) endRow = rows;
        if (startRow >= rows) return;

        // Rolling base offsets to reduce repeated mul in the row loop.
        uint32_t base = startRow * cols;
        for (uint32_t r = startRow; r < endRow; ++r) {
            ProcessRowBase(base);
            base += cols;
        }
    }

private:
    __aicore__ inline void ScanUnrolled16(const AscendC::LocalTensor<float> &x,
                                         const AscendC::LocalTensor<float> &y,
                                         uint32_t n, float &carry)
    {
        float acc = carry;

        uint32_t i = 0;
        for (; i + 16 <= n; i += 16) {
            // Load to registers first; then do a straight-line dependency chain.
            float v0  = x(i + 0);
            float v1  = x(i + 1);
            float v2  = x(i + 2);
            float v3  = x(i + 3);
            float v4  = x(i + 4);
            float v5  = x(i + 5);
            float v6  = x(i + 6);
            float v7  = x(i + 7);
            float v8  = x(i + 8);
            float v9  = x(i + 9);
            float v10 = x(i + 10);
            float v11 = x(i + 11);
            float v12 = x(i + 12);
            float v13 = x(i + 13);
            float v14 = x(i + 14);
            float v15 = x(i + 15);

            acc += v0;  y(i + 0)  = acc;
            acc += v1;  y(i + 1)  = acc;
            acc += v2;  y(i + 2)  = acc;
            acc += v3;  y(i + 3)  = acc;
            acc += v4;  y(i + 4)  = acc;
            acc += v5;  y(i + 5)  = acc;
            acc += v6;  y(i + 6)  = acc;
            acc += v7;  y(i + 7)  = acc;
            acc += v8;  y(i + 8)  = acc;
            acc += v9;  y(i + 9)  = acc;
            acc += v10; y(i + 10) = acc;
            acc += v11; y(i + 11) = acc;
            acc += v12; y(i + 12) = acc;
            acc += v13; y(i + 13) = acc;
            acc += v14; y(i + 14) = acc;
            acc += v15; y(i + 15) = acc;
        }

        for (; i < n; ++i) {
            acc += x(i);
            y(i) = acc;
        }
        carry = acc;
    }

    __aicore__ inline void ProcessRowBase(uint32_t base)
    {
        AscendC::LocalTensor<float> xUb = xUbBuf.Get<float>();
        AscendC::LocalTensor<float> yUb = yUbBuf.Get<float>();

        float carry = 0.0f;

        uint32_t c = 0;
        // Full tiles.
        while (c + TILE_ELEMS <= cols) {
            AscendC::DataCopy(xUb, xGm[base + c], TILE_ELEMS);
            ScanUnrolled16(xUb, yUb, TILE_ELEMS, carry);
            AscendC::DataCopy(yGm[base + c], yUb, TILE_ELEMS);
            c += TILE_ELEMS;
        }

        // Tail.
        const uint32_t rem = cols - c;
        if (rem > 0) {
            AscendC::DataCopy(xUb, xGm[base + c], rem);
            ScanUnrolled16(xUb, yUb, rem, carry);
            AscendC::DataCopy(yGm[base + c], yUb, rem);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xUbBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yUbBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t totalElems = 0;
    uint32_t rowsPerCore = 0;
};

extern "C" __global__ __aicore__ void cumsum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumsumInclusive op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols, tiling_data.totalElems, tiling_data.rowsPerCore);
    op.Process();
}
