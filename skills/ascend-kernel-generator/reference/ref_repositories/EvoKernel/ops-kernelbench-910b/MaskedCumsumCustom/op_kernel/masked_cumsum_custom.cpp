
#include "kernel_operator.h"

class KernelMaskedCumsum {
public:
    __aicore__ inline KernelMaskedCumsum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                               uint32_t rank, int32_t dim,
                               uint32_t outerSize, uint32_t axisSize,
                               uint32_t totalElems, uint32_t tileElems)
    {
        (void)rank;
        (void)dim; // specialized for last dimension by python bind
        this->outerSize = outerSize;
        this->axisSize = axisSize;
        this->totalElems = totalElems;
        this->tileElems = tileElems;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
        mGm.SetGlobalBuffer((__gm__ uint8_t*)mask, totalElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);

        // Double-buffer UB: x0/x1, m0/m1, y0/y1
        pipe.InitBuffer(xBuf0, tileElems * sizeof(float));
        pipe.InitBuffer(xBuf1, tileElems * sizeof(float));
        pipe.InitBuffer(mBuf0, tileElems * sizeof(uint8_t));
        pipe.InitBuffer(mBuf1, tileElems * sizeof(uint8_t));
        pipe.InitBuffer(yBuf0, tileElems * sizeof(float));
        pipe.InitBuffer(yBuf1, tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (outerSize == 0 || axisSize == 0) return;

        const uint32_t core = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        for (uint32_t row = core; row < outerSize; row += coreNum) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void LoadTile(uint32_t base, uint32_t c, uint32_t n, bool ping)
    {
        if (!ping) {
            AscendC::LocalTensor<float> xLocal = xBuf0.Get<float>();
            AscendC::LocalTensor<uint8_t> mLocal = mBuf0.Get<uint8_t>();
            AscendC::DataCopy(xLocal, xGm[base + c], n);
            AscendC::DataCopy(mLocal, mGm[base + c], n);
        } else {
            AscendC::LocalTensor<float> xLocal = xBuf1.Get<float>();
            AscendC::LocalTensor<uint8_t> mLocal = mBuf1.Get<uint8_t>();
            AscendC::DataCopy(xLocal, xGm[base + c], n);
            AscendC::DataCopy(mLocal, mGm[base + c], n);
        }
    }

    __aicore__ inline void StoreTile(uint32_t base, uint32_t c, uint32_t n, bool ping)
    {
        if (!ping) {
            AscendC::LocalTensor<float> yLocal = yBuf0.Get<float>();
            AscendC::DataCopy(yGm[base + c], yLocal, n);
        } else {
            AscendC::LocalTensor<float> yLocal = yBuf1.Get<float>();
            AscendC::DataCopy(yGm[base + c], yLocal, n);
        }
    }

    __aicore__ inline void ComputeTile(uint32_t n, float &acc, bool ping)
    {
        AscendC::LocalTensor<float> xLocal = (!ping) ? xBuf0.Get<float>() : xBuf1.Get<float>();
        AscendC::LocalTensor<uint8_t> mLocal = (!ping) ? mBuf0.Get<uint8_t>() : mBuf1.Get<uint8_t>();
        AscendC::LocalTensor<float> yLocal = (!ping) ? yBuf0.Get<float>() : yBuf1.Get<float>();

        // MTE->V sync for this tile.
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t i = 0;
        // 8-way unroll to reduce scalar loop overhead.
        for (; i + 7 < n; i += 8) {
            uint8_t m0 = mLocal(i + 0); float x0 = xLocal(i + 0);
            uint8_t m1 = mLocal(i + 1); float x1 = xLocal(i + 1);
            uint8_t m2 = mLocal(i + 2); float x2 = xLocal(i + 2);
            uint8_t m3 = mLocal(i + 3); float x3 = xLocal(i + 3);
            uint8_t m4 = mLocal(i + 4); float x4 = xLocal(i + 4);
            uint8_t m5 = mLocal(i + 5); float x5 = xLocal(i + 5);
            uint8_t m6 = mLocal(i + 6); float x6 = xLocal(i + 6);
            uint8_t m7 = mLocal(i + 7); float x7 = xLocal(i + 7);

            // Avoid forbidden float casts from uint8 in hot loop by selecting.
            float v0 = (m0 != 0) ? x0 : 0.0f; acc += v0; yLocal(i + 0) = acc;
            float v1 = (m1 != 0) ? x1 : 0.0f; acc += v1; yLocal(i + 1) = acc;
            float v2 = (m2 != 0) ? x2 : 0.0f; acc += v2; yLocal(i + 2) = acc;
            float v3 = (m3 != 0) ? x3 : 0.0f; acc += v3; yLocal(i + 3) = acc;
            float v4 = (m4 != 0) ? x4 : 0.0f; acc += v4; yLocal(i + 4) = acc;
            float v5 = (m5 != 0) ? x5 : 0.0f; acc += v5; yLocal(i + 5) = acc;
            float v6 = (m6 != 0) ? x6 : 0.0f; acc += v6; yLocal(i + 6) = acc;
            float v7 = (m7 != 0) ? x7 : 0.0f; acc += v7; yLocal(i + 7) = acc;
        }
        for (; i < n; ++i) {
            float v = (mLocal(i) != 0) ? xLocal(i) : 0.0f;
            acc += v;
            yLocal(i) = acc;
        }

        // V->MTE store sync.
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint32_t base = row * axisSize;
        float acc = 0.0f;

        uint32_t c = 0;
        if (axisSize == 0) return;

        // Prefetch first tile
        uint32_t n0 = axisSize;
        if (n0 > tileElems) n0 = tileElems;
        bool ping = false;
        LoadTile(base, c, n0, ping);

        while (c < axisSize) {
            uint32_t n = axisSize - c;
            if (n > tileElems) n = tileElems;

            // Prefetch next tile (overlap with compute)
            uint32_t cNext = c + n;
            uint32_t nNext = 0;
            if (cNext < axisSize) {
                nNext = axisSize - cNext;
                if (nNext > tileElems) nNext = tileElems;
                LoadTile(base, cNext, nNext, !ping);
            }

            ComputeTile(n, acc, ping);
            StoreTile(base, c, n, ping);

            c = cNext;
            ping = !ping;
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf1;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<uint8_t> mGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t outerSize = 0;
    uint32_t axisSize = 0;
    uint32_t totalElems = 0;
    uint32_t tileElems = 0;
};

extern "C" __global__ __aicore__ void masked_cumsum_custom(GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedCumsum op;
    op.Init(x, mask, y,
            tiling_data.rank, tiling_data.dim,
            tiling_data.outerSize, tiling_data.axisSize,
            tiling_data.totalElems, tiling_data.tileElems);
    op.Process();
}
