
#include "kernel_operator.h"

class KernelDAModuleCustom {
public:
    __aicore__ inline KernelDAModuleCustom() {}

    __aicore__ inline void Init(GM_ADDR p_out, GM_ADDR c_out, GM_ADDR y,
                               uint32_t bs, uint32_t c, uint32_t hw,
                               uint32_t totalRows, uint32_t rowsPerCore,
                               uint32_t cTile)
    {
        this->bs = bs;
        this->c = c;
        this->hw = hw;
        this->totalRows = totalRows;
        this->rowsPerCore = rowsPerCore;
        this->cTile = cTile;

        const uint64_t totalP = (uint64_t)bs * (uint64_t)hw * (uint64_t)c;
        const uint64_t totalC = (uint64_t)bs * (uint64_t)c  * (uint64_t)hw;

        pGm.SetGlobalBuffer((__gm__ float*)p_out, totalP); // [bs, hw, c]
        cGm.SetGlobalBuffer((__gm__ float*)c_out, totalC); // [bs, c, hw]
        yGm.SetGlobalBuffer((__gm__ float*)y,     totalC); // write as flat [bs, c, hw] (matches contiguous NCHW storage)

        // UB buffers: one p row, one c segment, one out segment
        pipe.InitBuffer(pRowBuf, c * sizeof(float));             // variable, but c is known at runtime; allocate max needed
        pipe.InitBuffer(cSegBuf, cTile * sizeof(float));
        pipe.InitBuffer(outSegBuf, cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t rowStart = core * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > totalRows) rowEnd = totalRows;

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            const uint32_t b = r / hw;
            const uint32_t pos = r - b * hw;

            // Load p_out[b, pos, :] contiguously into UB once
            AscendC::LocalTensor<float> pRow = pRowBuf.Get<float>();
            const uint64_t pBase = ((uint64_t)b * (uint64_t)hw + (uint64_t)pos) * (uint64_t)c;
            AscendC::DataCopy(pRow, pGm[pBase], c);

            // Walk channels in tiles; c_out and y are contiguous for fixed (b,pos) in channel-major
            for (uint32_t c0 = 0; c0 < c; c0 += cTile) {
                const uint32_t cLen = (c0 + cTile <= c) ? cTile : (c - c0);

                AscendC::LocalTensor<float> cSeg = cSegBuf.Get<float>();
                AscendC::LocalTensor<float> outSeg = outSegBuf.Get<float>();

                // Load c_out[b, c0:c0+cLen, pos] which is a strided gather in GM if we view [bs,c,hw],
                // but in flattened layout it is still regular with stride hw between consecutive channels.
                // We avoid 2D stride DataCopy (fragile) and instead do scalar reads from GM for hw=49? No.
                // Instead, we copy each channel element with small scalar loop but only for cLen (<=128),
                // and we eliminated the heavy div/mod and p-gather from GM; remaining scalar cost is much smaller.
                const uint64_t cBase = ((uint64_t)b * (uint64_t)c + (uint64_t)c0) * (uint64_t)hw + (uint64_t)pos;
                for (uint32_t i = 0; i < cLen; ++i) {
                    float cv = cGm.GetValue(cBase + (uint64_t)i * (uint64_t)hw);
                    cSeg.SetValue(i, cv);
                }

                // Build p segment transposed value: p_out[b,pos,c0+i] corresponds to output channel (c0+i)
                for (uint32_t i = 0; i < cLen; ++i) {
                    float pv = pRow.GetValue(c0 + i);
                    outSeg.SetValue(i, pv);
                }

                AscendC::Add(outSeg, outSeg, cSeg, cLen);

                const uint64_t yBase = ((uint64_t)b * (uint64_t)c + (uint64_t)c0) * (uint64_t)hw + (uint64_t)pos;
                // Store back with stride hw between channels: scalar loop store (cLen<=128)
                for (uint32_t i = 0; i < cLen; ++i) {
                    yGm.SetValue(yBase + (uint64_t)i * (uint64_t)hw, outSeg.GetValue(i));
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> pRowBuf;
    AscendC::TBuf<> cSegBuf;
    AscendC::TBuf<> outSegBuf;

    AscendC::GlobalTensor<float> pGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t bs, c, hw;
    uint32_t totalRows, rowsPerCore;
    uint32_t cTile;
};

extern "C" __global__ __aicore__ void da_module_custom(GM_ADDR p_out, GM_ADDR c_out, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDAModuleCustom op;
    op.Init(p_out, c_out, y,
            tiling_data.bs, tiling_data.c, tiling_data.hw,
            tiling_data.totalRows, tiling_data.rowsPerCore,
            tiling_data.cTile);
    op.Process();
}
