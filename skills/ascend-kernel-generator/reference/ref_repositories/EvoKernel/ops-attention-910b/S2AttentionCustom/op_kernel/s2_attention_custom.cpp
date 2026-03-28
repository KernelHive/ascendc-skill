
#include "kernel_operator.h"
#include <cstdint>

class KernelS2Attention {
public:
    __aicore__ inline KernelS2Attention() {}

    __aicore__ inline void Init(GM_ADDR attn, GM_ADDR x_all, GM_ADDR y,
                               uint32_t B, uint32_t K, uint32_t H, uint32_t W, uint32_t C,
                               uint32_t HW, uint32_t rows, uint32_t rowsPerCore, uint32_t cTile)
    {
        this->B = B;
        this->K = K; // 3
        this->H = H;
        this->W = W;
        this->C = C;
        this->HW = HW;
        this->rows = rows;
        this->rowsPerCore = rowsPerCore;
        this->cTile = cTile;

        const uint64_t attnSize = (uint64_t)B * (uint64_t)K * (uint64_t)C; // [B,3,C]
        const uint64_t xSize    = (uint64_t)B * (uint64_t)K * (uint64_t)HW * (uint64_t)C; // [B,3,HW,C]
        const uint64_t ySize    = (uint64_t)B * (uint64_t)HW * (uint64_t)C; // [B,HW,C]

        attnGm.SetGlobalBuffer((__gm__ float*)attn, attnSize);
        xGm.SetGlobalBuffer((__gm__ float*)x_all, xSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB layout: x0,x1,x2,a0,out each of size cTile floats.
        pipe.InitBuffer(ubuf, (uint64_t)5 * (uint64_t)this->cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();
        if (coreNum == 0) return;

        const uint32_t rowStart = coreId * this->rowsPerCore;
        if (rowStart >= this->rows) return;
        uint32_t rowEnd = rowStart + this->rowsPerCore;
        if (rowEnd > this->rows) rowEnd = this->rows;

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            ComputeRow(r);
        }
    }

private:
    __aicore__ inline uint64_t AttnBase(uint32_t b) const
    {
        // base for attn[b,0,0]
        return (uint64_t)b * (uint64_t)this->K * (uint64_t)this->C;
    }

    __aicore__ inline uint64_t XBase(uint32_t b, uint32_t k, uint32_t hw) const
    {
        // base for x_all[b,k,hw,0] with flatten [B,K,HW,C]
        return (((uint64_t)b * (uint64_t)this->K + (uint64_t)k) * (uint64_t)this->HW + (uint64_t)hw) * (uint64_t)this->C;
    }

    __aicore__ inline uint64_t YBase(uint32_t b, uint32_t hw) const
    {
        // base for y[b,hw,0] flatten [B,HW,C]
        return ((uint64_t)b * (uint64_t)this->HW + (uint64_t)hw) * (uint64_t)this->C;
    }

    __aicore__ inline void ComputeRow(uint32_t row)
    {
        const uint32_t b  = row / this->HW;
        const uint32_t hw = row - b * this->HW;

        AscendC::LocalTensor<float> base = ubuf.Get<float>();
        AscendC::LocalTensor<float> x0L  = base;                       // cTile
        AscendC::LocalTensor<float> x1L  = base[this->cTile];          // cTile
        AscendC::LocalTensor<float> x2L  = base[2 * this->cTile];      // cTile
        AscendC::LocalTensor<float> a0L  = base[3 * this->cTile];      // cTile
        AscendC::LocalTensor<float> outL = base[4 * this->cTile];      // cTile

        const uint64_t attnB = AttnBase(b);
        const uint64_t xB0   = XBase(b, 0, hw);
        const uint64_t xB1   = XBase(b, 1, hw);
        const uint64_t xB2   = XBase(b, 2, hw);
        const uint64_t yB    = YBase(b, hw);

        // Iterate channels in contiguous tiles.
        for (uint32_t c0 = 0; c0 < this->C; c0 += this->cTile) {
            uint32_t cLen = this->C - c0;
            if (cLen > this->cTile) cLen = this->cTile;

            // Load x for each k as contiguous vector.
            AscendC::DataCopy(x0L, xGm[(uint32_t)(xB0 + c0)], cLen);
            AscendC::DataCopy(x1L, xGm[(uint32_t)(xB1 + c0)], cLen);
            AscendC::DataCopy(x2L, xGm[(uint32_t)(xB2 + c0)], cLen);

            // Load attn for k=0,1,2 into outL,a0L,x2L (reuse buffers to avoid extra UB).
            // a0L <- attn0, outL <- attn1, x2L <- attn2 (overwrite x2 after it's safe: we still need x2 values)
            // So instead keep attn0 in a0L, attn1 in outL, attn2 in base scratch by reusing x0L tail? Too complex.
            // Use a0L for attn0, outL for attn1, and reuse x1L (after multiply) for attn2 later. Do math stepwise.

            AscendC::DataCopy(a0L, attnGm[(uint32_t)(attnB + 0ULL * this->C + c0)], cLen);
            AscendC::Mul(x0L, x0L, a0L, (int32_t)cLen);

            AscendC::DataCopy(a0L, attnGm[(uint32_t)(attnB + 1ULL * this->C + c0)], cLen);
            AscendC::Mul(x1L, x1L, a0L, (int32_t)cLen);

            AscendC::DataCopy(a0L, attnGm[(uint32_t)(attnB + 2ULL * this->C + c0)], cLen);
            AscendC::Mul(x2L, x2L, a0L, (int32_t)cLen);

            AscendC::Add(outL, x0L, x1L, (int32_t)cLen);
            AscendC::Add(outL, outL, x2L, (int32_t)cLen);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopy(yGm[(uint32_t)(yB + c0)], outL, cLen);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> ubuf;

    AscendC::GlobalTensor<float> attnGm;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B{0}, K{0}, H{0}, W{0}, C{0};
    uint32_t HW{0}, rows{0}, rowsPerCore{0};
    uint32_t cTile{0};
};

extern "C" __global__ __aicore__ void s2_attention_custom(GM_ADDR attn,
                                                         GM_ADDR x_all,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelS2Attention op;
    op.Init(attn, x_all, y,
            tiling_data.B, tiling_data.K, tiling_data.H, tiling_data.W, tiling_data.C,
            tiling_data.HW, tiling_data.rows, tiling_data.rowsPerCore, tiling_data.cTile);
    op.Process();
}
