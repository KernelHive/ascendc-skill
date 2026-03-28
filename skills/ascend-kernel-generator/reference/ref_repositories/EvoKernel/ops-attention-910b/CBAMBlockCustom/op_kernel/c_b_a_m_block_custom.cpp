
#include "kernel_operator.h"

class KernelCBAMBlockCustom {
public:
    __aicore__ inline KernelCBAMBlockCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR ca, GM_ADDR sa, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t strideB, uint32_t cTile)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->B = B;
        this->C = C;
        this->H = H;
        this->W = W;
        this->HW = HW;
        this->strideB = strideB;
        this->cTile = cTile;

        // Split batch across cores.
        uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blkNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t bPerCore = (B + blkNum - 1) / blkNum;
        bStart = blk * bPerCore;
        bEnd = bStart + bPerCore;
        if (bEnd > B) bEnd = B;

        uint64_t xTotal = static_cast<uint64_t>(B) * static_cast<uint64_t>(strideB);
        xGm.SetGlobalBuffer((__gm__ float*)x, xTotal);
        yGm.SetGlobalBuffer((__gm__ float*)y, xTotal);

        // ca is [B,C,1,1] contiguous => treat as [B*C]
        caGm.SetGlobalBuffer((__gm__ float*)ca, static_cast<uint64_t>(B) * static_cast<uint64_t>(C));
        // sa is [B,1,H,W] contiguous => treat as [B*HW]
        saGm.SetGlobalBuffer((__gm__ float*)sa, static_cast<uint64_t>(B) * static_cast<uint64_t>(HW));
    }

    __aicore__ inline void Process()
    {
        if (bStart >= bEnd) return;
        for (uint32_t b = bStart; b < bEnd; ++b) {
            ComputeOneBatch(b);
        }
    }

private:
    __aicore__ inline void ComputeOneBatch(uint32_t b)
    {
        // y = x * ca * sa + x
        // Flatten within batch: idx = c*HW + hw
        uint64_t baseB = static_cast<uint64_t>(b) * static_cast<uint64_t>(strideB);
        uint64_t baseCa = static_cast<uint64_t>(b) * static_cast<uint64_t>(C);
        uint64_t baseSa = static_cast<uint64_t>(b) * static_cast<uint64_t>(HW);

        uint32_t numCTiles = (C + cTile - 1) / cTile;
        for (uint32_t tc = 0; tc < numCTiles; ++tc) {
            uint32_t c0 = tc * cTile;
            uint32_t curC = (tc == numCTiles - 1) ? (C - c0) : cTile;
            if (curC == 0) continue;

            for (uint32_t cc = 0; cc < curC; ++cc) {
                uint32_t c = c0 + cc;
                float caVal = caGm.GetValue(baseCa + static_cast<uint64_t>(c));
                uint64_t baseXc = baseB + static_cast<uint64_t>(c) * static_cast<uint64_t>(HW);

                for (uint32_t hw = 0; hw < HW; ++hw) {
                    float saVal = saGm.GetValue(baseSa + static_cast<uint64_t>(hw));
                    float xv = xGm.GetValue(baseXc + static_cast<uint64_t>(hw));
                    float outv = xv * caVal * saVal + xv;
                    yGm.SetValue(baseXc + static_cast<uint64_t>(hw), outv);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> caGm;
    AscendC::GlobalTensor<float> saGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B, C, H, W, HW, strideB, cTile;
    uint32_t bStart, bEnd;
};

extern "C" __global__ __aicore__ void cbam_block_custom(GM_ADDR x, GM_ADDR ca, GM_ADDR sa, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelCBAMBlockCustom op;
    op.Init(x, ca, sa, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.strideB, tiling_data.cTile);
    op.Process();
}
