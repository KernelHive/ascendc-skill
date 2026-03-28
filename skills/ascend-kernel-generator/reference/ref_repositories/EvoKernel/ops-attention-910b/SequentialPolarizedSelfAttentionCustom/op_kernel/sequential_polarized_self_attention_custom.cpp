
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSequentialPolarizedSelfAttentionCustom {
public:
    __aicore__ inline KernelSequentialPolarizedSelfAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR channel_weight, GM_ADDR spatial_weight, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W, uint32_t HW,
                               uint32_t totalPos, uint32_t blockDim, uint32_t posPerCore, uint32_t cTile)
    {
        (void)H; (void)W; (void)blockDim;
        this->B = B;
        this->C = C;
        this->HW = HW;
        this->totalPos = totalPos;
        this->posPerCore = posPerCore;
        this->cTile = cTile;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        this->posStart = coreIdx * posPerCore;
        const uint32_t posEnd = (this->posStart + posPerCore < totalPos) ? (this->posStart + posPerCore) : totalPos;
        this->posCount = (posEnd > this->posStart) ? (posEnd - this->posStart) : 0;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B * (uint64_t)C * (uint64_t)HW);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)B * (uint64_t)C * (uint64_t)HW);
        cwGm.SetGlobalBuffer((__gm__ float*)channel_weight, (uint64_t)B * (uint64_t)C);
        swGm.SetGlobalBuffer((__gm__ float*)spatial_weight, (uint64_t)B * (uint64_t)HW);

        pipe.InitBuffer(xBuf,  this->cTile * sizeof(float));
        pipe.InitBuffer(cwBuf, this->cTile * sizeof(float));
        pipe.InitBuffer(swBuf, this->cTile * sizeof(float)); // swVec / gate reuse
        pipe.InitBuffer(yBuf,  this->cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (posCount == 0) return;

        const uint32_t tilesPerPos = (C + cTile - 1) / cTile;

        for (uint32_t posLocal = 0; posLocal < posCount; ++posLocal) {
            const uint32_t pos = posStart + posLocal;
            const uint32_t b = pos / HW;
            const uint32_t hw = pos - b * HW;

            // Load sw once per position; expand to swVec for each tileLen via Duplicate
            const float sw = swGm.GetValue((uint64_t)b * (uint64_t)HW + (uint64_t)hw);

            for (uint32_t tileId = 0; tileId < tilesPerPos; ++tileId) {
                const uint32_t c0 = tileId * cTile;
                const uint32_t tileLen = (c0 + cTile <= C) ? cTile : (C - c0);
                if (tileLen == 0) continue;

                // UB tensors
                AscendC::LocalTensor<float> xLocal  = xBuf.Get<float>();
                AscendC::LocalTensor<float> cwLocal = cwBuf.Get<float>();
                AscendC::LocalTensor<float> swVec   = swBuf.Get<float>(); // will become gate in-place
                AscendC::LocalTensor<float> yLocal  = yBuf.Get<float>();

                // Copy cw tile (contiguous)
                const uint64_t cwBase = (uint64_t)b * (uint64_t)C + (uint64_t)c0;
                AscendC::DataCopy(cwLocal, cwGm[cwBase], tileLen);

                // Gather x[b, c0:c0+tileLen, hw] with stride HW; unroll by 4
                const uint64_t xBase = ((uint64_t)b * (uint64_t)C + (uint64_t)c0) * (uint64_t)HW + (uint64_t)hw;
                uint32_t i = 0;
                for (; i + 3 < tileLen; i += 4) {
                    const uint64_t off0 = xBase + (uint64_t)(i + 0) * (uint64_t)HW;
                    const uint64_t off1 = xBase + (uint64_t)(i + 1) * (uint64_t)HW;
                    const uint64_t off2 = xBase + (uint64_t)(i + 2) * (uint64_t)HW;
                    const uint64_t off3 = xBase + (uint64_t)(i + 3) * (uint64_t)HW;
                    xLocal.SetValue(i + 0, xGm.GetValue(off0));
                    xLocal.SetValue(i + 1, xGm.GetValue(off1));
                    xLocal.SetValue(i + 2, xGm.GetValue(off2));
                    xLocal.SetValue(i + 3, xGm.GetValue(off3));
                }
                for (; i < tileLen; ++i) {
                    const uint64_t off = xBase + (uint64_t)i * (uint64_t)HW;
                    xLocal.SetValue(i, xGm.GetValue(off));
                }

                // Vector compute:
                // swVec = sw (broadcast)
                AscendC::Duplicate(swVec, sw, tileLen);
                // swVec = cw * swVec (gate)
                AscendC::Mul(swVec, cwLocal, swVec, tileLen);
                // y = x * gate
                AscendC::Mul(yLocal, xLocal, swVec, tileLen);

                // Scatter y with stride HW; unroll by 4
                const uint64_t yBase = xBase;
                i = 0;
                for (; i + 3 < tileLen; i += 4) {
                    const uint64_t off0 = yBase + (uint64_t)(i + 0) * (uint64_t)HW;
                    const uint64_t off1 = yBase + (uint64_t)(i + 1) * (uint64_t)HW;
                    const uint64_t off2 = yBase + (uint64_t)(i + 2) * (uint64_t)HW;
                    const uint64_t off3 = yBase + (uint64_t)(i + 3) * (uint64_t)HW;
                    yGm.SetValue(off0, yLocal.GetValue(i + 0));
                    yGm.SetValue(off1, yLocal.GetValue(i + 1));
                    yGm.SetValue(off2, yLocal.GetValue(i + 2));
                    yGm.SetValue(off3, yLocal.GetValue(i + 3));
                }
                for (; i < tileLen; ++i) {
                    const uint64_t off = yBase + (uint64_t)i * (uint64_t)HW;
                    yGm.SetValue(off, yLocal.GetValue(i));
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> xBuf;
    AscendC::TBuf<> cwBuf;
    AscendC::TBuf<> swBuf;
    AscendC::TBuf<> yBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> cwGm;
    AscendC::GlobalTensor<float> swGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B = 0, C = 0, HW = 0;
    uint32_t totalPos = 0;
    uint32_t posPerCore = 0;
    uint32_t cTile = 0;

    uint32_t posStart = 0;
    uint32_t posCount = 0;
};

extern "C" __global__ __aicore__ void sequential_polarized_self_attention_custom(
    GM_ADDR x, GM_ADDR channel_weight, GM_ADDR spatial_weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSequentialPolarizedSelfAttentionCustom op;
    op.Init(x, channel_weight, spatial_weight, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W, tiling_data.HW,
            tiling_data.totalPos, tiling_data.blockDim, tiling_data.posPerCore, tiling_data.cTile);
    op.Process();
}
