
#include "kernel_operator.h"

class KernelMobileViTAttentionCustom {
public:
    __aicore__ inline KernelMobileViTAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR att_delta, GM_ADDR ffn_delta, GM_ADDR y,
                               uint32_t totalElems, uint32_t totalAligned,
                               uint32_t tileElems, uint32_t tileNum)
    {
        this->totalElems = totalElems;
        this->totalAligned = totalAligned;
        this->tileElems = tileElems;
        this->tileNum = tileNum;

        const uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blkNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        this->tileStart = blk;
        this->tileStride = blkNum;

        xGm.SetGlobalBuffer((__gm__ float*)x,            static_cast<uint64_t>(totalAligned));
        attGm.SetGlobalBuffer((__gm__ float*)att_delta,  static_cast<uint64_t>(totalAligned));
        ffnGm.SetGlobalBuffer((__gm__ float*)ffn_delta,  static_cast<uint64_t>(totalAligned));
        yGm.SetGlobalBuffer((__gm__ float*)y,            static_cast<uint64_t>(totalAligned));

        // 3 UB tiles: out ping/pong + tmp scratch.
        pipe.InitBuffer(outBuf0, tileElems * sizeof(float));
        pipe.InitBuffer(outBuf1, tileElems * sizeof(float));
        pipe.InitBuffer(tmpBuf,  tileElems * sizeof(float));
    }

    __aicore__ inline void ProcessOne(uint32_t base, uint32_t lenAligned, uint32_t outLen,
                                     AscendC::LocalTensor<float>& outLocal,
                                     AscendC::LocalTensor<float>& tmpLocal)
    {
        // Stream deltas through tmp and accumulate to out.
        AscendC::DataCopy(tmpLocal, attGm[base], lenAligned);
        AscendC::Add(outLocal, outLocal, tmpLocal, static_cast<int32_t>(lenAligned));

        AscendC::DataCopy(tmpLocal, ffnGm[base], lenAligned);
        AscendC::Add(outLocal, outLocal, tmpLocal, static_cast<int32_t>(lenAligned));

        if (outLen > 0) {
            AscendC::DataCopy(yGm[base], outLocal, outLen);
        }
    }

    __aicore__ inline void Process()
    {
        if (totalElems == 0) return;

        // Ping-pong prefetch of x only (mandatory copy), overlap with vector adds of current tile.
        bool ping = true;
        uint32_t t = tileStart;

        // Prime first tile
        if (t < tileNum) {
            const uint32_t base0 = t * tileElems;
            uint32_t lenAligned0 = totalAligned - base0;
            if (lenAligned0 > tileElems) lenAligned0 = tileElems;

            AscendC::LocalTensor<float> out0 = outBuf0.Get<float>();
            AscendC::DataCopy(out0, xGm[base0], lenAligned0);
        }

        for (; t < tileNum; t += tileStride) {
            const uint32_t base = t * tileElems;

            uint32_t lenAligned = 0;
            if (base < totalAligned) {
                lenAligned = totalAligned - base;
                if (lenAligned > tileElems) lenAligned = tileElems;
            }
            if (lenAligned == 0) continue;

            uint32_t outLen = 0;
            if (base < totalElems) {
                outLen = totalElems - base;
                if (outLen > tileElems) outLen = tileElems;
            }

            // Prefetch next tile's x into the other buffer (if exists).
            const uint32_t tNext = t + tileStride;
            if (tNext < tileNum) {
                const uint32_t baseN = tNext * tileElems;
                uint32_t lenAlignedN = totalAligned - baseN;
                if (lenAlignedN > tileElems) lenAlignedN = tileElems;

                AscendC::LocalTensor<float> outNext = ping ? outBuf1.Get<float>() : outBuf0.Get<float>();
                AscendC::DataCopy(outNext, xGm[baseN], lenAlignedN);
            }

            AscendC::LocalTensor<float> outLocal = ping ? outBuf0.Get<float>() : outBuf1.Get<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

            // Ensure x copy into outLocal is visible before vector adds on it.
            AscendC::PipeBarrier<PIPE_MTE2>();
            ProcessOne(base, lenAligned, outLen, outLocal, tmpLocal);

            ping = !ping;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> outBuf0, outBuf1, tmpBuf;

    AscendC::GlobalTensor<float> xGm, attGm, ffnGm, yGm;

    uint32_t totalElems = 0;
    uint32_t totalAligned = 0;
    uint32_t tileElems = 0;
    uint32_t tileNum = 0;
    uint32_t tileStart = 0;
    uint32_t tileStride = 1;
};

extern "C" __global__ __aicore__ void mobile_vi_t_attention_custom(
    GM_ADDR x, GM_ADDR att_delta, GM_ADDR ffn_delta,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMobileViTAttentionCustom op;
    op.Init(x, att_delta, ffn_delta, y,
            tiling_data.totalElems, tiling_data.totalAligned,
            tiling_data.tileElems, tiling_data.tileNum);
    op.Process();
}
