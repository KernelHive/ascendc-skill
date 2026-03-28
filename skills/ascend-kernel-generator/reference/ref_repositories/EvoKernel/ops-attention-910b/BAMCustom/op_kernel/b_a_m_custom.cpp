
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelBAMCustom {
public:
    __aicore__ inline KernelBAMCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR c, GM_ADDR s, GM_ADDR y,
                               uint32_t totalElems, uint32_t elemsPerCore, uint32_t tileElems)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalElems = totalElems;
        this->elemsPerCore = elemsPerCore;
        this->tileElems = tileElems;

        uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint64_t start64 = static_cast<uint64_t>(blk) * static_cast<uint64_t>(elemsPerCore);
        uint64_t end64 = start64 + static_cast<uint64_t>(elemsPerCore);
        if (start64 > static_cast<uint64_t>(totalElems)) start64 = static_cast<uint64_t>(totalElems);
        if (end64 > static_cast<uint64_t>(totalElems)) end64 = static_cast<uint64_t>(totalElems);
        this->coreStart = static_cast<uint32_t>(start64);
        this->coreEnd = static_cast<uint32_t>(end64);

        xGm.SetGlobalBuffer((__gm__ float*)x, static_cast<uint64_t>(totalElems));
        cGm.SetGlobalBuffer((__gm__ float*)c, static_cast<uint64_t>(totalElems));
        sGm.SetGlobalBuffer((__gm__ float*)s, static_cast<uint64_t>(totalElems));
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(totalElems));

        // UB buffers (single buffering for stability)
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(inQueueS, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileElems * sizeof(float));

        pipe.InitBuffer(attnBuf, this->tileElems * sizeof(float));
        pipe.InitBuffer(sigmoidTmpBuf, this->tileElems * 2);
    }

    __aicore__ inline void Process()
    {
        if (coreStart >= coreEnd) return;

        constexpr uint32_t ALIGN_ELEMS = 64;

        uint32_t remain = coreEnd - coreStart;
        // Only do vector tiling on the aligned prefix. The tiny remainder is handled scalar in GM.
        uint32_t vecRemain = (remain / ALIGN_ELEMS) * ALIGN_ELEMS;
        uint32_t tailRemain = remain - vecRemain;

        // Vector part
        if (vecRemain > 0) {
            uint32_t vecEnd = coreStart + vecRemain;
            uint32_t vecTiles = (vecRemain + tileElems - 1) / tileElems;
            for (uint32_t t = 0; t < vecTiles; ++t) {
                uint32_t base = coreStart + t * tileElems;
                uint32_t len = tileElems;
                if (base + len > vecEnd) len = vecEnd - base; // len is multiple of 64

                CopyInVec(base, len);
                ComputeVec(len);
                CopyOutVec(base, len);
            }
        }

        // Scalar tail (no UB, no vector ops, avoids any out-of-bounds risk)
        if (tailRemain > 0) {
            uint32_t base = coreStart + vecRemain;
            for (uint32_t i = 0; i < tailRemain; ++i) {
                uint64_t idx = static_cast<uint64_t>(base + i);
                float xv = xGm.GetValue(idx);
                float cv = cGm.GetValue(idx);
                float sv = sGm.GetValue(idx);
                // Use scalar sigmoid approximation via exp is risky; instead, reuse device intrinsic Sigmoid is not available scalar.
                // For tail only, fall back to a numerically-stable rational approximation is also risky.
                // Therefore, keep correctness by doing the tail with a 64-wide padded vector micro-tile in UB.
                // (Handled below in ProcessTailUB)
            }
            ProcessTailUB(base, tailRemain);
        }
    }

private:
    __aicore__ inline void CopyInVec(uint32_t base, uint32_t len)
    {
        // len is guaranteed <= tileElems and multiple of 64 and fully in-range.
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[base], len);
        AscendC::DataCopy(cLocal, cGm[base], len);
        AscendC::DataCopy(sLocal, sGm[base], len);

        inQueueX.EnQue(xLocal);
        inQueueC.EnQue(cLocal);
        inQueueS.EnQue(sLocal);
    }

    __aicore__ inline void ComputeVec(uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.DeQue<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.DeQue<float>();

        AscendC::LocalTensor<float> attnLocal = attnBuf.Get<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> sigmoidTmp = sigmoidTmpBuf.Get<uint8_t>();

        AscendC::Add(attnLocal, cLocal, sLocal, static_cast<int32_t>(len));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Sigmoid<float>(attnLocal, attnLocal, sigmoidTmp, len);
        AscendC::PipeBarrier<PIPE_V>();

        // y = x * (1 + attn)
        AscendC::Adds(attnLocal, attnLocal, 1.0f, static_cast<int32_t>(len));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal, xLocal, attnLocal, static_cast<int32_t>(len));

        inQueueX.FreeTensor(xLocal);
        inQueueC.FreeTensor(cLocal);
        inQueueS.FreeTensor(sLocal);

        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOutVec(uint32_t base, uint32_t len)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[base], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessTailUB(uint32_t base, uint32_t tailLen)
    {
        // Tail correctness without scalar sigmoid: do one UB micro-tile of 64 elements.
        // This never goes out-of-range in GM because we only read/write valid tailLen scalars,
        // and vector ops only touch UB.
        constexpr uint32_t MICRO = 64;

        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> cLocal = inQueueC.AllocTensor<float>();
        AscendC::LocalTensor<float> sLocal = inQueueS.AllocTensor<float>();

        // Fill UB micro tile with zeros then scalar-load valid tail.
        AscendC::Duplicate(xLocal, 0.0f, MICRO);
        AscendC::Duplicate(cLocal, 0.0f, MICRO);
        AscendC::Duplicate(sLocal, 0.0f, MICRO);

        for (uint32_t i = 0; i < tailLen; ++i) {
            uint64_t idx = static_cast<uint64_t>(base + i);
            xLocal.SetValue(i, xGm.GetValue(idx));
            cLocal.SetValue(i, cGm.GetValue(idx));
            sLocal.SetValue(i, sGm.GetValue(idx));
        }

        inQueueX.EnQue(xLocal);
        inQueueC.EnQue(cLocal);
        inQueueS.EnQue(sLocal);

        // Compute on MICRO
        AscendC::LocalTensor<float> xL = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> cL = inQueueC.DeQue<float>();
        AscendC::LocalTensor<float> sL = inQueueS.DeQue<float>();

        AscendC::LocalTensor<float> attnLocal = attnBuf.Get<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> sigmoidTmp = sigmoidTmpBuf.Get<uint8_t>();

        AscendC::Add(attnLocal, cL, sL, static_cast<int32_t>(MICRO));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid<float>(attnLocal, attnLocal, sigmoidTmp, MICRO);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(attnLocal, attnLocal, 1.0f, static_cast<int32_t>(MICRO));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal, xL, attnLocal, static_cast<int32_t>(MICRO));

        inQueueX.FreeTensor(xL);
        inQueueC.FreeTensor(cL);
        inQueueS.FreeTensor(sL);

        outQueueY.EnQue(yLocal);

        // Store only valid tailLen elements to GM
        AscendC::LocalTensor<float> yL = outQueueY.DeQue<float>();
        for (uint32_t i = 0; i < tailLen; ++i) {
            yGm.SetValue(static_cast<uint64_t>(base + i), yL.GetValue(i));
        }
        outQueueY.FreeTensor(yL);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueC;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueS;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<> attnBuf;
    AscendC::TBuf<> sigmoidTmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> cGm;
    AscendC::GlobalTensor<float> sGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalElems = 0;
    uint32_t elemsPerCore = 0;
    uint32_t tileElems = 0;
    uint32_t coreStart = 0;
    uint32_t coreEnd = 0;
};

extern "C" __global__ __aicore__ void bam_custom(GM_ADDR x, GM_ADDR channel_map, GM_ADDR spatial_map, GM_ADDR y,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelBAMCustom op;
    op.Init(x, channel_map, spatial_map, y,
            tiling_data.totalElems, tiling_data.elemsPerCore, tiling_data.tileElems);
    op.Process();
}
