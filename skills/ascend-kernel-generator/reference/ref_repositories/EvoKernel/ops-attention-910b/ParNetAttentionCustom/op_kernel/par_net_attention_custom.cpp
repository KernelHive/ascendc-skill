
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelParNetAttentionCustom {
public:
    __aicore__ inline KernelParNetAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                               uint32_t totalElems, uint32_t elemsPerCore, uint32_t tileElems)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->totalElems = totalElems;
        this->elemsPerCore = elemsPerCore;
        this->tileElems = tileElems;

        uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint64_t start64 = static_cast<uint64_t>(blk) * static_cast<uint64_t>(elemsPerCore);
        uint64_t end64 = start64 + static_cast<uint64_t>(elemsPerCore);
        if (start64 > totalElems) start64 = totalElems;
        if (end64 > totalElems) end64 = totalElems;
        this->coreStart = static_cast<uint32_t>(start64);
        this->coreEnd = static_cast<uint32_t>(end64);

        x1Gm.SetGlobalBuffer((__gm__ float*)x1, static_cast<uint64_t>(totalElems));
        x2Gm.SetGlobalBuffer((__gm__ float*)x2, static_cast<uint64_t>(totalElems));
        x3Gm.SetGlobalBuffer((__gm__ float*)x3, static_cast<uint64_t>(totalElems));
        yGm.SetGlobalBuffer((__gm__ float*)y,  static_cast<uint64_t>(totalElems));

        // UB queues
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(inQueueX3, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileElems * sizeof(float));

        // UB intermediates
        pipe.InitBuffer(sumBuf, this->tileElems * sizeof(float));
        pipe.InitBuffer(sigBuf, this->tileElems * sizeof(float));

        // Sigmoid shared tmp (conservative heuristic, similar to BAM example).
        pipe.InitBuffer(sigmoidTmpBuf, this->tileElems * 2);
    }

    __aicore__ inline void Process()
    {
        if (coreStart >= coreEnd) return;
        uint32_t remain = coreEnd - coreStart;
        uint32_t tiles = (remain + tileElems - 1) / tileElems;
        for (uint32_t t = 0; t < tiles; ++t) {
            CopyIn(t);
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t t)
    {
        AscendC::LocalTensor<float> x1Local = inQueueX1.AllocTensor<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.AllocTensor<float>();
        AscendC::LocalTensor<float> x3Local = inQueueX3.AllocTensor<float>();

        uint32_t base = coreStart + t * tileElems;
        uint32_t tileLen = tileElems;
        if (base + tileLen > coreEnd) tileLen = coreEnd - base;

        // Fill full tile to keep vector ops safe; tail handled via scalar loop.
        AscendC::Duplicate(x1Local, 0.0f, tileElems);
        AscendC::Duplicate(x2Local, 0.0f, tileElems);
        AscendC::Duplicate(x3Local, 0.0f, tileElems);

        if (tileLen == tileElems) {
            AscendC::DataCopy(x1Local, x1Gm[base], tileElems);
            AscendC::DataCopy(x2Local, x2Gm[base], tileElems);
            AscendC::DataCopy(x3Local, x3Gm[base], tileElems);
        } else {
            for (uint32_t i = 0; i < tileLen; ++i) {
                x1Local.SetValue(i, x1Gm.GetValue(static_cast<uint64_t>(base + i)));
                x2Local.SetValue(i, x2Gm.GetValue(static_cast<uint64_t>(base + i)));
                x3Local.SetValue(i, x3Gm.GetValue(static_cast<uint64_t>(base + i)));
            }
        }

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
        inQueueX3.EnQue(x3Local);
    }

    __aicore__ inline void Compute(uint32_t t)
    {
        (void)t;
        AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
        AscendC::LocalTensor<float> x3Local = inQueueX3.DeQue<float>();

        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> sigLocal = sigBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> sigmoidTmp = sigmoidTmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // sum = x1 + x2 + x3
        AscendC::Add(sumLocal, x1Local, x2Local, static_cast<int32_t>(tileElems));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sumLocal, sumLocal, x3Local, static_cast<int32_t>(tileElems));
        AscendC::PipeBarrier<PIPE_V>();

        // sig = sigmoid(sum)   (dst/src must not overlap)
        AscendC::Sigmoid<float>(sigLocal, sumLocal, sigmoidTmp, tileElems);
        AscendC::PipeBarrier<PIPE_V>();

        // y = sum * sig  (SiLU)
        AscendC::Mul(yLocal, sumLocal, sigLocal, static_cast<int32_t>(tileElems));

        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        inQueueX3.FreeTensor(x3Local);

        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t t)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();

        uint32_t base = coreStart + t * tileElems;
        uint32_t tileLen = tileElems;
        if (base + tileLen > coreEnd) tileLen = coreEnd - base;

        if (tileLen == tileElems) {
            AscendC::DataCopy(yGm[base], yLocal, tileElems);
        } else {
            for (uint32_t i = 0; i < tileLen; ++i) {
                yGm.SetValue(static_cast<uint64_t>(base + i), yLocal.GetValue(i));
            }
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX3;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<> sumBuf;
    AscendC::TBuf<> sigBuf;
    AscendC::TBuf<> sigmoidTmpBuf;

    AscendC::GlobalTensor<float> x1Gm;
    AscendC::GlobalTensor<float> x2Gm;
    AscendC::GlobalTensor<float> x3Gm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalElems = 0;
    uint32_t elemsPerCore = 0;
    uint32_t tileElems = 0;
    uint32_t coreStart = 0;
    uint32_t coreEnd = 0;
};

extern "C" __global__ __aicore__ void par_net_attention_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelParNetAttentionCustom op;
    op.Init(x1, x2, x3, y,
            tiling_data.totalElems, tiling_data.elemsPerCore, tiling_data.tileElems);
    op.Process();
}
