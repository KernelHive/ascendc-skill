
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGN_ELEMS = 8; // 32B / sizeof(float)

class KernelTripletAttentionCustom {
public:
    __aicore__ inline KernelTripletAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x_ch, GM_ADDR x_cw, GM_ADDR x_hw, GM_ADDR y,
                               uint32_t totalElems, uint32_t coreElemsAligned, uint32_t tileElems)
    {
        this->totalElems = totalElems;
        this->coreElemsAligned = coreElemsAligned;
        this->tileElems = tileElems;

        const uint32_t coreIdx  = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();

        // Divide work into contiguous ranges using proportional split.
        const uint32_t start = (uint32_t)(((uint64_t)totalElems * coreIdx) / blockNum);
        const uint32_t end   = (uint32_t)(((uint64_t)totalElems * (coreIdx + 1)) / blockNum);

        coreStart = start;
        coreEnd = end;
        coreValid = (end > start) ? (end - start) : 0;

        // Vectorizable length: aligned down to ALIGN_ELEMS and capped by coreElemsAligned.
        uint32_t vec = (coreValid / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (vec > coreElemsAligned) vec = coreElemsAligned;
        vecValid = vec;

        // Offset GM pointers to coreStart, only coreValid elements are in-range.
        xchGm.SetGlobalBuffer((__gm__ float*)x_ch + coreStart, (uint64_t)coreValid);
        xcwGm.SetGlobalBuffer((__gm__ float*)x_cw + coreStart, (uint64_t)coreValid);
        xhwGm.SetGlobalBuffer((__gm__ float*)x_hw + coreStart, (uint64_t)coreValid);
        yGm.SetGlobalBuffer(  (__gm__ float*)y    + coreStart, (uint64_t)coreValid);

        pipe.InitBuffer(qch, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qcw, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qhw, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qy,  BUFFER_NUM, this->tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreValid == 0) return;

        if (vecValid > 0) {
            const uint32_t tiles = vecValid / tileElems + ((vecValid % tileElems) ? 1 : 0);
            for (uint32_t t = 0; t < tiles; ++t) {
                CopyInVec(t);
                ComputeVec(t);
                CopyOutVec(t);
            }
        }

        // Scalar tail (safe for any remainder)
        const float oneThird = 0.333333343f;
        for (uint32_t i = vecValid; i < coreValid; ++i) {
            float a = xchGm.GetValue(i);
            float b = xcwGm.GetValue(i);
            float c = xhwGm.GetValue(i);
            yGm.SetValue(i, (a + b + c) * oneThird);
        }
    }

private:
    __aicore__ inline uint32_t TileBase(uint32_t t) const { return t * tileElems; }

    __aicore__ inline uint32_t GetTileLenAligned(uint32_t t) const
    {
        const uint32_t base = TileBase(t);
        if (base >= vecValid) return 0;
        uint32_t remain = vecValid - base;
        uint32_t len = (remain < tileElems) ? remain : tileElems;
        // len is aligned because vecValid, tileElems, base are aligned.
        return len;
    }

    __aicore__ inline void CopyInVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> chLocal = qch.AllocTensor<float>();
        AscendC::LocalTensor<float> cwLocal = qcw.AllocTensor<float>();
        AscendC::LocalTensor<float> hwLocal = qhw.AllocTensor<float>();

        const uint32_t base = TileBase(t);
        AscendC::DataCopy(chLocal, xchGm[base], len);
        AscendC::DataCopy(cwLocal, xcwGm[base], len);
        AscendC::DataCopy(hwLocal, xhwGm[base], len);

        qch.EnQue(chLocal);
        qcw.EnQue(cwLocal);
        qhw.EnQue(hwLocal);
    }

    __aicore__ inline void ComputeVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> chLocal = qch.DeQue<float>();
        AscendC::LocalTensor<float> cwLocal = qcw.DeQue<float>();
        AscendC::LocalTensor<float> hwLocal = qhw.DeQue<float>();

        AscendC::LocalTensor<float> yLocal = qy.AllocTensor<float>();

        const float oneThird = 0.333333343f;

        // y = (ch + cw + hw) * (1/3)
        AscendC::Add(yLocal, chLocal, cwLocal, (int32_t)len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(yLocal, yLocal, hwLocal, (int32_t)len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(yLocal, yLocal, oneThird, (int32_t)len);

        qch.FreeTensor(chLocal);
        qcw.FreeTensor(cwLocal);
        qhw.FreeTensor(hwLocal);

        qy.EnQue(yLocal);
    }

    __aicore__ inline void CopyOutVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> yLocal = qy.DeQue<float>();
        const uint32_t base = TileBase(t);
        AscendC::DataCopy(yGm[base], yLocal, len);
        qy.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qch;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qcw;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qhw;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qy;

    AscendC::GlobalTensor<float> xchGm;
    AscendC::GlobalTensor<float> xcwGm;
    AscendC::GlobalTensor<float> xhwGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalElems = 0;
    uint32_t coreElemsAligned = 0;

    uint32_t coreStart = 0;
    uint32_t coreEnd = 0;
    uint32_t coreValid = 0;
    uint32_t vecValid = 0;

    uint32_t tileElems = 0;
};

extern "C" __global__ __aicore__ void triplet_attention_custom(
    GM_ADDR x_ch, GM_ADDR x_cw, GM_ADDR x_hw,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelTripletAttentionCustom op;
    op.Init(x_ch, x_cw, x_hw, y,
            tiling_data.totalElems, tiling_data.coreElemsAligned, tiling_data.tileElems);
    op.Process();
}
