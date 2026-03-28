
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGN_ELEMS = 8; // 32B / sizeof(float)

class KernelCrissCrossAttentionCustom {
public:
    __aicore__ inline KernelCrissCrossAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out_h, GM_ADDR out_w, GM_ADDR gamma, GM_ADDR y,
                               uint32_t totalElems, uint32_t coreElemsAligned, uint32_t tileElems)
    {
        this->totalElems = totalElems;
        this->coreElemsAligned = coreElemsAligned;
        this->tileElems = tileElems;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();

        // Divide tensor into contiguous nominal ranges; vectorize only aligned chunk.
        const uint32_t start = (uint32_t)(((uint64_t)totalElems * coreIdx) / blockNum);
        const uint32_t end   = (uint32_t)(((uint64_t)totalElems * (coreIdx + 1)) / blockNum);

        coreStart = start;
        coreEnd = end;
        coreValid = (end > start) ? (end - start) : 0;

        // Vectorizable length: floor to ALIGN_ELEMS and also cap by coreElemsAligned.
        uint32_t vec = (coreValid / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (vec > coreElemsAligned) vec = coreElemsAligned;
        vecValid = vec;

        // Set GM buffers at coreStart; size uses coreValid for bounds checks (but we only vectorize vecValid).
        xGm.SetGlobalBuffer((__gm__ float*)x + coreStart, (uint64_t)coreValid);
        ohGm.SetGlobalBuffer((__gm__ float*)out_h + coreStart, (uint64_t)coreValid);
        owGm.SetGlobalBuffer((__gm__ float*)out_w + coreStart, (uint64_t)coreValid);
        yGm.SetGlobalBuffer((__gm__ float*)y + coreStart, (uint64_t)coreValid);

        gGm.SetGlobalBuffer((__gm__ float*)gamma, (uint64_t)1);

        pipe.InitBuffer(qx,  BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qoh, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qow, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qy,  BUFFER_NUM, this->tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreValid == 0) return;

        // Deterministic scalar GM read for gamma.
        AscendC::PipeBarrier<PIPE_MTE2>();
        gammaVal = gGm.GetValue(0);

        // Vector path: only aligned sizes to avoid partial-tile vector primitive hazards.
        if (vecValid > 0) {
            const uint32_t tiles = vecValid / tileElems + ((vecValid % tileElems) ? 1 : 0);
            for (uint32_t t = 0; t < tiles; ++t) {
                CopyInVec(t);
                ComputeVec(t);
                CopyOutVec(t);
            }
        }

        // Scalar tail path: handle any remaining elements safely without UB.
        for (uint32_t i = vecValid; i < coreValid; ++i) {
            float xv = xGm.GetValue(i);
            float ohv = ohGm.GetValue(i);
            float owv = owGm.GetValue(i);
            float yv = xv + gammaVal * (ohv + owv);
            yGm.SetValue(i, yv);
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
        // len is always aligned because vecValid and tileElems are aligned and base is aligned.
        return len;
    }

    __aicore__ inline void CopyInVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> xLocal  = qx.AllocTensor<float>();
        AscendC::LocalTensor<float> ohLocal = qoh.AllocTensor<float>();
        AscendC::LocalTensor<float> owLocal = qow.AllocTensor<float>();

        const uint32_t base = TileBase(t);
        AscendC::DataCopy(xLocal,  xGm[base],  len);
        AscendC::DataCopy(ohLocal, ohGm[base], len);
        AscendC::DataCopy(owLocal, owGm[base], len);

        qx.EnQue(xLocal);
        qoh.EnQue(ohLocal);
        qow.EnQue(owLocal);
    }

    __aicore__ inline void ComputeVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> xLocal  = qx.DeQue<float>();
        AscendC::LocalTensor<float> ohLocal = qoh.DeQue<float>();
        AscendC::LocalTensor<float> owLocal = qow.DeQue<float>();

        AscendC::LocalTensor<float> yLocal = qy.AllocTensor<float>();

        // In-place into yLocal: y = x + gamma*(oh + ow)
        AscendC::Add(yLocal, ohLocal, owLocal, (int32_t)len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(yLocal, yLocal, gammaVal, (int32_t)len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(yLocal, yLocal, xLocal, (int32_t)len);

        qx.FreeTensor(xLocal);
        qoh.FreeTensor(ohLocal);
        qow.FreeTensor(owLocal);

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
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qx;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qoh;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qow;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qy;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> ohGm;
    AscendC::GlobalTensor<float> owGm;
    AscendC::GlobalTensor<float> gGm;
    AscendC::GlobalTensor<float> yGm;

    float gammaVal = 0.0f;

    uint32_t totalElems = 0;
    uint32_t coreElemsAligned = 0;

    uint32_t coreStart = 0;
    uint32_t coreEnd = 0;
    uint32_t coreValid = 0;
    uint32_t vecValid = 0;

    uint32_t tileElems = 0;
};

extern "C" __global__ __aicore__ void criss_cross_attention_custom(
    GM_ADDR x, GM_ADDR out_h, GM_ADDR out_w, GM_ADDR gamma,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrissCrossAttentionCustom op;
    op.Init(x, out_h, out_w, gamma, y,
            tiling_data.totalElems, tiling_data.coreElemsAligned, tiling_data.tileElems);
    op.Process();
}
