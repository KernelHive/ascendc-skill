
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGN_ELEMS = 8; // 32B / sizeof(float)

class KernelCoordAttCustom {
public:
    __aicore__ inline KernelCoordAttCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR aw, GM_ADDR ah, GM_ADDR y,
                               uint32_t totalElems, uint32_t coreElemsAligned, uint32_t tileElems)
    {
        this->totalElems = totalElems;
        this->coreElemsAligned = coreElemsAligned;
        this->tileElems = tileElems;

        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t start = (uint32_t)(((uint64_t)totalElems * coreIdx) / blockNum);
        const uint32_t end   = (uint32_t)(((uint64_t)totalElems * (coreIdx + 1)) / blockNum);

        coreStart = start;
        coreEnd = end;
        coreValid = (end > start) ? (end - start) : 0;

        uint32_t vec = (coreValid / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (vec > coreElemsAligned) vec = coreElemsAligned;
        vecValid = vec;

        xGm.SetGlobalBuffer((__gm__ float*)x + coreStart, (uint64_t)coreValid);
        awGm.SetGlobalBuffer((__gm__ float*)aw + coreStart, (uint64_t)coreValid);
        ahGm.SetGlobalBuffer((__gm__ float*)ah + coreStart, (uint64_t)coreValid);
        yGm.SetGlobalBuffer((__gm__ float*)y + coreStart, (uint64_t)coreValid);

        // UB buffers:
        // qx:  x tile
        // qw:  a_w tile
        // qh:  a_h tile
        // qtmp: tmp tile for (x*aw)
        // qy:  y tile
        pipe.InitBuffer(qx,   BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qw,   BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qh,   BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qtmp, BUFFER_NUM, this->tileElems * sizeof(float));
        pipe.InitBuffer(qy,   BUFFER_NUM, this->tileElems * sizeof(float));
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

        // Tail: scalar safe.
        for (uint32_t i = vecValid; i < coreValid; ++i) {
            float xv  = xGm.GetValue(i);
            float awv = awGm.GetValue(i);
            float ahv = ahGm.GetValue(i);
            yGm.SetValue(i, xv * awv * ahv);
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
        len = (len / ALIGN_ELEMS) * ALIGN_ELEMS;
        return len;
    }

    __aicore__ inline void CopyInVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;
        const uint32_t base = TileBase(t);

        AscendC::LocalTensor<float> xLocal  = qx.AllocTensor<float>();
        AscendC::LocalTensor<float> awLocal = qw.AllocTensor<float>();
        AscendC::LocalTensor<float> ahLocal = qh.AllocTensor<float>();

        AscendC::DataCopy(xLocal,  xGm[base],  len);
        AscendC::DataCopy(awLocal, awGm[base], len);
        AscendC::DataCopy(ahLocal, ahGm[base], len);

        AscendC::PipeBarrier<PIPE_MTE2>();

        qx.EnQue(xLocal);
        qw.EnQue(awLocal);
        qh.EnQue(ahLocal);
    }

    __aicore__ inline void ComputeVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        AscendC::LocalTensor<float> xLocal  = qx.DeQue<float>();
        AscendC::LocalTensor<float> awLocal = qw.DeQue<float>();
        AscendC::LocalTensor<float> ahLocal = qh.DeQue<float>();

        AscendC::LocalTensor<float> tmpLocal = qtmp.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal   = qy.AllocTensor<float>();

        // tmp = x * aw
        AscendC::Mul(tmpLocal, xLocal, awLocal, (int32_t)len);
        // y = tmp * ah
        AscendC::Mul(yLocal, tmpLocal, ahLocal, (int32_t)len);

        qx.FreeTensor(xLocal);
        qw.FreeTensor(awLocal);
        qh.FreeTensor(ahLocal);
        qtmp.FreeTensor(tmpLocal);

        qy.EnQue(yLocal);
    }

    __aicore__ inline void CopyOutVec(uint32_t t)
    {
        const uint32_t len = GetTileLenAligned(t);
        if (len == 0) return;

        const uint32_t base = TileBase(t);
        AscendC::LocalTensor<float> yLocal = qy.DeQue<float>();
        AscendC::DataCopy(yGm[base], yLocal, len);
        qy.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qx;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qw;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qh;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> qtmp;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qy;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> awGm;
    AscendC::GlobalTensor<float> ahGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t totalElems = 0;
    uint32_t coreElemsAligned = 0;

    uint32_t coreStart = 0;
    uint32_t coreEnd = 0;
    uint32_t coreValid = 0;
    uint32_t vecValid = 0;

    uint32_t tileElems = 0;
};

extern "C" __global__ __aicore__ void coord_att_custom(
    GM_ADDR x, GM_ADDR a_w, GM_ADDR a_h,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCoordAttCustom op;
    op.Init(x, a_w, a_h, y,
            tiling_data.totalElems, tiling_data.coreElemsAligned, tiling_data.tileElems);
    op.Process();
}
