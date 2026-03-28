
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
// Batch streams to reduce scalar loop/queue overhead; common N=4 => 2 batches.
constexpr uint32_t UNROLL_N = 2;

class KernelStreamWriteCustom {
public:
    __aicore__ inline KernelStreamWriteCustom() {}

    __aicore__ inline void Init(GM_ADDR y, GM_ADDR h, GM_ADDR out,
                               uint32_t B, uint32_t T, uint32_t N, uint32_t C,
                               uint32_t BT, uint32_t tileC, uint32_t cTiles,
                               uint32_t totalTiles, uint32_t Npad)
    {
        this->B = B;
        this->T = T;
        this->N = N;
        this->C = C;
        this->BT = BT;
        this->tileC = tileC;
        this->cTiles = cTiles;
        this->totalTiles = totalTiles;
        this->Npad = Npad;

        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, BT * C);
        hGm.SetGlobalBuffer((__gm__ DTYPE_H_POST*)h, BT * N);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, BT * N * C);

        // Double-buffer y tiles.
        pipe.InitBuffer(yQueue, BUFFER_NUM, this->tileC * sizeof(DTYPE_Y));
        // Double-buffer output tiles; reused across n-batches safely via TQue ordering.
        pipe.InitBuffer(oQueue, BUFFER_NUM, this->tileC * sizeof(DTYPE_OUT));
        // Coeff vector for one bt (padded copy so GM->UB is legal even when N is small).
        pipe.InitBuffer(hBuf, this->Npad * sizeof(DTYPE_H_POST));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t prevBt = 0xFFFFFFFF;
        AscendC::LocalTensor<DTYPE_H_POST> hLocal = hBuf.Get<DTYPE_H_POST>();

        for (uint32_t tileIdx = blockIdx; tileIdx < totalTiles; tileIdx += blockNum) {
            const uint32_t bt = tileIdx / cTiles;
            const uint32_t ct = tileIdx - bt * cTiles;

            const uint32_t c0 = ct * tileC;
            uint32_t len = C - c0;
            if (len > tileC) len = tileC;
            if (len == 0) continue;

            // Load h_post[bt, :] once when bt changes for this block.
            if (bt != prevBt) {
                const uint32_t hOff = bt * N;
                // Safe because Npad is multiple of 8 (>=8): at least 32B for fp32.
                AscendC::DataCopy(hLocal, hGm[hOff], Npad);
                prevBt = bt;
            }

            // Load y tile once
            CopyInY(bt, c0, len);
            // For all streams, compute out = y * h[n] and store; batch UNROLL_N streams.
            ComputeAndStoreAllN(bt, c0, len, hLocal);
        }

        // Ensure all queued ops complete before exit (queues enforce ordering; nothing extra needed).
    }

private:
    __aicore__ inline void CopyInY(uint32_t bt, uint32_t c0, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = yQueue.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(yLocal, yGm[bt * C + c0], len);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void ComputeAndStoreAllN(uint32_t bt, uint32_t c0, uint32_t len,
                                              const AscendC::LocalTensor<DTYPE_H_POST>& hLocal)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = yQueue.DeQue<DTYPE_Y>();

        uint32_t n = 0;
        for (; n + UNROLL_N <= N; n += UNROLL_N) {
            // stream n
            {
                AscendC::LocalTensor<DTYPE_OUT> o0 = oQueue.AllocTensor<DTYPE_OUT>();
                const DTYPE_OUT h0 = (DTYPE_OUT)hLocal.GetValue(n);
                AscendC::Muls(o0, yLocal, h0, len);
                oQueue.EnQue(o0);
                CopyOut(bt, n, c0, len);
            }
            // stream n+1
            {
                AscendC::LocalTensor<DTYPE_OUT> o1 = oQueue.AllocTensor<DTYPE_OUT>();
                const DTYPE_OUT h1 = (DTYPE_OUT)hLocal.GetValue(n + 1);
                AscendC::Muls(o1, yLocal, h1, len);
                oQueue.EnQue(o1);
                CopyOut(bt, n + 1, c0, len);
            }
        }

        // tail streams
        for (; n < N; ++n) {
            AscendC::LocalTensor<DTYPE_OUT> o = oQueue.AllocTensor<DTYPE_OUT>();
            const DTYPE_OUT hv = (DTYPE_OUT)hLocal.GetValue(n);
            AscendC::Muls(o, yLocal, hv, len);
            oQueue.EnQue(o);
            CopyOut(bt, n, c0, len);
        }

        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t bt, uint32_t n, uint32_t c0, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_OUT> oLocal = oQueue.DeQue<DTYPE_OUT>();
        const uint32_t outOff = (bt * N + n) * C + c0;
        AscendC::DataCopy(outGm[outOff], oLocal, len);
        oQueue.FreeTensor(oLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> yQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> oQueue;

    AscendC::TBuf<AscendC::TPosition::VECIN> hBuf;

    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_H_POST> hGm;
    AscendC::GlobalTensor<DTYPE_OUT> outGm;

    uint32_t B, T, N, C;
    uint32_t BT, tileC, cTiles, totalTiles, Npad;
};

extern "C" __global__ __aicore__ void stream_write_custom(GM_ADDR y, GM_ADDR h_post, GM_ADDR out,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelStreamWriteCustom op;
    op.Init(y, h_post, out,
            tiling_data.B, tiling_data.T, tiling_data.N, tiling_data.C,
            tiling_data.BT, tiling_data.tileC, tiling_data.cTiles,
            tiling_data.totalTiles, tiling_data.Npad);
    op.Process();
}
