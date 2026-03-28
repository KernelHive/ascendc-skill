
#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;

class KernelStreamMixCustom {
public:
    __aicore__ inline KernelStreamMixCustom() {}

    __aicore__ inline void Init(GM_ADDR x_stream, GM_ADDR h_res, GM_ADDR out,
                               uint32_t B, uint32_t Tseq, uint32_t N, uint32_t C,
                               uint32_t BT, uint32_t cTile)
    {
        this->B = B;
        this->Tseq = Tseq;
        this->N = N;
        this->C = C;
        this->BT = BT;
        this->cTile = (cTile == 0) ? 128 : cTile;

        const uint64_t xSize = (uint64_t)B * (uint64_t)Tseq * (uint64_t)N * (uint64_t)C;
        const uint64_t hSize = (uint64_t)B * (uint64_t)Tseq * (uint64_t)N * (uint64_t)N;
        const uint64_t oSize = xSize;

        xGm.SetGlobalBuffer((__gm__ float*)x_stream, xSize);
        hGm.SetGlobalBuffer((__gm__ float*)h_res,    hSize);
        outGm.SetGlobalBuffer((__gm__ float*)out,    oSize);

        // hQ: 16 floats. xQ: packed 4*cTile floats. yQ: packed 4*cTile floats.
        pipe.InitBuffer(hQ, 2, 16 * sizeof(float));
        pipe.InitBuffer(xQ, 2, 4 * this->cTile * sizeof(float));
        pipe.InitBuffer(yQ, 2, 4 * this->cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = (AscendC::GetBlockNum() == 0) ? 1 : AscendC::GetBlockNum();

        if (N == 4) {
            ProcessN4PackedPanel(core, cores);
        } else {
            ProcessScalarFallback(core, cores);
        }
    }

private:
    __aicore__ inline void ProcessN4PackedPanel(uint32_t core, uint32_t cores)
    {
        const uint32_t ct = this->cTile;

        for (uint32_t bt = core; bt < BT; bt += cores) {
            const uint32_t b = bt / Tseq;
            const uint32_t t = bt - b * Tseq;

            // Base offsets for (b,t)
            const uint64_t btBaseX = ((uint64_t)b * (uint64_t)Tseq + (uint64_t)t) * 4ull * (uint64_t)C;
            const uint64_t btBaseH = ((uint64_t)b * (uint64_t)Tseq + (uint64_t)t) * 16ull;
            const uint64_t btBaseO = btBaseX;

            // Load H once -> registers
            LocalTensor<float> hLoc = hQ.AllocTensor<float>();
            AscendC::DataCopy(hLoc, hGm[btBaseH], 16);
            hQ.EnQue(hLoc);
            hLoc = hQ.DeQue<float>();

            const float h00 = hLoc.GetValue(0);  const float h01 = hLoc.GetValue(1);  const float h02 = hLoc.GetValue(2);  const float h03 = hLoc.GetValue(3);
            const float h10 = hLoc.GetValue(4);  const float h11 = hLoc.GetValue(5);  const float h12 = hLoc.GetValue(6);  const float h13 = hLoc.GetValue(7);
            const float h20 = hLoc.GetValue(8);  const float h21 = hLoc.GetValue(9);  const float h22 = hLoc.GetValue(10); const float h23 = hLoc.GetValue(11);
            const float h30 = hLoc.GetValue(12); const float h31 = hLoc.GetValue(13); const float h32 = hLoc.GetValue(14); const float h33 = hLoc.GetValue(15);

            hQ.FreeTensor(hLoc);

            // Double-buffered stream over C tiles with single-panel GM<->UB copies
            const uint32_t numTiles = (C + ct - 1u) / ct;
            uint64_t xOff = btBaseX;
            uint64_t oOff = btBaseO;

            // Prefetch tile 0
            uint32_t c0 = 0;
            uint32_t len0 = (C < ct) ? C : ct;
            {
                LocalTensor<float> xPack0 = xQ.AllocTensor<float>();
                // x layout is contiguous for N-major: [x0(C), x1(C), x2(C), x3(C)]
                // For each c0 tile, the panel is also contiguous: 4*len0 floats.
                AscendC::DataCopy(xPack0, xGm[xOff], 4u * len0);
                xQ.EnQue(xPack0);
            }

            for (uint32_t tile = 0; tile < numTiles; ++tile) {
                const uint32_t cBase = tile * ct;
                const uint32_t len = (C - cBase < ct) ? (C - cBase) : ct;

                // Prefetch next tile
                if (tile + 1u < numTiles) {
                    const uint32_t cNext = cBase + ct;
                    const uint32_t lenN = (C - cNext < ct) ? (C - cNext) : ct;
                    LocalTensor<float> xPackN = xQ.AllocTensor<float>();
                    AscendC::DataCopy(xPackN, xGm[xOff + (uint64_t)ct], 4u * lenN);
                    xQ.EnQue(xPackN);
                }

                // Consume current tile
                LocalTensor<float> xPack = xQ.DeQue<float>();

                LocalTensor<float> yPack = yQ.AllocTensor<float>();

                LocalTensor<float> x0v = xPack;
                LocalTensor<float> x1v = xPack[len];
                LocalTensor<float> x2v = xPack[2u * len];
                LocalTensor<float> x3v = xPack[3u * len];

                LocalTensor<float> y0v = yPack;
                LocalTensor<float> y1v = yPack[len];
                LocalTensor<float> y2v = yPack[2u * len];
                LocalTensor<float> y3v = yPack[3u * len];

                AscendC::Muls(y0v, x0v, h00, len);
                AscendC::Axpy(y0v, x1v, h01, len);
                AscendC::Axpy(y0v, x2v, h02, len);
                AscendC::Axpy(y0v, x3v, h03, len);

                AscendC::Muls(y1v, x0v, h10, len);
                AscendC::Axpy(y1v, x1v, h11, len);
                AscendC::Axpy(y1v, x2v, h12, len);
                AscendC::Axpy(y1v, x3v, h13, len);

                AscendC::Muls(y2v, x0v, h20, len);
                AscendC::Axpy(y2v, x1v, h21, len);
                AscendC::Axpy(y2v, x2v, h22, len);
                AscendC::Axpy(y2v, x3v, h23, len);

                AscendC::Muls(y3v, x0v, h30, len);
                AscendC::Axpy(y3v, x1v, h31, len);
                AscendC::Axpy(y3v, x2v, h32, len);
                AscendC::Axpy(y3v, x3v, h33, len);

                yQ.EnQue(yPack);
                yPack = yQ.DeQue<float>();

                // Single contiguous store of 4*len floats
                AscendC::DataCopy(outGm[oOff], yPack, 4u * len);

                yQ.FreeTensor(yPack);
                xQ.FreeTensor(xPack);

                xOff += (uint64_t)ct;
                oOff += (uint64_t)ct;
            }
        }
    }

    __aicore__ inline void ProcessScalarFallback(uint32_t core, uint32_t cores)
    {
        const uint32_t BTN = B * Tseq * N;
        for (uint32_t row = core; row < BTN; row += cores) {
            uint32_t i = row % N;
            uint32_t tmp = row / N;
            uint32_t t = tmp % Tseq;
            uint32_t b = tmp / Tseq;

            const uint64_t btBaseX = ((uint64_t)b * (uint64_t)Tseq + (uint64_t)t) * (uint64_t)N * (uint64_t)C;
            const uint64_t btBaseH = ((uint64_t)b * (uint64_t)Tseq + (uint64_t)t) * (uint64_t)N * (uint64_t)N;
            const uint64_t outBase = (((uint64_t)b * (uint64_t)Tseq + (uint64_t)t) * (uint64_t)N + (uint64_t)i) * (uint64_t)C;
            const uint64_t hRowBase = btBaseH + (uint64_t)i * (uint64_t)N;

            for (uint32_t c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < N; ++j) {
                    float hij = hGm.GetValue(hRowBase + (uint64_t)j);
                    acc += hij * xGm.GetValue(btBaseX + (uint64_t)j * (uint64_t)C + (uint64_t)c);
                }
                outGm.SetValue(outBase + (uint64_t)c, acc);
            }
        }
    }

private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> hGm;
    GlobalTensor<float> outGm;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2>  hQ;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2>  xQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> yQ;

    uint32_t B = 0, Tseq = 0, N = 0, C = 0, BT = 0, cTile = 0;
};

extern "C" __global__ __aicore__ void stream_mix_custom(GM_ADDR x_stream, GM_ADDR h_res, GM_ADDR out,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelStreamMixCustom op;
    op.Init(x_stream, h_res, out,
            tiling_data.B, tiling_data.T, tiling_data.N, tiling_data.C,
            tiling_data.BT, tiling_data.cTile);
    op.Process();
}
