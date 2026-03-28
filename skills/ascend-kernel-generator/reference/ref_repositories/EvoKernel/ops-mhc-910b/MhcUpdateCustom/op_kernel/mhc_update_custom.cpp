
#include "kernel_operator.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::TQue;
using AscendC::TPipe;
using AscendC::DataCopyParams;

class KernelMhcUpdateCustom {
public:
    __aicore__ inline KernelMhcUpdateCustom() {}

    __aicore__ inline void Init(GM_ADDR x_stream, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR y,
                               GM_ADDR out,
                               uint32_t B, uint32_t T, uint32_t I, uint32_t J, uint32_t C,
                               uint32_t BT, uint32_t Vc)
    {
        this->B = B;
        this->T = T;
        this->I = I;
        this->J = J;
        this->C = C;
        this->BT = BT;
        this->Vc = (Vc == 0 ? 256 : Vc);

        const uint64_t xSize  = (uint64_t)B * (uint64_t)T * (uint64_t)J * (uint64_t)C;
        const uint64_t hpSize = (uint64_t)B * (uint64_t)T * (uint64_t)I;
        const uint64_t hrSize = (uint64_t)B * (uint64_t)T * (uint64_t)I * (uint64_t)J;
        const uint64_t ySize  = (uint64_t)B * (uint64_t)T * (uint64_t)C;
        const uint64_t oSize  = (uint64_t)B * (uint64_t)T * (uint64_t)I * (uint64_t)C;

        xGm.SetGlobalBuffer((__gm__ float*)x_stream, xSize);
        hPostGm.SetGlobalBuffer((__gm__ float*)h_post, hpSize);
        hResGm.SetGlobalBuffer((__gm__ float*)h_res, hrSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);
        outGm.SetGlobalBuffer((__gm__ float*)out, oSize);

        // Ping-pong packed buffers to overlap DMA and compute.
        // in panel layout: [ y | x0 | x1 | x2 | x3 ] (5*Vc)
        // out panel layout: [ o0 | o1 | o2 | o3 ] (4*Vc)
        const uint64_t inBytes = (uint64_t)5 * (uint64_t)this->Vc * sizeof(float);
        const uint64_t outBytes = (uint64_t)4 * (uint64_t)this->Vc * sizeof(float);

        pipe.InitBuffer(qIn,  2, inBytes);
        pipe.InitBuffer(qOut, 2, outBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = (AscendC::GetBlockNum() == 0) ? 1 : AscendC::GetBlockNum();

        for (uint32_t btIdx = core; btIdx < BT; btIdx += cores) {
            uint32_t t = btIdx % T;
            uint32_t b = btIdx / T;
            const uint64_t bt = (uint64_t)b * (uint64_t)T + (uint64_t)t;

            const uint64_t yBase   = bt * (uint64_t)C;
            const uint64_t hpBase  = bt * (uint64_t)I;
            const uint64_t hrBase  = bt * (uint64_t)I * (uint64_t)J;
            const uint64_t xBaseBT = bt * (uint64_t)J * (uint64_t)C;
            const uint64_t oBaseBT = bt * (uint64_t)I * (uint64_t)C;

            const bool fast = (I == 4) && (J == 4) && ((C % 8) == 0) && ((Vc % 8) == 0);

            if (!fast) {
                // Generic scalar fallback for full correctness.
                for (uint32_t i = 0; i < I; ++i) {
                    float hpv = hPostGm.GetValue(hpBase + i);
                    const uint64_t hrRow = hrBase + (uint64_t)i * (uint64_t)J;
                    const uint64_t oBase = oBaseBT + (uint64_t)i * (uint64_t)C;
                    for (uint32_t c = 0; c < C; ++c) {
                        float acc = hpv * yGm.GetValue(yBase + c);
                        for (uint32_t j = 0; j < J; ++j) {
                            float hij = hResGm.GetValue(hrRow + j);
                            acc += hij * xGm.GetValue(xBaseBT + (uint64_t)j * (uint64_t)C + c);
                        }
                        outGm.SetValue(oBase + c, acc);
                    }
                }
                continue;
            }

            // Load scalars once per (b,t)
            float hp0 = hPostGm.GetValue(hpBase + 0);
            float hp1 = hPostGm.GetValue(hpBase + 1);
            float hp2 = hPostGm.GetValue(hpBase + 2);
            float hp3 = hPostGm.GetValue(hpBase + 3);

            float h00 = hResGm.GetValue(hrBase + 0 * 4 + 0);
            float h01 = hResGm.GetValue(hrBase + 0 * 4 + 1);
            float h02 = hResGm.GetValue(hrBase + 0 * 4 + 2);
            float h03 = hResGm.GetValue(hrBase + 0 * 4 + 3);

            float h10 = hResGm.GetValue(hrBase + 1 * 4 + 0);
            float h11 = hResGm.GetValue(hrBase + 1 * 4 + 1);
            float h12 = hResGm.GetValue(hrBase + 1 * 4 + 2);
            float h13 = hResGm.GetValue(hrBase + 1 * 4 + 3);

            float h20 = hResGm.GetValue(hrBase + 2 * 4 + 0);
            float h21 = hResGm.GetValue(hrBase + 2 * 4 + 1);
            float h22 = hResGm.GetValue(hrBase + 2 * 4 + 2);
            float h23 = hResGm.GetValue(hrBase + 2 * 4 + 3);

            float h30 = hResGm.GetValue(hrBase + 3 * 4 + 0);
            float h31 = hResGm.GetValue(hrBase + 3 * 4 + 1);
            float h32 = hResGm.GetValue(hrBase + 3 * 4 + 2);
            float h33 = hResGm.GetValue(hrBase + 3 * 4 + 3);

            const uint32_t tiles = (C + Vc - 1) / Vc; // with aligned C, typically exact
            // We only support aligned case for fast path; tiling ensures Vc multiple-of-8, and C%8==0.
            // So tail tile won't happen when Vc==C or Vc divides C; if not, last partial would exist.
            // For safety, we assume Vc divides C in fast path.
            if ((C % Vc) != 0) {
                // Fallback to scalar if non-divisible.
                for (uint32_t i = 0; i < I; ++i) {
                    float hpv = hPostGm.GetValue(hpBase + i);
                    const uint64_t hrRow = hrBase + (uint64_t)i * (uint64_t)J;
                    const uint64_t oBase = oBaseBT + (uint64_t)i * (uint64_t)C;
                    for (uint32_t c = 0; c < C; ++c) {
                        float acc = hpv * yGm.GetValue(yBase + c);
                        for (uint32_t j = 0; j < J; ++j) {
                            float hij = hResGm.GetValue(hrRow + j);
                            acc += hij * xGm.GetValue(xBaseBT + (uint64_t)j * (uint64_t)C + c);
                        }
                        outGm.SetValue(oBase + c, acc);
                    }
                }
                continue;
            }

            // Prefetch tile 0
            {
                LocalTensor<float> in0 = qIn.AllocTensor<float>();
                PrefetchInPanel(in0, yBase, xBaseBT, 0);
                qIn.EnQue(in0);
            }

            for (uint32_t k = 0; k < tiles; ++k) {
                // Prefetch next tile early (ping-pong), except for last
                if (k + 1 < tiles) {
                    LocalTensor<float> inN = qIn.AllocTensor<float>();
                    PrefetchInPanel(inN, yBase, xBaseBT, k + 1);
                    qIn.EnQue(inN);
                }

                // Dequeue current input
                LocalTensor<float> in = qIn.DeQue<float>();

                // Allocate output panel for this tile
                LocalTensor<float> outP = qOut.AllocTensor<float>();

                ComputePanel(outP, in,
                             hp0, hp1, hp2, hp3,
                             h00,h01,h02,h03,
                             h10,h11,h12,h13,
                             h20,h21,h22,h23,
                             h30,h31,h32,h33);

                qOut.EnQue(outP);

                // Dequeue and store previous output (k-1), overlapped with current compute by ping-ponging
                LocalTensor<float> outStore = qOut.DeQue<float>();
                StoreOutPanel(outStore, oBaseBT, k);
                qOut.FreeTensor(outStore);

                qIn.FreeTensor(in);
            }
        }
    }

private:
    __aicore__ inline void PrefetchInPanel(LocalTensor<float> &inP,
                                          uint64_t yBase, uint64_t xBaseBT,
                                          uint32_t tileK)
    {
        const uint64_t c0 = (uint64_t)tileK * (uint64_t)Vc;

        // Slices
        LocalTensor<float> yUb  = inP[0];
        LocalTensor<float> x0Ub = inP[(uint64_t)1 * Vc];
        LocalTensor<float> x1Ub = inP[(uint64_t)2 * Vc];
        LocalTensor<float> x2Ub = inP[(uint64_t)3 * Vc];
        LocalTensor<float> x3Ub = inP[(uint64_t)4 * Vc];

        CopyGmToUbAligned(yUb,  yGm[yBase + c0], Vc);
        CopyGmToUbAligned(x0Ub, xGm[xBaseBT + (uint64_t)0 * (uint64_t)C + c0], Vc);
        CopyGmToUbAligned(x1Ub, xGm[xBaseBT + (uint64_t)1 * (uint64_t)C + c0], Vc);
        CopyGmToUbAligned(x2Ub, xGm[xBaseBT + (uint64_t)2 * (uint64_t)C + c0], Vc);
        CopyGmToUbAligned(x3Ub, xGm[xBaseBT + (uint64_t)3 * (uint64_t)C + c0], Vc);
    }

    __aicore__ inline void ComputePanel(LocalTensor<float> &outP,
                                        const LocalTensor<float> &inP,
                                        float hp0, float hp1, float hp2, float hp3,
                                        float h00,float h01,float h02,float h03,
                                        float h10,float h11,float h12,float h13,
                                        float h20,float h21,float h22,float h23,
                                        float h30,float h31,float h32,float h33)
    {
        LocalTensor<float> yUb  = inP[0];
        LocalTensor<float> x0Ub = inP[(uint64_t)1 * Vc];
        LocalTensor<float> x1Ub = inP[(uint64_t)2 * Vc];
        LocalTensor<float> x2Ub = inP[(uint64_t)3 * Vc];
        LocalTensor<float> x3Ub = inP[(uint64_t)4 * Vc];

        LocalTensor<float> o0 = outP[0];
        LocalTensor<float> o1 = outP[(uint64_t)1 * Vc];
        LocalTensor<float> o2 = outP[(uint64_t)2 * Vc];
        LocalTensor<float> o3 = outP[(uint64_t)3 * Vc];

        AscendC::Muls(o0, yUb, hp0, Vc);
        AscendC::Muls(o1, yUb, hp1, Vc);
        AscendC::Muls(o2, yUb, hp2, Vc);
        AscendC::Muls(o3, yUb, hp3, Vc);

        AscendC::Axpy(o0, x0Ub, h00, Vc);
        AscendC::Axpy(o0, x1Ub, h01, Vc);
        AscendC::Axpy(o0, x2Ub, h02, Vc);
        AscendC::Axpy(o0, x3Ub, h03, Vc);

        AscendC::Axpy(o1, x0Ub, h10, Vc);
        AscendC::Axpy(o1, x1Ub, h11, Vc);
        AscendC::Axpy(o1, x2Ub, h12, Vc);
        AscendC::Axpy(o1, x3Ub, h13, Vc);

        AscendC::Axpy(o2, x0Ub, h20, Vc);
        AscendC::Axpy(o2, x1Ub, h21, Vc);
        AscendC::Axpy(o2, x2Ub, h22, Vc);
        AscendC::Axpy(o2, x3Ub, h23, Vc);

        AscendC::Axpy(o3, x0Ub, h30, Vc);
        AscendC::Axpy(o3, x1Ub, h31, Vc);
        AscendC::Axpy(o3, x2Ub, h32, Vc);
        AscendC::Axpy(o3, x3Ub, h33, Vc);
    }

    __aicore__ inline void StoreOutPanel(const LocalTensor<float> &outP,
                                         uint64_t oBaseBT, uint32_t tileK)
    {
        const uint64_t c0 = (uint64_t)tileK * (uint64_t)Vc;
        LocalTensor<float> o0 = outP[0];
        LocalTensor<float> o1 = outP[(uint64_t)1 * Vc];
        LocalTensor<float> o2 = outP[(uint64_t)2 * Vc];
        LocalTensor<float> o3 = outP[(uint64_t)3 * Vc];

        CopyUbToGmAligned(outGm[oBaseBT + (uint64_t)0 * (uint64_t)C + c0], o0, Vc);
        CopyUbToGmAligned(outGm[oBaseBT + (uint64_t)1 * (uint64_t)C + c0], o1, Vc);
        CopyUbToGmAligned(outGm[oBaseBT + (uint64_t)2 * (uint64_t)C + c0], o2, Vc);
        CopyUbToGmAligned(outGm[oBaseBT + (uint64_t)3 * (uint64_t)C + c0], o3, Vc);
    }

    __aicore__ inline void CopyGmToUbAligned(LocalTensor<float> &dst, const GlobalTensor<float> src, uint32_t len)
    {
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = (uint16_t)(len / 8); // 8 fp32 = 32B
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, src, p);
    }

    __aicore__ inline void CopyUbToGmAligned(GlobalTensor<float> dst, const LocalTensor<float> &src, uint32_t len)
    {
        DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = (uint16_t)(len / 8);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, src, p);
    }

private:
    TPipe pipe;

    // Ping-pong queues for packed panels
    TQue<AscendC::QuePosition::VECIN, 2>  qIn;
    TQue<AscendC::QuePosition::VECOUT, 2> qOut;

    GlobalTensor<float> xGm;
    GlobalTensor<float> hPostGm;
    GlobalTensor<float> hResGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> outGm;

    uint32_t B = 0, T = 0, I = 0, J = 0, C = 0, BT = 0, Vc = 0;
};

extern "C" __global__ __aicore__ void mhc_update_custom(GM_ADDR x_stream, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR y,
                                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMhcUpdateCustom op;
    op.Init(x_stream, h_post, h_res, y, out,
            tiling_data.B, tiling_data.T, tiling_data.I, tiling_data.J, tiling_data.C,
            tiling_data.BT, tiling_data.Vc);
    op.Process();
}
