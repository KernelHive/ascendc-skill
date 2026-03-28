
#include "kernel_operator.h"

// Fixed specialization:
// x: [32, 64, 131072] fp32 NCL contiguous
// w: [128, 64, 3] fp32 contiguous
// y: [32, 128, 131070] fp32
//
// Key change: partition work over linear y index => disjoint output ranges per block,
// avoiding any possibility of overlapping writes that caused non-deterministic corruption.
// Keep the 4-output sliding-window inner structure to reduce redundant x loads.

class KernelConvStandard1dLinearY {
public:
    __aicore__ inline KernelConvStandard1dLinearY() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t totalY, uint32_t totalX, uint32_t totalW,
                               uint32_t blockTiles)
    {
        totalY_ = totalY;
        (void)totalX;
        (void)totalW;
        blockTiles_ = blockTiles;
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        constexpr int N = 32;
        constexpr int CIN = 64;
        constexpr int COUT = 128;
        constexpr int K = 3;
        constexpr int LIN = 131072;
        constexpr int LOUT = LIN - K + 1; // 131070

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t totalY = (int64_t)totalY_;
        int64_t start = blockIdx * (int64_t)blockTiles_;
        int64_t end = start + (int64_t)blockTiles_;
        if (start > totalY) start = totalY;
        if (end > totalY) end = totalY;

        // Process in chunks where (n, oc) stay constant and l advances linearly.
        // linear index order is y[((n*COUT + oc)*LOUT + l)] in contiguous layout.
        int64_t idx = start;
        while (idx < end) {
            // Decode idx -> n, oc, l using division at chunk boundaries only.
            int64_t tmp = idx;
            const int64_t l = tmp % LOUT;
            tmp /= LOUT;
            const int64_t oc = tmp % COUT;
            const int64_t n = tmp / COUT;

            // Maximum contiguous l we can do before (n,oc) changes or block range ends.
            int64_t remainL = (int64_t)LOUT - l;
            int64_t remainBlock = end - idx;
            int64_t run = remainL < remainBlock ? remainL : remainBlock;

            const int64_t xBaseN = (n * CIN) * (int64_t)LIN;            // x[n,0,0]
            const int64_t yBase  = (n * COUT + oc) * (int64_t)LOUT;     // y[n,oc,0]
            const int64_t wBaseOc = (oc * CIN) * (int64_t)K;            // w[oc,0,0]

            int64_t lcur = l;
            // Vectorize over length: do 4 outputs at a time, but don't cross run.
            for (; lcur + 3 < l + run; lcur += 4) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

                #pragma unroll
                for (int ci = 0; ci < CIN; ++ci) {
                    const int64_t xb = xBaseN + (int64_t)ci * LIN + lcur;
                    const int64_t wb = wBaseOc + (int64_t)ci * K;

                    float a = wGm.GetValue((int)(wb + 0));
                    float b = wGm.GetValue((int)(wb + 1));
                    float c = wGm.GetValue((int)(wb + 2));

                    float x0v = xGm.GetValue((int)(xb + 0));
                    float x1v = xGm.GetValue((int)(xb + 1));
                    float x2v = xGm.GetValue((int)(xb + 2));
                    float x3v = xGm.GetValue((int)(xb + 3));
                    float x4v = xGm.GetValue((int)(xb + 4));
                    float x5v = xGm.GetValue((int)(xb + 5));

                    acc0 += x0v * a + x1v * b + x2v * c;
                    acc1 += x1v * a + x2v * b + x3v * c;
                    acc2 += x2v * a + x3v * b + x4v * c;
                    acc3 += x3v * a + x4v * b + x5v * c;
                }

                yGm.SetValue((int)(yBase + lcur + 0), acc0);
                yGm.SetValue((int)(yBase + lcur + 1), acc1);
                yGm.SetValue((int)(yBase + lcur + 2), acc2);
                yGm.SetValue((int)(yBase + lcur + 3), acc3);
            }

            // Tail within this run
            for (; lcur < l + run; ++lcur) {
                float acc = 0.0f;
                const int64_t li0 = lcur + 0;
                const int64_t li1 = lcur + 1;
                const int64_t li2 = lcur + 2;
                #pragma unroll
                for (int ci = 0; ci < CIN; ++ci) {
                    const int64_t xBaseNc = xBaseN + (int64_t)ci * LIN;
                    const int64_t wb = wBaseOc + (int64_t)ci * K;

                    float a = wGm.GetValue((int)(wb + 0));
                    float b = wGm.GetValue((int)(wb + 1));
                    float c = wGm.GetValue((int)(wb + 2));

                    float x0v = xGm.GetValue((int)(xBaseNc + li0));
                    float x1v = xGm.GetValue((int)(xBaseNc + li1));
                    float x2v = xGm.GetValue((int)(xBaseNc + li2));
                    acc += x0v * a + x1v * b + x2v * c;
                }
                yGm.SetValue((int)(yBase + lcur), acc);
            }

            idx += run;
        }
    }

private:
    uint32_t totalY_{0};
    uint32_t blockTiles_{0};
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;
};

extern "C" __global__ __aicore__ void conv_standard1d_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvStandard1dLinearY op;
    op.Init(x, weight, y,
            tiling_data.totalY, tiling_data.totalX, tiling_data.totalW,
            tiling_data.blockTiles);
    op.Process();
}
