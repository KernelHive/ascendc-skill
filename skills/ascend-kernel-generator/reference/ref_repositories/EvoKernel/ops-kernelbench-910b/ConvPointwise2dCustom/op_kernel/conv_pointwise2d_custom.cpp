
#include "kernel_operator.h"

// Pointwise Conv2d (1x1) specialized:
// x: [16, 64, 1024, 1024] NCHW, float32 contiguous
// w: [128, 64, 1, 1]     OIHW, float32 contiguous
// y: [16, 128, 1024, 1024] NCHW
//
// Mapping:
//  - task -> (nIdx, hIdx, wGroupIdx)
//  - each task computes WGROUP adjacent stripes, each WTILE contiguous W positions
//  - compute all COUT for the WGROUP*WTILE positions
//
// Key points:
//  - weights cached to UB once per block
//  - per-ci load two x lines (WTILE each) -> update two accumulator stripes
//  - store outputs as contiguous DataCopy bursts (WTILE) for each stripe

class KernelConvPointwise2dCustom_WT16_WG2_COB8 {
public:
    __aicore__ inline KernelConvPointwise2dCustom_WT16_WG2_COB8() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t n, uint32_t cin, uint32_t cout,
                               uint32_t h, uint32_t w_in,
                               uint32_t w_tile, uint32_t w_group,
                               uint32_t w_group_len, uint32_t w_groups,
                               uint32_t tasks)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        n_ = n; cin_ = cin; cout_ = cout;
        h_ = h; w_ = w_in;

        wTile_ = w_tile;
        wGroup_ = w_group;
        wGroupLen_ = w_group_len;
        wGroups_ = w_groups;
        tasks_ = tasks;

        // UB:
        // - wBuf: cout*cin
        // - xLine0/xLine1: WTILE each
        // - yLine: WTILE for staging stores
        pipe_.InitBuffer(wBuf_, cout_ * cin_ * sizeof(float));
        pipe_.InitBuffer(xLine0Buf_, wTile_ * sizeof(float));
        pipe_.InitBuffer(xLine1Buf_, wTile_ * sizeof(float));
        pipe_.InitBuffer(yLineBuf_, wTile_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        // Cache weights to UB once per core.
        AscendC::LocalTensor<float> wLocal = wBuf_.Get<float>(0);
        AscendC::DataCopy(wLocal, wGm, cout_ * cin_);
        AscendC::PipeBarrier<PIPE_ALL>();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t tStart = coreId * tasksPerCore;
        uint32_t tEnd = tStart + tasksPerCore;
        if (tEnd > tasks_) tEnd = tasks_;
        if (tStart >= tEnd) return;

        AscendC::LocalTensor<float> xLine0 = xLine0Buf_.Get<float>(0);
        AscendC::LocalTensor<float> xLine1 = xLine1Buf_.Get<float>(0);
        AscendC::LocalTensor<float> yLine = yLineBuf_.Get<float>(0);

        const uint32_t COBLOCK = 8;   // divides COUT=128
        const uint32_t LANES = 16;    // WTILE=16
        const uint32_t WG = 2;        // WGROUP=2 specialized

        for (uint32_t t = tStart; t < tEnd; ++t) {
            uint32_t tmp = t;
            const uint32_t wGroupIdx = tmp % wGroups_;
            tmp /= wGroups_;
            const uint32_t hIdx = tmp % h_;
            const uint32_t nIdx = tmp / h_;
            if (nIdx >= n_) continue;

            const uint32_t w0 = wGroupIdx * wGroupLen_;
            if (w0 >= w_) continue;

            // In this specialization W is divisible by 32, so both stripes are full.
            // Keep generic bounds for safety.
            uint32_t wLen0 = wTile_;
            uint32_t wLen1 = wTile_;
            const uint32_t w1 = w0 + wTile_;
            if (w0 + wLen0 > w_) wLen0 = w_ - w0;
            if (w1 >= w_) wLen1 = 0;
            else if (w1 + wLen1 > w_) wLen1 = w_ - w1;

            const uint32_t xNHBase = (nIdx * cin_ * h_ + hIdx) * w_;   // x[n,0,h,0]
            const uint32_t yNHBase = (nIdx * cout_ * h_ + hIdx) * w_;  // y[n,0,h,0]

            for (uint32_t co0 = 0; co0 < cout_; co0 += COBLOCK) {
                float acc0[COBLOCK][LANES];
                float acc1[COBLOCK][LANES];
#pragma unroll
                for (uint32_t j = 0; j < COBLOCK; ++j) {
#pragma unroll
                    for (uint32_t lane = 0; lane < LANES; ++lane) {
                        acc0[j][lane] = 0.0f;
                        acc1[j][lane] = 0.0f;
                    }
                }

                for (uint32_t ci = 0; ci < cin_; ++ci) {
                    // Load x stripes for this ci (two short contiguous bursts)
                    const uint32_t xBase = xNHBase + ci * h_ * w_;
                    if (wLen0) {
                        AscendC::DataCopy(xLine0, xGm[xBase + w0], wLen0);
                    }
                    if (wLen1) {
                        AscendC::DataCopy(xLine1, xGm[xBase + w1], wLen1);
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // Load COBLOCK weights for this ci once into regs
                    float wv[COBLOCK];
#pragma unroll
                    for (uint32_t j = 0; j < COBLOCK; ++j) {
                        wv[j] = wLocal.GetValue((co0 + j) * cin_ + ci);
                    }

                    // Update stripe 0
#pragma unroll
                    for (uint32_t lane = 0; lane < LANES; ++lane) {
                        if (lane >= wLen0) break;
                        const float xv = xLine0.GetValue(lane);
#pragma unroll
                        for (uint32_t j = 0; j < COBLOCK; ++j) {
                            acc0[j][lane] += wv[j] * xv;
                        }
                    }

                    // Update stripe 1
#pragma unroll
                    for (uint32_t lane = 0; lane < LANES; ++lane) {
                        if (lane >= wLen1) break;
                        const float xv = xLine1.GetValue(lane);
#pragma unroll
                        for (uint32_t j = 0; j < COBLOCK; ++j) {
                            acc1[j][lane] += wv[j] * xv;
                        }
                    }
                }

                // Store COBLOCK output channels, two stripes each
#pragma unroll
                for (uint32_t j = 0; j < COBLOCK; ++j) {
                    const uint32_t yBase = yNHBase + (co0 + j) * h_ * w_;
                    if (wLen0) {
#pragma unroll
                        for (uint32_t lane = 0; lane < LANES; ++lane) {
                            if (lane >= wLen0) break;
                            yLine.SetValue(lane, acc0[j][lane]);
                        }
                        AscendC::DataCopy(yGm[yBase + w0], yLine, wLen0);
                    }
                    if (wLen1) {
#pragma unroll
                        for (uint32_t lane = 0; lane < LANES; ++lane) {
                            if (lane >= wLen1) break;
                            yLine.SetValue(lane, acc1[j][lane]);
                        }
                        AscendC::DataCopy(yGm[yBase + w1], yLine, wLen1);
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xLine0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xLine1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yLineBuf_;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n_{0}, cin_{0}, cout_{0}, h_{0}, w_{0};
    uint32_t wTile_{0}, wGroup_{0}, wGroupLen_{0}, wGroups_{0}, tasks_{0};
};

extern "C" __global__ __aicore__ void conv_pointwise2d_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvPointwise2dCustom_WT16_WG2_COB8 op;
    op.Init(x, weight, y,
            tiling_data.n, tiling_data.cin, tiling_data.cout,
            tiling_data.h, tiling_data.w,
            tiling_data.w_tile, tiling_data.w_group,
            tiling_data.w_group_len, tiling_data.w_groups,
            tiling_data.tasks);
    op.Process();
}
