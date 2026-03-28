
#include "kernel_operator.h"

class KernelConvDepthwiseSeparable2d_Fused_OW8_CO16 {
public:
    __aicore__ inline KernelConvDepthwiseSeparable2d_Fused_OW8_CO16() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wdw, GM_ADDR wpw, GM_ADDR y,
                               uint32_t tasks,
                               uint32_t N, uint32_t CIN, uint32_t COUT,
                               uint32_t H, uint32_t W, uint32_t OH, uint32_t OW,
                               uint32_t tileOw, uint32_t owTiles,
                               uint32_t coTile, uint32_t coTiles,
                               uint32_t owInteriorStart, uint32_t owInteriorEnd,
                               uint32_t ohInteriorStart, uint32_t ohInteriorEnd)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wDwGm.SetGlobalBuffer((__gm__ float*)wdw);
        wPwGm.SetGlobalBuffer((__gm__ float*)wpw);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        tasks_ = tasks;
        N_ = N; CIN_ = CIN; COUT_ = COUT;
        H_ = H; W_ = W; OH_ = OH; OW_ = OW;
        tileOw_ = tileOw; owTiles_ = owTiles;
        coTile_ = coTile; coTiles_ = coTiles;
        owInteriorStart_ = owInteriorStart; owInteriorEnd_ = owInteriorEnd;
        ohInteriorStart_ = ohInteriorStart; ohInteriorEnd_ = ohInteriorEnd;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t tStart = coreId * tasksPerCore;
        uint32_t tEnd = tStart + tasksPerCore;
        if (tEnd > tasks_) tEnd = tasks_;

        constexpr int32_t PAD = 1;
        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;

        const uint32_t CIN = CIN_;
        const uint32_t COUT = COUT_;
        const uint32_t H = H_;
        const uint32_t W = W_;
        const uint32_t OH = OH_;
        const uint32_t OW = OW_;
        const uint32_t tileOw = tileOw_;
        const uint32_t owTiles = owTiles_;
        const uint32_t coTile = coTile_;
        const uint32_t coTiles = coTiles_;

        for (uint32_t t = tStart; t < tEnd; ++t) {
            uint32_t tmp = t;
            const uint32_t coTileId = tmp % coTiles;
            tmp /= coTiles;
            const uint32_t tile = tmp % owTiles;
            tmp /= owTiles;
            const uint32_t oh = tmp % OH;
            const uint32_t n  = tmp / OH;
            if (n >= N_) continue;

            const uint32_t co0 = coTileId * coTile;
            if (co0 >= COUT) continue;

            const uint32_t ow0 = tile * tileOw;
            uint32_t owLen = tileOw;
            if (ow0 + owLen > OW) owLen = OW - ow0;

            const bool owInterior = (ow0 >= owInteriorStart_) && (ow0 < owInteriorEnd_);
            const bool ohInterior = (oh >= ohInteriorStart_) && (oh < ohInteriorEnd_);
            const bool fullInterior = owInterior && ohInterior && (owLen == 8);

            // Accumulators: 16 output channels x 8 lanes
            float acc[16][8];
#pragma unroll
            for (uint32_t j = 0; j < 16; ++j) {
#pragma unroll
                for (uint32_t lane = 0; lane < 8; ++lane) {
                    acc[j][lane] = 0.0f;
                }
            }

            const uint32_t xNBase = n * CIN * H * W;
            const int32_t ih0 = (int32_t)oh - PAD;

            // Hoist row bases for general path checks
            const int32_t ihA = ih0 + 0;
            const int32_t ihB = ih0 + 1;
            const int32_t ihC = ih0 + 2;

            for (uint32_t ci = 0; ci < CIN; ++ci) {
                const uint32_t xNcBase = xNBase + ci * H * W;
                const uint32_t wdwBase = ci * (KH * KW);

                // Depthwise weights (registers)
                const float w00 = wDwGm.GetValue(wdwBase + 0);
                const float w01 = wDwGm.GetValue(wdwBase + 1);
                const float w02 = wDwGm.GetValue(wdwBase + 2);
                const float w10 = wDwGm.GetValue(wdwBase + 3);
                const float w11 = wDwGm.GetValue(wdwBase + 4);
                const float w12 = wDwGm.GetValue(wdwBase + 5);
                const float w20 = wDwGm.GetValue(wdwBase + 6);
                const float w21 = wDwGm.GetValue(wdwBase + 7);
                const float w22 = wDwGm.GetValue(wdwBase + 8);

                // Compute 8 depthwise outputs for this ci (dwLane[0..7])
                float dwLane[8];
#pragma unroll
                for (uint32_t lane = 0; lane < 8; ++lane) dwLane[lane] = 0.0f;

                if (fullInterior) {
                    const uint32_t ihAu = (uint32_t)ihA;
                    const uint32_t ihBu = (uint32_t)ihB;
                    const uint32_t ihCu = (uint32_t)ihC;

                    const uint32_t xRowA = xNcBase + ihAu * W;
                    const uint32_t xRowB = xNcBase + ihBu * W;
                    const uint32_t xRowC = xNcBase + ihCu * W;

                    const uint32_t iwStart = ow0 - 1U;

                    float a0 = xGm.GetValue(xRowA + (iwStart + 0));
                    float a1 = xGm.GetValue(xRowA + (iwStart + 1));
                    float a2 = xGm.GetValue(xRowA + (iwStart + 2));

                    float b0 = xGm.GetValue(xRowB + (iwStart + 0));
                    float b1 = xGm.GetValue(xRowB + (iwStart + 1));
                    float b2 = xGm.GetValue(xRowB + (iwStart + 2));

                    float c0 = xGm.GetValue(xRowC + (iwStart + 0));
                    float c1 = xGm.GetValue(xRowC + (iwStart + 1));
                    float c2 = xGm.GetValue(xRowC + (iwStart + 2));

#pragma unroll
                    for (uint32_t lane = 0; lane < 8; ++lane) {
                        float dw =
                            (a0 * w00 + a1 * w01 + a2 * w02) +
                            (b0 * w10 + b1 * w11 + b2 * w12) +
                            (c0 * w20 + c1 * w21 + c2 * w22);
                        dwLane[lane] = dw;

                        const uint32_t tcol = iwStart + lane + 3U;
                        const float a3 = xGm.GetValue(xRowA + tcol);
                        const float b3 = xGm.GetValue(xRowB + tcol);
                        const float c3 = xGm.GetValue(xRowC + tcol);

                        a0 = a1; a1 = a2; a2 = a3;
                        b0 = b1; b1 = b2; b2 = b3;
                        c0 = c1; c1 = c2; c2 = c3;
                    }
                } else {
                    // boundary-safe per lane (only up to owLen)
#pragma unroll
                    for (uint32_t lane = 0; lane < 8; ++lane) {
                        if (lane >= owLen) break;
                        const int32_t ow = (int32_t)ow0 + (int32_t)lane;
                        const int32_t iw0 = ow - PAD;

                        float dw = 0.0f;

                        if ((uint32_t)ihA < H) {
                            const uint32_t rowBase = xNcBase + (uint32_t)ihA * W;
                            const int32_t i0 = iw0 + 0, i1 = iw0 + 1, i2 = iw0 + 2;
                            if ((uint32_t)i0 < W) dw += xGm.GetValue(rowBase + (uint32_t)i0) * w00;
                            if ((uint32_t)i1 < W) dw += xGm.GetValue(rowBase + (uint32_t)i1) * w01;
                            if ((uint32_t)i2 < W) dw += xGm.GetValue(rowBase + (uint32_t)i2) * w02;
                        }
                        if ((uint32_t)ihB < H) {
                            const uint32_t rowBase = xNcBase + (uint32_t)ihB * W;
                            const int32_t i0 = iw0 + 0, i1 = iw0 + 1, i2 = iw0 + 2;
                            if ((uint32_t)i0 < W) dw += xGm.GetValue(rowBase + (uint32_t)i0) * w10;
                            if ((uint32_t)i1 < W) dw += xGm.GetValue(rowBase + (uint32_t)i1) * w11;
                            if ((uint32_t)i2 < W) dw += xGm.GetValue(rowBase + (uint32_t)i2) * w12;
                        }
                        if ((uint32_t)ihC < H) {
                            const uint32_t rowBase = xNcBase + (uint32_t)ihC * W;
                            const int32_t i0 = iw0 + 0, i1 = iw0 + 1, i2 = iw0 + 2;
                            if ((uint32_t)i0 < W) dw += xGm.GetValue(rowBase + (uint32_t)i0) * w20;
                            if ((uint32_t)i1 < W) dw += xGm.GetValue(rowBase + (uint32_t)i1) * w21;
                            if ((uint32_t)i2 < W) dw += xGm.GetValue(rowBase + (uint32_t)i2) * w22;
                        }
                        dwLane[lane] = dw;
                    }
                }

                // Load pointwise weights for 16 output channels for this ci, then update all 8 lanes.
                float pw[16];
#pragma unroll
                for (uint32_t j = 0; j < 16; ++j) {
                    const uint32_t co = co0 + j;
                    pw[j] = (co < COUT) ? wPwGm.GetValue(co * CIN + ci) : 0.0f;
                }

#pragma unroll
                for (uint32_t j = 0; j < 16; ++j) {
                    const float wv = pw[j];
#pragma unroll
                    for (uint32_t lane = 0; lane < 8; ++lane) {
                        // lane beyond owLen has dwLane=0 from boundary path; safe for fullInterior too.
                        acc[j][lane] += dwLane[lane] * wv;
                    }
                }
            } // ci

            // Store
#pragma unroll
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t co = co0 + j;
                if (co >= COUT) continue;
                const uint32_t yBase = ((n * COUT + co) * OH + oh) * OW + ow0;
                if (owLen > 0) yGm.SetValue(yBase + 0, acc[j][0]);
                if (owLen > 1) yGm.SetValue(yBase + 1, acc[j][1]);
                if (owLen > 2) yGm.SetValue(yBase + 2, acc[j][2]);
                if (owLen > 3) yGm.SetValue(yBase + 3, acc[j][3]);
                if (owLen > 4) yGm.SetValue(yBase + 4, acc[j][4]);
                if (owLen > 5) yGm.SetValue(yBase + 5, acc[j][5]);
                if (owLen > 6) yGm.SetValue(yBase + 6, acc[j][6]);
                if (owLen > 7) yGm.SetValue(yBase + 7, acc[j][7]);
            }
        } // tasks
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wDwGm;
    AscendC::GlobalTensor<float> wPwGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t tasks_{0};
    uint32_t N_{0}, CIN_{0}, COUT_{0};
    uint32_t H_{0}, W_{0}, OH_{0}, OW_{0};
    uint32_t tileOw_{0}, owTiles_{0};
    uint32_t coTile_{0}, coTiles_{0};
    uint32_t owInteriorStart_{0}, owInteriorEnd_{0};
    uint32_t ohInteriorStart_{0}, ohInteriorEnd_{0};
};

extern "C" __global__ __aicore__ void conv_depthwise_separable2d_custom(
    GM_ADDR x,
    GM_ADDR w_depthwise,
    GM_ADDR w_pointwise,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvDepthwiseSeparable2d_Fused_OW8_CO16 op;
    op.Init(x, w_depthwise, w_pointwise, y,
            tiling_data.tasks,
            tiling_data.n, tiling_data.cin, tiling_data.cout,
            tiling_data.h, tiling_data.w, tiling_data.oh, tiling_data.ow,
            tiling_data.tile_ow, tiling_data.ow_tiles,
            tiling_data.co_tile, tiling_data.co_tiles,
            tiling_data.ow_interior_start, tiling_data.ow_interior_end,
            tiling_data.oh_interior_start, tiling_data.oh_interior_end);
    op.Process();
}
