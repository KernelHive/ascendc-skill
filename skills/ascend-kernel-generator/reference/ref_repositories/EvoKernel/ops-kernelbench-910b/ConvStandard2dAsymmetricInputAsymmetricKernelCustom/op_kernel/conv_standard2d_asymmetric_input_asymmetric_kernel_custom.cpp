
#include "kernel_operator.h"

class KernelConvStandard2dAsymAsym5x7_TaskTiled {
public:
    __aicore__ inline KernelConvStandard2dAsymAsym5x7_TaskTiled() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t rows, uint32_t ow,
                               uint32_t H, uint32_t W,
                               uint32_t CIN, uint32_t COUT, uint32_t OH,
                               uint32_t tileOw, uint32_t owTiles, uint32_t tasks)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        rows_ = rows;
        ow_ = ow;
        H_ = H;
        W_ = W;
        CIN_ = CIN;
        COUT_ = COUT;
        OH_ = OH;
        tileOw_ = tileOw;
        owTiles_ = owTiles;
        tasks_ = tasks;
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t N = 8;
        constexpr uint32_t KH = 5;
        constexpr uint32_t KW = 7;

        const uint32_t coreId = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t tStart = coreId * tasksPerCore;
        uint32_t tEnd = tStart + tasksPerCore;
        if (tEnd > tasks_) tEnd = tasks_;

        const uint32_t OW = ow_;
        const uint32_t W = W_;
        const uint32_t H = H_;
        const uint32_t CIN = CIN_;
        const uint32_t COUT = COUT_;
        const uint32_t OH = OH_;
        const uint32_t TILE_OW = tileOw_;
        const uint32_t OW_TILES = owTiles_;

        for (uint32_t task = tStart; task < tEnd; ++task) {
            const uint32_t row = task / OW_TILES;
            const uint32_t owTileId = task - row * OW_TILES;
            const uint32_t ow0 = owTileId * TILE_OW;

            uint32_t tile = OW - ow0;
            if (tile > TILE_OW) tile = TILE_OW;

            // row index order: ((n*COUT + co)*OH + oh)
            uint32_t tmp = row;
            const uint32_t oh = tmp % OH;
            tmp /= OH;
            const uint32_t co = tmp % COUT;
            const uint32_t n  = tmp / COUT;
            if (n >= N) continue;

            const uint32_t yBase = ((n * COUT + co) * OH + oh) * OW + ow0;

            // Hoist invariant bases.
            const uint32_t xNBase = n * (CIN * H * W);
            const uint32_t wCoBase = co * (CIN * KH * KW);

            float acc[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

            if (tile == 16) {
                // Fast path: branch-free.
                for (uint32_t ci = 0; ci < CIN; ++ci) {
                    const uint32_t xNcBase = xNBase + ci * (H * W);
                    const uint32_t wCociBase = wCoBase + ci * (KH * KW);

                    const uint32_t ih0 = oh + 0U;
                    const uint32_t ih1 = oh + 1U;
                    const uint32_t ih2 = oh + 2U;
                    const uint32_t ih3 = oh + 3U;
                    const uint32_t ih4 = oh + 4U;

                    const uint32_t xRow0 = xNcBase + ih0 * W + ow0;
                    const uint32_t xRow1 = xNcBase + ih1 * W + ow0;
                    const uint32_t xRow2 = xNcBase + ih2 * W + ow0;
                    const uint32_t xRow3 = xNcBase + ih3 * W + ow0;
                    const uint32_t xRow4 = xNcBase + ih4 * W + ow0;

#pragma unroll
                    for (uint32_t kh = 0; kh < KH; ++kh) {
                        const uint32_t wRowBase = wCociBase + kh * KW;

                        const float w0 = wGm.GetValue(wRowBase + 0U);
                        const float w1 = wGm.GetValue(wRowBase + 1U);
                        const float w2 = wGm.GetValue(wRowBase + 2U);
                        const float w3 = wGm.GetValue(wRowBase + 3U);
                        const float w4 = wGm.GetValue(wRowBase + 4U);
                        const float w5 = wGm.GetValue(wRowBase + 5U);
                        const float w6 = wGm.GetValue(wRowBase + 6U);

                        const uint32_t xBase = (kh == 0 ? xRow0 : (kh == 1 ? xRow1 : (kh == 2 ? xRow2 : (kh == 3 ? xRow3 : xRow4))));

                        // For 16 outputs we need 22 input values (16 + KW - 1).
                        const float x0  = xGm.GetValue(xBase + 0U);
                        const float x1  = xGm.GetValue(xBase + 1U);
                        const float x2  = xGm.GetValue(xBase + 2U);
                        const float x3  = xGm.GetValue(xBase + 3U);
                        const float x4  = xGm.GetValue(xBase + 4U);
                        const float x5  = xGm.GetValue(xBase + 5U);
                        const float x6  = xGm.GetValue(xBase + 6U);
                        const float x7  = xGm.GetValue(xBase + 7U);
                        const float x8  = xGm.GetValue(xBase + 8U);
                        const float x9  = xGm.GetValue(xBase + 9U);
                        const float x10 = xGm.GetValue(xBase + 10U);
                        const float x11 = xGm.GetValue(xBase + 11U);
                        const float x12 = xGm.GetValue(xBase + 12U);
                        const float x13 = xGm.GetValue(xBase + 13U);
                        const float x14 = xGm.GetValue(xBase + 14U);
                        const float x15 = xGm.GetValue(xBase + 15U);
                        const float x16 = xGm.GetValue(xBase + 16U);
                        const float x17 = xGm.GetValue(xBase + 17U);
                        const float x18 = xGm.GetValue(xBase + 18U);
                        const float x19 = xGm.GetValue(xBase + 19U);
                        const float x20 = xGm.GetValue(xBase + 20U);
                        const float x21 = xGm.GetValue(xBase + 21U);

                        acc[0]  += x0*w0  + x1*w1  + x2*w2  + x3*w3  + x4*w4  + x5*w5  + x6*w6;
                        acc[1]  += x1*w0  + x2*w1  + x3*w2  + x4*w3  + x5*w4  + x6*w5  + x7*w6;
                        acc[2]  += x2*w0  + x3*w1  + x4*w2  + x5*w3  + x6*w4  + x7*w5  + x8*w6;
                        acc[3]  += x3*w0  + x4*w1  + x5*w2  + x6*w3  + x7*w4  + x8*w5  + x9*w6;
                        acc[4]  += x4*w0  + x5*w1  + x6*w2  + x7*w3  + x8*w4  + x9*w5  + x10*w6;
                        acc[5]  += x5*w0  + x6*w1  + x7*w2  + x8*w3  + x9*w4  + x10*w5 + x11*w6;
                        acc[6]  += x6*w0  + x7*w1  + x8*w2  + x9*w3  + x10*w4 + x11*w5 + x12*w6;
                        acc[7]  += x7*w0  + x8*w1  + x9*w2  + x10*w3 + x11*w4 + x12*w5 + x13*w6;
                        acc[8]  += x8*w0  + x9*w1  + x10*w2 + x11*w3 + x12*w4 + x13*w5 + x14*w6;
                        acc[9]  += x9*w0  + x10*w1 + x11*w2 + x12*w3 + x13*w4 + x14*w5 + x15*w6;
                        acc[10] += x10*w0 + x11*w1 + x12*w2 + x13*w3 + x14*w4 + x15*w5 + x16*w6;
                        acc[11] += x11*w0 + x12*w1 + x13*w2 + x14*w3 + x15*w4 + x16*w5 + x17*w6;
                        acc[12] += x12*w0 + x13*w1 + x14*w2 + x15*w3 + x16*w4 + x17*w5 + x18*w6;
                        acc[13] += x13*w0 + x14*w1 + x15*w2 + x16*w3 + x17*w4 + x18*w5 + x19*w6;
                        acc[14] += x14*w0 + x15*w1 + x16*w2 + x17*w3 + x18*w4 + x19*w5 + x20*w6;
                        acc[15] += x15*w0 + x16*w1 + x17*w2 + x18*w3 + x19*w4 + x20*w5 + x21*w6;
                    }
                }

#pragma unroll
                for (uint32_t i = 0; i < 16; ++i) {
                    yGm.SetValue(yBase + i, acc[i]);
                }
            } else {
                // Tail path: minimal bounds checks outside the hot path.
                for (uint32_t ci = 0; ci < CIN; ++ci) {
                    const uint32_t xNcBase = xNBase + ci * (H * W);
                    const uint32_t wCociBase = wCoBase + ci * (KH * KW);

                    for (uint32_t kh = 0; kh < KH; ++kh) {
                        const uint32_t ih = oh + kh;
                        const uint32_t xBase = xNcBase + ih * W + ow0;
                        const uint32_t wRowBase = wCociBase + kh * KW;

                        const float w0 = wGm.GetValue(wRowBase + 0U);
                        const float w1 = wGm.GetValue(wRowBase + 1U);
                        const float w2 = wGm.GetValue(wRowBase + 2U);
                        const float w3 = wGm.GetValue(wRowBase + 3U);
                        const float w4 = wGm.GetValue(wRowBase + 4U);
                        const float w5 = wGm.GetValue(wRowBase + 5U);
                        const float w6 = wGm.GetValue(wRowBase + 6U);

                        // For tile outputs, need tile+6 input values.
                        for (uint32_t o = 0; o < tile; ++o) {
                            const float a0 = xGm.GetValue(xBase + o + 0U);
                            const float a1 = xGm.GetValue(xBase + o + 1U);
                            const float a2 = xGm.GetValue(xBase + o + 2U);
                            const float a3 = xGm.GetValue(xBase + o + 3U);
                            const float a4 = xGm.GetValue(xBase + o + 4U);
                            const float a5 = xGm.GetValue(xBase + o + 5U);
                            const float a6 = xGm.GetValue(xBase + o + 6U);
                            acc[o] += a0*w0 + a1*w1 + a2*w2 + a3*w3 + a4*w4 + a5*w5 + a6*w6;
                        }
                    }
                }

                for (uint32_t i = 0; i < tile; ++i) {
                    yGm.SetValue(yBase + i, acc[i]);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t rows_{0};
    uint32_t ow_{0};
    uint32_t H_{0};
    uint32_t W_{0};
    uint32_t CIN_{0};
    uint32_t COUT_{0};
    uint32_t OH_{0};
    uint32_t tileOw_{16};
    uint32_t owTiles_{0};
    uint32_t tasks_{0};
};

extern "C" __global__ __aicore__ void conv_standard2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard2dAsymAsym5x7_TaskTiled op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.ow,
            tiling_data.h, tiling_data.w,
            tiling_data.cin, tiling_data.cout, tiling_data.oh,
            tiling_data.tile_ow, tiling_data.ow_tiles, tiling_data.tasks);
    op.Process();
}
