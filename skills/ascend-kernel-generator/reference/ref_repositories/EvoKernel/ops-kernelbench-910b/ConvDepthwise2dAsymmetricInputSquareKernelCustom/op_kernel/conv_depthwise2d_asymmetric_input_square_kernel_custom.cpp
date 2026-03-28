
#include "kernel_operator.h"

// Depthwise Conv2d specialized for:
// x: [64,128,256,512], w: [128,1,3,3], y: [64,128,254,510]
// stride=1, pad=0, dil=1, bias=False
// Optimized by tiling OW and mapping tasks across cores.

class KernelConvDepthwise2dAsymInSquareOwTiled {
public:
    __aicore__ inline KernelConvDepthwise2dAsymInSquareOwTiled() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t OH, uint32_t OW,
                               uint32_t tileOw, uint32_t owTiles,
                               uint32_t rows, uint32_t tasks)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        N_ = N; C_ = C; H_ = H; W_ = W;
        OH_ = OH; OW_ = OW;
        tileOw_ = tileOw;
        owTiles_ = owTiles;
        rows_ = rows;
        tasks_ = tasks;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t taskStart = coreId * tasksPerCore;
        uint32_t taskEnd = taskStart + tasksPerCore;
        if (taskEnd > tasks_) taskEnd = tasks_;

        const uint32_t C = C_;
        const uint32_t H = H_;
        const uint32_t W = W_;
        const uint32_t OH = OH_;
        const uint32_t OW = OW_;
        const uint32_t tileOw = tileOw_;
        const uint32_t owTiles = owTiles_;

        const uint32_t HW = H * W;
        const uint32_t OHW = OH * OW;

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / owTiles;
            const uint32_t owTileId = task - row * owTiles;
            const uint32_t ow0 = owTileId * tileOw;

            uint32_t tile = OW - ow0;
            if (tile > tileOw) tile = tileOw;

            // row -> (n,c,oh) where row = (n*C + c)*OH + oh
            uint32_t tmp = row;
            const uint32_t oh = tmp % OH;
            tmp /= OH;
            const uint32_t c = tmp % C;
            const uint32_t n = tmp / C;

            const uint32_t xNcBase = (n * C + c) * HW;
            const uint32_t yNcBase = (n * C + c) * OHW;

            const uint32_t ih0 = oh;
            const uint32_t ih1 = oh + 1U;
            const uint32_t ih2 = oh + 2U;

            const uint32_t xRow0 = xNcBase + ih0 * W + ow0;
            const uint32_t xRow1 = xNcBase + ih1 * W + ow0;
            const uint32_t xRow2 = xNcBase + ih2 * W + ow0;

            const uint32_t yBase = yNcBase + oh * OW + ow0;

            // weights: [C,1,3,3] flattened per channel
            const uint32_t wBase = c * 9U;
            const float w00 = wGm.GetValue(wBase + 0U);
            const float w01 = wGm.GetValue(wBase + 1U);
            const float w02 = wGm.GetValue(wBase + 2U);
            const float w10 = wGm.GetValue(wBase + 3U);
            const float w11 = wGm.GetValue(wBase + 4U);
            const float w12 = wGm.GetValue(wBase + 5U);
            const float w20 = wGm.GetValue(wBase + 6U);
            const float w21 = wGm.GetValue(wBase + 7U);
            const float w22 = wGm.GetValue(wBase + 8U);

            float acc[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

            if (tile == 16) {
                // Fast path: fully unrolled, no per-lane bounds check.
#pragma unroll
                for (uint32_t o = 0; o < 16; ++o) {
                    const float a00 = xGm.GetValue(xRow0 + o + 0U);
                    const float a01 = xGm.GetValue(xRow0 + o + 1U);
                    const float a02 = xGm.GetValue(xRow0 + o + 2U);

                    const float a10 = xGm.GetValue(xRow1 + o + 0U);
                    const float a11 = xGm.GetValue(xRow1 + o + 1U);
                    const float a12 = xGm.GetValue(xRow1 + o + 2U);

                    const float a20 = xGm.GetValue(xRow2 + o + 0U);
                    const float a21 = xGm.GetValue(xRow2 + o + 1U);
                    const float a22 = xGm.GetValue(xRow2 + o + 2U);

                    float out = 0.0f;
                    out += a00 * w00; out += a01 * w01; out += a02 * w02;
                    out += a10 * w10; out += a11 * w11; out += a12 * w12;
                    out += a20 * w20; out += a21 * w21; out += a22 * w22;
                    acc[o] = out;
                }
#pragma unroll
                for (uint32_t i = 0; i < 16; ++i) {
                    yGm.SetValue(yBase + i, acc[i]);
                }
            } else {
                // Tail path.
                for (uint32_t o = 0; o < tile; ++o) {
                    const float a00 = xGm.GetValue(xRow0 + o + 0U);
                    const float a01 = xGm.GetValue(xRow0 + o + 1U);
                    const float a02 = xGm.GetValue(xRow0 + o + 2U);

                    const float a10 = xGm.GetValue(xRow1 + o + 0U);
                    const float a11 = xGm.GetValue(xRow1 + o + 1U);
                    const float a12 = xGm.GetValue(xRow1 + o + 2U);

                    const float a20 = xGm.GetValue(xRow2 + o + 0U);
                    const float a21 = xGm.GetValue(xRow2 + o + 1U);
                    const float a22 = xGm.GetValue(xRow2 + o + 2U);

                    float out = 0.0f;
                    out += a00 * w00; out += a01 * w01; out += a02 * w02;
                    out += a10 * w10; out += a11 * w11; out += a12 * w12;
                    out += a20 * w20; out += a21 * w21; out += a22 * w22;
                    yGm.SetValue(yBase + o, out);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t N_{0}, C_{0}, H_{0}, W_{0};
    uint32_t OH_{0}, OW_{0};
    uint32_t tileOw_{16};
    uint32_t owTiles_{0};
    uint32_t rows_{0};
    uint32_t tasks_{0};
};

extern "C" __global__ __aicore__ void conv_depthwise2d_asymmetric_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvDepthwise2dAsymInSquareOwTiled op;
    op.Init(x, weight, y,
            tiling_data.n, tiling_data.c, tiling_data.h, tiling_data.w,
            tiling_data.oh, tiling_data.ow,
            tiling_data.tile_ow, tiling_data.ow_tiles,
            tiling_data.rows, tiling_data.tasks);
    op.Process();
}
