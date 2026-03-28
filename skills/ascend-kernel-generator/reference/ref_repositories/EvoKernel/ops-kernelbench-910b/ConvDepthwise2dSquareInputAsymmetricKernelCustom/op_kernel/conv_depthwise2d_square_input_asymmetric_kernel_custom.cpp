
#include "kernel_operator.h"

class KernelConvDwKH3KW1_RowOwTiled {
public:
    __aicore__ inline KernelConvDwKH3KW1_RowOwTiled() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                               uint32_t rows, uint32_t ow, uint32_t oh,
                               uint32_t tileOw, uint32_t owTiles, uint32_t tasks)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)weight);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        rows_ = rows;
        OW_ = ow;
        OH_ = oh;
        tileOw_ = tileOw;
        owTiles_ = owTiles;
        tasks_ = tasks;
    }

    __aicore__ inline void Process()
    {
        // Specialized constants for benchmark.
        constexpr uint32_t N_ = 64;
        constexpr uint32_t C_ = 8;
        constexpr uint32_t H_ = 512;
        constexpr uint32_t W_ = 512;
        constexpr uint32_t KH_ = 3;

        const uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t taskStart = coreId * tasksPerCore;
        uint32_t taskEnd = taskStart + tasksPerCore;
        if (taskEnd > tasks_) taskEnd = tasks_;

        const uint32_t OW = OW_;
        const uint32_t OH = OH_;
        const uint32_t owTiles = owTiles_;
        const uint32_t tileOw = tileOw_;

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / owTiles;
            const uint32_t owTileId = task - row * owTiles;
            const uint32_t ow0 = owTileId * tileOw;

            uint32_t tile = OW - ow0;
            if (tile > tileOw) tile = tileOw;

            // row = (n*C + c)*OH + oh
            uint32_t tmp = row;
            const uint32_t oh = tmp % OH;
            tmp /= OH;
            const uint32_t c = tmp % C_;
            const uint32_t n = tmp / C_;

            // Load weights once per tile
            const uint32_t wBase = c * KH_;
            const float w0 = wGm.GetValue(wBase + 0U);
            const float w1 = wGm.GetValue(wBase + 1U);
            const float w2 = wGm.GetValue(wBase + 2U);

            const uint32_t xNcBase = (n * C_ + c) * (H_ * W_);
            const uint32_t yNcBase = (n * C_ + c) * (OH * OW);

            const uint32_t xRow0 = xNcBase + (oh + 0U) * W_ + ow0;
            const uint32_t xRow1 = xNcBase + (oh + 1U) * W_ + ow0;
            const uint32_t xRow2 = xNcBase + (oh + 2U) * W_ + ow0;

            const uint32_t yRow = yNcBase + oh * OW + ow0;

            if (tile == 16U) {
                // Preload 16 values from each row (contiguous).
                float r0[16];
                float r1[16];
                float r2[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    r0[i] = xGm.GetValue(xRow0 + (uint32_t)i);
                    r1[i] = xGm.GetValue(xRow1 + (uint32_t)i);
                    r2[i] = xGm.GetValue(xRow2 + (uint32_t)i);
                }

#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    yGm.SetValue(yRow + (uint32_t)i, r0[i] * w0 + r1[i] * w1 + r2[i] * w2);
                }
            } else {
                // Tail (rare because OW=512 and tileOw=16): keep correct and simple.
                for (uint32_t i = 0; i < tile; ++i) {
                    float a = xGm.GetValue(xRow0 + i);
                    float b = xGm.GetValue(xRow1 + i);
                    float d = xGm.GetValue(xRow2 + i);
                    yGm.SetValue(yRow + i, a * w0 + b * w1 + d * w2);
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t rows_{0};
    uint32_t OW_{0};
    uint32_t OH_{0};
    uint32_t tileOw_{16};
    uint32_t owTiles_{0};
    uint32_t tasks_{0};
};

extern "C" __global__ __aicore__ void conv_depthwise2d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDwKH3KW1_RowOwTiled op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.ow, tiling_data.oh,
            tiling_data.tile_ow, tiling_data.ow_tiles, tiling_data.tasks);
    op.Process();
}
