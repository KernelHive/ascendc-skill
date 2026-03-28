
#include "kernel_operator.h"

class KernelConvStandard2dAsymSquareOwTiledFastPath {
public:
    __aicore__ inline KernelConvStandard2dAsymSquareOwTiledFastPath() {}

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
        const uint32_t coreId = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t tasksPerCore = (tasks_ + coreNum - 1U) / coreNum;
        uint32_t taskStart = coreId * tasksPerCore;
        uint32_t taskEnd = taskStart + tasksPerCore;
        if (taskEnd > tasks_) taskEnd = tasks_;

        const uint32_t ow = ow_;
        const uint32_t W = W_;
        const uint32_t H = H_;
        const uint32_t CIN = CIN_;
        const uint32_t COUT = COUT_;
        const uint32_t OH = OH_;
        const uint32_t tileOw = tileOw_;
        const uint32_t owTiles = owTiles_;

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / owTiles;
            const uint32_t owTileId = task - row * owTiles;
            const uint32_t ow0 = owTileId * tileOw;

            uint32_t tile = ow - ow0;
            if (tile > tileOw) tile = tileOw;

            uint32_t tmp = row;
            const uint32_t oh = tmp % OH;
            tmp /= OH;
            const uint32_t co = tmp % COUT;
            const uint32_t n  = tmp / COUT;

            const uint32_t yBase = ((n * COUT + co) * OH + oh) * ow + ow0;

            const uint32_t ih0 = oh;
            const uint32_t ih1 = oh + 1U;
            const uint32_t ih2 = oh + 2U;

            float acc16[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) acc16[i] = 0.0f;

            // Hoist base pointers to reduce address-gen scalar pressure.
            const uint32_t wCoBase = co * (CIN * 9U);
            const uint32_t xNBase = n * (CIN * H * W);

            if (tile == 16) {
                // Fast path: branch-free inner loop.
                for (uint32_t ci = 0; ci < CIN; ++ci) {
                    const uint32_t wBase = wCoBase + ci * 9U;

                    const float w00 = wGm.GetValue(wBase + 0U);
                    const float w01 = wGm.GetValue(wBase + 1U);
                    const float w02 = wGm.GetValue(wBase + 2U);
                    const float w10 = wGm.GetValue(wBase + 3U);
                    const float w11 = wGm.GetValue(wBase + 4U);
                    const float w12 = wGm.GetValue(wBase + 5U);
                    const float w20 = wGm.GetValue(wBase + 6U);
                    const float w21 = wGm.GetValue(wBase + 7U);
                    const float w22 = wGm.GetValue(wBase + 8U);

                    const uint32_t xNcBase = xNBase + ci * (H * W);
                    const uint32_t xRow0 = xNcBase + ih0 * W + ow0;
                    const uint32_t xRow1 = xNcBase + ih1 * W + ow0;
                    const uint32_t xRow2 = xNcBase + ih2 * W + ow0;

#pragma unroll
                    for (uint32_t o = 0; o < 16; ++o) {
                        const float a0 = xGm.GetValue(xRow0 + o + 0U);
                        const float a1 = xGm.GetValue(xRow0 + o + 1U);
                        const float a2 = xGm.GetValue(xRow0 + o + 2U);

                        const float b0 = xGm.GetValue(xRow1 + o + 0U);
                        const float b1 = xGm.GetValue(xRow1 + o + 1U);
                        const float b2 = xGm.GetValue(xRow1 + o + 2U);

                        const float c0 = xGm.GetValue(xRow2 + o + 0U);
                        const float c1 = xGm.GetValue(xRow2 + o + 1U);
                        const float c2 = xGm.GetValue(xRow2 + o + 2U);

                        float sum = 0.0f;
                        sum += a0 * w00; sum += a1 * w01; sum += a2 * w02;
                        sum += b0 * w10; sum += b1 * w11; sum += b2 * w12;
                        sum += c0 * w20; sum += c1 * w21; sum += c2 * w22;

                        acc16[o] += sum;
                    }
                }

#pragma unroll
                for (uint32_t i = 0; i < 16; ++i) {
                    yGm.SetValue(yBase + i, acc16[i]);
                }
            } else {
                // Tail path: minimal bounds checks, not in the common-case hot loop.
                for (uint32_t ci = 0; ci < CIN; ++ci) {
                    const uint32_t wBase = wCoBase + ci * 9U;

                    const float w00 = wGm.GetValue(wBase + 0U);
                    const float w01 = wGm.GetValue(wBase + 1U);
                    const float w02 = wGm.GetValue(wBase + 2U);
                    const float w10 = wGm.GetValue(wBase + 3U);
                    const float w11 = wGm.GetValue(wBase + 4U);
                    const float w12 = wGm.GetValue(wBase + 5U);
                    const float w20 = wGm.GetValue(wBase + 6U);
                    const float w21 = wGm.GetValue(wBase + 7U);
                    const float w22 = wGm.GetValue(wBase + 8U);

                    const uint32_t xNcBase = xNBase + ci * (H * W);
                    const uint32_t xRow0 = xNcBase + ih0 * W + ow0;
                    const uint32_t xRow1 = xNcBase + ih1 * W + ow0;
                    const uint32_t xRow2 = xNcBase + ih2 * W + ow0;

                    for (uint32_t o = 0; o < tile; ++o) {
                        const float a0 = xGm.GetValue(xRow0 + o + 0U);
                        const float a1 = xGm.GetValue(xRow0 + o + 1U);
                        const float a2 = xGm.GetValue(xRow0 + o + 2U);

                        const float b0 = xGm.GetValue(xRow1 + o + 0U);
                        const float b1 = xGm.GetValue(xRow1 + o + 1U);
                        const float b2 = xGm.GetValue(xRow1 + o + 2U);

                        const float c0 = xGm.GetValue(xRow2 + o + 0U);
                        const float c1 = xGm.GetValue(xRow2 + o + 1U);
                        const float c2 = xGm.GetValue(xRow2 + o + 2U);

                        float sum = 0.0f;
                        sum += a0 * w00; sum += a1 * w01; sum += a2 * w02;
                        sum += b0 * w10; sum += b1 * w11; sum += b2 * w12;
                        sum += c0 * w20; sum += c1 * w21; sum += c2 * w22;

                        acc16[o] += sum;
                    }
                }

                for (uint32_t i = 0; i < tile; ++i) {
                    yGm.SetValue(yBase + i, acc16[i]);
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

extern "C" __global__ __aicore__ void conv_standard2d_asymmetric_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvStandard2dAsymSquareOwTiledFastPath op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.ow,
            tiling_data.h, tiling_data.w,
            tiling_data.cin, tiling_data.cout, tiling_data.oh,
            tiling_data.tile_ow, tiling_data.ow_tiles, tiling_data.tasks);
    op.Process();
}
