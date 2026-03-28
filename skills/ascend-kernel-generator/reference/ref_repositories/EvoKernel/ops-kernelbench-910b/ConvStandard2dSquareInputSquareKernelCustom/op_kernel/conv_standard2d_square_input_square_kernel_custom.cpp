
#include "kernel_operator.h"

class KernelConvStandard2dSquareInputSquareK3x3OwTiledSliding {
public:
    __aicore__ inline KernelConvStandard2dSquareInputSquareK3x3OwTiledSliding() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t rows, uint32_t ow,
                               uint32_t H, uint32_t W,
                               uint32_t CIN, uint32_t COUT, uint32_t HO,
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
        HO_ = HO;
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

        constexpr uint32_t KH = 3;
        constexpr uint32_t KW = 3;

        const uint32_t ow = ow_;
        const uint32_t W = W_;
        const uint32_t H = H_;
        const uint32_t CIN = CIN_;
        const uint32_t COUT = COUT_;
        const uint32_t HO = HO_;
        const uint32_t tileOw = tileOw_;
        const uint32_t owTiles = owTiles_;

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / owTiles;
            const uint32_t owTileId = task - row * owTiles;
            const uint32_t ow0 = owTileId * tileOw;

            uint32_t tile = ow - ow0;
            if (tile > tileOw) tile = tileOw;
            if (tile == 0) continue;

            uint32_t tmp = row;
            const uint32_t ho = tmp % HO;
            tmp /= HO;
            const uint32_t co = tmp % COUT;
            const uint32_t n  = tmp / COUT;

            const uint32_t yBase = ((n * COUT + co) * HO + ho) * ow + ow0;

            float acc16[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) acc16[i] = 0.0f;

            const uint32_t xNBase = n * (CIN * H * W);
            const uint32_t wCoBase = co * (CIN * KH * KW);

            // Sliding strip length = tile + (KW - 1) <= 18 when tile<=16.
            float xbuf[18];

            for (uint32_t ci = 0; ci < CIN; ++ci) {
                const uint32_t xNcBase = xNBase + ci * (H * W);
                const uint32_t wCociBase = wCoBase + ci * (KH * KW);

#pragma unroll
                for (uint32_t kh = 0; kh < KH; ++kh) {
                    const uint32_t ih = ho + kh;
                    const uint32_t xRowBase = xNcBase + ih * W + ow0;

                    const uint32_t loadLen = tile + (KW - 1U); // <= 18
                    for (uint32_t t = 0; t < loadLen; ++t) {
                        xbuf[t] = xGm.GetValue(xRowBase + t);
                    }

                    const uint32_t wRowBase = wCociBase + kh * KW;
                    const float w0 = wGm.GetValue(wRowBase + 0U);
                    const float w1 = wGm.GetValue(wRowBase + 1U);
                    const float w2 = wGm.GetValue(wRowBase + 2U);

                    if (tile == 16) {
#pragma unroll
                        for (uint32_t o = 0; o < 16; ++o) {
                            acc16[o] += xbuf[o + 0U] * w0 + xbuf[o + 1U] * w1 + xbuf[o + 2U] * w2;
                        }
                    } else {
                        for (uint32_t o = 0; o < tile; ++o) {
                            acc16[o] += xbuf[o + 0U] * w0 + xbuf[o + 1U] * w1 + xbuf[o + 2U] * w2;
                        }
                    }
                }
            }

            if (tile == 16) {
#pragma unroll
                for (uint32_t i = 0; i < 16; ++i) {
                    yGm.SetValue(yBase + i, acc16[i]);
                }
            } else {
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
    uint32_t HO_{0};
    uint32_t tileOw_{16};
    uint32_t owTiles_{0};
    uint32_t tasks_{0};
};

extern "C" __global__ __aicore__ void conv_standard2d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard2dSquareInputSquareK3x3OwTiledSliding op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.ow,
            tiling_data.h, tiling_data.w,
            tiling_data.cin, tiling_data.cout, tiling_data.ho,
            tiling_data.tile_ow, tiling_data.ow_tiles, tiling_data.tasks);
    op.Process();
}
