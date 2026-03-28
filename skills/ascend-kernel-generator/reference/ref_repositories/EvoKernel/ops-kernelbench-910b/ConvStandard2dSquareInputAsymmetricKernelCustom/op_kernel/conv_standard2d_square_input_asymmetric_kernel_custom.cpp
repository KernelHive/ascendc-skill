
#include "kernel_operator.h"

class KernelConvStandard2dSquareInputAsym5x9WoTiledSliding {
public:
    __aicore__ inline KernelConvStandard2dSquareInputAsym5x9WoTiledSliding() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t rows, uint32_t wo,
                               uint32_t H, uint32_t W,
                               uint32_t CIN, uint32_t COUT, uint32_t HO,
                               uint32_t tileWo, uint32_t woTiles, uint32_t tasks)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        rows_ = rows;
        wo_ = wo;
        H_ = H;
        W_ = W;
        CIN_ = CIN;
        COUT_ = COUT;
        HO_ = HO;
        tileWo_ = tileWo;
        woTiles_ = woTiles;
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

        constexpr uint32_t KH = 5;
        constexpr uint32_t KW = 9;

        const uint32_t wo = wo_;
        const uint32_t W = W_;
        const uint32_t H = H_;
        const uint32_t CIN = CIN_;
        const uint32_t COUT = COUT_;
        const uint32_t HO = HO_;
        const uint32_t tileWo = tileWo_;
        const uint32_t woTiles = woTiles_;

        for (uint32_t task = taskStart; task < taskEnd; ++task) {
            const uint32_t row = task / woTiles;
            const uint32_t woTileId = task - row * woTiles;
            const uint32_t ow0 = woTileId * tileWo;

            uint32_t tile = wo - ow0;
            if (tile > tileWo) tile = tileWo;

            uint32_t tmp = row;
            const uint32_t ho = tmp % HO;
            tmp /= HO;
            const uint32_t co = tmp % COUT;
            const uint32_t n  = tmp / COUT;

            const uint32_t yBase = ((n * COUT + co) * HO + ho) * wo + ow0;

            float acc16[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) acc16[i] = 0.0f;

            const uint32_t xNBase = n * (CIN * H * W);
            const uint32_t wCoBase = co * (CIN * KH * KW);

            // Sliding buffer length = tile + KW - 1, capped at 16+8=24.
            // We keep it in registers to avoid UB alloc/copies.
            float xbuf[24];

            // Main accumulation: for each input channel, for each kh-row, load contiguous strip once and reuse for all kw.
            for (uint32_t ci = 0; ci < CIN; ++ci) {
                const uint32_t xNcBase = xNBase + ci * (H * W);
                const uint32_t wCociBase = wCoBase + ci * (KH * KW);

#pragma unroll
                for (uint32_t kh = 0; kh < KH; ++kh) {
                    const uint32_t ih = ho + kh;
                    const uint32_t xRowBase = xNcBase + ih * W + ow0;

                    const uint32_t loadLen = tile + (KW - 1U); // <= 24
                    // Load x strip once.
                    for (uint32_t t = 0; t < loadLen; ++t) {
                        xbuf[t] = xGm.GetValue(xRowBase + t);
                    }

                    const uint32_t wRowBase = wCociBase + kh * KW;

                    // Load weights for this kh row (9 scalars).
                    const float w0 = wGm.GetValue(wRowBase + 0U);
                    const float w1 = wGm.GetValue(wRowBase + 1U);
                    const float w2 = wGm.GetValue(wRowBase + 2U);
                    const float w3 = wGm.GetValue(wRowBase + 3U);
                    const float w4 = wGm.GetValue(wRowBase + 4U);
                    const float w5 = wGm.GetValue(wRowBase + 5U);
                    const float w6 = wGm.GetValue(wRowBase + 6U);
                    const float w7 = wGm.GetValue(wRowBase + 7U);
                    const float w8 = wGm.GetValue(wRowBase + 8U);

                    if (tile == 16) {
#pragma unroll
                        for (uint32_t o = 0; o < 16; ++o) {
                            // Reuse xbuf sliding window.
                            float sum = 0.0f;
                            sum += xbuf[o + 0U] * w0;
                            sum += xbuf[o + 1U] * w1;
                            sum += xbuf[o + 2U] * w2;
                            sum += xbuf[o + 3U] * w3;
                            sum += xbuf[o + 4U] * w4;
                            sum += xbuf[o + 5U] * w5;
                            sum += xbuf[o + 6U] * w6;
                            sum += xbuf[o + 7U] * w7;
                            sum += xbuf[o + 8U] * w8;
                            acc16[o] += sum;
                        }
                    } else {
                        for (uint32_t o = 0; o < tile; ++o) {
                            float sum = 0.0f;
                            sum += xbuf[o + 0U] * w0;
                            sum += xbuf[o + 1U] * w1;
                            sum += xbuf[o + 2U] * w2;
                            sum += xbuf[o + 3U] * w3;
                            sum += xbuf[o + 4U] * w4;
                            sum += xbuf[o + 5U] * w5;
                            sum += xbuf[o + 6U] * w6;
                            sum += xbuf[o + 7U] * w7;
                            sum += xbuf[o + 8U] * w8;
                            acc16[o] += sum;
                        }
                    }
                }
            }

            // Single GM store per output element.
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
    uint32_t wo_{0};
    uint32_t H_{0};
    uint32_t W_{0};
    uint32_t CIN_{0};
    uint32_t COUT_{0};
    uint32_t HO_{0};
    uint32_t tileWo_{16};
    uint32_t woTiles_{0};
    uint32_t tasks_{0};
};

extern "C" __global__ __aicore__ void conv_standard2d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelConvStandard2dSquareInputAsym5x9WoTiledSliding op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.wo,
            tiling_data.h, tiling_data.w,
            tiling_data.cin, tiling_data.cout, tiling_data.ho,
            tiling_data.tile_wo, tiling_data.wo_tiles, tiling_data.tasks);
    op.Process();
}
