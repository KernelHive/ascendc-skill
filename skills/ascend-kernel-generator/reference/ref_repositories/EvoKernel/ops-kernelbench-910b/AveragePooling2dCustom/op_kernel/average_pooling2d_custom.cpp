
#include "kernel_operator.h"

// AvgPool2d specialized: kernel=(11,11), stride=(11,11), padding=0, ceil_mode=False.
// Optimization vs baseline:
// - 4-wide micro-batching per iteration to increase ILP (4 independent accumulators).
// - Odometer walk over (ow->oh->c->n) to avoid repeated div/mod decoding.
// - Fast path when 4 outputs remain in same (n,c,oh) row, minimizing pointer recomputation.

class KernelAveragePooling2dCustom {
public:
    __aicore__ inline KernelAveragePooling2dCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t n, uint32_t c,
                               uint32_t h_in, uint32_t w_in,
                               uint32_t h_out, uint32_t w_out,
                               uint32_t totalY, uint32_t elemsPerBlock)
    {
        this->n = n;
        this->c = c;
        this->h_in = h_in;
        this->w_in = w_in;
        this->h_out = h_out;
        this->w_out = w_out;
        this->totalY = totalY;
        this->elemsPerBlock = elemsPerBlock;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t KH = 11;
        constexpr uint32_t KW = 11;
        constexpr uint32_t SH = 11;
        constexpr uint32_t SW = 11;
        constexpr uint32_t MICRO = 4;

        const float invK = 1.0f / 121.0f;

        const uint32_t bid = AscendC::GetBlockIdx();
        const uint32_t start = bid * elemsPerBlock;
        uint32_t end = start + elemsPerBlock;
        if (end > totalY) end = totalY;
        if (start >= end) return;

        const uint32_t hw_in = h_in * w_in;
        const uint32_t hw_out = h_out * w_out;
        const uint32_t chw_in = c * hw_in;
        const uint32_t chw_out = c * hw_out;

        // Decode start once.
        uint32_t outIdx = start;
        uint32_t ow = outIdx % w_out;
        uint32_t t1 = outIdx / w_out;
        uint32_t oh = t1 % h_out;
        uint32_t t2 = t1 / h_out;
        uint32_t ci = t2 % c;
        uint32_t ni = t2 / c;

        uint32_t xChanBase = ni * chw_in + ci * hw_in;
        uint32_t yChanBase = ni * chw_out + ci * hw_out;

        while (outIdx < end) {
            // Micro-batch of 4 outputs if they stay in the same row.
            const uint32_t remain = end - outIdx;
            const uint32_t canDo4 = (remain >= MICRO) && (ow + (MICRO - 1) < w_out);
            if (canDo4) {
                const uint32_t ih_start = oh * SH;
                const uint32_t iw0 = ow * SW;

                const uint32_t xWinBase0 = xChanBase + ih_start * w_in + iw0;
                const uint32_t xWinBase1 = xWinBase0 + SW;
                const uint32_t xWinBase2 = xWinBase1 + SW;
                const uint32_t xWinBase3 = xWinBase2 + SW;

                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

#pragma unroll
                for (uint32_t r = 0; r < KH; ++r) {
                    const uint32_t rowOff = r * w_in;
                    const uint32_t p0 = xWinBase0 + rowOff;
                    const uint32_t p1 = xWinBase1 + rowOff;
                    const uint32_t p2 = xWinBase2 + rowOff;
                    const uint32_t p3 = xWinBase3 + rowOff;
#pragma unroll
                    for (uint32_t s = 0; s < KW; ++s) {
                        acc0 += xGm.GetValue((uint64_t)(p0 + s));
                        acc1 += xGm.GetValue((uint64_t)(p1 + s));
                        acc2 += xGm.GetValue((uint64_t)(p2 + s));
                        acc3 += xGm.GetValue((uint64_t)(p3 + s));
                    }
                }

                const uint32_t yBaseRow = yChanBase + oh * w_out + ow;
                yGm.SetValue((uint64_t)(yBaseRow + 0), acc0 * invK);
                yGm.SetValue((uint64_t)(yBaseRow + 1), acc1 * invK);
                yGm.SetValue((uint64_t)(yBaseRow + 2), acc2 * invK);
                yGm.SetValue((uint64_t)(yBaseRow + 3), acc3 * invK);

                // Advance odometer by 4 in ow dimension only.
                outIdx += MICRO;
                ow += MICRO;
                // ow stays < w_out by construction.
                continue;
            }

            // Scalar fallback (also handles row boundary / tail).
            const uint32_t ih_start = oh * SH;
            const uint32_t iw_start = ow * SW;
            const uint32_t xWinBase = xChanBase + ih_start * w_in + iw_start;

            float acc = 0.0f;
#pragma unroll
            for (uint32_t r = 0; r < KH; ++r) {
                const uint32_t xLine = xWinBase + r * w_in;
#pragma unroll
                for (uint32_t s = 0; s < KW; ++s) {
                    acc += xGm.GetValue((uint64_t)(xLine + s));
                }
            }
            const uint32_t yOff = yChanBase + oh * w_out + ow;
            yGm.SetValue((uint64_t)yOff, acc * invK);

            // Advance odometer by 1 element.
            ++outIdx;
            ++ow;
            if (ow == w_out) {
                ow = 0;
                ++oh;
                if (oh == h_out) {
                    oh = 0;
                    ++ci;
                    if (ci == c) {
                        ci = 0;
                        ++ni;
                    }
                    xChanBase = ni * chw_in + ci * hw_in;
                    yChanBase = ni * chw_out + ci * hw_out;
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, c, h_in, w_in, h_out, w_out;
    uint32_t totalY, elemsPerBlock;
};

extern "C" __global__ __aicore__ void average_pooling2d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelAveragePooling2dCustom op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c,
            tiling_data.h_in, tiling_data.w_in,
            tiling_data.h_out, tiling_data.w_out,
            tiling_data.totalY, tiling_data.elemsPerBlock);
    op.Process();
}
