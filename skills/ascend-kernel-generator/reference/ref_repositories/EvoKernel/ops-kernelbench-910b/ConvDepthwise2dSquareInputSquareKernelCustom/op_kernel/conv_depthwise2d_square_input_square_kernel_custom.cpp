
#include "kernel_operator.h"

class KernelConvDepthwise2dSquareSquare {
public:
    __aicore__ inline KernelConvDepthwise2dSquareSquare() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                               uint32_t rows, uint32_t ow,
                               uint32_t H, uint32_t W, uint32_t C, uint32_t OH)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)w);
        yGm.SetGlobalBuffer((__gm__ float*)y);
        rows_ = rows;
        ow_ = ow;
        H_ = H;
        W_ = W;
        C_ = C;
        OH_ = OH;
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();

        const uint32_t rowsPerCore = (rows_ + coreNum - 1u) / coreNum;
        uint32_t rowStart = coreId * rowsPerCore;
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > rows_) rowEnd = rows_;
        if (rowStart >= rowEnd) return;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            // row indexes (n, c, oh)
            uint32_t tmp = row;
            const uint32_t oh = tmp % OH_;
            tmp /= OH_;
            const uint32_t c = tmp % C_;
            const uint32_t n = tmp / C_;

            // Load weights for channel c: [C,1,3,3] flattened
            const uint32_t wBase = c * 9u;
            const float w00 = wGm.GetValue(wBase + 0u);
            const float w01 = wGm.GetValue(wBase + 1u);
            const float w02 = wGm.GetValue(wBase + 2u);
            const float w10 = wGm.GetValue(wBase + 3u);
            const float w11 = wGm.GetValue(wBase + 4u);
            const float w12 = wGm.GetValue(wBase + 5u);
            const float w20 = wGm.GetValue(wBase + 6u);
            const float w21 = wGm.GetValue(wBase + 7u);
            const float w22 = wGm.GetValue(wBase + 8u);

            // Base offsets
            const uint32_t xNcBase = (n * C_ + c) * H_ * W_;
            const uint32_t xRow0 = xNcBase + (oh + 0u) * W_;
            const uint32_t xRow1 = xNcBase + (oh + 1u) * W_;
            const uint32_t xRow2 = xNcBase + (oh + 2u) * W_;
            const uint32_t yBase = ((n * C_ + c) * OH_ + oh) * ow_;

            uint32_t ow = 0u;

            // Sliding-window over width; unroll by 2 to reduce loop/control overhead.
            if (ow_ >= 2u) {
                // preload first window (ow=0)
                float a0 = xGm.GetValue(xRow0 + 0u);
                float a1 = xGm.GetValue(xRow0 + 1u);
                float a2 = xGm.GetValue(xRow0 + 2u);

                float b0 = xGm.GetValue(xRow1 + 0u);
                float b1 = xGm.GetValue(xRow1 + 1u);
                float b2 = xGm.GetValue(xRow1 + 2u);

                float c0 = xGm.GetValue(xRow2 + 0u);
                float c1 = xGm.GetValue(xRow2 + 1u);
                float c2 = xGm.GetValue(xRow2 + 2u);

                const uint32_t owEnd2 = (ow_ / 2u) * 2u;
                for (; ow < owEnd2; ow += 2u) {
                    // output at ow
                    float s0 = 0.0f;
                    s0 += a0 * w00 + a1 * w01 + a2 * w02;
                    s0 += b0 * w10 + b1 * w11 + b2 * w12;
                    s0 += c0 * w20 + c1 * w21 + c2 * w22;
                    yGm.SetValue(yBase + ow, s0);

                    // advance window by 1 to compute ow+1
                    const uint32_t loadIdx1 = ow + 3u;
                    a0 = a1; a1 = a2; a2 = xGm.GetValue(xRow0 + loadIdx1);
                    b0 = b1; b1 = b2; b2 = xGm.GetValue(xRow1 + loadIdx1);
                    c0 = c1; c1 = c2; c2 = xGm.GetValue(xRow2 + loadIdx1);

                    float s1 = 0.0f;
                    s1 += a0 * w00 + a1 * w01 + a2 * w02;
                    s1 += b0 * w10 + b1 * w11 + b2 * w12;
                    s1 += c0 * w20 + c1 * w21 + c2 * w22;
                    yGm.SetValue(yBase + ow + 1u, s1);

                    // advance window by 1 for next loop iteration start (ow+2)
                    const uint32_t loadIdx2 = ow + 4u;
                    a0 = a1; a1 = a2; a2 = xGm.GetValue(xRow0 + loadIdx2);
                    b0 = b1; b1 = b2; b2 = xGm.GetValue(xRow1 + loadIdx2);
                    c0 = c1; c1 = c2; c2 = xGm.GetValue(xRow2 + loadIdx2);
                }

                // If ow_ is odd, handle the last element (window already positioned at owEnd2)
                if (ow < ow_) {
                    float s = 0.0f;
                    s += a0 * w00 + a1 * w01 + a2 * w02;
                    s += b0 * w10 + b1 * w11 + b2 * w12;
                    s += c0 * w20 + c1 * w21 + c2 * w22;
                    yGm.SetValue(yBase + ow, s);
                }
            } else if (ow_ == 1u) {
                // Degenerate (not for our specialized OW=510, but keep safe)
                const float a0 = xGm.GetValue(xRow0 + 0u);
                const float a1 = xGm.GetValue(xRow0 + 1u);
                const float a2 = xGm.GetValue(xRow0 + 2u);

                const float b0 = xGm.GetValue(xRow1 + 0u);
                const float b1 = xGm.GetValue(xRow1 + 1u);
                const float b2 = xGm.GetValue(xRow1 + 2u);

                const float c0 = xGm.GetValue(xRow2 + 0u);
                const float c1 = xGm.GetValue(xRow2 + 1u);
                const float c2 = xGm.GetValue(xRow2 + 2u);

                float s = 0.0f;
                s += a0 * w00 + a1 * w01 + a2 * w02;
                s += b0 * w10 + b1 * w11 + b2 * w12;
                s += c0 * w20 + c1 * w21 + c2 * w22;
                yGm.SetValue(yBase + 0u, s);
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
    uint32_t C_{0};
    uint32_t OH_{0};
};

extern "C" __global__ __aicore__ void conv_depthwise2d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwise2dSquareSquare op;
    op.Init(x, weight, y,
            tiling_data.rows, tiling_data.ow,
            tiling_data.h, tiling_data.w, tiling_data.c, tiling_data.oh);
    op.Process();
}
