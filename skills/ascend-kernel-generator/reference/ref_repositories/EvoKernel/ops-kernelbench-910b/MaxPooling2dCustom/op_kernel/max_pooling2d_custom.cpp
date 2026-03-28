
#include "kernel_operator.h"

// MaxPool2d specialized for:
// kernel=4, stride=1, padding=1, dilation=1, ceil_mode=False.
// Input/Output: NCHW contiguous stored as ND.
//
// Optimizations in this round:
// - Keep original flattened tiling coverage (no gaps) but process with plane-row traversal.
// - Split border (checked) and interior (branchless) paths; interior is fully unrolled 4x4.
// - Reduce scalar-heavy "odometer across N/C/H/W" to simple within-plane row/col updates.

class KernelMaxPooling2dCustom {
public:
    __aicore__ inline KernelMaxPooling2dCustom() {}

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

    __aicore__ inline float MaxPoolChecked(uint32_t xPlaneBase, int32_t ih0, int32_t iw0) const
    {
        constexpr int32_t K = 4;
        const float neg_inf = -3.402823466e+38f;
        float m = neg_inf;

#pragma unroll
        for (int32_t kh = 0; kh < K; ++kh) {
            const int32_t ih = ih0 + kh;
            if ((uint32_t)ih >= h_in) continue;
            const uint32_t xRowBase = xPlaneBase + static_cast<uint32_t>(ih) * w_in;
#pragma unroll
            for (int32_t kw = 0; kw < K; ++kw) {
                const int32_t iw = iw0 + kw;
                if ((uint32_t)iw >= w_in) continue;
                const float v = xGm.GetValue((uint64_t)(xRowBase + static_cast<uint32_t>(iw)));
                m = (v > m) ? v : m;
            }
        }
        return m;
    }

    __aicore__ inline float MaxPoolInterior(uint32_t xRow0, uint32_t xRow1, uint32_t xRow2, uint32_t xRow3, uint32_t iw0u) const
    {
        float m = xGm.GetValue((uint64_t)(xRow0 + iw0u + 0U));
        float v;

        v = xGm.GetValue((uint64_t)(xRow0 + iw0u + 1U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow0 + iw0u + 2U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow0 + iw0u + 3U)); m = (v > m) ? v : m;

        v = xGm.GetValue((uint64_t)(xRow1 + iw0u + 0U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow1 + iw0u + 1U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow1 + iw0u + 2U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow1 + iw0u + 3U)); m = (v > m) ? v : m;

        v = xGm.GetValue((uint64_t)(xRow2 + iw0u + 0U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow2 + iw0u + 1U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow2 + iw0u + 2U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow2 + iw0u + 3U)); m = (v > m) ? v : m;

        v = xGm.GetValue((uint64_t)(xRow3 + iw0u + 0U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow3 + iw0u + 1U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow3 + iw0u + 2U)); m = (v > m) ? v : m;
        v = xGm.GetValue((uint64_t)(xRow3 + iw0u + 3U)); m = (v > m) ? v : m;

        return m;
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t P = 1;

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

        // Traverse within planes; when leaving a plane, re-decode (rare compared to per-element).
        while (outIdx < end) {
            const uint32_t xPlaneBase = ni * chw_in + ci * hw_in;
            const uint32_t yPlaneBase = ni * chw_out + ci * hw_out;

            // Process remainder of this plane or until end.
            uint32_t planeLinear = oh * w_out + ow;
            uint32_t remainingInPlane = hw_out - planeLinear;
            uint32_t remainingInBlock = end - outIdx;
            uint32_t work = (remainingInPlane < remainingInBlock) ? remainingInPlane : remainingInBlock;

            uint32_t yOffset = yPlaneBase + planeLinear;

            // Determine interior ranges safely.
            // For k=4,s=1,p=1: interior outputs satisfy oh in [1, h_out-2] and ow in [1, w_out-2]
            const bool hasInterior = (h_in >= 4U) && (w_in >= 4U) && (h_out >= 3U) && (w_out >= 3U);
            const uint32_t ohIntBeg = hasInterior ? 1U : 0U;
            const uint32_t ohIntEndEx = hasInterior ? (h_out - 1U) : 0U; // exclusive, so last interior oh is h_out-2
            const uint32_t owIntBeg = hasInterior ? 1U : 0U;
            const uint32_t owIntEndEx = hasInterior ? (w_out - 1U) : 0U; // exclusive, so last interior ow is w_out-2

            // Row/col traversal with simple updates.
            uint32_t local = 0;
            while (local < work) {
                // How many contiguous outputs remain in this row from current ow.
                uint32_t rowRemain = w_out - ow;
                uint32_t chunk = (rowRemain < (work - local)) ? rowRemain : (work - local);

                // For this row, optionally split into left border, interior, right border for current ow..ow+chunk
                if (!hasInterior || oh < ohIntBeg || oh >= ohIntEndEx) {
                    // Entire chunk is border row => checked path
                    const int32_t ih0 = static_cast<int32_t>(oh) - (int32_t)P;
                    int32_t iw0 = static_cast<int32_t>(ow) - (int32_t)P;
                    for (uint32_t i = 0; i < chunk; ++i) {
                        const float m = MaxPoolChecked(xPlaneBase, ih0, iw0);
                        yGm.SetValue((uint64_t)(yOffset + i), m);
                        ++iw0; // stride=1
                    }
                } else {
                    // Middle row: split by columns
                    const int32_t ih0 = static_cast<int32_t>(oh) - (int32_t)P;

                    // Left border part
                    uint32_t col = ow;
                    uint32_t outPos = 0;
                    while (outPos < chunk && col < owIntBeg) {
                        const int32_t iw0 = static_cast<int32_t>(col) - (int32_t)P;
                        const float m = MaxPoolChecked(xPlaneBase, ih0, iw0);
                        yGm.SetValue((uint64_t)(yOffset + outPos), m);
                        ++col;
                        ++outPos;
                    }

                    // Interior part
                    if (hasInterior) {
                        const uint32_t ih0u = oh - 1U;
                        const uint32_t xRow0 = xPlaneBase + (ih0u + 0U) * w_in;
                        const uint32_t xRow1 = xPlaneBase + (ih0u + 1U) * w_in;
                        const uint32_t xRow2 = xPlaneBase + (ih0u + 2U) * w_in;
                        const uint32_t xRow3 = xPlaneBase + (ih0u + 3U) * w_in;

                        while (outPos < chunk && col < owIntEndEx) {
                            const uint32_t iw0u = col - 1U;
                            const float m = MaxPoolInterior(xRow0, xRow1, xRow2, xRow3, iw0u);
                            yGm.SetValue((uint64_t)(yOffset + outPos), m);
                            ++col;
                            ++outPos;
                        }
                    }

                    // Right border part
                    while (outPos < chunk) {
                        const int32_t iw0 = static_cast<int32_t>(col) - (int32_t)P;
                        const float m = MaxPoolChecked(xPlaneBase, ih0, iw0);
                        yGm.SetValue((uint64_t)(yOffset + outPos), m);
                        ++col;
                        ++outPos;
                    }
                }

                // Advance to next row or continue.
                outIdx += chunk;
                local += chunk;
                yOffset += chunk;

                ow += chunk;
                if (ow == w_out) {
                    ow = 0;
                    ++oh;
                    // if we finished the plane, break to advance (n,c)
                    if (oh == h_out) {
                        oh = 0;
                        ++ci;
                        if (ci == c) {
                            ci = 0;
                            ++ni;
                        }
                        break; // plane ended
                    }
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

extern "C" __global__ __aicore__ void max_pooling2d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelMaxPooling2dCustom op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c,
            tiling_data.h_in, tiling_data.w_in,
            tiling_data.h_out, tiling_data.w_out,
            tiling_data.totalY, tiling_data.elemsPerBlock);
    op.Process();
}
