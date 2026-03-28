
#include "kernel_operator.h"

// MaxPool1d specialized for:
// kernel_size=8, stride=1, padding=4, dilation=1, return_indices=false
//
// This round:
// - Multi-block parallelization over rows (N*C) via host tiling.
// - Quad-output unrolling in the steady in-range region to amortize loop/branch overhead
//   and increase ILP while keeping the same sliding-window + "recompute if outgoing==max" logic.

class KernelMaxPooling1dCustom {
public:
    __aicore__ inline KernelMaxPooling1dCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t n, uint32_t c,
                               uint32_t l_in, uint32_t l_out,
                               uint32_t totalRows, uint32_t blockDim, uint32_t rowsPerBlock)
    {
        this->n = n;
        this->c = c;
        this->l_in = l_in;
        this->l_out = l_out;
        this->totalRows = totalRows;
        this->blockDim = blockDim;
        this->rowsPerBlock = rowsPerBlock;

        xPtr = (__gm__ float*)x;
        yPtr = (__gm__ float*)y;
    }

    __aicore__ inline float ReduceMax8(const float ring[8]) const
    {
        float m = ring[0];
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            m = (ring[i] > m) ? ring[i] : m;
        }
        return m;
    }

    __aicore__ inline void StepOne(float ring[8], int &head, float &maxv, float incoming) const
    {
        const float outgoing = ring[head];
        ring[head] = incoming;
        head = (head + 1) & 7;

        if (outgoing == maxv) {
            maxv = ReduceMax8(ring);
        } else {
            maxv = (incoming > maxv) ? incoming : maxv;
        }
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t K = 8;
        constexpr int32_t P = 4;
        const float neg_inf = -3.402823466e+38f;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t startRow = bid * rowsPerBlock;
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > totalRows) endRow = totalRows;

        for (uint32_t row = startRow; row < endRow; ++row) {
            __gm__ float* xRow = xPtr + static_cast<uint64_t>(row) * static_cast<uint64_t>(l_in);
            __gm__ float* yRow = yPtr + static_cast<uint64_t>(row) * static_cast<uint64_t>(l_out);

            // Ring buffer for last K values (register-resident)
            float ring[K];
#pragma unroll
            for (int i = 0; i < K; ++i) ring[i] = neg_inf;
            int head = 0;

            // out=0 window: input indices [-4..3]
            float maxv = neg_inf;
#pragma unroll
            for (int i = 0; i < K; ++i) {
                const int in_pos = i - P; // -4..3
                float v = (in_pos >= 0 && in_pos < static_cast<int>(l_in)) ? xRow[in_pos] : neg_inf;
                ring[i] = v;
                maxv = (v > maxv) ? v : maxv;
            }
            if (l_out == 0) continue;
            yRow[0] = maxv;

            uint32_t out = 1;

            // Small left prefix (at most P-1=3 outputs): keep robust checks for very small l_in.
            for (; out < static_cast<uint32_t>(P) && out < l_out; ++out) {
                const int in_pos = static_cast<int>(out) + (K - P - 1); // out+3
                float incoming = (in_pos >= 0 && in_pos < static_cast<int>(l_in)) ? xRow[in_pos] : neg_inf;
                StepOne(ring, head, maxv, incoming);
                yRow[out] = maxv;
            }

            // Steady region: incoming index always valid and equals (out+3).
            // Steady valid for out <= l_in + P - K, and also out < l_out.
            const int64_t steady_end64 = static_cast<int64_t>(l_in) + P - K; // l_in - 4
            if (steady_end64 >= static_cast<int64_t>(out) && l_in >= 1) {
                uint32_t steady_end = static_cast<uint32_t>(steady_end64);
                if (steady_end >= l_out) steady_end = l_out - 1;

                // Quad unroll: process 4 outputs at once to reduce loop overhead.
                // incoming positions are (out+3), (out+4), (out+5), (out+6)
                for (; out + 3 <= steady_end; out += 4) {
                    const float in0 = xRow[out + 3];
                    const float in1 = xRow[out + 4];
                    const float in2 = xRow[out + 5];
                    const float in3 = xRow[out + 6];

                    StepOne(ring, head, maxv, in0);
                    yRow[out + 0] = maxv;

                    StepOne(ring, head, maxv, in1);
                    yRow[out + 1] = maxv;

                    StepOne(ring, head, maxv, in2);
                    yRow[out + 2] = maxv;

                    StepOne(ring, head, maxv, in3);
                    yRow[out + 3] = maxv;
                }

                // Tail in steady region
                for (; out <= steady_end; ++out) {
                    const float incoming = xRow[out + 3];
                    StepOne(ring, head, maxv, incoming);
                    yRow[out] = maxv;
                }
            }

            // Right suffix: incoming out-of-range => -inf
            for (; out < l_out; ++out) {
                // outgoing may drop the max; if it does, recompute; otherwise max stays.
                const float outgoing = ring[head];
                ring[head] = neg_inf;
                head = (head + 1) & 7;

                if (outgoing == maxv) {
                    maxv = ReduceMax8(ring);
                }
                yRow[out] = maxv;
            }
        }
    }

private:
    __gm__ float* xPtr;
    __gm__ float* yPtr;

    uint32_t n, c, l_in, l_out;
    uint32_t totalRows, blockDim, rowsPerBlock;
};

extern "C" __global__ __aicore__ void max_pooling1d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelMaxPooling1dCustom op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c,
            tiling_data.l_in, tiling_data.l_out,
            tiling_data.totalRows, tiling_data.blockDim, tiling_data.rowsPerBlock);
    op.Process();
}
