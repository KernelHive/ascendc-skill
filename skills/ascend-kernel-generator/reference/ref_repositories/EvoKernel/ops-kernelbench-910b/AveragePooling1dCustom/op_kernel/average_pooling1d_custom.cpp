
#include "kernel_operator.h"

// Specialized AvgPool1d for this benchmark/model instance:
// kernel_size=8, stride=1, padding=4
// count_include_pad=True: always divide by 8, padded positions contribute 0.
//
// Optimization:
// 1) Multi-block: each block processes a contiguous range of flattened output indices.
// 2) Sliding window running-sum within each (n,c) row segment to reduce loads to ~2/output.

class KernelAveragePooling1dCustom {
public:
    __aicore__ inline KernelAveragePooling1dCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t totalX, uint32_t totalY,
                               uint32_t n, uint32_t c,
                               uint32_t l_in, uint32_t l_out,
                               uint32_t nc)
    {
        (void)totalX;
        this->totalY = totalY;
        this->n = n;
        this->c = c;
        this->l_in = l_in;
        this->l_out = l_out;
        this->nc = nc;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline float LoadPadZero(uint64_t xBase, int32_t pos) const
    {
        // Unsigned range check is fast and handles negative pos.
        if ((uint32_t)pos < l_in) {
            return xGm.GetValue(xBase + (uint32_t)pos);
        }
        return 0.0f;
    }

    __aicore__ inline float InitWindowSum(uint64_t xBase, int32_t in_start) const
    {
        float sum = 0.0f;
        sum += LoadPadZero(xBase, in_start + 0);
        sum += LoadPadZero(xBase, in_start + 1);
        sum += LoadPadZero(xBase, in_start + 2);
        sum += LoadPadZero(xBase, in_start + 3);
        sum += LoadPadZero(xBase, in_start + 4);
        sum += LoadPadZero(xBase, in_start + 5);
        sum += LoadPadZero(xBase, in_start + 6);
        sum += LoadPadZero(xBase, in_start + 7);
        return sum;
    }

    __aicore__ inline void ProcessRowSegment(uint32_t nc_idx, uint32_t out_begin, uint32_t out_end)
    {
        constexpr int32_t K = 8;
        constexpr int32_t P = 4;
        const float invK = 1.0f / 8.0f;

        const uint64_t xBase = static_cast<uint64_t>(nc_idx) * l_in;
        const uint64_t yBase = static_cast<uint64_t>(nc_idx) * l_out;

        // Initialize sum for the first output of this segment.
        int32_t in_start = static_cast<int32_t>(out_begin) - P; // stride=1
        float sum = InitWindowSum(xBase, in_start);

        // Write first output.
        yGm.SetValue(yBase + out_begin, sum * invK);

        // Slide within [out_begin+1, out_end)
        for (uint32_t out_idx = out_begin + 1; out_idx < out_end; ++out_idx) {
            const int32_t out_pos = static_cast<int32_t>(out_idx);
            const int32_t removePos = (out_pos - 1) - P;
            const int32_t addPos    = (out_pos)     - P + (K - 1);

            sum -= LoadPadZero(xBase, removePos);
            sum += LoadPadZero(xBase, addPos);

            yGm.SetValue(yBase + out_idx, sum * invK);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockDim = AscendC::GetBlockNum();

        // Each block works on a contiguous output interval [start, end) in flattened Y.
        // This keeps writes contiguous and reduces scalar overhead.
        const uint32_t start = (uint64_t)blockIdx * (uint64_t)totalY / (uint64_t)blockDim;
        const uint32_t end   = (uint64_t)(blockIdx + 1) * (uint64_t)totalY / (uint64_t)blockDim;

        if (start >= end) return;

        // Flatten index: y_index = nc_idx * l_out + out_idx
        uint32_t idx = start;
        while (idx < end) {
            const uint32_t nc_idx = idx / l_out;
            const uint32_t out_idx = idx - nc_idx * l_out;

            // Consume as much as possible within this row, but not past end.
            uint32_t row_remain = l_out - out_idx;
            uint32_t seg_len = end - idx;
            uint32_t take = (seg_len < row_remain) ? seg_len : row_remain;

            ProcessRowSegment(nc_idx, out_idx, out_idx + take);
            idx += take;
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalY;
    uint32_t n, c, l_in, l_out, nc;
};

extern "C" __global__ __aicore__ void average_pooling1d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelAveragePooling1dCustom op;
    op.Init(x, y,
            tiling_data.totalX, tiling_data.totalY,
            tiling_data.n, tiling_data.c,
            tiling_data.l_in, tiling_data.l_out,
            tiling_data.nc);
    op.Process();
}
