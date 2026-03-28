
#include "kernel_operator.h"

class KernelResidualAttentionCustom {
public:
    __aicore__ inline KernelResidualAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR la, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t totalRows, uint32_t rowsPerCore,
                               float invHW)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->HW = HW;
        this->totalRows = totalRows;
        this->rowsPerCore = rowsPerCore;
        this->invHW = invHW;

        uint64_t totalX = static_cast<uint64_t>(B) * static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);
        xGm.SetGlobalBuffer((__gm__ float*)x, totalX);
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(B) * static_cast<uint64_t>(C));
        laGm.SetGlobalBuffer((__gm__ float*)la, 1);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t startRow = blk * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > totalRows) endRow = totalRows;
        if (startRow >= endRow) return;

        const float laVal = laGm.GetValue(0);

        for (uint32_t row = startRow; row < endRow; ++row) {
            ComputeOneRowStreamGm(row, laVal);
        }
    }

private:
    __aicore__ inline void ComputeOneRowStreamGm(uint32_t row, float laVal)
    {
        const uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(HW);

        float sum = 0.0f;
        float mx = -3.402823466e+38f; // -FLT_MAX

        // Fast path for common small HW (e.g., 49): single tight loop, unrolled.
        if (HW <= 256) {
            uint32_t i = 0;
            for (; i + 7 < HW; i += 8) {
                float v0 = xGm.GetValue(base + i + 0);
                float v1 = xGm.GetValue(base + i + 1);
                float v2 = xGm.GetValue(base + i + 2);
                float v3 = xGm.GetValue(base + i + 3);
                float v4 = xGm.GetValue(base + i + 4);
                float v5 = xGm.GetValue(base + i + 5);
                float v6 = xGm.GetValue(base + i + 6);
                float v7 = xGm.GetValue(base + i + 7);

                sum += (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7);

                mx = (v0 > mx) ? v0 : mx;
                mx = (v1 > mx) ? v1 : mx;
                mx = (v2 > mx) ? v2 : mx;
                mx = (v3 > mx) ? v3 : mx;
                mx = (v4 > mx) ? v4 : mx;
                mx = (v5 > mx) ? v5 : mx;
                mx = (v6 > mx) ? v6 : mx;
                mx = (v7 > mx) ? v7 : mx;
            }
            for (; i < HW; ++i) {
                float v = xGm.GetValue(base + i);
                sum += v;
                mx = (v > mx) ? v : mx;
            }
        } else {
            // Generic path: still unrolled, no UB staging (avoids GM->UB copy cost).
            uint32_t i = 0;
            for (; i + 15 < HW; i += 16) {
                float v0  = xGm.GetValue(base + i + 0);
                float v1  = xGm.GetValue(base + i + 1);
                float v2  = xGm.GetValue(base + i + 2);
                float v3  = xGm.GetValue(base + i + 3);
                float v4  = xGm.GetValue(base + i + 4);
                float v5  = xGm.GetValue(base + i + 5);
                float v6  = xGm.GetValue(base + i + 6);
                float v7  = xGm.GetValue(base + i + 7);
                float v8  = xGm.GetValue(base + i + 8);
                float v9  = xGm.GetValue(base + i + 9);
                float v10 = xGm.GetValue(base + i + 10);
                float v11 = xGm.GetValue(base + i + 11);
                float v12 = xGm.GetValue(base + i + 12);
                float v13 = xGm.GetValue(base + i + 13);
                float v14 = xGm.GetValue(base + i + 14);
                float v15 = xGm.GetValue(base + i + 15);

                sum += (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 +
                        v8 + v9 + v10 + v11 + v12 + v13 + v14 + v15);

                mx = (v0 > mx) ? v0 : mx;   mx = (v1 > mx) ? v1 : mx;
                mx = (v2 > mx) ? v2 : mx;   mx = (v3 > mx) ? v3 : mx;
                mx = (v4 > mx) ? v4 : mx;   mx = (v5 > mx) ? v5 : mx;
                mx = (v6 > mx) ? v6 : mx;   mx = (v7 > mx) ? v7 : mx;
                mx = (v8 > mx) ? v8 : mx;   mx = (v9 > mx) ? v9 : mx;
                mx = (v10 > mx) ? v10 : mx; mx = (v11 > mx) ? v11 : mx;
                mx = (v12 > mx) ? v12 : mx; mx = (v13 > mx) ? v13 : mx;
                mx = (v14 > mx) ? v14 : mx; mx = (v15 > mx) ? v15 : mx;
            }
            for (; i < HW; ++i) {
                float v = xGm.GetValue(base + i);
                sum += v;
                mx = (v > mx) ? v : mx;
            }
        }

        const float outv = (sum * invHW) + laVal * mx;
        yGm.SetValue(static_cast<uint64_t>(row), outv);
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> laGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t HW;
    uint32_t totalRows, rowsPerCore;
    float invHW;
};

extern "C" __global__ __aicore__ void residual_attention_custom(GM_ADDR x, GM_ADDR la, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelResidualAttentionCustom op;
    op.Init(x, la, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.totalRows, tiling_data.rowsPerCore,
            tiling_data.invHW);
    op.Process();
}
