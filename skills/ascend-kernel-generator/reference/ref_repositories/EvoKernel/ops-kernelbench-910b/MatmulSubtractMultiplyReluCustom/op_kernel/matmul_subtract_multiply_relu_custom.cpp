
#include "kernel_operator.h"

class KernelMatmulSubtractMultiplyReluCustom {
public:
    __aicore__ inline KernelMatmulSubtractMultiplyReluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR sub, GM_ADDR mul, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t rowsPerBlock, uint32_t vecN)
    {
        M_ = M; K_ = K; N_ = N;
        rowsPerBlock_ = (rowsPerBlock == 0 ? 1 : rowsPerBlock);
        vecN_ = (vecN == 0 ? 1 : vecN);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        subGm_.SetGlobalBuffer((__gm__ float*)sub);
        mulGm_.SetGlobalBuffer((__gm__ float*)mul);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0) return;

        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();

        // Scalars once per block (sub/mul are scalar tensors)
        const float subVal = subGm_.GetValue(0);
        const float mulVal = mulGm_.GetValue(0);

        uint32_t rowStart = blockIdx * rowsPerBlock_;
        if (rowStart >= M_) return;
        uint32_t rowEnd = rowStart + rowsPerBlock_;
        if (rowEnd > M_) rowEnd = M_;

        for (uint32_t m = rowStart; m < rowEnd; ++m) {
            const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
            const uint64_t outBase = (uint64_t)m * (uint64_t)N_;

            if (vecN_ == 2) {
                for (uint32_t n = 0; n + 1U < N_; n += 2U) {
                    ComputeTwoAndStore(xBase, outBase, n, subVal, mulVal);
                }
                if (N_ & 1U) {
                    ComputeOneAndStore(xBase, outBase, N_ - 1U, subVal, mulVal);
                }
            } else {
                for (uint32_t n = 0; n < N_; ++n) {
                    ComputeOneAndStore(xBase, outBase, n, subVal, mulVal);
                }
            }
        }
    }

private:
    __aicore__ inline void FinishAndStore(uint64_t outBase, uint32_t n, float acc, float subVal, float mulVal)
    {
        acc += bGm_.GetValue((uint64_t)n);
        acc -= subVal;
        acc *= mulVal;
        if (acc < 0.0f) acc = 0.0f;
        yGm_.SetValue(outBase + (uint64_t)n, acc);
    }

    __aicore__ inline void ComputeOneAndStore(uint64_t xBase, uint64_t outBase, uint32_t n,
                                             float subVal, float mulVal)
    {
        const uint64_t wBase = (uint64_t)n * (uint64_t)K_;
        float acc = 0.0f;

        uint32_t k = 0;
        for (; k + 3U < K_; k += 4U) {
            const float x0 = xGm_.GetValue(xBase + (uint64_t)(k + 0U));
            const float x1 = xGm_.GetValue(xBase + (uint64_t)(k + 1U));
            const float x2 = xGm_.GetValue(xBase + (uint64_t)(k + 2U));
            const float x3 = xGm_.GetValue(xBase + (uint64_t)(k + 3U));

            const float w0 = wGm_.GetValue(wBase + (uint64_t)(k + 0U));
            const float w1 = wGm_.GetValue(wBase + (uint64_t)(k + 1U));
            const float w2 = wGm_.GetValue(wBase + (uint64_t)(k + 2U));
            const float w3 = wGm_.GetValue(wBase + (uint64_t)(k + 3U));

            acc += x0 * w0;
            acc += x1 * w1;
            acc += x2 * w2;
            acc += x3 * w3;
        }
        for (; k < K_; ++k) {
            const float xv = xGm_.GetValue(xBase + (uint64_t)k);
            const float wv = wGm_.GetValue(wBase + (uint64_t)k);
            acc += xv * wv;
        }

        FinishAndStore(outBase, n, acc, subVal, mulVal);
    }

    __aicore__ inline void ComputeTwoAndStore(uint64_t xBase, uint64_t outBase, uint32_t n,
                                             float subVal, float mulVal)
    {
        const uint64_t wBase0 = (uint64_t)n * (uint64_t)K_;
        const uint64_t wBase1 = (uint64_t)(n + 1U) * (uint64_t)K_;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        uint32_t k = 0;
        for (; k + 3U < K_; k += 4U) {
            const float x0 = xGm_.GetValue(xBase + (uint64_t)(k + 0U));
            const float x1 = xGm_.GetValue(xBase + (uint64_t)(k + 1U));
            const float x2 = xGm_.GetValue(xBase + (uint64_t)(k + 2U));
            const float x3 = xGm_.GetValue(xBase + (uint64_t)(k + 3U));

            const float w00 = wGm_.GetValue(wBase0 + (uint64_t)(k + 0U));
            const float w01 = wGm_.GetValue(wBase0 + (uint64_t)(k + 1U));
            const float w02 = wGm_.GetValue(wBase0 + (uint64_t)(k + 2U));
            const float w03 = wGm_.GetValue(wBase0 + (uint64_t)(k + 3U));

            const float w10 = wGm_.GetValue(wBase1 + (uint64_t)(k + 0U));
            const float w11 = wGm_.GetValue(wBase1 + (uint64_t)(k + 1U));
            const float w12 = wGm_.GetValue(wBase1 + (uint64_t)(k + 2U));
            const float w13 = wGm_.GetValue(wBase1 + (uint64_t)(k + 3U));

            acc0 += x0 * w00; acc1 += x0 * w10;
            acc0 += x1 * w01; acc1 += x1 * w11;
            acc0 += x2 * w02; acc1 += x2 * w12;
            acc0 += x3 * w03; acc1 += x3 * w13;
        }
        for (; k < K_; ++k) {
            const float xv = xGm_.GetValue(xBase + (uint64_t)k);
            const float wv0 = wGm_.GetValue(wBase0 + (uint64_t)k);
            const float wv1 = wGm_.GetValue(wBase1 + (uint64_t)k);
            acc0 += xv * wv0;
            acc1 += xv * wv1;
        }

        FinishAndStore(outBase, n + 0U, acc0, subVal, mulVal);
        FinishAndStore(outBase, n + 1U, acc1, subVal, mulVal);
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> subGm_;
    AscendC::GlobalTensor<float> mulGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t rowsPerBlock_{1};
    uint32_t vecN_{2};
};

extern "C" __global__ __aicore__ void matmul_subtract_multiply_relu_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR sub, GM_ADDR mul,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulSubtractMultiplyReluCustom op;
    op.Init(x, w, b, sub, mul, y, td.M, td.K, td.N, td.rowsPerBlock, td.vecN);
    op.Process();
}
