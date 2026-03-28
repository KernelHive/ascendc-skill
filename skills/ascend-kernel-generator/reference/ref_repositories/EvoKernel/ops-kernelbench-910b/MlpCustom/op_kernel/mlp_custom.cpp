
#include "kernel_operator.h"

class KernelMlpCustom {
public:
    __aicore__ inline KernelMlpCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wPacked, GM_ADDR bPacked, GM_ADDR y,
                               uint32_t batch, uint32_t inSize, uint32_t hidden1,
                               uint32_t hidden2, uint32_t outSize, uint32_t rowsPerBlock)
    {
        batch_ = batch;
        inSize_ = inSize;
        hidden1_ = hidden1;
        hidden2_ = hidden2;
        outSize_ = outSize;
        rowsPerBlock_ = (rowsPerBlock == 0 ? 1 : rowsPerBlock);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)wPacked);
        bGm_.SetGlobalBuffer((__gm__ float*)bPacked);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0 || inSize_ == 0 || hidden1_ == 0 || hidden2_ == 0 || outSize_ == 0) return;

        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        uint32_t rowStart = blockIdx * rowsPerBlock_;
        uint32_t rowEnd = rowStart + rowsPerBlock_;
        if (rowStart >= batch_) return;
        if (rowEnd > batch_) rowEnd = batch_;

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            RunRow_(r);
        }
    }

private:
    __aicore__ inline uint64_t WOff0_() const { return 0ULL; }
    __aicore__ inline uint64_t WOff1_() const { return (uint64_t)hidden1_ * (uint64_t)inSize_; }
    __aicore__ inline uint64_t WOff2_() const
    {
        return (uint64_t)hidden1_ * (uint64_t)inSize_ +
               (uint64_t)hidden2_ * (uint64_t)hidden1_;
    }

    __aicore__ inline uint64_t BOff0_() const { return 0ULL; }
    __aicore__ inline uint64_t BOff1_() const { return (uint64_t)hidden1_; }
    __aicore__ inline uint64_t BOff2_() const { return (uint64_t)hidden1_ + (uint64_t)hidden2_; }

    __aicore__ inline uint32_t MaxDim_() const
    {
        uint32_t m = inSize_;
        if (hidden1_ > m) m = hidden1_;
        if (hidden2_ > m) m = hidden2_;
        if (outSize_ > m) m = outSize_;
        return m;
    }

    __aicore__ inline void RunRow_(uint32_t r)
    {
        const uint64_t xBase = (uint64_t)r * (uint64_t)inSize_;
        const uint64_t yBase = (uint64_t)r * (uint64_t)outSize_;

        // One TPipe per kernel function instance; do not create multiple TPipe objects.
        AscendC::TPipe pipe;
        pipe.Init();

        // UB ping-pong activation buffers. Never use output GM as scratch.
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, (uint32_t)(2U * MaxDim_() * sizeof(float)));

        AscendC::LocalTensor<float> act0 = buf.Get<float>();
        AscendC::LocalTensor<float> act1 = act0[(uint32_t)MaxDim_()];

        // Load input row into act0
        for (uint32_t i = 0; i < inSize_; ++i) {
            act0.SetValue(i, xGm_.GetValue(xBase + (uint64_t)i));
        }

        // Layer0: IN -> H1 with ReLU (act1)
        ComputeLinearRelu_(WOff0_(), BOff0_(), inSize_, hidden1_, act0, act1);

        // Layer1: H1 -> H2 with ReLU (act0)
        ComputeLinearRelu_(WOff1_(), BOff1_(), hidden1_, hidden2_, act1, act0);

        // Layer2: H2 -> OUT no ReLU, write to GM
        ComputeLinearOutToGm_(WOff2_(), BOff2_(), hidden2_, outSize_, act0, yBase);
    }

    __aicore__ inline void ComputeLinearRelu_(uint64_t wOff, uint64_t bOff,
                                             uint32_t inDim, uint32_t outDim,
                                             const AscendC::LocalTensor<float>& actIn,
                                             const AscendC::LocalTensor<float>& actOut)
    {
        for (uint32_t o = 0; o < outDim; ++o) {
            const uint64_t wRow = wOff + (uint64_t)o * (uint64_t)inDim;
            float acc = bGm_.GetValue(bOff + (uint64_t)o);

            uint32_t k = 0;
            for (; k + 3 < inDim; k += 4) {
                const float a0 = actIn.GetValue(k + 0);
                const float a1 = actIn.GetValue(k + 1);
                const float a2 = actIn.GetValue(k + 2);
                const float a3 = actIn.GetValue(k + 3);

                const float w0 = wGm_.GetValue(wRow + (uint64_t)(k + 0));
                const float w1 = wGm_.GetValue(wRow + (uint64_t)(k + 1));
                const float w2 = wGm_.GetValue(wRow + (uint64_t)(k + 2));
                const float w3 = wGm_.GetValue(wRow + (uint64_t)(k + 3));

                acc += a0 * w0;
                acc += a1 * w1;
                acc += a2 * w2;
                acc += a3 * w3;
            }
            for (; k < inDim; ++k) {
                acc += actIn.GetValue(k) * wGm_.GetValue(wRow + (uint64_t)k);
            }

            if (acc < 0.0f) acc = 0.0f;
            actOut.SetValue(o, acc);
        }
    }

    __aicore__ inline void ComputeLinearOutToGm_(uint64_t wOff, uint64_t bOff,
                                                uint32_t inDim, uint32_t outDim,
                                                const AscendC::LocalTensor<float>& actIn,
                                                uint64_t yBase)
    {
        for (uint32_t o = 0; o < outDim; ++o) {
            const uint64_t wRow = wOff + (uint64_t)o * (uint64_t)inDim;
            float acc = bGm_.GetValue(bOff + (uint64_t)o);

            uint32_t k = 0;
            for (; k + 3 < inDim; k += 4) {
                const float a0 = actIn.GetValue(k + 0);
                const float a1 = actIn.GetValue(k + 1);
                const float a2 = actIn.GetValue(k + 2);
                const float a3 = actIn.GetValue(k + 3);

                const float w0 = wGm_.GetValue(wRow + (uint64_t)(k + 0));
                const float w1 = wGm_.GetValue(wRow + (uint64_t)(k + 1));
                const float w2 = wGm_.GetValue(wRow + (uint64_t)(k + 2));
                const float w3 = wGm_.GetValue(wRow + (uint64_t)(k + 3));

                acc += a0 * w0;
                acc += a1 * w1;
                acc += a2 * w2;
                acc += a3 * w3;
            }
            for (; k < inDim; ++k) {
                acc += actIn.GetValue(k) * wGm_.GetValue(wRow + (uint64_t)k);
            }

            yGm_.SetValue(yBase + (uint64_t)o, acc);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t batch_{0}, inSize_{0}, hidden1_{0}, hidden2_{0}, outSize_{0};
    uint32_t rowsPerBlock_{1};
};

extern "C" __global__ __aicore__ void mlp_custom(
    GM_ADDR x, GM_ADDR w_packed, GM_ADDR b_packed, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMlpCustom op;
    op.Init(x, w_packed, b_packed, y,
            td.batch, td.inSize, td.hidden1, td.hidden2, td.outSize, td.rowsPerBlock);
    op.Process();
}
