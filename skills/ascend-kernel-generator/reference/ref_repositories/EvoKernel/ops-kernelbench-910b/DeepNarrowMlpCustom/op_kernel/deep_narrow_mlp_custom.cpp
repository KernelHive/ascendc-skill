
#include "kernel_operator.h"

class KernelDeepNarrowMlpCustom {
public:
    __aicore__ inline KernelDeepNarrowMlpCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR wPacked, GM_ADDR bPacked, GM_ADDR y,
                               uint32_t batch, uint32_t inSize, uint32_t hiddenSize, uint32_t outSize,
                               uint32_t numHidden, uint32_t rowsPerBlock)
    {
        batch_ = batch;
        inSize_ = inSize;
        hiddenSize_ = hiddenSize;
        outSize_ = outSize;
        numHidden_ = numHidden;
        rowsPerBlock_ = (rowsPerBlock == 0 ? 1 : rowsPerBlock);

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)wPacked);
        bGm_.SetGlobalBuffer((__gm__ float*)bPacked);
        yGm_.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline void Process()
    {
        if (batch_ == 0 || inSize_ == 0 || hiddenSize_ == 0 || outSize_ == 0 || numHidden_ == 0) return;

        const uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
        uint32_t rowStart = blockIdx * rowsPerBlock_;
        uint32_t rowEnd = rowStart + rowsPerBlock_;
        if (rowStart >= batch_) return;
        if (rowEnd > batch_) rowEnd = batch_;

        for (uint32_t r = rowStart; r < rowEnd; ++r) {
            RunRow(r);
        }
    }

private:
    __aicore__ inline uint64_t WOffsetLayer0() const
    {
        return 0ULL;
    }

    __aicore__ inline uint64_t WOffsetHidden(uint32_t hiddenLayerIdx /*1..numHidden_-1*/) const
    {
        // After W0 [H,IN], each hidden layer is [H,H].
        // hiddenLayerIdx=1 -> first [H,H] block.
        const uint64_t base = (uint64_t)hiddenSize_ * (uint64_t)inSize_;
        return base + (uint64_t)(hiddenLayerIdx - 1U) * (uint64_t)hiddenSize_ * (uint64_t)hiddenSize_;
    }

    __aicore__ inline uint64_t WOffsetFinal() const
    {
        const uint64_t base = (uint64_t)hiddenSize_ * (uint64_t)inSize_ +
                              (uint64_t)(numHidden_ - 1U) * (uint64_t)hiddenSize_ * (uint64_t)hiddenSize_;
        return base;
    }

    __aicore__ inline uint64_t BOffsetHidden(uint32_t layerIdx /*0..numHidden_-1*/) const
    {
        return (uint64_t)layerIdx * (uint64_t)hiddenSize_;
    }

    __aicore__ inline uint64_t BOffsetFinal() const
    {
        return (uint64_t)numHidden_ * (uint64_t)hiddenSize_;
    }

    __aicore__ inline void RunRow(uint32_t r)
    {
        // GM-only reference-like implementation to minimize UB usage.
        // Layout:
        //   x: [batch, in]
        //   y: [batch, out]
        // Packed weights are row-major flattened per layer: [outDim, inDim]
        // Bias per layer: [outDim]
        const uint64_t xBase = (uint64_t)r * (uint64_t)inSize_;
        const uint64_t yBase = (uint64_t)r * (uint64_t)outSize_;

        // activations are conceptually:
        // a0 = x row
        // a1..a16 = hidden activations
        // out = final

        // We'll store the current activation in a scratch region in GM? Not available.
        // Instead, compute each layer output element-by-element using x directly for first,
        // and for subsequent layers recompute from previous layer outputs would be O(n^3).
        //
        // Therefore, we keep a small per-row UB buffer for activations, but sized only to hidden/out.
        // This keeps memory bounded and avoids large multi-queue allocations.

        AscendC::TPipe pipe;
        pipe.Init();

        // UB buffers: actIn (max(in,hidden,out)=8192) and actOut (same).
        // To reduce UB pressure, we allocate exactly needed for each stage by reusing a single max buffer.
        // Note: 8192 floats = 32KB; two buffers = 64KB, generally safe.
        AscendC::TBuf<AscendC::TPosition::VECCALC> buf;
        pipe.InitBuffer(buf, (uint32_t)(2U * MaxDim_() * sizeof(float)));

        AscendC::LocalTensor<float> act0 = buf.Get<float>();
        AscendC::LocalTensor<float> act1 = act0[(uint32_t)MaxDim_()];

        // Load input row into act0 (size inSize_)
        for (uint32_t i = 0; i < inSize_; ++i) {
            act0.SetValue(i, xGm_.GetValue(xBase + (uint64_t)i));
        }

        // Layer 0: IN -> HIDDEN with ReLU into act1
        ComputeLinearRelu(/*wOff=*/WOffsetLayer0(), /*bOff=*/BOffsetHidden(0),
                          /*inDim=*/inSize_, /*outDim=*/hiddenSize_,
                          /*actIn=*/act0, /*actOut=*/act1);

        // Hidden layers 1..numHidden_-1: HIDDEN -> HIDDEN with ReLU, ping-pong act0/act1
        for (uint32_t l = 1; l < numHidden_; ++l) {
            if (l & 1U) {
                ComputeLinearRelu(WOffsetHidden(l), BOffsetHidden(l),
                                  hiddenSize_, hiddenSize_, act1, act0);
            } else {
                ComputeLinearRelu(WOffsetHidden(l), BOffsetHidden(l),
                                  hiddenSize_, hiddenSize_, act0, act1);
            }
        }

        // Final: HIDDEN -> OUT, no ReLU, write to GM
        const bool lastInActIsAct0 = ((numHidden_ - 1U) & 1U) != 0;
        AscendC::LocalTensor<float> lastAct = lastInActIsAct0 ? act0 : act1;
        ComputeLinearOutToGm(WOffsetFinal(), BOffsetFinal(),
                             hiddenSize_, outSize_, lastAct, yBase);
    }

    __aicore__ inline uint32_t MaxDim_() const
    {
        uint32_t m = inSize_;
        if (hiddenSize_ > m) m = hiddenSize_;
        if (outSize_ > m) m = outSize_;
        return m;
    }

    __aicore__ inline void ComputeLinearRelu(uint64_t wOff, uint64_t bOff,
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

    __aicore__ inline void ComputeLinearOutToGm(uint64_t wOff, uint64_t bOff,
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

    uint32_t batch_{0}, inSize_{0}, hiddenSize_{0}, outSize_{0}, numHidden_{0};
    uint32_t rowsPerBlock_{1};
};

extern "C" __global__ __aicore__ void deep_narrow_mlp_custom(
    GM_ADDR x, GM_ADDR w_packed, GM_ADDR b_packed, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelDeepNarrowMlpCustom op;
    op.Init(x, w_packed, b_packed, y,
            td.batch, td.inSize, td.hiddenSize, td.outSize, td.numHidden, td.rowsPerBlock);
    op.Process();
}
