
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [128,32768] float32
// w: [N,K] = [32768,32768] float32 (PyTorch Linear weight [out,in])
// b: [N]   = [32768] float32
// scaling: [1] float32 (scalar in tensor)
// y: [M,N] = [128,32768] float32
//
// PyTorch reference:
//   z = x @ w^T + b
//   y = (z * sigmoid(z)) * scaling
//
// Key guardrail: do NOT use scalar expf(); use vector UB Exp() + Reciprocal().

class KernelMatmulSwishScalingCustom {
public:
    __aicore__ inline KernelMatmulSwishScalingCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
                               GM_ADDR y, uint32_t M, uint32_t K, uint32_t N,
                               uint32_t totalElems)
    {
        M_ = M; K_ = K; N_ = N; totalElems_ = totalElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        sGm_.SetGlobalBuffer((__gm__ float*)scaling);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        scale_ = sGm_.GetValue(0);

        // UB buffers for vector sigmoid on a tile.
        // Two queues: one for temps, one for output tile.
        pipe_.InitBuffer(tmpQ_, 1, kTileElems * sizeof(float));     // tmp: stores -z and exp(-z) and (1+exp(-z))
        pipe_.InitBuffer(outQ_, 1, kTileElems * sizeof(float));     // out: stores final y tile
    }

    __aicore__ inline void Process()
    {
        if (M_ == 0 || N_ == 0 || K_ == 0 || totalElems_ == 0) return;

        const int64_t blockNum = (int64_t)AscendC::GetBlockNum();
        const int64_t blockIdx = (int64_t)AscendC::GetBlockIdx();

        const int64_t total = (int64_t)totalElems_;
        const int64_t chunk = (total + blockNum - 1) / blockNum;
        int64_t start = blockIdx * chunk;
        int64_t end = start + chunk;
        if (end > total) end = total;
        if (start >= end) return;

        // Tile over flattened output indices for vector activation.
        for (int64_t base = start; base < end; base += (int64_t)kTileElems) {
            uint32_t tile = (uint32_t)((end - base) > (int64_t)kTileElems ? kTileElems : (end - base));

            // Allocate UB.
            AscendC::LocalTensor<float> tmp = tmpQ_.AllocTensor<float>();
            AscendC::LocalTensor<float> out = outQ_.AllocTensor<float>();

            // Compute z and store into 'out' (temporarily).
            for (uint32_t i = 0; i < tile; ++i) {
                const int64_t outIdx = base + (int64_t)i;
                const uint32_t m = (uint32_t)(outIdx / (int64_t)N_);
                const uint32_t n = (uint32_t)(outIdx - (int64_t)m * (int64_t)N_);

                float acc = 0.0f;
                const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
                const uint64_t wBase = (uint64_t)n * (uint64_t)K_;
                for (uint32_t k = 0; k < K_; ++k) {
                    const float xv = xGm_.GetValue(xBase + (uint64_t)k);
                    const float wv = wGm_.GetValue(wBase + (uint64_t)k);
                    acc += xv * wv;
                }
                acc += bGm_.GetValue((uint64_t)n);

                out.SetValue(i, acc); // z
            }

            // sigmoid(z) = 1 / (1 + exp(-z))
            // tmp = -z
            AscendC::Muls(tmp, out, -1.0f, tile);
            // tmp = exp(-z)
            AscendC::Exp(tmp, tmp, tile);
            // tmp = 1 + exp(-z)
            AscendC::Adds(tmp, tmp, 1.0f, tile);
            // tmp = 1 / (1 + exp(-z))  (sigmoid)
            AscendC::Reciprocal(tmp, tmp, tile);

            // out = z * sigmoid(z)
            AscendC::Mul(out, out, tmp, tile);
            // out = out * scale
            AscendC::Muls(out, out, scale_, tile);

            // Write back to GM.
            for (uint32_t i = 0; i < tile; ++i) {
                yGm_.SetValue((uint64_t)(base + (int64_t)i), out.GetValue(i));
            }

            // Free UB.
            tmpQ_.FreeTensor(tmp);
            outQ_.FreeTensor(out);
        }
    }

private:
    static constexpr uint32_t kTileElems = 256; // small UB tile; safe and amortizes vector ops

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQ_;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> outQ_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    float scale_{1.0f};
};

extern "C" __global__ __aicore__ void matmul_swish_scaling_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMatmulSwishScalingCustom op;
    op.Init(x, w, b, scaling, y, td.M, td.K, td.N, td.totalElems);
    op.Process();
}
