
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [1024,8192] float32
// w: [N,K] = [8192,8192] float32 (Linear weight [out,in])
// b: [N]   = [8192] float32
// scaling: [1] float32 (scalar in tensor)
// y: [M,N] = [1024,8192] float32
//
// Computes:
//   orig = sum_k x[m,k]*w[n,k] + b[n]
//   y = sigmoid(orig) * scaling + orig
//
// Avoid scalar Exp (not available); use LocalTensor intrinsics for sigmoid:
//   sigmoid(x) = 1 / (1 + exp(-x))
//
// Parallelizes by flattened output element index.

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelGemmSigmoidScalingResidualAddCustom {
public:
    __aicore__ inline KernelGemmSigmoidScalingResidualAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
                               GM_ADDR y, uint32_t M, uint32_t K, uint32_t N,
                               uint32_t totalElems, uint32_t tileElems)
    {
        M_ = M; K_ = K; N_ = N;
        totalElems_ = totalElems;
        tileElems_ = (tileElems == 0) ? 1 : tileElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        sGm_.SetGlobalBuffer((__gm__ float*)scaling);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        // Allocate UB buffers for post-ops.
        pipe_.InitBuffer(qOrig_, 1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qTmp_,  1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qOut_,  1, tileElems_ * sizeof(float));

        // Load scaling scalar once.
        // (Use simple GM scalar read; safe for float and avoids extra UB allocation.)
        scale_ = sGm_.GetValue(0);
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

        const uint32_t myLen = (uint32_t)(end - start);
        const uint32_t tiles = CeilDivU32(myLen, tileElems_);

        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t off = t * tileElems_;
            const uint32_t len = (off + tileElems_ <= myLen) ? tileElems_ : (myLen - off);
            const uint64_t baseOut = (uint64_t)start + (uint64_t)off;

            ComputeOrigTile(baseOut, len);
            PostOpTile(len);
            StoreTile(baseOut, len);
        }
    }

private:
    __aicore__ inline void ComputeOrigTile(uint64_t baseOut, uint32_t len)
    {
        AscendC::LocalTensor<float> orig = qOrig_.AllocTensor<float>();

        for (uint32_t i = 0; i < len; ++i) {
            const uint64_t outIdx = baseOut + (uint64_t)i;
            const uint32_t m = (uint32_t)(outIdx / (uint64_t)N_);
            const uint32_t n = (uint32_t)(outIdx - (uint64_t)m * (uint64_t)N_);

            float acc = 0.0f;
            const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
            const uint64_t wBase = (uint64_t)n * (uint64_t)K_;

            for (uint32_t k = 0; k < K_; ++k) {
                const float xv = xGm_.GetValue(xBase + (uint64_t)k);
                const float wv = wGm_.GetValue(wBase + (uint64_t)k);
                acc += xv * wv;
            }
            acc += bGm_.GetValue((uint64_t)n);
            orig(i) = acc;
        }
        qOrig_.EnQue<float>(orig);
    }

    __aicore__ inline void PostOpTile(uint32_t len)
    {
        AscendC::LocalTensor<float> orig = qOrig_.DeQue<float>();
        AscendC::LocalTensor<float> tmp  = qTmp_.AllocTensor<float>();
        AscendC::LocalTensor<float> out  = qOut_.AllocTensor<float>();

        // tmp = -orig
        AscendC::Muls(tmp, orig, -1.0f, (int32_t)len);

        // tmp = exp(tmp) = exp(-orig)
        AscendC::Exp(tmp, tmp, (int32_t)len);

        // tmp = tmp + 1
        AscendC::Adds(tmp, tmp, 1.0f, (int32_t)len);

        // tmp = 1 / tmp  (sigmoid)
        AscendC::Reciprocal(tmp, tmp, (int32_t)len);

        // tmp = tmp * scale
        AscendC::Muls(tmp, tmp, scale_, (int32_t)len);

        // out = tmp + orig  (residual add)
        AscendC::Add(out, tmp, orig, (int32_t)len);

        qOut_.EnQue<float>(out);
        qTmp_.FreeTensor(tmp);
        qOrig_.FreeTensor(orig);
    }

    __aicore__ inline void StoreTile(uint64_t baseOut, uint32_t len)
    {
        AscendC::LocalTensor<float> out = qOut_.DeQue<float>();
        for (uint32_t i = 0; i < len; ++i) {
            yGm_.SetValue(baseOut + (uint64_t)i, out(i));
        }
        qOut_.FreeTensor(out);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOrig_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qTmp_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOut_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    uint32_t tileElems_{256};
    float scale_{1.0f};
};

extern "C" __global__ __aicore__ void gemm_sigmoid_scaling_residual_add_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR scaling,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelGemmSigmoidScalingResidualAddCustom op;
    op.Init(x, w, b, scaling, y, td.M, td.K, td.N, td.totalElems, td.tileElems);
    op.Process();
}
