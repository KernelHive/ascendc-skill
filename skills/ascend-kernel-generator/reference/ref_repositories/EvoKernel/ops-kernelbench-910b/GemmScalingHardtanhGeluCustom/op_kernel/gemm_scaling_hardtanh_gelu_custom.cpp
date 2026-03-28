
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [2048,8192] float32
// w: [N,K] = [8192,8192] float32 (Linear weight [out,in])
// b: [N]   = [8192] float32
// scaling/min/max: [1] float32
// y: [M,N] = [2048,8192] float32
//
// Computes:
//   z = sum_k x[m,k]*w[n,k] + b[n]
//   z = z * scaling
//   z = clamp(z, hardtanh_min, hardtanh_max)
//   y = GELU(z)  (exact erf version): 0.5*z*(1 + erf(z / sqrt(2)))

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelGemmScalingHardtanhGeluCustom {
public:
    __aicore__ inline KernelGemmScalingHardtanhGeluCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b,
                               GM_ADDR scaling, GM_ADDR htMin, GM_ADDR htMax,
                               GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t totalElems, uint32_t tileElems)
    {
        M_ = M; K_ = K; N_ = N;
        totalElems_ = totalElems;
        tileElems_ = (tileElems == 0) ? 1 : tileElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        sGm_.SetGlobalBuffer((__gm__ float*)scaling);
        minGm_.SetGlobalBuffer((__gm__ float*)htMin);
        maxGm_.SetGlobalBuffer((__gm__ float*)htMax);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        // UB buffers: orig/clamped, tmp, out, erfTmp bytes.
        pipe_.InitBuffer(qZ_,   1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qTmp_, 1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qOut_, 1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(erfTmpBuf_, 8192); // conservative tmp buffer for Erf

        // Load scalar parameters once per core.
        scale_ = sGm_.GetValue(0);
        htMin_ = minGm_.GetValue(0);
        htMax_ = maxGm_.GetValue(0);
        if (htMin_ > htMax_) { float t = htMin_; htMin_ = htMax_; htMax_ = t; }
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

            ComputeGemmTile(baseOut, len);
            PostOpsTile(len);
            StoreTile(baseOut, len);
        }
    }

private:
    __aicore__ inline void ComputeGemmTile(uint64_t baseOut, uint32_t len)
    {
        AscendC::LocalTensor<float> z = qZ_.AllocTensor<float>();

        for (uint32_t i = 0; i < len; ++i) {
            const uint64_t outIdx = baseOut + (uint64_t)i;
            const uint32_t m = (uint32_t)(outIdx / (uint64_t)N_);
            const uint32_t n = (uint32_t)(outIdx - (uint64_t)m * (uint64_t)N_);

            float acc = 0.0f;
            const uint64_t xBase = (uint64_t)m * (uint64_t)K_;
            const uint64_t wBase = (uint64_t)n * (uint64_t)K_; // w is [N,K]

            for (uint32_t k = 0; k < K_; ++k) {
                const float xv = xGm_.GetValue(xBase + (uint64_t)k);
                const float wv = wGm_.GetValue(wBase + (uint64_t)k);
                acc += xv * wv;
            }
            acc += bGm_.GetValue((uint64_t)n);
            z(i) = acc;
        }
        qZ_.EnQue<float>(z);
    }

    __aicore__ inline void PostOpsTile(uint32_t len)
    {
        constexpr float kInvSqrt2 = 0.7071067811865475f; // 1/sqrt(2)
        constexpr float kHalf = 0.5f;

        AscendC::LocalTensor<float> z = qZ_.DeQue<float>();
        AscendC::LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
        AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> erfTmp = erfTmpBuf_.Get<uint8_t>();

        // z = z * scale
        AscendC::Muls(z, z, scale_, (int32_t)len);

        // z = clamp(z, htMin_, htMax_)  (hardtanh)
        AscendC::Mins(z, z, htMax_, (int32_t)len);
        AscendC::Maxs(z, z, htMin_, (int32_t)len);

        // tmp = z / sqrt(2)
        AscendC::Muls(tmp, z, kInvSqrt2, (int32_t)len);

        // tmp = erf(tmp)
        AscendC::Erf(tmp, tmp, erfTmp, (int32_t)len);

        // tmp = 1 + tmp
        AscendC::Adds(tmp, tmp, 1.0f, (int32_t)len);

        // tmp = 0.5 * tmp
        AscendC::Muls(tmp, tmp, kHalf, (int32_t)len);

        // out = z * tmp
        AscendC::Mul(out, z, tmp, (int32_t)len);

        qOut_.EnQue<float>(out);
        qTmp_.FreeTensor(tmp);
        qZ_.FreeTensor(z);
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
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qZ_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qTmp_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> qOut_;
    AscendC::TBuf<> erfTmpBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> minGm_;
    AscendC::GlobalTensor<float> maxGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    uint32_t tileElems_{1024};
    float scale_{1.0f};
    float htMin_{-2.0f};
    float htMax_{2.0f};
};

extern "C" __global__ __aicore__ void gemm_scaling_hardtanh_gelu_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b,
    GM_ADDR scaling, GM_ADDR hardtanh_min, GM_ADDR hardtanh_max,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelGemmScalingHardtanhGeluCustom op;
    op.Init(x, w, b, scaling, hardtanh_min, hardtanh_max, y,
            td.M, td.K, td.N, td.totalElems, td.tileElems);
    op.Process();
}
