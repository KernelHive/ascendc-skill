
#include "kernel_operator.h"

// Specialized contract:
// x: [M,K] = [1024,8192] float32
// w: [N,K] = [8192,8192] float32 (Linear weight [out,in])
// b: [N]   = [8192] float32
// y: [M,N] = [1024,8192] float32
//
// Computes:
//   z = sum_k x[m,k]*w[n,k] + b[n]
//   z = z * sigmoid(z)      (Swish / SiLU)
//   z = z / 2
//   z = clamp(z, -1, 1)
//   z = tanh(z)
//   y = clamp(z, -1, 1)

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

class KernelGemmSwishDivideClampTanhClampCustom {
public:
    __aicore__ inline KernelGemmSwishDivideClampTanhClampCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t M, uint32_t K, uint32_t N,
                               uint32_t totalElems, uint32_t tileElems,
                               uint32_t tanhTmpBytes)
    {
        M_ = M; K_ = K; N_ = N;
        totalElems_ = totalElems;
        tileElems_ = (tileElems == 0) ? 1 : tileElems;

        xGm_.SetGlobalBuffer((__gm__ float*)x);
        wGm_.SetGlobalBuffer((__gm__ float*)w);
        bGm_.SetGlobalBuffer((__gm__ float*)b);
        yGm_.SetGlobalBuffer((__gm__ float*)y);

        pipe_.InitBuffer(qZ_,   1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qTmp_, 1, tileElems_ * sizeof(float));
        pipe_.InitBuffer(qOut_, 1, tileElems_ * sizeof(float));

        tanhTmpBytes_ = tanhTmpBytes;
        if (tanhTmpBytes_ < 1024) tanhTmpBytes_ = 1024;
        pipe_.InitBuffer(tanhTmpBuf_, tanhTmpBytes_);
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
        constexpr float kHalf = 0.5f;
        constexpr float kClampMin = -1.0f;
        constexpr float kClampMax =  1.0f;

        AscendC::LocalTensor<float> z = qZ_.DeQue<float>();
        AscendC::LocalTensor<float> tmp = qTmp_.AllocTensor<float>();
        AscendC::LocalTensor<float> out = qOut_.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tanhTmp = tanhTmpBuf_.Get<uint8_t>(tanhTmpBytes_);

        // out = silu(z) = z * sigmoid(z)
        AscendC::Silu(out, z, (int32_t)len);

        // out = out / 2
        AscendC::Muls(out, out, kHalf, (int32_t)len);

        // out = clamp(out, -1, 1)
        AscendC::Mins(out, out, kClampMax, (int32_t)len);
        AscendC::Maxs(out, out, kClampMin, (int32_t)len);

        // out = tanh(out)
        // (use tmp as a safe non-overlap staging in case tanh forbids overlap on some versions)
        AscendC::DataCopy(tmp, out, (uint32_t)len);
        AscendC::Tanh(out, tmp, tanhTmp, (int32_t)len);

        // out = clamp(out, -1, 1)
        AscendC::Mins(out, out, kClampMax, (int32_t)len);
        AscendC::Maxs(out, out, kClampMin, (int32_t)len);

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tanhTmpBuf_;

    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> wGm_;
    AscendC::GlobalTensor<float> bGm_;
    AscendC::GlobalTensor<float> yGm_;

    uint32_t M_{0}, K_{0}, N_{0};
    uint32_t totalElems_{0};
    uint32_t tileElems_{2048};
    uint32_t tanhTmpBytes_{8192};
};

extern "C" __global__ __aicore__ void gemm_swish_divide_clamp_tanh_clamp_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR b,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelGemmSwishDivideClampTanhClampCustom op;
    op.Init(x, w, b, y, td.M, td.K, td.N, td.totalElems, td.tileElems, td.tanhTmpBytes);
    op.Process();
}
