
#include "kernel_operator.h"

// Fused Cosformer Attention on [B,H,S,D] float32 contiguous.
//
// Math:
// decay[t] = exp(-t / S)
// q1[t] = relu(q[t]) * decay[t]
// k1[t] = relu(k[t]) * decay[t]
// KV    = sum_t outer(k1[t], v[t])   -> [D,D]
// Ksum  = sum_t k1[t]               -> [D]
// out[t] = (q1[t] @ KV) / (dot(q1[t], Ksum) + eps)
//
// Key optimization this round:
// Build KV and Ksum once per (B,H) by streaming K/V once (O(S*D) GM reads),
// then compute outputs by streaming Q once with small D×D matvec (O(S*D) GM reads).
// This removes the previous O(S^2) rereads of K/V and reduces scalar inner loops.

static constexpr uint32_t MAX_S = 1024;
static constexpr uint32_t MAX_D = 64;

class KernelCosformerAttentionCustom {
public:
    __aicore__ inline KernelCosformerAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               float eps)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->d = d;
        this->eps = eps;

        const uint64_t totalElems = static_cast<uint64_t>(b) * h * s * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, totalElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, totalElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, totalElems);
        oGm.SetGlobalBuffer((__gm__ float*)out, totalElems);

        pipe.InitBuffer(bufDecay,  MAX_S * sizeof(float));            // decay[S]
        pipe.InitBuffer(bufKSum,   MAX_D * sizeof(float));            // Ksum[D]
        pipe.InitBuffer(bufKV,     MAX_D * MAX_D * sizeof(float));    // KV[D*D]
        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufKRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (b == 0 || h == 0 || s == 0 || d == 0) return;
        if (s > MAX_S || d > MAX_D) return;

        BuildDecay();

        const uint64_t totalBH = static_cast<uint64_t>(b) * h;
        const uint32_t grid = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());

        for (uint64_t bh = bid; bh < totalBH; bh += grid) {
            const uint32_t batch = static_cast<uint32_t>(bh / h);
            const uint32_t head  = static_cast<uint32_t>(bh - static_cast<uint64_t>(batch) * h);
            ComputeOneBH(batch, head);
        }
    }

private:
    __aicore__ inline uint64_t BaseBH(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(s) * d);
    }

    __aicore__ inline void ZeroVec(const AscendC::LocalTensor<float>& t, uint32_t len) const
    {
        for (uint32_t i = 0; i < len; ++i) t(i) = 0.0f;
    }

    __aicore__ inline void ZeroMat(const AscendC::LocalTensor<float>& t, uint32_t rows, uint32_t cols) const
    {
        const uint32_t len = rows * cols;
        for (uint32_t i = 0; i < len; ++i) t(i) = 0.0f;
    }

    __aicore__ inline void BuildDecay()
    {
        auto decayL = bufDecay.Get<float>();
        const float s_f = (s == 0u) ? 1.0f : (1.0f * s);
        const float invS = 1.0f / s_f;

        float neg_t = 0.0f;
        for (uint32_t t = 0; t < s; ++t) {
            decayL(t) = neg_t;
            neg_t -= invS;
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(decayL, decayL, static_cast<int32_t>(s));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadRow(const AscendC::LocalTensor<float>& ub,
                                  const AscendC::GlobalTensor<float>& gm,
                                  uint64_t off) const
    {
        AscendC::DataCopy(ub, gm[off], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void StoreRow(uint64_t off,
                                   const AscendC::LocalTensor<float>& ub) const
    {
        AscendC::DataCopy(oGm[off], ub, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ReluInplace(const AscendC::LocalTensor<float>& row) const
    {
        AscendC::Relu(row, row, static_cast<int32_t>(d));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScaleInplace(const AscendC::LocalTensor<float>& row, float alpha) const
    {
#if defined(ASCENDC_SUPPORTS_MULS)
        AscendC::Muls(row, row, alpha, static_cast<int32_t>(d));
        AscendC::PipeBarrier<PIPE_V>();
#else
        for (uint32_t i = 0; i < d; ++i) row(i) = row(i) * alpha;
        AscendC::PipeBarrier<PIPE_V>();
#endif
    }

    __aicore__ inline float Dot(const AscendC::LocalTensor<float>& a,
                                const AscendC::LocalTensor<float>& bvec) const
    {
        float acc = 0.0f;
        for (uint32_t i = 0; i < d; ++i) acc += a(i) * bvec(i);
        return acc;
    }

    __aicore__ inline void MatVec(const AscendC::LocalTensor<float>& out,
                                 const AscendC::LocalTensor<float>& vec,
                                 const AscendC::LocalTensor<float>& mat) const
    {
        // out[j] = sum_i vec[i] * mat[i,j], mat stored row-major [D,D]
        for (uint32_t j = 0; j < d; ++j) out(j) = 0.0f;
        for (uint32_t i = 0; i < d; ++i) {
            const float vi = vec(i);
            const uint32_t rowBase = i * d;
            for (uint32_t j = 0; j < d; ++j) {
                out(j) = out(j) + vi * mat(rowBase + j);
            }
        }
    }

    __aicore__ inline void Rank1Update(const AscendC::LocalTensor<float>& mat,
                                      const AscendC::LocalTensor<float>& kvec,
                                      const AscendC::LocalTensor<float>& vvec) const
    {
        // mat[i,j] += kvec[i] * vvec[j]
        for (uint32_t i = 0; i < d; ++i) {
            const float ki = kvec(i);
            const uint32_t rowBase = i * d;
            for (uint32_t j = 0; j < d; ++j) {
                mat(rowBase + j) = mat(rowBase + j) + ki * vvec(j);
            }
        }
    }

    __aicore__ inline void ComputeOneBH(uint32_t batch, uint32_t head)
    {
        auto decayL = bufDecay.Get<float>();
        auto qRow   = bufQRow.Get<float>();
        auto kRow   = bufKRow.Get<float>();
        auto vRow   = bufVRow.Get<float>();
        auto outRow = bufOutRow.Get<float>();
        auto ksum   = bufKSum.Get<float>();
        auto kvMat  = bufKV.Get<float>();

        const uint64_t base = BaseBH(batch, head);

        // Pass 1: build Ksum and KV with one stream over (K,V)
        ZeroVec(ksum, d);
        ZeroMat(kvMat, d, d);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t t = 0; t < s; ++t) {
            const uint64_t off = base + static_cast<uint64_t>(t) * d;

            LoadRow(kRow, kGm, off);
            LoadRow(vRow, vGm, off);

            ReluInplace(kRow);
            ScaleInplace(kRow, decayL(t));

            // Ksum += k1
            for (uint32_t i = 0; i < d; ++i) ksum(i) = ksum(i) + kRow(i);

            // KV += outer(k1, v)
            Rank1Update(kvMat, kRow, vRow);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Pass 2: stream Q and compute outputs
        for (uint32_t tout = 0; tout < s; ++tout) {
            const uint64_t offQ = base + static_cast<uint64_t>(tout) * d;

            LoadRow(qRow, qGm, offQ);
            ReluInplace(qRow);
            ScaleInplace(qRow, decayL(tout));

            const float denom = eps + Dot(qRow, ksum);
            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);

            MatVec(outRow, qRow, kvMat);

            for (uint32_t j = 0; j < d; ++j) outRow(j) = outRow(j) * invDen;
            AscendC::PipeBarrier<PIPE_V>();

            StoreRow(offQ, outRow);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufDecay;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKSum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKV;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b {0}, h {0}, s {0}, d {0};
    float eps {1e-6f};
};

extern "C" __global__ __aicore__ void cosformer_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCosformerAttentionCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d,
            tiling_data.eps);
    op.Process();
}
