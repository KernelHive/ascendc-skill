
#include "kernel_operator.h"

// Q:[B,H,NQ,D], K:[B,H,NK,D], V:[B,H,NK,D] -> O:[B,H,NQ,D]
static constexpr uint32_t MAX_NQ = 512;
static constexpr uint32_t MAX_NK = 512;
static constexpr uint32_t MAX_D  = 128;

class KernelSimplifiedSelfAttentionCustom {
public:
    __aicore__ inline KernelSimplifiedSelfAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t nq, uint32_t nk,
                               uint32_t d, float scale)
    {
        this->b = b;
        this->h = h;
        this->nq = nq;
        this->nk = nk;
        this->d = d;
        this->scale = scale;

        const uint64_t qTotal = static_cast<uint64_t>(b) * h * nq * d;
        const uint64_t kTotal = static_cast<uint64_t>(b) * h * nk * d;
        const uint64_t vTotal = static_cast<uint64_t>(b) * h * nk * d;
        const uint64_t oTotal = static_cast<uint64_t>(b) * h * nq * d;

        qGm.SetGlobalBuffer((__gm__ float*)q, qTotal);
        kGm.SetGlobalBuffer((__gm__ float*)k, kTotal);
        vGm.SetGlobalBuffer((__gm__ float*)v, vTotal);
        oGm.SetGlobalBuffer((__gm__ float*)out, oTotal);

        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));

        // Double-buffer K/V rows to overlap MTE2 with compute across iterations.
        pipe.InitBuffer(bufKRow0,  MAX_D * sizeof(float));
        pipe.InitBuffer(bufKRow1,  MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow0,  MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow1,  MAX_D * sizeof(float));

        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (nq == 0 || nk == 0 || d == 0) return;
        if (d > MAX_D || nk > MAX_NK || nq > MAX_NQ) return;

        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t grid = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint64_t totalRows = static_cast<uint64_t>(b) * h * nq;

        for (uint64_t row = bid; row < totalRows; row += static_cast<uint64_t>(grid)) {
            const uint32_t qi   = static_cast<uint32_t>(row % nq);
            const uint32_t bh   = static_cast<uint32_t>(row / nq);
            const uint32_t head = (h == 0) ? 0 : (bh % h);
            const uint32_t batch = (h == 0) ? 0 : (bh / h);
            if (batch >= b) continue;

            // Load Q row (single copy).
            LoadRowWithBarrier(bufQRow, qGm, QBase(batch, head, qi), d);
            AscendC::LocalTensor<float> qRow   = bufQRow.Get<float>();
            AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

            // init out
            if (d == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; di += 8u) {
                    outRow(di+0)=0.0f; outRow(di+1)=0.0f; outRow(di+2)=0.0f; outRow(di+3)=0.0f;
                    outRow(di+4)=0.0f; outRow(di+5)=0.0f; outRow(di+6)=0.0f; outRow(di+7)=0.0f;
                }
            } else {
                for (uint32_t di = 0; di < d; ++di) outRow(di) = 0.0f;
            }
            AscendC::PipeBarrier<PIPE_V>();

            float m = -3.402823466e+38f; // -FLT_MAX
            float denom = 0.0f;

            // Prefetch key/value j=0 into buffer 0, then one MTE2 barrier to ensure ready.
            if (nk > 0) {
                LoadRowNoBarrier(bufKRow0, kGm, KBase(batch, head, 0), d);
                LoadRowNoBarrier(bufVRow0, vGm, VBase(batch, head, 0), d);
                AscendC::PipeBarrier<PIPE_MTE2>();
            }

            for (uint32_t kj = 0; kj < nk; ++kj) {
                const uint32_t cur = (kj & 1u);
                const uint32_t nxt = cur ^ 1u;

                // Prefetch next (no barrier here; we'll sync once before use).
                if (kj + 1u < nk) {
                    const uint64_t kOffN = KBase(batch, head, kj + 1u);
                    const uint64_t vOffN = VBase(batch, head, kj + 1u);
                    if (nxt == 0u) {
                        LoadRowNoBarrier(bufKRow0, kGm, kOffN, d);
                        LoadRowNoBarrier(bufVRow0, vGm, vOffN, d);
                    } else {
                        LoadRowNoBarrier(bufKRow1, kGm, kOffN, d);
                        LoadRowNoBarrier(bufVRow1, vGm, vOffN, d);
                    }
                }

                // Ensure current buffers are ready before consumption.
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::LocalTensor<float> kRow = (cur == 0u) ? bufKRow0.Get<float>() : bufKRow1.Get<float>();
                AscendC::LocalTensor<float> vRow = (cur == 0u) ? bufVRow0.Get<float>() : bufVRow1.Get<float>();

                const float score = DotScaled(qRow, kRow);

                if (score > m) {
                    const float alpha = (m == -3.402823466e+38f) ? 0.0f : FastExpApprox(m - score);
                    denom = denom * alpha + 1.0f;

                    if (d == 64u) {
                        #pragma unroll
                        for (uint32_t di = 0; di < 64u; di += 8u) {
                            outRow(di+0) = outRow(di+0) * alpha + vRow(di+0);
                            outRow(di+1) = outRow(di+1) * alpha + vRow(di+1);
                            outRow(di+2) = outRow(di+2) * alpha + vRow(di+2);
                            outRow(di+3) = outRow(di+3) * alpha + vRow(di+3);
                            outRow(di+4) = outRow(di+4) * alpha + vRow(di+4);
                            outRow(di+5) = outRow(di+5) * alpha + vRow(di+5);
                            outRow(di+6) = outRow(di+6) * alpha + vRow(di+6);
                            outRow(di+7) = outRow(di+7) * alpha + vRow(di+7);
                        }
                    } else {
                        for (uint32_t di = 0; di < d; ++di) outRow(di) = outRow(di) * alpha + vRow(di);
                    }
                    m = score;
                } else {
                    const float p = FastExpApprox(score - m);
                    denom += p;

                    if (d == 64u) {
                        #pragma unroll
                        for (uint32_t di = 0; di < 64u; di += 8u) {
                            outRow(di+0) = outRow(di+0) + p * vRow(di+0);
                            outRow(di+1) = outRow(di+1) + p * vRow(di+1);
                            outRow(di+2) = outRow(di+2) + p * vRow(di+2);
                            outRow(di+3) = outRow(di+3) + p * vRow(di+3);
                            outRow(di+4) = outRow(di+4) + p * vRow(di+4);
                            outRow(di+5) = outRow(di+5) + p * vRow(di+5);
                            outRow(di+6) = outRow(di+6) + p * vRow(di+6);
                            outRow(di+7) = outRow(di+7) + p * vRow(di+7);
                        }
                    } else {
                        for (uint32_t di = 0; di < d; ++di) outRow(di) = outRow(di) + p * vRow(di);
                    }
                }

                // Order vector updates before the next iteration potentially rescales.
                AscendC::PipeBarrier<PIPE_V>();
            }

            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);
            if (d == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; di += 8u) {
                    outRow(di+0) *= invDen; outRow(di+1) *= invDen; outRow(di+2) *= invDen; outRow(di+3) *= invDen;
                    outRow(di+4) *= invDen; outRow(di+5) *= invDen; outRow(di+6) *= invDen; outRow(di+7) *= invDen;
                }
            } else {
                for (uint32_t di = 0; di < d; ++di) outRow(di) = outRow(di) * invDen;
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopy(oGm[OBase(batch, head, qi)], outRow, d);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t B, uint32_t H, uint32_t qi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nq) * d) +
               static_cast<uint64_t>(qi) * d;
    }
    __aicore__ inline uint64_t KBase(uint32_t B, uint32_t H, uint32_t ki) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nk) * d) +
               static_cast<uint64_t>(ki) * d;
    }
    __aicore__ inline uint64_t VBase(uint32_t B, uint32_t H, uint32_t vi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nk) * d) +
               static_cast<uint64_t>(vi) * d;
    }
    __aicore__ inline uint64_t OBase(uint32_t B, uint32_t H, uint32_t qi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nq) * d) +
               static_cast<uint64_t>(qi) * d;
    }

    __aicore__ inline void LoadRowNoBarrier(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
                                           AscendC::GlobalTensor<float>& gm, uint64_t off, uint32_t len)
    {
        AscendC::LocalTensor<float> t = buf.Get<float>();
        AscendC::DataCopy(t, gm[off], len);
    }

    __aicore__ inline void LoadRowWithBarrier(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
                                             AscendC::GlobalTensor<float>& gm, uint64_t off, uint32_t len)
    {
        LoadRowNoBarrier(buf, gm, off, len);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline float DotScaled(const AscendC::LocalTensor<float>& a,
                                      const AscendC::LocalTensor<float>& bvec) const
    {
        float acc = 0.0f;
        if (d == 64u) {
            #pragma unroll
            for (uint32_t di = 0; di < 64u; di += 8u) {
                acc += a(di + 0u) * bvec(di + 0u);
                acc += a(di + 1u) * bvec(di + 1u);
                acc += a(di + 2u) * bvec(di + 2u);
                acc += a(di + 3u) * bvec(di + 3u);
                acc += a(di + 4u) * bvec(di + 4u);
                acc += a(di + 5u) * bvec(di + 5u);
                acc += a(di + 6u) * bvec(di + 6u);
                acc += a(di + 7u) * bvec(di + 7u);
            }
        } else if (d == 128u) {
            #pragma unroll
            for (uint32_t di = 0; di < 128u; di += 8u) {
                acc += a(di + 0u) * bvec(di + 0u);
                acc += a(di + 1u) * bvec(di + 1u);
                acc += a(di + 2u) * bvec(di + 2u);
                acc += a(di + 3u) * bvec(di + 3u);
                acc += a(di + 4u) * bvec(di + 4u);
                acc += a(di + 5u) * bvec(di + 5u);
                acc += a(di + 6u) * bvec(di + 6u);
                acc += a(di + 7u) * bvec(di + 7u);
            }
        } else {
            for (uint32_t di = 0; di < d; ++di) acc += a(di) * bvec(di);
        }
        return acc * scale;
    }

    __aicore__ inline float FastExpApprox(float x) const
    {
        if (x < -20.0f) x = -20.0f;
        if (x >  0.0f)  x =  0.0f;

        const float inv_ln2 = 1.4426950408889634f;
        float y = x * inv_ln2;

        int32_t n = (int32_t)(y);
        float f = y - (float)n;

        const float c1 = 0.69314718056f;
        const float c2 = 0.24022650695f;
        const float c3 = 0.05550410866f;
        float p = 1.0f + f * (c1 + f * (c2 + f * c3));

        int32_t e = n + 127;
        if (e <= 0) return 0.0f;
        if (e >= 255) e = 255;
        uint32_t bits = ((uint32_t)e) << 23;
        union { uint32_t u; float f; } u;
        u.u = bits;
        return u.f * p;
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b = 0, h = 0, nq = 0, nk = 0, d = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void simplified_self_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSimplifiedSelfAttentionCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.nq, tiling_data.nk,
            tiling_data.d, tiling_data.scale);
    op.Process();
}
