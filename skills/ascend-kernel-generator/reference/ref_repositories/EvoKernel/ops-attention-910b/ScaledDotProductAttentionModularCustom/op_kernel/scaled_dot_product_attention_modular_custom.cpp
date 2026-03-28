
#include "kernel_operator.h"

// Q:[B,H,NQ,Dk], K:[B,H,NK,Dk], V:[B,H,NK,Dv] -> O:[B,H,NQ,Dv]
static constexpr uint32_t MAX_NQ = 512;
static constexpr uint32_t MAX_NK = 512;
static constexpr uint32_t MAX_DK = 128;
static constexpr uint32_t MAX_DV = 128;

class KernelScaledDotProductAttentionModularCustom {
public:
    __aicore__ inline KernelScaledDotProductAttentionModularCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t nq, uint32_t nk,
                               uint32_t dk, uint32_t dv, float scale)
    {
        this->b = b;
        this->h = h;
        this->nq = nq;
        this->nk = nk;
        this->dk = dk;
        this->dv = dv;
        this->scale = scale;

        const uint64_t qTotal = static_cast<uint64_t>(b) * h * nq * dk;
        const uint64_t kTotal = static_cast<uint64_t>(b) * h * nk * dk;
        const uint64_t vTotal = static_cast<uint64_t>(b) * h * nk * dv;
        const uint64_t oTotal = static_cast<uint64_t>(b) * h * nq * dv;

        qGm.SetGlobalBuffer((__gm__ float*)q, qTotal);
        kGm.SetGlobalBuffer((__gm__ float*)k, kTotal);
        vGm.SetGlobalBuffer((__gm__ float*)v, vTotal);
        oGm.SetGlobalBuffer((__gm__ float*)out, oTotal);

        pipe.InitBuffer(bufQRow, MAX_DK * sizeof(float));

        // Double-buffer K/V to overlap MTE2 and compute across iterations.
        pipe.InitBuffer(bufKRow0, MAX_DK * sizeof(float));
        pipe.InitBuffer(bufKRow1, MAX_DK * sizeof(float));
        pipe.InitBuffer(bufVRow0, MAX_DV * sizeof(float));
        pipe.InitBuffer(bufVRow1, MAX_DV * sizeof(float));

        pipe.InitBuffer(bufOutRow, MAX_DV * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (dk == 0 || dv == 0 || nk == 0 || nq == 0) return;
        if (dk > MAX_DK || dv > MAX_DV || nk > MAX_NK || nq > MAX_NQ) return;

        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t grid = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint64_t totalRows = static_cast<uint64_t>(b) * h * nq;

        for (uint64_t row = bid; row < totalRows; row += static_cast<uint64_t>(grid)) {
            const uint32_t qi   = static_cast<uint32_t>(row % nq);
            const uint32_t bh   = static_cast<uint32_t>(row / nq);
            const uint32_t head = (h == 0) ? 0 : (bh % h);
            const uint32_t batch= (h == 0) ? 0 : (bh / h);
            if (batch >= b) continue;

            const uint64_t qOff = QBase(batch, head, qi);
            LoadRow(bufQRow, qGm, qOff, dk);

            AscendC::LocalTensor<float> qRow   = bufQRow.Get<float>();
            AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();

            // init out
            if (dv == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; di += 8u) {
                    outRow(di+0)=0.0f; outRow(di+1)=0.0f; outRow(di+2)=0.0f; outRow(di+3)=0.0f;
                    outRow(di+4)=0.0f; outRow(di+5)=0.0f; outRow(di+6)=0.0f; outRow(di+7)=0.0f;
                }
            } else {
                for (uint32_t di = 0; di < dv; ++di) outRow(di) = 0.0f;
            }
            AscendC::PipeBarrier<PIPE_V>();

            float m = -3.402823466e+38f; // -FLT_MAX
            float denom = 0.0f;

            // Prefetch j=0 into buffer 0
            if (nk > 0) {
                const uint64_t kOff0 = KBase(batch, head, 0);
                const uint64_t vOff0 = VBase(batch, head, 0);
                LoadRow(bufKRow0, kGm, kOff0, dk);
                LoadRow(bufVRow0, vGm, vOff0, dv);
            }

            for (uint32_t kj = 0; kj < nk; ++kj) {
                const uint32_t cur = (kj & 1u);
                const uint32_t nxt = cur ^ 1u;

                // Prefetch next while we compute current (issue MTE2 early).
                if (kj + 1u < nk) {
                    const uint64_t kOffN = KBase(batch, head, kj + 1u);
                    const uint64_t vOffN = VBase(batch, head, kj + 1u);
                    if (nxt == 0u) {
                        LoadRow(bufKRow0, kGm, kOffN, dk);
                        LoadRow(bufVRow0, vGm, vOffN, dv);
                    } else {
                        LoadRow(bufKRow1, kGm, kOffN, dk);
                        LoadRow(bufVRow1, vGm, vOffN, dv);
                    }
                }

                // Use current buffers
                AscendC::LocalTensor<float> kRow = (cur == 0u) ? bufKRow0.Get<float>() : bufKRow1.Get<float>();
                AscendC::LocalTensor<float> vRow = (cur == 0u) ? bufVRow0.Get<float>() : bufVRow1.Get<float>();

                const float score = DotQK(qRow, kRow);

                if (score > m) {
                    const float alpha = (m == -3.402823466e+38f) ? 0.0f : FastExpApprox(m - score);
                    denom = denom * alpha + 1.0f;

                    if (dv == 64u) {
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
                        for (uint32_t di = 0; di < dv; ++di) outRow(di) = outRow(di) * alpha + vRow(di);
                    }
                    m = score;
                } else {
                    const float p = FastExpApprox(score - m);
                    denom += p;

                    if (dv == 64u) {
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
                        for (uint32_t di = 0; di < dv; ++di) outRow(di) = outRow(di) + p * vRow(di);
                    }
                }

                // One V barrier per iteration is enough to order math before next overwrite.
                AscendC::PipeBarrier<PIPE_V>();
            }

            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);
            if (dv == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; di += 8u) {
                    outRow(di+0) *= invDen; outRow(di+1) *= invDen; outRow(di+2) *= invDen; outRow(di+3) *= invDen;
                    outRow(di+4) *= invDen; outRow(di+5) *= invDen; outRow(di+6) *= invDen; outRow(di+7) *= invDen;
                }
            } else {
                for (uint32_t di = 0; di < dv; ++di) outRow(di) = outRow(di) * invDen;
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopy(oGm[OBase(batch, head, qi)], outRow, dv);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t B, uint32_t H, uint32_t qi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nq) * dk) +
               static_cast<uint64_t>(qi) * dk;
    }
    __aicore__ inline uint64_t KBase(uint32_t B, uint32_t H, uint32_t ki) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nk) * dk) +
               static_cast<uint64_t>(ki) * dk;
    }
    __aicore__ inline uint64_t VBase(uint32_t B, uint32_t H, uint32_t vi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nk) * dv) +
               static_cast<uint64_t>(vi) * dv;
    }
    __aicore__ inline uint64_t OBase(uint32_t B, uint32_t H, uint32_t qi) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(nq) * dv) +
               static_cast<uint64_t>(qi) * dv;
    }

    __aicore__ inline void LoadRow(AscendC::TBuf<AscendC::TPosition::VECCALC>& buf,
                                  AscendC::GlobalTensor<float>& gm, uint64_t off, uint32_t len)
    {
        AscendC::LocalTensor<float> t = buf.Get<float>();
        AscendC::DataCopy(t, gm[off], len);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline float DotQK(const AscendC::LocalTensor<float>& qRow,
                                 const AscendC::LocalTensor<float>& kRow) const
    {
        float acc = 0.0f;
        if (dk == 64u) {
            #pragma unroll
            for (uint32_t di = 0; di < 64u; di += 8u) {
                acc += qRow(di + 0u) * kRow(di + 0u);
                acc += qRow(di + 1u) * kRow(di + 1u);
                acc += qRow(di + 2u) * kRow(di + 2u);
                acc += qRow(di + 3u) * kRow(di + 3u);
                acc += qRow(di + 4u) * kRow(di + 4u);
                acc += qRow(di + 5u) * kRow(di + 5u);
                acc += qRow(di + 6u) * kRow(di + 6u);
                acc += qRow(di + 7u) * kRow(di + 7u);
            }
        } else {
            for (uint32_t di = 0; di < dk; ++di) acc += qRow(di) * kRow(di);
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

    uint32_t b = 0, h = 0, nq = 0, nk = 0, dk = 0, dv = 0;
    float scale = 1.0f;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_modular_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaledDotProductAttentionModularCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.nq, tiling_data.nk,
            tiling_data.dk, tiling_data.dv, tiling_data.scale);
    op.Process();
}
