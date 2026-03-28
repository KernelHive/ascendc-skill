
#include "kernel_operator.h"

// q_phi: [B,H,S,F], k_phi: [B,H,S,F], v: [B,H,S,D] -> y: [B,H,S,D]
//
// FAVOR+ linear attention core:
//   ksum[f]   = sum_s k[s,f]
//   kv[f,d]   = sum_s k[s,f] * v[s,d]
//   denom[s]  = eps + sum_f q[s,f] * ksum[f]
//   numer[s,d]= sum_f q[s,f] * kv[f,d]
//   y[s,d]    = numer[s,d] / denom[s]
//
// Constraints (must match host/python checks):
//   S<=4096, F<=256, D<=128, float32 only.
// Mapping: one block per (B,H).

static constexpr uint32_t MAX_S = 4096;
static constexpr uint32_t MAX_F = 256;
static constexpr uint32_t MAX_D = 128;

class KernelPerformerAttentionCustom {
public:
    __aicore__ inline KernelPerformerAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q_phi, GM_ADDR k_phi, GM_ADDR v, GM_ADDR y,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t f, uint32_t d,
                               float eps, uint32_t blockDim, uint32_t dTile)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->f = f;
        this->d = d;
        this->eps = eps;
        this->blockDim = blockDim;
        this->dTile = (dTile == 0 ? 1 : dTile);

        const uint64_t qElems = static_cast<uint64_t>(b) * h * s * f;
        const uint64_t kElems = static_cast<uint64_t>(b) * h * s * f;
        const uint64_t vElems = static_cast<uint64_t>(b) * h * s * d;
        const uint64_t yElems = static_cast<uint64_t>(b) * h * s * d;

        qGm.SetGlobalBuffer((__gm__ float*)q_phi, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k_phi, kElems);
        vGm.SetGlobalBuffer((__gm__ float*)v,     vElems);
        yGm.SetGlobalBuffer((__gm__ float*)y,     yElems);

        // Persistent accumulators per (B,H)
        pipe.InitBuffer(bufKsum, f * sizeof(float));                 // [F]
        pipe.InitBuffer(bufKV,   (f * d) * sizeof(float));           // [F*D]

        // Per-row temporaries
        pipe.InitBuffer(bufQRow,  f * sizeof(float));                // [F]
        pipe.InitBuffer(bufOutRow, d * sizeof(float));               // [D]
        pipe.InitBuffer(bufTmpD,   this->dTile * sizeof(float));     // [dTile]
    }

    __aicore__ inline void Process()
    {
        if (s > MAX_S || f > MAX_F || d > MAX_D) return;

        const uint32_t bh = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (h == 0) return;
        const uint32_t head  = bh % h;
        const uint32_t batch = bh / h;
        if (batch >= b) return;

        BuildKsumAndKV(batch, head);
        ComputeY(batch, head);
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t B, uint32_t H, uint32_t sIdx) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(s) * f) +
               static_cast<uint64_t>(sIdx) * f;
    }
    __aicore__ inline uint64_t KBase(uint32_t B, uint32_t H, uint32_t sIdx) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(s) * f) +
               static_cast<uint64_t>(sIdx) * f;
    }
    __aicore__ inline uint64_t VBase(uint32_t B, uint32_t H, uint32_t sIdx) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(s) * d) +
               static_cast<uint64_t>(sIdx) * d;
    }
    __aicore__ inline uint64_t YBase(uint32_t B, uint32_t H, uint32_t sIdx) const
    {
        return (static_cast<uint64_t>(B) * h + H) * (static_cast<uint64_t>(s) * d) +
               static_cast<uint64_t>(sIdx) * d;
    }

    __aicore__ inline void Zero1D(const AscendC::LocalTensor<float>& t, uint32_t n) const
    {
        // Simple scalar loop (safe for any n<=32768).
        for (uint32_t i = 0; i < n; ++i) {
            t(i) = 0.0f;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildKsumAndKV(uint32_t B, uint32_t H)
    {
        // ksum[F], kv[F*D]
        AscendC::LocalTensor<float> ksum = bufKsum.Get<float>();
        AscendC::LocalTensor<float> kv   = bufKV.Get<float>();

        AscendC::LocalTensor<float> tmpD = bufTmpD.Get<float>();

        Zero1D(ksum, f);
        Zero1D(kv, f * d);

        // Stream s one row at a time to keep UB bounded and simple.
        // For each s:
        //   ksum += k_row
        //   kv[f,:] += k_row[f] * v_row[:]
        AscendC::LocalTensor<float> vRowUb = bufOutRow.Get<float>(); // reuse as V row scratch [D]
        for (uint32_t sIdx = 0; sIdx < s; ++sIdx) {
            AscendC::LocalTensor<float> qRowUb = bufQRow.Get<float>(); // reuse as K row scratch [F]
            AscendC::DataCopy(qRowUb, kGm[KBase(B, H, sIdx)], f);
            AscendC::DataCopy(vRowUb, vGm[VBase(B, H, sIdx)], d);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // ksum += k_row
            AscendC::Add(ksum, ksum, qRowUb, static_cast<int32_t>(f));
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t fi = 0; fi < f; ++fi) {
                const float ksf = qRowUb(fi);
                const uint32_t kvBase = fi * d;

                uint32_t d0 = 0;
                for (; d0 + dTile <= d; d0 += dTile) {
                    AscendC::Muls(tmpD, vRowUb[d0], ksf, static_cast<int32_t>(dTile));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(kv[kvBase + d0], kv[kvBase + d0], tmpD, static_cast<int32_t>(dTile));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (; d0 < d; ++d0) {
                    kv(kvBase + d0) = kv(kvBase + d0) + ksf * vRowUb(d0);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeY(uint32_t B, uint32_t H)
    {
        AscendC::LocalTensor<float> ksum  = bufKsum.Get<float>();   // [F]
        AscendC::LocalTensor<float> kv    = bufKV.Get<float>();     // [F*D]
        AscendC::LocalTensor<float> qRow  = bufQRow.Get<float>();   // [F]
        AscendC::LocalTensor<float> outRow= bufOutRow.Get<float>(); // [D]
        AscendC::LocalTensor<float> tmpD  = bufTmpD.Get<float>();   // [dTile]

        for (uint32_t sIdx = 0; sIdx < s; ++sIdx) {
            AscendC::DataCopy(qRow, qGm[QBase(B, H, sIdx)], f);
            AscendC::PipeBarrier<PIPE_MTE2>();

            float denom = eps;
            for (uint32_t fi = 0; fi < f; ++fi) {
                denom += qRow(fi) * ksum(fi);
            }
            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);

            // outRow = sum_f q[f] * kv[f,:]
            Zero1D(outRow, d);

            for (uint32_t fi = 0; fi < f; ++fi) {
                const float qsf = qRow(fi);
                const uint32_t kvBase = fi * d;

                uint32_t d0 = 0;
                for (; d0 + dTile <= d; d0 += dTile) {
                    AscendC::Muls(tmpD, kv[kvBase + d0], qsf, static_cast<int32_t>(dTile));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(outRow[d0], outRow[d0], tmpD, static_cast<int32_t>(dTile));
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (; d0 < d; ++d0) {
                    outRow(d0) = outRow(d0) + qsf * kv(kvBase + d0);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::Muls(outRow, outRow, invDen, static_cast<int32_t>(d));
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::DataCopy(yGm[YBase(B, H, sIdx)], outRow, d);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKsum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKV;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECOUT>  bufOutRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmpD;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t b = 0, h = 0, s = 0, f = 0, d = 0;
    float eps = 1e-6f;
    uint32_t blockDim = 0;
    uint32_t dTile = 1;
};

extern "C" __global__ __aicore__ void performer_attention_custom(
    GM_ADDR q_phi, GM_ADDR k_phi, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelPerformerAttentionCustom op;
    op.Init(q_phi, k_phi, v, y,
            tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.f, tiling_data.d,
            tiling_data.eps, tiling_data.block_dim, tiling_data.d_tile);
    op.Process();
}
