
#include "kernel_operator.h"

// q: [B,H,NQ,DK], k: [B,H,DK,NK], v:[B,H,NK,DV], y:[B,H,NQ,DV]
static constexpr uint32_t MAX_NQ = 64;
static constexpr uint32_t MAX_NK = 64;
static constexpr uint32_t MAX_DK = 64;
static constexpr uint32_t MAX_DV = 64;

class KernelEMSACustom {
public:
    __aicore__ inline KernelEMSACustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t NQ, uint32_t DK,
                               uint32_t NK, uint32_t DV, float scale,
                               uint32_t totalBH)
    {
        this->B = B;
        this->H = H;
        this->NQ = NQ;
        this->DK = DK;
        this->NK = NK;
        this->DV = DV;
        this->scale = scale;
        this->totalBH = totalBH;

        const uint64_t qSize = static_cast<uint64_t>(B) * H * NQ * DK;
        const uint64_t kSize = static_cast<uint64_t>(B) * H * DK * NK;
        const uint64_t vSize = static_cast<uint64_t>(B) * H * NK * DV;
        const uint64_t ySize = static_cast<uint64_t>(B) * H * NQ * DV;

        qGm.SetGlobalBuffer((__gm__ float*)q, qSize);
        kGm.SetGlobalBuffer((__gm__ float*)k, kSize);
        vGm.SetGlobalBuffer((__gm__ float*)v, vSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB buffers. K/V are cached once per (b,h) and reused for all NQ.
        pipe.InitBuffer(bufQRow,   MAX_DK * sizeof(float));                 // [DK]
        pipe.InitBuffer(bufKAll,   MAX_DK * MAX_NK * sizeof(float));        // [DK*NK]
        pipe.InitBuffer(bufVAll,   MAX_NK * MAX_DV * sizeof(float));        // [NK*DV]
        pipe.InitBuffer(bufScores, MAX_NK * sizeof(float));                 // [NK] scores/probs
        pipe.InitBuffer(bufExps,   MAX_NK * sizeof(float));                 // [NK]
        pipe.InitBuffer(bufOut,    MAX_DV * sizeof(float));                 // [DV]
    }

    __aicore__ inline void Process()
    {
        if (B == 0 || H == 0 || NQ == 0 || DK == 0 || NK == 0 || DV == 0) return;
        if (NQ > MAX_NQ || NK > MAX_NK || DK > MAX_DK || DV > MAX_DV) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (bid >= totalBH) return;

        const uint32_t b = bid / H;
        const uint32_t h = bid - b * H;
        if (b >= B) return;

        LoadKAll(b, h);
        LoadVAll(b, h);

        // For this (b,h), process all query rows; reuse cached K/V in UB.
        for (uint32_t iq = 0; iq < NQ; ++iq) {
            LoadQRow(b, h, iq);
            ComputeScoresFromUB();   // scores in bufScores
            SoftmaxStableInplace();  // scores -> probs
            ComputeOutFromUB();      // out in bufOut
            StoreOut(b, h, iq);
        }
    }

private:
    __aicore__ inline uint64_t QOff(uint32_t b, uint32_t h, uint32_t iq, uint32_t dk) const
    {
        return ((((uint64_t)b * H + h) * NQ + iq) * DK + dk);
    }
    __aicore__ inline uint64_t KOff(uint32_t b, uint32_t h, uint32_t dk, uint32_t ik) const
    {
        return ((((uint64_t)b * H + h) * DK + dk) * NK + ik);
    }
    __aicore__ inline uint64_t VOff(uint32_t b, uint32_t h, uint32_t ik, uint32_t dv) const
    {
        return ((((uint64_t)b * H + h) * NK + ik) * DV + dv);
    }
    __aicore__ inline uint64_t YOff(uint32_t b, uint32_t h, uint32_t iq, uint32_t dv) const
    {
        return ((((uint64_t)b * H + h) * NQ + iq) * DV + dv);
    }

    __aicore__ inline void LoadQRow(uint32_t b, uint32_t h, uint32_t iq)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t qBase = QOff(b, h, iq, 0);
        AscendC::DataCopy(qRow, qGm[qBase], DK);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadKAll(uint32_t b, uint32_t h)
    {
        AscendC::LocalTensor<float> kAll = bufKAll.Get<float>();
        const uint64_t kBase = KOff(b, h, 0, 0);
        AscendC::DataCopy(kAll, kGm[kBase], DK * NK);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadVAll(uint32_t b, uint32_t h)
    {
        AscendC::LocalTensor<float> vAll = bufVAll.Get<float>();
        const uint64_t vBase = VOff(b, h, 0, 0);
        AscendC::DataCopy(vAll, vGm[vBase], NK * DV);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeScoresFromUB()
    {
        AscendC::LocalTensor<float> qRow   = bufQRow.Get<float>();    // [DK]
        AscendC::LocalTensor<float> kAll   = bufKAll.Get<float>();    // [DK*NK]
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();  // [NK]

        for (uint32_t ik = 0; ik < NK; ++ik) {
            float acc = 0.0f;
            #pragma unroll
            for (uint32_t dk = 0; dk < MAX_DK; ++dk) {
                if (dk >= DK) break;
                acc += qRow(dk) * kAll(dk * NK + ik);
            }
            scores(ik) = acc * scale;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SoftmaxStableInplace()
    {
        AscendC::LocalTensor<float> scores = bufScores.Get<float>(); // in/out
        AscendC::LocalTensor<float> exps   = bufExps.Get<float>();

        float m = scores(0);
        for (uint32_t i = 1; i < NK; ++i) {
            float v = scores(i);
            if (v > m) m = v;
        }

        for (uint32_t i = 0; i < NK; ++i) scores(i) = scores(i) - m;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(exps, scores, static_cast<int32_t>(NK));
        AscendC::PipeBarrier<PIPE_V>();

        float s = 0.0f;
        for (uint32_t i = 0; i < NK; ++i) s += exps(i);
        float inv = (s == 0.0f) ? 0.0f : (1.0f / s);

        for (uint32_t i = 0; i < NK; ++i) scores(i) = exps(i) * inv;
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeOutFromUB()
    {
        AscendC::LocalTensor<float> probs = bufScores.Get<float>(); // [NK]
        AscendC::LocalTensor<float> vAll  = bufVAll.Get<float>();   // [NK*DV]
        AscendC::LocalTensor<float> out   = bufOut.Get<float>();    // [DV]

        for (uint32_t dv = 0; dv < DV; ++dv) out(dv) = 0.0f;

        for (uint32_t ik = 0; ik < NK; ++ik) {
            const float p = probs(ik);
            const uint32_t base = ik * DV;
            #pragma unroll
            for (uint32_t dv = 0; dv < MAX_DV; ++dv) {
                if (dv >= DV) break;
                out(dv) = out(dv) + p * vAll(base + dv);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreOut(uint32_t b, uint32_t h, uint32_t iq)
    {
        AscendC::LocalTensor<float> out = bufOut.Get<float>();
        const uint64_t yBase = YOff(b, h, iq, 0);
        AscendC::DataCopy(yGm[yBase], out, DV);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVAll;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufExps;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B, H, NQ, DK, NK, DV;
    float scale;
    uint32_t totalBH;
};

extern "C" __global__ __aicore__ void emsa_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelEMSACustom op;
    op.Init(q, k, v, y,
            tiling_data.B, tiling_data.H, tiling_data.NQ, tiling_data.DK,
            tiling_data.NK, tiling_data.DV, tiling_data.scale,
            tiling_data.totalBH);
    op.Process();
}
