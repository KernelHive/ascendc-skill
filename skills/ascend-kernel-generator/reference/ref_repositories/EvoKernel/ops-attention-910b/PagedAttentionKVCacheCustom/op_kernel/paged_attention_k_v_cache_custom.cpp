
#include "kernel_operator.h"

static constexpr uint32_t MAX_SQ  = 4;      // decoding-oriented
static constexpr uint32_t MAX_D   = 128;    // common head dim in LLMs
static constexpr uint32_t MAX_SEQ = 1024;   // max cached tokens supported by this UB-bounded kernel

class KernelPagedAttentionKVCacheCustom {
public:
    __aicore__ inline KernelPagedAttentionKVCacheCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,
                               GM_ADDR cache_seqlens, GM_ADDR page_table, GM_ADDR causal_flag,
                               GM_ADDR out,
                               uint32_t B, uint32_t Sq, uint32_t Hq, uint32_t D,
                               uint32_t NB, uint32_t PBS, uint32_t Hkv, uint32_t MBS,
                               uint32_t groups, float scale)
    {
        this->B = B; this->Sq = Sq; this->Hq = Hq; this->D = D;
        this->NB = NB; this->PBS = PBS; this->Hkv = Hkv; this->MBS = MBS;
        this->groups = groups == 0u ? 1u : groups;
        this->maxSeq = MBS * PBS;
        this->scale = scale;

        const uint64_t qElems = static_cast<uint64_t>(B) * Sq * Hq * D;
        qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q, qElems);
        outGm.SetGlobalBuffer((__gm__ bfloat16_t*)out, qElems);

        const uint64_t cacheElems = static_cast<uint64_t>(NB) * PBS * Hkv * D;
        kGm.SetGlobalBuffer((__gm__ bfloat16_t*)k_cache, cacheElems);
        vGm.SetGlobalBuffer((__gm__ bfloat16_t*)v_cache, cacheElems);

        seqlenGm.SetGlobalBuffer((__gm__ int32_t*)cache_seqlens, B);
        pageTableGm.SetGlobalBuffer((__gm__ int32_t*)page_table, static_cast<uint64_t>(B) * MBS);
        causalFlagGm.SetGlobalBuffer((__gm__ int32_t*)causal_flag, 1);

        pipe.InitBuffer(bufQ,      MAX_D * sizeof(float));
        pipe.InitBuffer(bufK,      MAX_D * sizeof(float));
        pipe.InitBuffer(bufV,      MAX_D * sizeof(float));
        pipe.InitBuffer(bufScores, MAX_SEQ * sizeof(float));
        pipe.InitBuffer(bufOut,    MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (Sq > MAX_SQ || D > MAX_D || maxSeq > MAX_SEQ) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t bnum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (bnum == 0) bnum = 1;

        const uint32_t totalTasks = B * Hq;
        const bool causal = (causalFlagGm.GetValue(0) != 0);

        for (uint32_t task = bid; task < totalTasks; task += bnum) {
            const uint32_t b = task / Hq;
            const uint32_t hq = task - b * Hq;

            uint32_t seqLen = static_cast<uint32_t>(seqlenGm.GetValue(b));
            if (seqLen == 0u) {
                ZeroOut(b, hq);
                continue;
            }
            if (seqLen > maxSeq) seqLen = maxSeq;

            for (uint32_t qpos = 0; qpos < Sq; ++qpos) {
                ComputeOne(b, qpos, hq, seqLen, causal);
            }
        }
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t b, uint32_t qpos, uint32_t hq) const
    {
        // q/out: [B,Sq,Hq,D]
        return (((static_cast<uint64_t>(b) * Sq + qpos) * Hq + hq) * D);
    }

    __aicore__ inline uint32_t GetPhysBlock(uint32_t b, uint32_t logicalBlk) const
    {
        const uint64_t idx = static_cast<uint64_t>(b) * MBS + logicalBlk;
        return static_cast<uint32_t>(pageTableGm.GetValue(idx));
    }

    __aicore__ inline void LoadQ(const AscendC::LocalTensor<float>& qFP, uint32_t b, uint32_t qpos, uint32_t hq)
    {
        const uint64_t base = QBase(b, qpos, hq);
        for (uint32_t i = 0; i < D; ++i) {
            qFP.SetValue(i, AscendC::ToFloat(qGm.GetValue(base + i)));
        }
    }

    __aicore__ inline void LoadKRow(const AscendC::LocalTensor<float>& kFP, uint32_t b, uint32_t t, uint32_t hkv)
    {
        const uint32_t blk = t / PBS;
        const uint32_t off = t - blk * PBS;
        const uint32_t phys = GetPhysBlock(b, blk);

        // k_cache: [NB,PBS,Hkv,D]
        const uint64_t base = (((static_cast<uint64_t>(phys) * PBS + off) * Hkv + hkv) * D);
        for (uint32_t i = 0; i < D; ++i) {
            kFP.SetValue(i, AscendC::ToFloat(kGm.GetValue(base + i)));
        }
    }

    __aicore__ inline void LoadVRow(const AscendC::LocalTensor<float>& vFP, uint32_t b, uint32_t t, uint32_t hkv)
    {
        const uint32_t blk = t / PBS;
        const uint32_t off = t - blk * PBS;
        const uint32_t phys = GetPhysBlock(b, blk);

        // v_cache: [NB,PBS,Hkv,D]
        const uint64_t base = (((static_cast<uint64_t>(phys) * PBS + off) * Hkv + hkv) * D);
        for (uint32_t i = 0; i < D; ++i) {
            vFP.SetValue(i, AscendC::ToFloat(vGm.GetValue(base + i)));
        }
    }

    __aicore__ inline float Dot(const AscendC::LocalTensor<float>& a, const AscendC::LocalTensor<float>& b) const
    {
        float acc = 0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            acc += a.GetValue(i) * b.GetValue(i);
        }
        return acc;
    }

    __aicore__ inline void ZeroOut(uint32_t b, uint32_t hq)
    {
        for (uint32_t qpos = 0; qpos < Sq; ++qpos) {
            const uint64_t base = QBase(b, qpos, hq);
            for (uint32_t i = 0; i < D; ++i) {
                outGm.SetValue(base + i, AscendC::ToBfloat16(0.0f));
            }
        }
    }

    __aicore__ inline void ComputeOne(uint32_t b, uint32_t qpos, uint32_t hq, uint32_t seqLen, bool causal)
    {
        AscendC::LocalTensor<float> qFP    = bufQ.Get<float>();
        AscendC::LocalTensor<float> kFP    = bufK.Get<float>();
        AscendC::LocalTensor<float> vFP    = bufV.Get<float>();
        AscendC::LocalTensor<float> scores = bufScores.Get<float>();
        AscendC::LocalTensor<float> outFP  = bufOut.Get<float>();

        const uint32_t hkv = hq / groups;

        LoadQ(qFP, b, qpos, hq);

        // decoding mapping: query positions correspond to last Sq tokens
        const uint32_t q_abs = (seqLen >= Sq) ? (seqLen - Sq + qpos) : qpos;

        // pass1: scores + max (fp32 scalars)
        float maxScore = -3.402823466e+38f; // -FLT_MAX
        for (uint32_t t = 0; t < seqLen; ++t) {
            if (causal && (t > q_abs)) {
                scores.SetValue(t, -3.402823466e+38f);
                continue;
            }
            LoadKRow(kFP, b, t, hkv);
            const float s = Dot(qFP, kFP) * scale;
            scores.SetValue(t, s);
            if (s > maxScore) maxScore = s;
        }
        AscendC::PipeBarrier<PIPE_V>();

        // scores = exp(scores - maxScore), masked -> 0
        for (uint32_t t = 0; t < seqLen; ++t) {
            const float s = scores.GetValue(t);
            if (s <= -3.0e+38f) {
                scores.SetValue(t, 0.0f);
            } else {
                scores.SetValue(t, s - maxScore);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(scores, scores, static_cast<int32_t>(seqLen));
        AscendC::PipeBarrier<PIPE_V>();

        float denom = 0.0f;
        for (uint32_t t = 0; t < seqLen; ++t) {
            denom += scores.GetValue(t);
        }
        const float invDen = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        for (uint32_t i = 0; i < D; ++i) outFP.SetValue(i, 0.0f);

        for (uint32_t t = 0; t < seqLen; ++t) {
            const float w = scores.GetValue(t) * invDen;
            if (w == 0.0f) continue;
            LoadVRow(vFP, b, t, hkv);
            for (uint32_t i = 0; i < D; ++i) {
                outFP.SetValue(i, outFP.GetValue(i) + w * vFP.GetValue(i));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t outBase = QBase(b, qpos, hq);
        for (uint32_t i = 0; i < D; ++i) {
            outGm.SetValue(outBase + i, AscendC::ToBfloat16(outFP.GetValue(i)));
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufK;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufV;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut;

    AscendC::GlobalTensor<bfloat16_t> qGm;
    AscendC::GlobalTensor<bfloat16_t> kGm;
    AscendC::GlobalTensor<bfloat16_t> vGm;
    AscendC::GlobalTensor<int32_t>    seqlenGm;
    AscendC::GlobalTensor<int32_t>    pageTableGm;
    AscendC::GlobalTensor<int32_t>    causalFlagGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;

    uint32_t B, Sq, Hq, D, NB, PBS, Hkv, MBS, maxSeq, groups;
    float scale;
};

extern "C" __global__ __aicore__ void paged_attention_kv_cache_custom(
    GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR cache_seqlens, GM_ADDR page_table, GM_ADDR causal_flag,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelPagedAttentionKVCacheCustom op;
    op.Init(q, k_cache, v_cache,
            cache_seqlens, page_table, causal_flag,
            out,
            tiling_data.B, tiling_data.Sq, tiling_data.Hq, tiling_data.D,
            tiling_data.NB, tiling_data.PBS, tiling_data.Hkv, tiling_data.MBS,
            tiling_data.groups,
            tiling_data.scale);
    op.Process();
}
