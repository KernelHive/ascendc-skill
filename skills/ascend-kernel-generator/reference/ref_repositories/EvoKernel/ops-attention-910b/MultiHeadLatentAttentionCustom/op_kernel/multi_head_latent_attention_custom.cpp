
#include "kernel_operator.h"

// Decode-only specialization.
static constexpr uint32_t SPEC_SQ  = 1;
static constexpr uint32_t SPEC_DQK = 576;
static constexpr uint32_t SPEC_DV  = 512;
static constexpr uint32_t SPEC_PBS = 16;
static constexpr uint32_t MAX_MBS  = 64;
static constexpr uint32_t MAX_SEQ  = 1024;
static constexpr float NEG_INF = -3.402823466e+38f;

class KernelMultiHeadLatentAttentionCustom {
public:
    __aicore__ inline KernelMultiHeadLatentAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kv_cache, GM_ADDR block_table,
                               GM_ADDR cache_seqlens, GM_ADDR causal_flag,
                               GM_ADDR out,
                               uint32_t B, uint32_t Sq, uint32_t Hq,
                               uint32_t Dqk, uint32_t Dv,
                               uint32_t NB, uint32_t PBS, uint32_t MBS,
                               float scale)
    {
        this->B = B; this->Sq = Sq; this->Hq = Hq;
        this->Dqk = Dqk; this->Dv = Dv;
        this->NB = NB; this->PBS = PBS; this->MBS = MBS;
        this->maxSeq = MBS * PBS;
        this->scale = scale;

        const uint64_t qElems   = static_cast<uint64_t>(B) * Sq * Hq * Dqk;
        const uint64_t outElems = static_cast<uint64_t>(B) * Sq * Hq * Dv;
        qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q, qElems);
        outGm.SetGlobalBuffer((__gm__ bfloat16_t*)out, outElems);

        const uint64_t kvElems = static_cast<uint64_t>(NB) * PBS * 1u * Dqk;
        kvGm.SetGlobalBuffer((__gm__ bfloat16_t*)kv_cache, kvElems);

        btGm.SetGlobalBuffer((__gm__ int32_t*)block_table, static_cast<uint64_t>(B) * MBS);
        slGm.SetGlobalBuffer((__gm__ int32_t*)cache_seqlens, B);
        causalGm.SetGlobalBuffer((__gm__ int32_t*)causal_flag, 1);

        // UB buffers
        pipe.InitBuffer(bufQbf16,  SPEC_DQK * sizeof(bfloat16_t));
        pipe.InitBuffer(bufQfp32,  SPEC_DQK * sizeof(float));

        // KV kept as bf16 only (avoid full-page fp32 conversion loop).
        pipe.InitBuffer(bufKVbf16, SPEC_PBS * SPEC_DQK * sizeof(bfloat16_t));

        pipe.InitBuffer(bufScores, MAX_SEQ * sizeof(float));
        pipe.InitBuffer(bufOut,    SPEC_DV  * sizeof(float));
        pipe.InitBuffer(bufPhys,   MAX_MBS * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (Sq != SPEC_SQ || Dqk != SPEC_DQK || Dv != SPEC_DV || PBS != SPEC_PBS) return;
        if (maxSeq == 0u || maxSeq > MAX_SEQ) return;
        if (MBS > MAX_MBS) return;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t bnum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (bnum == 0) bnum = 1;

        const uint32_t totalTasks = B * Hq;
        const bool causal = (causalGm.GetValue(0) != 0);

        for (uint32_t task = bid; task < totalTasks; task += bnum) {
            const uint32_t b = task / Hq;
            const uint32_t h = task - b * Hq;

            uint32_t seqLen = static_cast<uint32_t>(slGm.GetValue(b));
            if (seqLen == 0u) { StoreZero(b, 0, h); continue; }
            if (seqLen > maxSeq) seqLen = maxSeq;

            ComputeOne(b, 0, h, seqLen, causal);
        }
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t b, uint32_t qpos, uint32_t h) const
    {
        return (((static_cast<uint64_t>(b) * Sq + qpos) * Hq + h) * Dqk);
    }
    __aicore__ inline uint64_t OutBase(uint32_t b, uint32_t qpos, uint32_t h) const
    {
        return (((static_cast<uint64_t>(b) * Sq + qpos) * Hq + h) * Dv);
    }
    __aicore__ inline uint64_t KVBase(uint32_t phys, uint32_t off) const
    {
        return (((static_cast<uint64_t>(phys) * PBS + off) * 1u + 0u) * Dqk);
    }

    __aicore__ inline void StoreZero(uint32_t b, uint32_t qpos, uint32_t h)
    {
        const uint64_t base = OutBase(b, qpos, h);
        for (uint32_t i = 0; i < Dv; ++i) outGm.SetValue(base + i, AscendC::ToBfloat16(0.0f));
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void LoadQ(const AscendC::LocalTensor<bfloat16_t>& qBf16,
                                 const AscendC::LocalTensor<float>& qFp32,
                                 uint32_t b, uint32_t qpos, uint32_t h)
    {
        const uint64_t base = QBase(b, qpos, h);
        AscendC::DataCopy(qBf16, qGm[base], Dqk);
        AscendC::PipeBarrier<PIPE_MTE2>();
        for (uint32_t i = 0; i < Dqk; ++i) qFp32.SetValue(i, AscendC::ToFloat(qBf16.GetValue(i)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GatherPhysBlocks(const AscendC::LocalTensor<int32_t>& physUb,
                                           uint32_t b, uint32_t numBlocks)
    {
        const uint64_t btBase = static_cast<uint64_t>(b) * MBS;
        AscendC::DataCopy(physUb, btGm[btBase], numBlocks);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadKVBlockBf16(const AscendC::LocalTensor<bfloat16_t>& kvBf16,
                                          uint32_t phys)
    {
        const uint64_t base = KVBase(phys, 0);
        AscendC::DataCopy(kvBf16, kvGm[base], PBS * Dqk);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline float DotQKRowBf16(const AscendC::LocalTensor<float>& qFp32,
                                        const AscendC::LocalTensor<bfloat16_t>& kvBf16,
                                        uint32_t tokInBlock) const
    {
        const uint32_t rowBase = tokInBlock * Dqk;
        float acc = 0.0f;
        for (uint32_t i = 0; i < Dqk; ++i) {
            acc += qFp32.GetValue(i) * AscendC::ToFloat(kvBf16.GetValue(rowBase + i));
        }
        return acc;
    }

    __aicore__ inline void AccumVRowBf16(AscendC::LocalTensor<float>& outFp32,
                                        const AscendC::LocalTensor<bfloat16_t>& kvBf16,
                                        uint32_t tokInBlock, float w) const
    {
        const uint32_t rowBase = tokInBlock * Dqk;
        for (uint32_t i = 0; i < Dv; ++i) {
            outFp32.SetValue(i, outFp32.GetValue(i) + w * AscendC::ToFloat(kvBf16.GetValue(rowBase + i)));
        }
    }

    __aicore__ inline void ComputeOne(uint32_t b, uint32_t qpos, uint32_t h,
                                      uint32_t seqLen, bool causal)
    {
        auto qBf16   = bufQbf16.Get<bfloat16_t>();
        auto qFp32   = bufQfp32.Get<float>();
        auto kvBf16  = bufKVbf16.Get<bfloat16_t>();
        auto scores  = bufScores.Get<float>();
        auto outFp32 = bufOut.Get<float>();
        auto physUb  = bufPhys.Get<int32_t>();

        const uint32_t q_abs = (seqLen >= Sq) ? (seqLen - Sq + qpos) : qpos;
        const uint32_t numBlocks = (seqLen + PBS - 1u) / PBS;

        GatherPhysBlocks(physUb, b, numBlocks);
        LoadQ(qBf16, qFp32, b, qpos, h);

        // Pass 1: scores + max
        float maxScore = NEG_INF;
        uint32_t tGlobal = 0;
        for (uint32_t blk = 0; blk < numBlocks; ++blk) {
            const uint32_t phys = static_cast<uint32_t>(physUb.GetValue(blk));
            LoadKVBlockBf16(kvBf16, phys);

            const uint32_t tBase = blk * PBS;
            const uint32_t valid = (tBase + PBS <= seqLen) ? PBS : (seqLen - tBase);

            for (uint32_t j = 0; j < valid; ++j) {
                const uint32_t t = tBase + j;
                float s = NEG_INF;
                if (!causal || (t <= q_abs)) {
                    s = DotQKRowBf16(qFp32, kvBf16, j) * scale;
                }
                scores.SetValue(tGlobal, s);
                if (s > maxScore) maxScore = s;
                ++tGlobal;
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Shift by max; masked -inf -> 0 so Exp() safe
        for (uint32_t t = 0; t < seqLen; ++t) {
            const float s = scores.GetValue(t);
            if (s <= NEG_INF * 0.5f) scores.SetValue(t, 0.0f);
            else scores.SetValue(t, s - maxScore);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(scores, scores, static_cast<int32_t>(seqLen));
        AscendC::PipeBarrier<PIPE_V>();

        float denom = 0.0f;
        for (uint32_t t = 0; t < seqLen; ++t) denom += scores.GetValue(t);
        const float invDen = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        for (uint32_t i = 0; i < Dv; ++i) outFp32.SetValue(i, 0.0f);

        // Pass 2: re-stream KV bf16 and accumulate V using weights.
        tGlobal = 0;
        for (uint32_t blk = 0; blk < numBlocks; ++blk) {
            const uint32_t phys = static_cast<uint32_t>(physUb.GetValue(blk));
            LoadKVBlockBf16(kvBf16, phys);

            const uint32_t tBase = blk * PBS;
            const uint32_t valid = (tBase + PBS <= seqLen) ? PBS : (seqLen - tBase);

            for (uint32_t j = 0; j < valid; ++j) {
                const float w = scores.GetValue(tGlobal) * invDen;
                if (w != 0.0f) AccumVRowBf16(outFp32, kvBf16, j, w);
                ++tGlobal;
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t outBase = OutBase(b, qpos, h);
        for (uint32_t i = 0; i < Dv; ++i) {
            outGm.SetValue(outBase + i, AscendC::ToBfloat16(outFp32.GetValue(i)));
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQbf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQfp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKVbf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufPhys;

    AscendC::GlobalTensor<bfloat16_t> qGm;
    AscendC::GlobalTensor<bfloat16_t> kvGm;
    AscendC::GlobalTensor<int32_t>    btGm;
    AscendC::GlobalTensor<int32_t>    slGm;
    AscendC::GlobalTensor<int32_t>    causalGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;

    uint32_t B, Sq, Hq, Dqk, Dv, NB, PBS, MBS, maxSeq;
    float scale;
};

extern "C" __global__ __aicore__ void multi_head_latent_attention_custom(
    GM_ADDR q, GM_ADDR kv_cache, GM_ADDR block_table, GM_ADDR cache_seqlens, GM_ADDR causal_flag,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMultiHeadLatentAttentionCustom op;
    op.Init(q, kv_cache, block_table, cache_seqlens, causal_flag,
            out,
            td.B, td.Sq, td.Hq,
            td.Dqk, td.Dv,
            td.NB, td.PBS, td.MBS,
            td.scale);
    op.Process();
}
