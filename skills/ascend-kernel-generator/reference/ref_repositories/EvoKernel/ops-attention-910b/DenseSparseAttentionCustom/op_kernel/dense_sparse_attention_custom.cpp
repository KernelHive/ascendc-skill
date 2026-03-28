
#include "kernel_operator.h"

static constexpr uint32_t SPEC_SQ   = 1;
static constexpr uint32_t SPEC_DQK  = 576;
static constexpr uint32_t SPEC_DV   = 512;
static constexpr uint32_t SPEC_PBS  = 16;
static constexpr uint32_t MAX_TOPK  = 32;
static constexpr float NEG_INF = -3.402823466e+38f;

class KernelDenseSparseAttentionCustom {
public:
    __aicore__ inline KernelDenseSparseAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kv_cache, GM_ADDR indices,
                               GM_ADDR out,
                               uint32_t B, uint32_t Sq, uint32_t H,
                               uint32_t Dqk, uint32_t Dv,
                               uint32_t NB, uint32_t PBS,
                               uint32_t topk, uint32_t flatKV,
                               uint32_t totalTasks, uint32_t coreNum,
                               float scale)
    {
        this->B = B; this->Sq = Sq; this->H = H;
        this->Dqk = Dqk; this->Dv = Dv;
        this->NB = NB; this->PBS = PBS;
        this->topk = topk; this->flatKV = flatKV;
        this->totalTasks = totalTasks;
        this->coreNum = coreNum;
        this->scale = scale;

        const uint64_t qElems   = static_cast<uint64_t>(B) * Sq * H * Dqk;
        const uint64_t kvElems  = static_cast<uint64_t>(NB) * PBS * 1u * Dqk;
        const uint64_t ixElems  = static_cast<uint64_t>(B) * Sq * topk;
        const uint64_t outElems = static_cast<uint64_t>(B) * Sq * H * Dv;

        qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q, qElems);
        kvGm.SetGlobalBuffer((__gm__ bfloat16_t*)kv_cache, kvElems);
        ixGm.SetGlobalBuffer((__gm__ int32_t*)indices, ixElems);
        outGm.SetGlobalBuffer((__gm__ bfloat16_t*)out, outElems);

        pipe.InitBuffer(bufQbf16,  SPEC_DQK * sizeof(bfloat16_t));
        pipe.InitBuffer(bufQfp32,  SPEC_DQK * sizeof(float));
        pipe.InitBuffer(bufIx,     MAX_TOPK * sizeof(int32_t));
        pipe.InitBuffer(bufRow,    SPEC_DQK * sizeof(bfloat16_t));
        pipe.InitBuffer(bufScores, MAX_TOPK * sizeof(float));
        pipe.InitBuffer(bufTmpDv,  SPEC_DV  * sizeof(float));
        pipe.InitBuffer(bufOutFp,  SPEC_DV  * sizeof(float));
        pipe.InitBuffer(bufOutBf,  SPEC_DV  * sizeof(bfloat16_t));
        pipe.InitBuffer(bufMaxVec, MAX_TOPK * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (Sq != SPEC_SQ || Dqk != SPEC_DQK || Dv != SPEC_DV || PBS != SPEC_PBS) return;
        if (topk == 0u || topk > MAX_TOPK) return;
        if (flatKV != NB * PBS) return;

        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t bnum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (bnum == 0u) bnum = 1u;

        for (uint32_t task = bid; task < totalTasks; task += bnum) {
            const uint32_t b = task / H;
            const uint32_t h = task - b * H;
            ComputeOne(b, 0u, h);
        }
    }

private:
    __aicore__ inline uint64_t QBase(uint32_t b, uint32_t s, uint32_t h) const
    {
        return (((static_cast<uint64_t>(b) * Sq + s) * H + h) * Dqk);
    }

    __aicore__ inline uint64_t OutBase(uint32_t b, uint32_t s, uint32_t h) const
    {
        return (((static_cast<uint64_t>(b) * Sq + s) * H + h) * Dv);
    }

    __aicore__ inline uint64_t IxBase(uint32_t b, uint32_t s) const
    {
        return ((static_cast<uint64_t>(b) * Sq + s) * topk);
    }

    __aicore__ inline uint32_t ClampIndex(int32_t raw) const
    {
        int32_t v = raw;
        if (v < 0) v = 0;
        const int32_t hi = static_cast<int32_t>(flatKV) - 1;
        if (v > hi) v = hi;
        return static_cast<uint32_t>(v);
    }

    __aicore__ inline uint64_t KVRowBase(uint32_t flatIdx) const
    {
        const uint32_t blk = flatIdx / PBS;
        const uint32_t off = flatIdx - blk * PBS;
        return (((static_cast<uint64_t>(blk) * PBS + off) * 1u + 0u) * Dqk);
    }

    __aicore__ inline void LoadQAndConvert(const AscendC::LocalTensor<bfloat16_t>& qBf16,
                                          const AscendC::LocalTensor<float>& qFp32,
                                          uint32_t b, uint32_t s, uint32_t h)
    {
        AscendC::DataCopy(qBf16, qGm[QBase(b, s, h)], SPEC_DQK);
        AscendC::PipeBarrier<PIPE_MTE2>();
        // Scalar conversion once; amortized across topk dot products
        for (uint32_t i = 0; i < SPEC_DQK; ++i) {
            qFp32.SetValue(i, AscendC::ToFloat(qBf16.GetValue(i)));
        }
    }

    __aicore__ inline float DotQK_Row(const AscendC::LocalTensor<float>& qFp32,
                                     const AscendC::LocalTensor<bfloat16_t>& rowBf16) const
    {
        float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
        uint32_t i = 0;
        for (; i + 3 < SPEC_DQK; i += 4) {
            acc0 += qFp32.GetValue(i + 0) * AscendC::ToFloat(rowBf16.GetValue(i + 0));
            acc1 += qFp32.GetValue(i + 1) * AscendC::ToFloat(rowBf16.GetValue(i + 1));
            acc2 += qFp32.GetValue(i + 2) * AscendC::ToFloat(rowBf16.GetValue(i + 2));
            acc3 += qFp32.GetValue(i + 3) * AscendC::ToFloat(rowBf16.GetValue(i + 3));
        }
        float acc = (acc0 + acc1) + (acc2 + acc3);
        for (; i < SPEC_DQK; ++i) acc += qFp32.GetValue(i) * AscendC::ToFloat(rowBf16.GetValue(i));
        return acc;
    }

    __aicore__ inline float ReduceSumTopk(const AscendC::LocalTensor<float>& scores, uint32_t k) const
    {
        float sum = 0.0f;
        // k <= 32, scalar reduction is cheap and avoids ReduceSum API constraints
        for (uint32_t j = 0; j < k; ++j) sum += scores.GetValue(j);
        return sum;
    }

    __aicore__ inline void ComputeOne(uint32_t b, uint32_t s, uint32_t h)
    {
        auto qBf16   = bufQbf16.Get<bfloat16_t>();
        auto qFp32   = bufQfp32.Get<float>();
        auto ixUb    = bufIx.Get<int32_t>();
        auto rowBf16 = bufRow.Get<bfloat16_t>();
        auto scores  = bufScores.Get<float>();    // exp(score-max) then weights
        auto tmpDv   = bufTmpDv.Get<float>();     // fp32 temp [Dv]
        auto outFp   = bufOutFp.Get<float>();     // fp32 accum [Dv]
        auto outBf   = bufOutBf.Get<bfloat16_t>();
        auto maxVec  = bufMaxVec.Get<float>();    // broadcast max for vector Sub

        LoadQAndConvert(qBf16, qFp32, b, s, h);

        AscendC::DataCopy(ixUb, ixGm[IxBase(b, s)], topk);
        AscendC::PipeBarrier<PIPE_MTE2>();
        for (uint32_t j = 0; j < topk; ++j) {
            ixUb.SetValue(j, static_cast<int32_t>(ClampIndex(ixUb.GetValue(j))));
        }

        // Pass 1: scores + max
        float maxScore = NEG_INF;
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t idx = static_cast<uint32_t>(ixUb.GetValue(j));
            const uint64_t kvBase = KVRowBase(idx);
            AscendC::DataCopy(rowBf16, kvGm[kvBase], SPEC_DQK);
            AscendC::PipeBarrier<PIPE_MTE2>();

            const float sc = DotQK_Row(qFp32, rowBf16) * scale;
            scores.SetValue(j, sc);
            if (sc > maxScore) maxScore = sc;
        }

        // Vector shift by max: scores -= max
        AscendC::Duplicate(maxVec, maxScore, static_cast<int32_t>(topk));
        AscendC::Sub(scores, scores, maxVec, static_cast<int32_t>(topk));
        AscendC::Exp(scores, scores, static_cast<int32_t>(topk));

        const float denom = ReduceSumTopk(scores, topk);
        const float invDen = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        AscendC::Muls(scores, scores, invDen, static_cast<int32_t>(topk));

        // out = 0 (vector)
        AscendC::Duplicate(outFp, 0.0f, static_cast<int32_t>(SPEC_DV));

        // Pass 2: stream KV rows; vectorized V accumulation
        for (uint32_t j = 0; j < topk; ++j) {
            const float w = scores.GetValue(j);
            if (w == 0.0f) continue;

            const uint32_t idx = static_cast<uint32_t>(ixUb.GetValue(j));
            const uint64_t kvBase = KVRowBase(idx);
            AscendC::DataCopy(rowBf16, kvGm[kvBase], SPEC_DQK);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // Convert V[0:512] to fp32 temp (scalar loop but only 512 and enables vector math thereafter)
            for (uint32_t i = 0; i < SPEC_DV; ++i) {
                tmpDv.SetValue(i, AscendC::ToFloat(rowBf16.GetValue(i)));
            }
            // tmpDv *= w; out += tmpDv (vector)
            AscendC::Muls(tmpDv, tmpDv, w, static_cast<int32_t>(SPEC_DV));
            AscendC::Add(outFp, outFp, tmpDv, static_cast<int32_t>(SPEC_DV));
        }

        // Cast and store (scalar cast; 512 elems)
        for (uint32_t i = 0; i < SPEC_DV; ++i) {
            outBf.SetValue(i, AscendC::ToBfloat16(outFp.GetValue(i)));
        }
        AscendC::DataCopy(outGm[OutBase(b, s, h)], outBf, SPEC_DV);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQbf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQfp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufIx;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScores;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmpDv;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutFp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutBf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMaxVec;

    AscendC::GlobalTensor<bfloat16_t> qGm;
    AscendC::GlobalTensor<bfloat16_t> kvGm;
    AscendC::GlobalTensor<int32_t>    ixGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;

    uint32_t B, Sq, H, Dqk, Dv, NB, PBS, topk, flatKV, totalTasks, coreNum;
    float scale;
};

extern "C" __global__ __aicore__ void dense_sparse_attention_custom(
    GM_ADDR q, GM_ADDR kv_cache, GM_ADDR indices,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    KernelDenseSparseAttentionCustom op;
    op.Init(q, kv_cache, indices,
            out,
            td.B, td.Sq, td.H,
            td.Dqk, td.Dv,
            td.NB, td.PBS,
            td.topk, td.flatKV,
            td.totalTasks, td.coreNum,
            td.scale);
    op.Process();
}
