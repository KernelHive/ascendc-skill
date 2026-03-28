
#include "kernel_operator.h"

static constexpr uint32_t MAX_N = 49;
static constexpr uint32_t S64   = 64;
static constexpr uint32_t MAX_TILE = MAX_N * S64; // 3136

class KernelExternalAttentionCustom {
public:
    __aicore__ inline KernelExternalAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t bs, uint32_t n, uint32_t s)
    {
        this->bs = bs;
        this->n = n;
        this->s = s;

        const uint64_t total = static_cast<uint64_t>(bs) * n * s;
        xGm.SetGlobalBuffer((__gm__ float*)x, total);
        yGm.SetGlobalBuffer((__gm__ float*)y, total);

        // UB buffers
        pipe.InitBuffer(bufTile,   MAX_TILE * sizeof(float));   // [n*s]
        pipe.InitBuffer(bufMaxS,   S64 * sizeof(float));        // [s]
        pipe.InitBuffer(bufSumS,   S64 * sizeof(float));        // [s]
        pipe.InitBuffer(bufInvS,   S64 * sizeof(float));        // [s]
        pipe.InitBuffer(bufOnes,   S64 * sizeof(float));        // [s]
        pipe.InitBuffer(bufRedWk,  256 * sizeof(float));        // reduce workspace (conservative)
        pipe.InitBuffer(bufRedOut, 8 * sizeof(float));          // scalar output container (aligned)
    }

    __aicore__ inline void Process()
    {
        const uint32_t b = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (b >= bs) return;
        if (n == 0 || s != S64 || n > MAX_N) return;

        LoadBatchTile(b);
        SoftmaxDim1_SafeLowScalar();
        L1NormalizeDim2_ReduceSumS64_Safe();
        StoreBatchTile(b);
    }

private:
    __aicore__ inline uint64_t Idx(uint32_t b, uint32_t ni, uint32_t si) const
    {
        return static_cast<uint64_t>(b) * n * s + static_cast<uint64_t>(ni) * s + si;
    }

    __aicore__ inline void LoadBatchTile(uint32_t b)
    {
        AscendC::LocalTensor<float> tile = bufTile.Get<float>();
        for (uint32_t ni = 0; ni < n; ++ni) {
            AscendC::DataCopy(tile[ni * s], xGm[Idx(b, ni, 0)], s);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void StoreBatchTile(uint32_t b)
    {
        AscendC::LocalTensor<float> tile = bufTile.Get<float>();
        for (uint32_t ni = 0; ni < n; ++ni) {
            AscendC::DataCopy(yGm[Idx(b, ni, 0)], tile[ni * s], s);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    // Softmax over dim=1 (N) for each column s (S=64).
    // Safe optimization:
    // - Keep only necessary barriers at true dependency points.
    // - Remove scalar reciprocal loop: invS = 1 / (sumS + eps) using vector Div.
    // - Keep barrier inside sum accumulation loop since sumS is loop-carried and shared.
    __aicore__ inline void SoftmaxDim1_SafeLowScalar()
    {
        AscendC::LocalTensor<float> tile = bufTile.Get<float>();
        AscendC::LocalTensor<float> maxS = bufMaxS.Get<float>();
        AscendC::LocalTensor<float> sumS = bufSumS.Get<float>();
        AscendC::LocalTensor<float> invS = bufInvS.Get<float>();
        AscendC::LocalTensor<float> ones = bufOnes.Get<float>();

        AscendC::Duplicate(ones, 1.0f, (int32_t)s);
        AscendC::PipeBarrier<PIPE_V>();

        // maxS = tile[0]
        AscendC::DataCopy(maxS, tile[0], s);
        AscendC::PipeBarrier<PIPE_V>();

        // max over rows
        for (uint32_t ni = 1; ni < n; ++ni) {
            const uint32_t base = ni * s;
            AscendC::Max(maxS, maxS, tile[base], (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>(); // dependency on maxS across iterations
        }

        // sumS = 0
        AscendC::Duplicate(sumS, 0.0f, (int32_t)s);
        AscendC::PipeBarrier<PIPE_V>();

        // exp(row-max) and accumulate into shared sumS
        for (uint32_t ni = 0; ni < n; ++ni) {
            const uint32_t base = ni * s;
            AscendC::Sub(tile[base], tile[base], maxS, (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tile[base], tile[base], (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(sumS, sumS, tile[base], (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>(); // loop-carried dependency on sumS
        }

        // invS = 1 / (sumS + eps) with vector ops (avoid scalar loop)
        AscendC::Adds(sumS, sumS, 1e-12f, (int32_t)s);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(invS, ones, sumS, (int32_t)s);
        AscendC::PipeBarrier<PIPE_V>();

        // normalize each row
        for (uint32_t ni = 0; ni < n; ++ni) {
            const uint32_t base = ni * s;
            AscendC::Mul(tile[base], tile[base], invS, (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>(); // tile[base] is consumed later by L1 reduce
        }
    }

    __aicore__ inline float RowReduceSumS64(AscendC::LocalTensor<float> row)
    {
        AscendC::LocalTensor<float> wk  = bufRedWk.Get<float>();
        AscendC::LocalTensor<float> out = bufRedOut.Get<float>();

        AscendC::ReduceSum(out, row, wk, 64);
        AscendC::PipeBarrier<PIPE_V>(); // strict: ensure scalar read sees final value
        return out(0);
    }

    // L1 normalization over dim=2 (S=64) for each row.
    // Keep strict barriers because ReduceSum uses shared wk/out buffers and produces a scalar readback.
    __aicore__ inline void L1NormalizeDim2_ReduceSumS64_Safe()
    {
        AscendC::LocalTensor<float> tile = bufTile.Get<float>();

        for (uint32_t ni = 0; ni < n; ++ni) {
            AscendC::LocalTensor<float> row = tile[ni * s];
            float sum = RowReduceSumS64(row);
            if (sum == 0.0f) sum = 1.0f;
            const float inv = 1.0f / sum;
            AscendC::Muls(row, row, inv, (int32_t)s);
            AscendC::PipeBarrier<PIPE_V>(); // ensure Muls completes before next ReduceSum reuses wk/out
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMaxS;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufSumS;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufInvS;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOnes;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRedWk;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufRedOut;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t bs, n, s;
};

extern "C" __global__ __aicore__ void external_attention_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelExternalAttentionCustom op;
    op.Init(x, y, tiling_data.bs, tiling_data.n, tiling_data.s);
    op.Process();
}
