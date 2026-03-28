
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;
using F16 = half;

class KernelMhcPostBlockCustom {
public:
    __aicore__ inline KernelMhcPostBlockCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x_fp16, GM_ADDR residual_fp16, GM_ADDR post_layer_mix, GM_ADDR comb_res_mix,
        GM_ADDR hidden_size, GM_ADDR hc_mult,
        GM_ADDR out_fp16,
        uint32_t N, uint32_t S, uint32_t H)
    {
        this->N = N; this->S = S; this->H = H;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t nPerBlock = (N + blockNum - 1u) / blockNum;
        nStart = blockIdx * nPerBlock;
        uint32_t nEnd0 = nStart + nPerBlock;
        if (nEnd0 > N) nEnd0 = N;
        nEnd = nEnd0;

        xGm.SetGlobalBuffer((__gm__ F16*)x_fp16, (uint64_t)N * (uint64_t)H);
        rGm.SetGlobalBuffer((__gm__ F16*)residual_fp16, (uint64_t)N * 4ull * (uint64_t)H);
        postGm.SetGlobalBuffer((__gm__ F32*)post_layer_mix, (uint64_t)N * 4ull);       // (N,4,1) flattened
        combGm.SetGlobalBuffer((__gm__ F32*)comb_res_mix, (uint64_t)N * 16ull);        // (N,4,4) flattened row-major [i][j]

        hsGm.SetGlobalBuffer((__gm__ I32*)hidden_size, (uint64_t)1);
        hmGm.SetGlobalBuffer((__gm__ I32*)hc_mult, (uint64_t)1);

        outGm.SetGlobalBuffer((__gm__ F16*)out_fp16, (uint64_t)N * 4ull * (uint64_t)H);

        supported = (S == 4u) && (H > 0u) && (N > 0u);
    }

    __aicore__ inline void AllocUb()
    {
        // Double-buffer FP16 inputs to overlap MTE2 (prefetch next token) with compute of current.
        pipe.InitBuffer(xUbQ, 2, H * sizeof(F16));
        pipe.InitBuffer(r0UbQ, 2, H * sizeof(F16));
        pipe.InitBuffer(r1UbQ, 2, H * sizeof(F16));
        pipe.InitBuffer(r2UbQ, 2, H * sizeof(F16));
        pipe.InitBuffer(r3UbQ, 2, H * sizeof(F16));

        // Single FP32 copies for compute (we cast current token's fp16 buffers into these).
        pipe.InitBuffer(xFbuf,  H * sizeof(F32));
        pipe.InitBuffer(r0Fbuf, H * sizeof(F32));
        pipe.InitBuffer(r1Fbuf, H * sizeof(F32));
        pipe.InitBuffer(r2Fbuf, H * sizeof(F32));
        pipe.InitBuffer(r3Fbuf, H * sizeof(F32));

        // Two output ping-pong buffers to avoid overwriting while MTE3 is still copying out.
        pipe.InitBuffer(yF0buf, H * sizeof(F32));
        pipe.InitBuffer(yF1buf, H * sizeof(F32));
        pipe.InitBuffer(yH0buf, H * sizeof(F16));
        pipe.InitBuffer(yH1buf, H * sizeof(F16));
    }

    __aicore__ inline void PrefetchToken(uint32_t n)
    {
        const uint64_t xBase = (uint64_t)n * (uint64_t)H;
        const uint64_t rBase = (uint64_t)n * 4ull * (uint64_t)H;

        auto xUb  = xUbQ.AllocTensor<F16>();
        auto r0Ub = r0UbQ.AllocTensor<F16>();
        auto r1Ub = r1UbQ.AllocTensor<F16>();
        auto r2Ub = r2UbQ.AllocTensor<F16>();
        auto r3Ub = r3UbQ.AllocTensor<F16>();

        AscendC::DataCopy(xUb,  xGm[xBase], H);
        AscendC::DataCopy(r0Ub, rGm[rBase + 0ull * (uint64_t)H], H);
        AscendC::DataCopy(r1Ub, rGm[rBase + 1ull * (uint64_t)H], H);
        AscendC::DataCopy(r2Ub, rGm[rBase + 2ull * (uint64_t)H], H);
        AscendC::DataCopy(r3Ub, rGm[rBase + 3ull * (uint64_t)H], H);

        xUbQ.EnQue(xUb);
        r0UbQ.EnQue(r0Ub);
        r1UbQ.EnQue(r1Ub);
        r2UbQ.EnQue(r2Ub);
        r3UbQ.EnQue(r3Ub);
    }

    __aicore__ inline void ProcessToken(uint32_t n)
    {
        const uint64_t postBase = (uint64_t)n * 4ull;
        const uint64_t combBase = (uint64_t)n * 16ull;
        const uint64_t oBase    = (uint64_t)n * 4ull * (uint64_t)H;

        // Scalar coefficients: tiny and safe (avoid UB staging hazards for small tensors).
        const F32 p0 = postGm.GetValue(postBase + 0ull);
        const F32 p1 = postGm.GetValue(postBase + 1ull);
        const F32 p2 = postGm.GetValue(postBase + 2ull);
        const F32 p3 = postGm.GetValue(postBase + 3ull);

        // comb row-major [i][j] at combBase + i*4 + j, and compute y_j += sum_i c[i][j]*r_i (comb^T @ residual).
        const F32 c00 = combGm.GetValue(combBase + 0ull);
        const F32 c01 = combGm.GetValue(combBase + 1ull);
        const F32 c02 = combGm.GetValue(combBase + 2ull);
        const F32 c03 = combGm.GetValue(combBase + 3ull);

        const F32 c10 = combGm.GetValue(combBase + 4ull);
        const F32 c11 = combGm.GetValue(combBase + 5ull);
        const F32 c12 = combGm.GetValue(combBase + 6ull);
        const F32 c13 = combGm.GetValue(combBase + 7ull);

        const F32 c20 = combGm.GetValue(combBase + 8ull);
        const F32 c21 = combGm.GetValue(combBase + 9ull);
        const F32 c22 = combGm.GetValue(combBase + 10ull);
        const F32 c23 = combGm.GetValue(combBase + 11ull);

        const F32 c30 = combGm.GetValue(combBase + 12ull);
        const F32 c31 = combGm.GetValue(combBase + 13ull);
        const F32 c32 = combGm.GetValue(combBase + 14ull);
        const F32 c33 = combGm.GetValue(combBase + 15ull);

        auto xUb  = xUbQ.DeQue<F16>();
        auto r0Ub = r0UbQ.DeQue<F16>();
        auto r1Ub = r1UbQ.DeQue<F16>();
        auto r2Ub = r2UbQ.DeQue<F16>();
        auto r3Ub = r3UbQ.DeQue<F16>();

        auto xF  = xFbuf.Get<F32>();
        auto r0F = r0Fbuf.Get<F32>();
        auto r1F = r1Fbuf.Get<F32>();
        auto r2F = r2Fbuf.Get<F32>();
        auto r3F = r3Fbuf.Get<F32>();

        // Cast current token to FP32 once.
        AscendC::Cast(xF,  xUb,  AscendC::RoundMode::CAST_NONE, H);
        AscendC::Cast(r0F, r0Ub, AscendC::RoundMode::CAST_NONE, H);
        AscendC::Cast(r1F, r1Ub, AscendC::RoundMode::CAST_NONE, H);
        AscendC::Cast(r2F, r2Ub, AscendC::RoundMode::CAST_NONE, H);
        AscendC::Cast(r3F, r3Ub, AscendC::RoundMode::CAST_NONE, H);

        // Release fp16 input tiles early (frees UB for upcoming prefetch).
        xUbQ.FreeTensor(xUb);
        r0UbQ.FreeTensor(r0Ub);
        r1UbQ.FreeTensor(r1Ub);
        r2UbQ.FreeTensor(r2Ub);
        r3UbQ.FreeTensor(r3Ub);

        // Ping-pong output buffers across streams to avoid MTE3 overwrite hazards.
        auto yF0 = yF0buf.Get<F32>();
        auto yF1 = yF1buf.Get<F32>();
        auto yH0 = yH0buf.Get<F16>();
        auto yH1 = yH1buf.Get<F16>();

        // Stream 0 -> buffers 0
        AscendC::Muls(yF0, xF, p0, H);
        AscendC::Axpy(yF0, r0F, c00, H);
        AscendC::Axpy(yF0, r1F, c10, H);
        AscendC::Axpy(yF0, r2F, c20, H);
        AscendC::Axpy(yF0, r3F, c30, H);
        AscendC::Cast(yH0, yF0, AscendC::RoundMode::CAST_NONE, H);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outGm[oBase + 0ull * (uint64_t)H], yH0, H);

        // Stream 1 -> buffers 1
        AscendC::Muls(yF1, xF, p1, H);
        AscendC::Axpy(yF1, r0F, c01, H);
        AscendC::Axpy(yF1, r1F, c11, H);
        AscendC::Axpy(yF1, r2F, c21, H);
        AscendC::Axpy(yF1, r3F, c31, H);
        AscendC::Cast(yH1, yF1, AscendC::RoundMode::CAST_NONE, H);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outGm[oBase + 1ull * (uint64_t)H], yH1, H);

        // Stream 2 -> buffers 0 (safe because stream0 copy has been fenced by PIPE_ALL before reuse)
        AscendC::Muls(yF0, xF, p2, H);
        AscendC::Axpy(yF0, r0F, c02, H);
        AscendC::Axpy(yF0, r1F, c12, H);
        AscendC::Axpy(yF0, r2F, c22, H);
        AscendC::Axpy(yF0, r3F, c32, H);
        AscendC::Cast(yH0, yF0, AscendC::RoundMode::CAST_NONE, H);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outGm[oBase + 2ull * (uint64_t)H], yH0, H);

        // Stream 3 -> buffers 1
        AscendC::Muls(yF1, xF, p3, H);
        AscendC::Axpy(yF1, r0F, c03, H);
        AscendC::Axpy(yF1, r1F, c13, H);
        AscendC::Axpy(yF1, r2F, c23, H);
        AscendC::Axpy(yF1, r3F, c33, H);
        AscendC::Cast(yH1, yF1, AscendC::RoundMode::CAST_NONE, H);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(outGm[oBase + 3ull * (uint64_t)H], yH1, H);
    }

    __aicore__ inline void Process()
    {
        if (!supported) return;
        if (nStart >= nEnd) return;

        // runtime guard: scalar inputs must match tiling
        const I32 hs = hsGm.GetValue(0);
        const I32 hm = hmGm.GetValue(0);
        if ((uint32_t)hs != H || (uint32_t)hm != S) return;

        // Prefetch first token (if any)
        uint32_t n = nStart;
        PrefetchToken(n);

        for (; n < nEnd; ++n) {
            const uint32_t next = n + 1u;
            if (next < nEnd) {
                // Overlap: issue next prefetch while computing current token.
                PrefetchToken(next);
            }
            ProcessToken(n);
        }
    }

private:
    AscendC::GlobalTensor<F16> xGm;
    AscendC::GlobalTensor<F16> rGm;
    AscendC::GlobalTensor<F32> postGm;
    AscendC::GlobalTensor<F32> combGm;

    AscendC::GlobalTensor<I32> hsGm;
    AscendC::GlobalTensor<I32> hmGm;

    AscendC::GlobalTensor<F16> outGm;

    AscendC::TPipe pipe;

    // Double-buffered input queues (VECIN position uses UB, allows pipelining with compute).
    AscendC::TQue<AscendC::TPosition::VECIN, 2> xUbQ, r0UbQ, r1UbQ, r2UbQ, r3UbQ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xFbuf, r0Fbuf, r1Fbuf, r2Fbuf, r3Fbuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yF0buf, yF1buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yH0buf, yH1buf;

    uint32_t N = 0, S = 0, H = 0;
    uint32_t nStart = 0, nEnd = 0;
    bool supported = false;
};

extern "C" __global__ __aicore__ void mhc_post_block_custom(
    GM_ADDR x_fp16, GM_ADDR residual_fp16, GM_ADDR post_layer_mix, GM_ADDR comb_res_mix,
    GM_ADDR hidden_size, GM_ADDR hc_mult,
    GM_ADDR out_fp16,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMhcPostBlockCustom op;
    op.Init(x_fp16, residual_fp16, post_layer_mix, comb_res_mix,
            hidden_size, hc_mult,
            out_fp16,
            td.N, td.S, td.H);
    op.AllocUb();
    op.Process();
}
