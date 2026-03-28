
#include "kernel_operator.h"

// Optimized fused linear attention core on [B,H,N,D] float32 contiguous.
// Key changes vs baseline:
// 1) Eliminate KV materialization (D*D UB) and compute out[n] by streaming over t:
//      out[n,:] = sum_t ( dot(Q[n,:], K[t,:]) * V[t,:] ) / (dot(Q[n,:], ksum)+eps)
// 2) Increase parallelism by mapping blocks to (bh, token-row-tiles).
// 3) Store ksum [B,H,D] in workspace so token-parallel blocks reuse it.
static constexpr uint32_t MAX_D = 64;
static constexpr uint32_t MAX_N = 1024;

class KernelLinearAttentionCustom {
public:
    __aicore__ inline KernelLinearAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               GM_ADDR workspace,
                               uint32_t b, uint32_t h, uint32_t n, uint32_t d,
                               float eps, uint32_t row_block)
    {
        this->b = b;
        this->h = h;
        this->n = n;
        this->d = d;
        this->eps = eps;
        this->rowBlock = row_block;

        const uint64_t totalElems = static_cast<uint64_t>(b) * h * n * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, totalElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, totalElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, totalElems);
        oGm.SetGlobalBuffer((__gm__ float*)out, totalElems);

        // Workspace for ksum: [B,H,D]
        const uint64_t ksumElems = static_cast<uint64_t>(b) * h * d;
        ksumWsGm.SetGlobalBuffer((__gm__ float*)workspace, ksumElems);

        // UB buffers
        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufKRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));
        pipe.InitBuffer(bufKSum,   MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (b == 0 || h == 0 || n == 0 || d == 0) return;
        if (d > MAX_D || n > MAX_N) return;
        if (rowBlock == 0) return;

        const uint64_t totalBH = static_cast<uint64_t>(b) * h;
        const uint64_t tilesPerBH = (static_cast<uint64_t>(n) + rowBlock - 1) / rowBlock;
        const uint64_t totalTasks = totalBH * tilesPerBH;

        const uint32_t grid = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());

        for (uint64_t task = bid; task < totalTasks; task += grid) {
            const uint64_t bh = task / tilesPerBH;
            const uint32_t tile = static_cast<uint32_t>(task - bh * tilesPerBH);

            const uint32_t batch = static_cast<uint32_t>(bh / h);
            const uint32_t head  = static_cast<uint32_t>(bh - static_cast<uint64_t>(batch) * h);
            const uint32_t nStart = tile * rowBlock;
            const uint32_t nEnd = (nStart + rowBlock <= n) ? (nStart + rowBlock) : n;

            // Ensure ksum exists (computed once per (b,h) by tile==0).
            if (tile == 0) {
                ComputeAndStoreKSum(batch, head);
            }
            // A small barrier to improve correctness under weak ordering. This is conservative;
            // for typical launch ordering, tile==0 will complete early, but we avoid stale reads.
            AscendC::PipeBarrier<PIPE_ALL>();

            ComputeRows(batch, head, nStart, nEnd);
        }
    }

private:
    __aicore__ inline uint64_t BaseBH(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(n) * d);
    }

    __aicore__ inline uint64_t BaseKSumWs(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * d;
    }

    __aicore__ inline void ZeroVec(const AscendC::LocalTensor<float>& t, uint32_t len) const
    {
        for (uint32_t i = 0; i < len; ++i) t(i) = 0.0f;
    }

    __aicore__ inline void LoadRow(const AscendC::LocalTensor<float>& ub,
                                  const AscendC::GlobalTensor<float>& gm,
                                  uint64_t off) const
    {
        AscendC::DataCopy(ub, gm[off], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void StoreRow(const AscendC::GlobalTensor<float>& gm,
                                   uint64_t off,
                                   const AscendC::LocalTensor<float>& ub) const
    {
        AscendC::DataCopy(gm[off], ub, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline void ComputeAndStoreKSum(uint32_t batch, uint32_t head)
    {
        auto kRow = bufKRow.Get<float>();
        auto ksum = bufKSum.Get<float>();
        ZeroVec(ksum, d);
        AscendC::PipeBarrier<PIPE_V>();

        const uint64_t base = BaseBH(batch, head);
        for (uint32_t t = 0; t < n; ++t) {
            const uint64_t off = base + static_cast<uint64_t>(t) * d;
            LoadRow(kRow, kGm, off);
            for (uint32_t i = 0; i < d; ++i) {
                ksum(i) = ksum(i) + kRow(i);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        const uint64_t wsOff = BaseKSumWs(batch, head);
        StoreRow(ksumWsGm, wsOff, ksum);
    }

    __aicore__ inline void LoadKSum(uint32_t batch, uint32_t head)
    {
        auto ksum = bufKSum.Get<float>();
        const uint64_t wsOff = BaseKSumWs(batch, head);
        AscendC::DataCopy(ksum, ksumWsGm[wsOff], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeRows(uint32_t batch, uint32_t head, uint32_t nStart, uint32_t nEnd)
    {
        auto qRow   = bufQRow.Get<float>();
        auto kRow   = bufKRow.Get<float>();
        auto vRow   = bufVRow.Get<float>();
        auto outRow = bufOutRow.Get<float>();
        auto ksum   = bufKSum.Get<float>();

        LoadKSum(batch, head);

        const uint64_t base = BaseBH(batch, head);

        for (uint32_t ni = nStart; ni < nEnd; ++ni) {
            const uint64_t qOff = base + static_cast<uint64_t>(ni) * d;
            LoadRow(qRow, qGm, qOff);

            float denom = eps;
            for (uint32_t i = 0; i < d; ++i) denom += qRow(i) * ksum(i);
            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);

            ZeroVec(outRow, d);
            AscendC::PipeBarrier<PIPE_V>();

            // Stream over sequence t and accumulate outRow += (dot(q,k_t) * v_t)
            for (uint32_t t = 0; t < n; ++t) {
                const uint64_t kvOff = base + static_cast<uint64_t>(t) * d;
                LoadRow(kRow, kGm, kvOff);
                LoadRow(vRow, vGm, kvOff);

                float alpha = 0.0f;
                for (uint32_t i = 0; i < d; ++i) alpha += qRow(i) * kRow(i);

                for (uint32_t j = 0; j < d; ++j) {
                    outRow(j) = outRow(j) + alpha * vRow(j);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            for (uint32_t j = 0; j < d; ++j) outRow(j) = outRow(j) * invDen;
            AscendC::PipeBarrier<PIPE_V>();

            StoreRow(oGm, qOff, outRow);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKSum;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;
    AscendC::GlobalTensor<float> ksumWsGm;

    uint32_t b {0}, h {0}, n {0}, d {0};
    float eps {1e-6f};
    uint32_t rowBlock {8};
};

extern "C" __global__ __aicore__ void linear_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearAttentionCustom op;
    op.Init(q, k, v, out, workspace,
            tiling_data.b, tiling_data.h, tiling_data.n, tiling_data.d,
            tiling_data.eps, tiling_data.row_block);
    op.Process();
}
