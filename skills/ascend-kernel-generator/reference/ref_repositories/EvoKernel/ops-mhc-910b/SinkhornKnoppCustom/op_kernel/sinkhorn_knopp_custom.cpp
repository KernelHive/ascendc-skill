
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t AlignUp32Bytes(uint32_t bytes)
{
    return (bytes + 31u) & ~31u;
}

class KernelSinkhornKnopp {
public:
    __aicore__ inline KernelSinkhornKnopp() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR out,
                               uint32_t totalLength,
                               uint32_t B, uint32_t N,
                               uint32_t tmax, float eps, float clampMin)
    {
        this->B = B;
        this->N = N;
        this->tmax = tmax;
        this->eps = eps;
        this->clampMin = clampMin;
        this->matSize = N * N;

        logitsGm.SetGlobalBuffer((__gm__ float *)logits, totalLength);
        outGm.SetGlobalBuffer((__gm__ float *)out, totalLength);

        // Keep queues (safe overlap + avoids single-buffer pitfalls), but remove redundant UB->UB copy:
        // compute directly into VECOUT buffer (mat).
        pipe.InitBuffer(outQueue, BUFFER_NUM, AlignUp32Bytes(this->matSize * sizeof(float)));

        // Scratch:
        // N==8: 128 floats (512B) -> [0:64)=rowScale broadcast, [64:128)=colScale broadcast
        // generic: transpose (N*N) + rowSum (N)
        if (N == 8) {
            pipe.InitBuffer(bufScale, AlignUp32Bytes(128u * sizeof(float)));
        } else {
            pipe.InitBuffer(bufTrans, AlignUp32Bytes(this->matSize * sizeof(float)));
            pipe.InitBuffer(bufRowSum, AlignUp32Bytes(this->N * sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = (AscendC::GetBlockNum() == 0) ? 1 : AscendC::GetBlockNum();

        for (uint32_t b = core; b < B; b += cores) {
            ComputeOne(b);
            CopyOut(b);
        }
    }

private:
    __aicore__ inline void ComputeOne(uint32_t b)
    {
        AscendC::LocalTensor<float> mat = outQueue.AllocTensor<float>();
        const uint64_t base = (uint64_t)b * (uint64_t)matSize;

        // GM -> UB once
        AscendC::DataCopy(mat, logitsGm[base], matSize);

        // Stabilize then exp once
        float maxv = ReduceMaxMatrixScalar(mat);
        AscendC::Adds(mat, mat, -maxv, matSize);
        AscendC::Exp(mat, mat, matSize);

        if (N == 8) {
            for (uint32_t t = 0; t < tmax; ++t) {
                SinkhornIter8_VectorHeavy(mat);
            }
            if (clampMin > 0.0f) {
                AscendC::Maxs(mat, mat, clampMin, 64);
                // As in reference: renormalize after clamp (row then col)
                SinkhornIter8_VectorHeavy(mat);
            }
        } else {
            for (uint32_t t = 0; t < tmax; ++t) {
                RowNormGeneric(mat);
                ColNormViaTransposeGeneric(mat);
            }
            if (clampMin > 0.0f) {
                AscendC::Maxs(mat, mat, clampMin, matSize);
                RowNormGeneric(mat);
                ColNormViaTransposeGeneric(mat);
            }
        }

        outQueue.EnQue(mat);
    }

    __aicore__ inline void CopyOut(uint32_t b)
    {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        const uint64_t base = (uint64_t)b * (uint64_t)matSize;
        AscendC::DataCopy(outGm[base], yLocal, matSize);
        outQueue.FreeTensor(yLocal);
    }

    __aicore__ inline float ReduceMaxMatrixScalar(const AscendC::LocalTensor<float> &mat)
    {
        float maxv = -3.402823466e+38f;
        for (uint32_t i = 0; i < matSize; ++i) {
            float v = mat.GetValue(i);
            if (v > maxv) maxv = v;
        }
        return maxv;
    }

    // One iteration for N==8:
    // 1) compute 8 row sums (scalar), build 64-element rowScale broadcast in UB, vector Mul(64)
    // 2) compute 8 col sums (scalar), build 64-element colScale broadcast in UB, vector Mul(64)
    __aicore__ inline void SinkhornIter8_VectorHeavy(AscendC::LocalTensor<float> &mat)
    {
        AscendC::LocalTensor<float> scale = bufScale.Get<float>();
        AscendC::LocalTensor<float> rowScale = scale;      // [0..63]
        AscendC::LocalTensor<float> colScale = scale[64];  // [64..127] (aligned: 64 floats = 256B)

        // ---- Row sums + rowScale broadcast ----
        float rs0 = eps, rs1 = eps, rs2 = eps, rs3 = eps, rs4 = eps, rs5 = eps, rs6 = eps, rs7 = eps;

        // Unrolled row reductions: still scalar reads but minimal and fixed.
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs0 += mat.GetValue(0 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs1 += mat.GetValue(1 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs2 += mat.GetValue(2 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs3 += mat.GetValue(3 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs4 += mat.GetValue(4 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs5 += mat.GetValue(5 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs6 += mat.GetValue(6 * 8 + c); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rs7 += mat.GetValue(7 * 8 + c); }

        float irs0 = 1.0f / rs0, irs1 = 1.0f / rs1, irs2 = 1.0f / rs2, irs3 = 1.0f / rs3;
        float irs4 = 1.0f / rs4, irs5 = 1.0f / rs5, irs6 = 1.0f / rs6, irs7 = 1.0f / rs7;

        // Fill rowScale: each row repeats its inverse across 8 cols.
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(0 * 8 + c, irs0); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(1 * 8 + c, irs1); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(2 * 8 + c, irs2); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(3 * 8 + c, irs3); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(4 * 8 + c, irs4); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(5 * 8 + c, irs5); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(6 * 8 + c, irs6); }
        #pragma unroll
        for (uint32_t c = 0; c < 8; ++c) { rowScale.SetValue(7 * 8 + c, irs7); }

        AscendC::Mul(mat, mat, rowScale, 64);

        // ---- Column sums + colScale broadcast ----
        float cs0 = eps, cs1 = eps, cs2 = eps, cs3 = eps, cs4 = eps, cs5 = eps, cs6 = eps, cs7 = eps;

        #pragma unroll
        for (uint32_t r = 0; r < 8; ++r) {
            const uint32_t base = r * 8;
            cs0 += mat.GetValue(base + 0);
            cs1 += mat.GetValue(base + 1);
            cs2 += mat.GetValue(base + 2);
            cs3 += mat.GetValue(base + 3);
            cs4 += mat.GetValue(base + 4);
            cs5 += mat.GetValue(base + 5);
            cs6 += mat.GetValue(base + 6);
            cs7 += mat.GetValue(base + 7);
        }

        float ics0 = 1.0f / cs0, ics1 = 1.0f / cs1, ics2 = 1.0f / cs2, ics3 = 1.0f / cs3;
        float ics4 = 1.0f / cs4, ics5 = 1.0f / cs5, ics6 = 1.0f / cs6, ics7 = 1.0f / cs7;

        // Fill colScale: each row repeats [ics0..ics7]
        #pragma unroll
        for (uint32_t r = 0; r < 8; ++r) {
            const uint32_t base = r * 8;
            colScale.SetValue(base + 0, ics0);
            colScale.SetValue(base + 1, ics1);
            colScale.SetValue(base + 2, ics2);
            colScale.SetValue(base + 3, ics3);
            colScale.SetValue(base + 4, ics4);
            colScale.SetValue(base + 5, ics5);
            colScale.SetValue(base + 6, ics6);
            colScale.SetValue(base + 7, ics7);
        }

        AscendC::Mul(mat, mat, colScale, 64);
    }

    // -------- Generic path (unchanged) --------
    __aicore__ inline void RowNormGeneric(AscendC::LocalTensor<float> &mat)
    {
        AscendC::LocalTensor<float> sum = bufRowSum.Get<float>();
        for (uint32_t r = 0; r < N; ++r) {
            const uint32_t base = r * N;
            float s = 0.f;
            for (uint32_t c = 0; c < N; ++c) s += mat.GetValue(base + c);
            sum.SetValue(r, s + eps);
        }
        for (uint32_t r = 0; r < N; ++r) {
            const uint32_t base = r * N;
            float inv = 1.0f / sum.GetValue(r);
            AscendC::Muls(mat[base], mat[base], inv, N);
        }
    }

    __aicore__ inline void TransposeGeneric(const AscendC::LocalTensor<float> &src, AscendC::LocalTensor<float> &dst)
    {
        for (uint32_t r = 0; r < N; ++r) {
            for (uint32_t c = 0; c < N; ++c) {
                dst.SetValue(c * N + r, src.GetValue(r * N + c));
            }
        }
    }

    __aicore__ inline void ColNormViaTransposeGeneric(AscendC::LocalTensor<float> &mat)
    {
        AscendC::LocalTensor<float> trans = bufTrans.Get<float>();
        TransposeGeneric(mat, trans);
        RowNormGeneric(trans);
        TransposeGeneric(trans, mat);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufScale;   // N==8: 128 floats
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTrans;   // generic: N*N
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufRowSum;  // generic: N

    AscendC::GlobalTensor<float> logitsGm, outGm;

    uint32_t B{}, N{}, tmax{}, matSize{};
    float eps{}, clampMin{};
};

extern "C" __global__ __aicore__ void sinkhorn_knopp_custom(GM_ADDR logits, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinkhornKnopp op;
    op.Init(logits, out,
            tiling_data.totalLength,
            tiling_data.B, tiling_data.N,
            tiling_data.tmax, tiling_data.eps, tiling_data.clampMin);
    op.Process();
}
