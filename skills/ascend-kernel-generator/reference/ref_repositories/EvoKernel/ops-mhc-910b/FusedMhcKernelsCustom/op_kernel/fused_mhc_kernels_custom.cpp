
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;

class KernelFusedMhcKernelsCustom {
public:
    __aicore__ inline KernelFusedMhcKernelsCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR phi, GM_ADDR bias, GM_ADDR scale,
        GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
        GM_ADDR iters, GM_ADDR eps_rms, GM_ADDR invD,
        GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR h_res,
        uint32_t B, uint32_t L, uint32_t D, uint32_t BL,
        uint32_t outCols, uint32_t n, uint32_t nn,
        uint32_t Dpad, uint32_t outPad, uint32_t nPad, uint32_t nnPad)
    {
        (void)B; (void)L;
        this->D = D; this->BL = BL;
        this->outCols = outCols; this->n = n; this->nn = nn;
        this->Dpad = Dpad; this->outPad = outPad; this->nPad = nPad; this->nnPad = nnPad;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t rowsPerBlock = (BL + blockNum - 1u) / blockNum;
        rowStart = blockIdx * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > BL) rowEnd = BL;
        rowCount = (rowEnd > rowStart) ? (rowEnd - rowStart) : 0u;

        xGm.SetGlobalBuffer((__gm__ F32*)x, (uint64_t)BL * (uint64_t)D);
        phiGm.SetGlobalBuffer((__gm__ F32*)phi, (uint64_t)D * (uint64_t)outCols);
        biasGm.SetGlobalBuffer((__gm__ F32*)bias, (uint64_t)outCols);
        scaleGm.SetGlobalBuffer((__gm__ F32*)scale, (uint64_t)D);

        aPreGm.SetGlobalBuffer((__gm__ F32*)alpha_pre, (uint64_t)1);
        aPostGm.SetGlobalBuffer((__gm__ F32*)alpha_post, (uint64_t)1);
        aResGm.SetGlobalBuffer((__gm__ F32*)alpha_res, (uint64_t)1);
        itersGm.SetGlobalBuffer((__gm__ I32*)iters, (uint64_t)1);
        epsRmsGm.SetGlobalBuffer((__gm__ F32*)eps_rms, (uint64_t)1);
        invDGm.SetGlobalBuffer((__gm__ F32*)invD, (uint64_t)1);

        hPreGm.SetGlobalBuffer((__gm__ F32*)h_pre, (uint64_t)BL * (uint64_t)n);
        hPostGm.SetGlobalBuffer((__gm__ F32*)h_post, (uint64_t)BL * (uint64_t)n);
        hResGm.SetGlobalBuffer((__gm__ F32*)h_res, (uint64_t)BL * (uint64_t)nn);

        // UB buffers
        pipe.InitBuffer(xUb,      (uint32_t)(Dpad * sizeof(F32)));
        pipe.InitBuffer(x2Ub,     (uint32_t)(Dpad * sizeof(F32)));
        pipe.InitBuffer(xNormUb,  (uint32_t)(Dpad * sizeof(F32)));

        pipe.InitBuffer(scaleUb,  (uint32_t)(Dpad * sizeof(F32)));
        pipe.InitBuffer(biasUb,   (uint32_t)(outPad * sizeof(F32)));

        pipe.InitBuffer(logitsUb, (uint32_t)(outPad * sizeof(F32)));
        pipe.InitBuffer(tmpUb,    (uint32_t)(outPad * sizeof(F32)));

        pipe.InitBuffer(preUb,    (uint32_t)(nPad * sizeof(F32)));
        pipe.InitBuffer(postUb,   (uint32_t)(nPad * sizeof(F32)));

        pipe.InitBuffer(resUb,    (uint32_t)(nnPad * sizeof(F32)));

        StageScaleBiasToUb();
    }

    __aicore__ inline void Process()
    {
        if (rowCount == 0u) return;

        const F32 aPre  = aPreGm.GetValue(0);
        const F32 aPost = aPostGm.GetValue(0);
        const F32 aRes  = aResGm.GetValue(0);
        I32 it = itersGm.GetValue(0);
        if (it < 0) it = 0;
        if (it > 64) it = 64; // guard for pathological inputs
        F32 eps = epsRmsGm.GetValue(0);
        if (eps < (F32)0) eps = (F32)0;
        const F32 invD = invDGm.GetValue(0);

        for (uint32_t r = 0; r < rowCount; ++r) {
            const uint32_t row = rowStart + r;
            RunOneRow(row, aPre, aPost, aRes, it, eps, invD);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

    __aicore__ inline void ZeroF32(AscendC::LocalTensor<F32>& t, uint32_t len)
    {
        for (uint32_t i = 0; i < len; ++i) t.SetValue(i, (F32)0);
    }

    // Approximations mirror the successful template (stable, no special intrinsics required).
    __aicore__ inline F32 ExpApprox(F32 x)
    {
        if (x > (F32)10.0f) x = (F32)10.0f;
        if (x < (F32)-10.0f) x = (F32)-10.0f;
        const F32 x2 = x * x;
        const F32 x3 = x2 * x;
        const F32 x4 = x2 * x2;
        const F32 x5 = x4 * x;
        return (F32)1.0f + x + x2 * (F32)0.5f + x3 * (F32)(1.0f/6.0f) + x4 * (F32)(1.0f/24.0f) + x5 * (F32)(1.0f/120.0f);
    }

    __aicore__ inline F32 LogApprox(F32 x)
    {
        if (x < (F32)1e-12f) x = (F32)1e-12f;
        F32 y = (x - (F32)1.0f) / (x + (F32)1.0f);
        const F32 y2 = y * y;
        const F32 y3 = y2 * y;
        const F32 y5 = y3 * y2;
        const F32 y7 = y5 * y2;
        return (F32)2.0f * (y + y3 * (F32)(1.0f/3.0f) + y5 * (F32)(1.0f/5.0f) + y7 * (F32)(1.0f/7.0f));
    }

    __aicore__ inline F32 RsqrtApprox(F32 x)
    {
        if (x < (F32)1e-12f) x = (F32)1e-12f;
        F32 y = (F32)1.0f;
        for (int it = 0; it < 3; ++it) {
            y = y * ((F32)1.5f - (F32)0.5f * x * y * y);
        }
        return y;
    }

    __aicore__ inline F32 SigmoidApprox(F32 x)
    {
        const F32 e = ExpApprox(-x);
        return (F32)1.0f / ((F32)1.0f + e);
    }

    __aicore__ inline void StageScaleBiasToUb()
    {
        // scale
        AscendC::LocalTensor<F32> sc = scaleUb.Get<F32>();
        ZeroF32(sc, Dpad);
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(D, 8u);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(sc, scaleGm[0], p);

        // bias
        AscendC::LocalTensor<F32> b = biasUb.Get<F32>();
        ZeroF32(b, outPad);
        AscendC::DataCopyParams pb;
        pb.blockCount = 1;
        pb.blockLen = CeilDiv(outCols, 8u);
        pb.srcStride = 0;
        pb.dstStride = 0;
        AscendC::DataCopy(b, biasGm[0], pb);
    }

    __aicore__ inline void LoadX(uint32_t row, AscendC::LocalTensor<F32>& xLocal)
    {
        ZeroF32(xLocal, Dpad);
        const uint64_t base = (uint64_t)row * (uint64_t)D;
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(D, 8u);
        p.srcStride = 0; p.dstStride = 0;
        AscendC::DataCopy(xLocal, xGm[base], p);
    }

    __aicore__ inline void RmsNormScale(const AscendC::LocalTensor<F32>& x,
                                       AscendC::LocalTensor<F32>& x2,
                                       AscendC::LocalTensor<F32>& xNorm,
                                       const AscendC::LocalTensor<F32>& scale,
                                       F32 invD, F32 eps)
    {
        // x2 = x*x (vectorized), padded tail is already 0
        AscendC::Mul(x2, x, x, Dpad);

        // sumsq over true D (scalar loop; D is moderate, avoids ReduceSum API pitfalls)
        F32 sumsq = (F32)0.0f;
        for (uint32_t i = 0; i < D; ++i) {
            const F32 v = x2.GetValue(i);
            sumsq += v;
        }
        const F32 invRms = RsqrtApprox(sumsq * invD + eps);

        // xNorm = x * invRms
        AscendC::Muls(xNorm, x, invRms, Dpad);
        // xNorm *= scale
        AscendC::Mul(xNorm, xNorm, scale, Dpad);
    }

    __aicore__ inline void InitLogitsFromBias(AscendC::LocalTensor<F32>& logits)
    {
        AscendC::LocalTensor<F32> b = biasUb.Get<F32>();
        // logits = bias (copy in UB to keep bias persistent)
        for (uint32_t i = 0; i < outPad; ++i) logits.SetValue(i, b.GetValue(i));
    }

    __aicore__ inline void GemvAccumulate(const AscendC::LocalTensor<F32>& xNorm,
                                         AscendC::LocalTensor<F32>& logits)
    {
        // Stream rows of phi from GM; outCols is small (typically n=4 => 24)
        AscendC::LocalTensor<F32> tmp = tmpUb.Get<F32>();
        for (uint32_t k = 0; k < D; ++k) {
            const F32 xv = xNorm.GetValue(k);
            const uint64_t base = (uint64_t)k * (uint64_t)outCols;

            ZeroF32(tmp, outPad);
            AscendC::DataCopyParams p;
            p.blockCount = 1;
            p.blockLen = CeilDiv(outCols, 8u);
            p.srcStride = 0; p.dstStride = 0;
            AscendC::DataCopy(tmp, phiGm[base], p);

            AscendC::Muls(tmp, tmp, xv, outPad);
            AscendC::Add(logits, logits, tmp, outPad);
        }
    }

    __aicore__ inline void WritePrePost(uint32_t row,
                                        const AscendC::LocalTensor<F32>& logits,
                                        F32 aPre, F32 aPost)
    {
        AscendC::LocalTensor<F32> pre = preUb.Get<F32>();
        AscendC::LocalTensor<F32> post = postUb.Get<F32>();
        ZeroF32(pre, nPad);
        ZeroF32(post, nPad);

        for (uint32_t i = 0; i < n; ++i) pre.SetValue(i, logits.GetValue(i));
        for (uint32_t i = 0; i < n; ++i) post.SetValue(i, logits.GetValue(n + i));

        // alpha scaling then sigmoid
        for (uint32_t i = 0; i < n; ++i) {
            const F32 v = aPre * pre.GetValue(i);
            pre.SetValue(i, SigmoidApprox(v));
        }
        for (uint32_t i = 0; i < n; ++i) {
            const F32 v = aPost * post.GetValue(i);
            post.SetValue(i, (F32)2.0f * SigmoidApprox(v));
        }

        const uint64_t oBase = (uint64_t)row * (uint64_t)n;

        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(n, 8u);
        p.srcStride = 0; p.dstStride = 0;
        AscendC::DataCopy(hPreGm[oBase], pre, p);
        AscendC::DataCopy(hPostGm[oBase], post, p);
    }

    __aicore__ inline void SinkhornLogSpace_N4(uint32_t row,
                                               const AscendC::LocalTensor<F32>& logits,
                                               F32 aRes, I32 iters)
    {
        // N=4 => nn=16, logits tail starts at 8
        const uint32_t base = 8u;
        F32 lg[16];
        F32 maxv = (F32)-1e30f;
        for (uint32_t i = 0; i < 16u; ++i) {
            const F32 v = aRes * logits.GetValue(base + i);
            lg[i] = v;
            if (v > maxv) maxv = v;
        }
        for (uint32_t i = 0; i < 16u; ++i) lg[i] -= maxv;

        for (I32 it = 0; it < iters; ++it) {
            // rows
            for (int r = 0; r < 4; ++r) {
                const int b = r * 4;
                F32 rmax = lg[b];
                if (lg[b+1] > rmax) rmax = lg[b+1];
                if (lg[b+2] > rmax) rmax = lg[b+2];
                if (lg[b+3] > rmax) rmax = lg[b+3];
                const F32 sumExp = ExpApprox(lg[b] - rmax) + ExpApprox(lg[b+1] - rmax) + ExpApprox(lg[b+2] - rmax) + ExpApprox(lg[b+3] - rmax);
                const F32 lse = rmax + LogApprox(sumExp);
                lg[b] -= lse; lg[b+1] -= lse; lg[b+2] -= lse; lg[b+3] -= lse;
            }
            // cols
            for (int c = 0; c < 4; ++c) {
                F32 cmax = lg[c];
                if (lg[4+c] > cmax) cmax = lg[4+c];
                if (lg[8+c] > cmax) cmax = lg[8+c];
                if (lg[12+c] > cmax) cmax = lg[12+c];
                const F32 sumExp = ExpApprox(lg[c] - cmax) + ExpApprox(lg[4+c] - cmax) + ExpApprox(lg[8+c] - cmax) + ExpApprox(lg[12+c] - cmax);
                const F32 lse = cmax + LogApprox(sumExp);
                lg[c] -= lse; lg[4+c] -= lse; lg[8+c] -= lse; lg[12+c] -= lse;
            }
        }

        const uint64_t oBase = (uint64_t)row * 16ull;
        for (uint32_t i = 0; i < 16u; ++i) {
            hResGm.SetValue(oBase + (uint64_t)i, ExpApprox(lg[i]));
        }
    }

    __aicore__ inline void SinkhornLogSpace_Generic(uint32_t row,
                                                    const AscendC::LocalTensor<F32>& logits,
                                                    F32 aRes, I32 iters)
    {
        const uint32_t base = 2u * n;
        AscendC::LocalTensor<F32> m = resUb.Get<F32>();
        ZeroF32(m, nnPad);

        // Build log matrix in UB (no exp yet), subtract global max for stability.
        F32 maxv = (F32)-1e30f;
        for (uint32_t i = 0; i < nn; ++i) {
            const F32 v = aRes * logits.GetValue(base + i);
            m.SetValue(i, v);
            if (v > maxv) maxv = v;
        }
        for (uint32_t i = 0; i < nn; ++i) m.SetValue(i, m.GetValue(i) - maxv);

        // Log-space Sinkhorn: normalize rows/cols by subtracting logsumexp.
        for (I32 it = 0; it < iters; ++it) {
            // rows
            for (uint32_t r = 0; r < n; ++r) {
                const uint32_t rb = r * n;
                F32 rmax = m.GetValue(rb);
                for (uint32_t c = 1; c < n; ++c) {
                    const F32 v = m.GetValue(rb + c);
                    if (v > rmax) rmax = v;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t c = 0; c < n; ++c) sumExp += ExpApprox(m.GetValue(rb + c) - rmax);
                const F32 lse = rmax + LogApprox(sumExp);
                for (uint32_t c = 0; c < n; ++c) m.SetValue(rb + c, m.GetValue(rb + c) - lse);
            }
            // cols
            for (uint32_t c = 0; c < n; ++c) {
                F32 cmax = m.GetValue(c);
                for (uint32_t r = 1; r < n; ++r) {
                    const F32 v = m.GetValue(r * n + c);
                    if (v > cmax) cmax = v;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t r = 0; r < n; ++r) sumExp += ExpApprox(m.GetValue(r * n + c) - cmax);
                const F32 lse = cmax + LogApprox(sumExp);
                for (uint32_t r = 0; r < n; ++r) {
                    const uint32_t idx = r * n + c;
                    m.SetValue(idx, m.GetValue(idx) - lse);
                }
            }
        }

        const uint64_t oBase = (uint64_t)row * (uint64_t)nn;
        for (uint32_t i = 0; i < nn; ++i) {
            hResGm.SetValue(oBase + (uint64_t)i, ExpApprox(m.GetValue(i)));
        }
    }

    __aicore__ inline void Sinkhorn(uint32_t row,
                                    const AscendC::LocalTensor<F32>& logits,
                                    F32 aRes, I32 iters)
    {
        if (n == 4u && nn == 16u && outCols >= 24u) {
            SinkhornLogSpace_N4(row, logits, aRes, iters);
        } else {
            SinkhornLogSpace_Generic(row, logits, aRes, iters);
        }
    }

    __aicore__ inline void RunOneRow(uint32_t row,
                                    F32 aPre, F32 aPost, F32 aRes,
                                    I32 iters, F32 eps, F32 invD)
    {
        AscendC::LocalTensor<F32> x = xUb.Get<F32>();
        AscendC::LocalTensor<F32> x2 = x2Ub.Get<F32>();
        AscendC::LocalTensor<F32> xNorm = xNormUb.Get<F32>();
        AscendC::LocalTensor<F32> sc = scaleUb.Get<F32>();

        AscendC::LocalTensor<F32> logits = logitsUb.Get<F32>();

        LoadX(row, x);
        RmsNormScale(x, x2, xNorm, sc, invD, eps);

        InitLogitsFromBias(logits);
        GemvAccumulate(xNorm, logits);

        WritePrePost(row, logits, aPre, aPost);
        Sinkhorn(row, logits, aRes, iters);
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<F32> xGm, phiGm, biasGm, scaleGm;
    AscendC::GlobalTensor<F32> aPreGm, aPostGm, aResGm;
    AscendC::GlobalTensor<I32> itersGm;
    AscendC::GlobalTensor<F32> epsRmsGm, invDGm;

    AscendC::GlobalTensor<F32> hPreGm, hPostGm, hResGm;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> xUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2Ub;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xNormUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> scaleUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> biasUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> logitsUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> preUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> postUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> resUb;

    uint32_t D{}, BL{}, outCols{}, n{}, nn{}, Dpad{}, outPad{}, nPad{}, nnPad{};
    uint32_t rowStart{}, rowCount{};
};

extern "C" __global__ __aicore__ void fused_mhc_kernels_custom(
    GM_ADDR x, GM_ADDR phi, GM_ADDR bias, GM_ADDR scale,
    GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
    GM_ADDR iters, GM_ADDR eps_rms, GM_ADDR invD,
    GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR h_res,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelFusedMhcKernelsCustom op;
    op.Init(x, phi, bias, scale,
            alpha_pre, alpha_post, alpha_res,
            iters, eps_rms, invD,
            h_pre, h_post, h_res,
            td.B, td.L, td.D, td.BL,
            td.outCols, td.n, td.nn,
            td.Dpad, td.outPad, td.nPad, td.nnPad);
    op.Process();
}
