
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;

class KernelStaticMhcHyperConnectionsCustom {
public:
    __aicore__ inline KernelStaticMhcHyperConnectionsCustom() {}

    __aicore__ inline void Init(
        GM_ADDR residuals,      // (B,S,D)
        GM_ADDR h_res_logits,   // (S,S)
        GM_ADDR h_pre_logits,   // (S)
        GM_ADDR h_post_logits,  // (S)
        GM_ADDR branch_weight,  // (D,D)
        GM_ADDR sinkhorn_iters, // (1) int32
        GM_ADDR tau,            // (1) float
        GM_ADDR log_s,          // (1) float (log(S))
        GM_ADDR out,            // (B,S,D)
        uint32_t B, uint32_t S, uint32_t D, uint32_t Dpad, uint32_t tileD)
    {
        this->B = B; this->S = S; this->D = D; this->Dpad = Dpad; this->tileD = tileD;

        residualsGm.SetGlobalBuffer((__gm__ F32*)residuals, (uint64_t)B * (uint64_t)S * (uint64_t)D);
        hResLogitsGm.SetGlobalBuffer((__gm__ F32*)h_res_logits, (uint64_t)S * (uint64_t)S);
        hPreLogitsGm.SetGlobalBuffer((__gm__ F32*)h_pre_logits, (uint64_t)S);
        hPostLogitsGm.SetGlobalBuffer((__gm__ F32*)h_post_logits, (uint64_t)S);
        wGm.SetGlobalBuffer((__gm__ F32*)branch_weight, (uint64_t)D * (uint64_t)D);

        itersGm.SetGlobalBuffer((__gm__ I32*)sinkhorn_iters, (uint64_t)1);
        tauGm.SetGlobalBuffer((__gm__ F32*)tau, (uint64_t)1);
        logSGm.SetGlobalBuffer((__gm__ F32*)log_s, (uint64_t)1);

        outGm.SetGlobalBuffer((__gm__ F32*)out, (uint64_t)B * (uint64_t)S * (uint64_t)D);

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t bPerBlock = (B + blockNum - 1u) / blockNum;
        bStart = blockIdx * bPerBlock;
        uint32_t bEnd = bStart + bPerBlock;
        if (bEnd > B) bEnd = B;
        bCount = (bEnd > bStart) ? (bEnd - bStart) : 0;

        // UB buffers: keep sizes small and aligned.
        pipe.InitBuffer(rTileUb,   (uint32_t)(tileD * sizeof(F32)));  // residual tile
        pipe.InitBuffer(accTileUb, (uint32_t)(tileD * sizeof(F32)));  // accumulator
        pipe.InitBuffer(brUb,      (uint32_t)(tileD * sizeof(F32)));  // branch_input tile
        pipe.InitBuffer(boUb,      (uint32_t)(tileD * sizeof(F32)));  // branch_output tile
        pipe.InitBuffer(wTileUb,   (uint32_t)(tileD * sizeof(F32)));  // W row tile

        pipe.InitBuffer(hVec0Ub,   (uint32_t)(16u * sizeof(F32)));    // hpre / u
        pipe.InitBuffer(hVec1Ub,   (uint32_t)(16u * sizeof(F32)));    // hpost / v
        pipe.InitBuffer(hMatUb,    (uint32_t)(256u * sizeof(F32)));   // SxS (padded to 16x16)
    }

    __aicore__ inline void Process()
    {
        if (bCount == 0) return;

        I32 iters = itersGm.GetValue(0);
        if (iters < 0) iters = 0;
        if (iters > 128) iters = 128;

        F32 tau = tauGm.GetValue(0);
        if (tau < (F32)1e-6f) tau = (F32)1e-6f;
        const F32 invTau = (F32)1.0f / tau;

        const F32 logS = logSGm.GetValue(0);
        const F32 logm = (F32)0.0f - logS;

        // Compute h_pre and h_post softmax (S<=16)
        F32 hpre[16];
        F32 hpost[16];
        SoftmaxS(hPreLogitsGm, hpre);
        SoftmaxS(hPostLogitsGm, hpost);

        // Compute h_res (Sinkhorn) into UB as float matrix (16x16 padded)
        AscendC::LocalTensor<F32> hres = hMatUb.Get<F32>();
        BuildHresSinkhornInUb(invTau, logm, iters, hres); // writes h_res * S

        // Main: for each b, compute:
        // residuals_mixed[b,t,:] = sum_s h_res[t,s] * residuals[b,s,:]
        // branch_input[b,:]      = sum_s h_pre[s]  * residuals[b,s,:]
        // branch_output[b,:]     = branch_input[b,:] @ W^T
        // out[b,t,:] = residuals_mixed[b,t,:] + h_post[t] * branch_output[b,:]
        for (uint32_t bi = 0; bi < bCount; ++bi) {
            const uint32_t b = bStart + bi;

            for (uint32_t d0 = 0; d0 < D; d0 += tileD) {
                const uint32_t dLen = (d0 + tileD <= D) ? tileD : (D - d0);
                const uint32_t dLenPad = CeilTo8(dLen);

                AscendC::LocalTensor<F32> branchIn = brUb.Get<F32>();
                ZeroTile(branchIn, tileD);

                // branchIn tile = sum_s hpre[s] * residuals[b,s, d0:d0+dLen]
                for (uint32_t s = 0; s < S; ++s) {
                    AscendC::LocalTensor<F32> rTile = rTileUb.Get<F32>();
                    LoadResidualTilePad(b, s, d0, dLen, dLenPad, rTile);
                    MulAddTile(branchIn, rTile, hpre[s], dLenPad);
                }

                // branchOut tile for output dims d0:d0+dLen:
                // For each outDim = d0+od: dot(branchIn(full D), W[outDim, :])
                // We do a blocked dot over input dims id0, using rTileUb for branchIn slices.
                AscendC::LocalTensor<F32> branchOut = boUb.Get<F32>();
                ZeroTile(branchOut, tileD);

                for (uint32_t od = 0; od < dLen; ++od) {
                    const uint32_t outDim = d0 + od;
                    const uint64_t wRowBase = (uint64_t)outDim * (uint64_t)D;

                    F32 acc = (F32)0.0f;
                    for (uint32_t id0 = 0; id0 < D; id0 += tileD) {
                        const uint32_t idLen = (id0 + tileD <= D) ? tileD : (D - id0);
                        const uint32_t idLenPad = CeilTo8(idLen);

                        AscendC::LocalTensor<F32> wTile = wTileUb.Get<F32>();
                        LoadWRowTilePad(wRowBase + (uint64_t)id0, idLen, idLenPad, wTile);

                        AscendC::LocalTensor<F32> xTile = rTileUb.Get<F32>();
                        LoadBranchInSlice(branchIn, id0, idLenPad, xTile);

                        // scalar reduce after elementwise mul (keeps codegen simple and robust)
                        for (uint32_t i = 0; i < idLen; ++i) {
                            acc += xTile.GetValue(i) * wTile.GetValue(i);
                        }
                    }
                    branchOut.SetValue(od, acc);
                }

                // For each target stream t: outTile = sum_s hres[t,s]*residual + hpost[t]*branchOut
                for (uint32_t t = 0; t < S; ++t) {
                    AscendC::LocalTensor<F32> outTile = accTileUb.Get<F32>();
                    ZeroTile(outTile, tileD);

                    const uint32_t rowBase = t * 16u;
                    for (uint32_t s = 0; s < S; ++s) {
                        const F32 hts = hres.GetValue(rowBase + s);
                        AscendC::LocalTensor<F32> rTile = rTileUb.Get<F32>();
                        LoadResidualTilePad(b, s, d0, dLen, dLenPad, rTile);
                        MulAddTile(outTile, rTile, hts, dLenPad);
                    }

                    // add hpost[t] * branchOut
                    const F32 scale = hpost[t];
                    for (uint32_t i = 0; i < dLen; ++i) {
                        outTile.SetValue(i, outTile.GetValue(i) + branchOut.GetValue(i) * scale);
                    }
                    // store only dLen (pad-safe store uses dLenPad but writes extra within row bounds? avoid)
                    StoreOutTile(b, t, d0, dLen, outTile);
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t CeilTo8(uint32_t x) { return ((x + 7u) / 8u) * 8u; }
    __aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

    // Polynomial approximations (scalar-safe, no AscendC transcendental intrinsics).
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

    __aicore__ inline void ZeroTile(AscendC::LocalTensor<F32>& t, uint32_t fullLen)
    {
        for (uint32_t i = 0; i < fullLen; ++i) t.SetValue(i, (F32)0.0f);
    }

    __aicore__ inline void MulAddTile(AscendC::LocalTensor<F32>& acc,
                                      const AscendC::LocalTensor<F32>& x,
                                      F32 scale,
                                      uint32_t lenPad)
    {
        for (uint32_t i = 0; i < lenPad; ++i) {
            acc.SetValue(i, acc.GetValue(i) + x.GetValue(i) * scale);
        }
    }

    __aicore__ inline void LoadResidualTilePad(uint32_t b, uint32_t s, uint32_t d0,
                                               uint32_t dLen, uint32_t dLenPad,
                                               AscendC::LocalTensor<F32>& dst)
    {
        for (uint32_t i = dLen; i < dLenPad; ++i) dst.SetValue(i, (F32)0.0f);
        for (uint32_t i = dLenPad; i < tileD; ++i) dst.SetValue(i, (F32)0.0f);

        const uint64_t base = ((uint64_t)b * (uint64_t)S + (uint64_t)s) * (uint64_t)D + (uint64_t)d0;
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDivU32(dLenPad, 8u); // copy padded 8-float blocks (safe)
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, residualsGm[base], p);
    }

    __aicore__ inline void StoreOutTile(uint32_t b, uint32_t t, uint32_t d0,
                                       uint32_t dLen,
                                       const AscendC::LocalTensor<F32>& src)
    {
        // store exact dLen (may be non-multiple of 8). Use scalar tail store to avoid overrunning.
        const uint64_t base = ((uint64_t)b * (uint64_t)S + (uint64_t)t) * (uint64_t)D + (uint64_t)d0;

        const uint32_t mainLen = (dLen / 8u) * 8u;
        if (mainLen > 0) {
            AscendC::DataCopyParams p;
            p.blockCount = 1;
            p.blockLen = mainLen / 8u;
            p.srcStride = 0;
            p.dstStride = 0;
            AscendC::DataCopy(outGm[base], src, p);
        }
        for (uint32_t i = mainLen; i < dLen; ++i) {
            outGm.SetValue(base + (uint64_t)i, src.GetValue(i));
        }
    }

    __aicore__ inline void LoadWRowTilePad(uint64_t wOffset, uint32_t len, uint32_t lenPad,
                                          AscendC::LocalTensor<F32>& dst)
    {
        for (uint32_t i = len; i < lenPad; ++i) dst.SetValue(i, (F32)0.0f);
        for (uint32_t i = lenPad; i < tileD; ++i) dst.SetValue(i, (F32)0.0f);

        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDivU32(lenPad, 8u);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, wGm[wOffset], p);
    }

    __aicore__ inline void LoadBranchInSlice(const AscendC::LocalTensor<F32>& branchIn,
                                             uint32_t offset, uint32_t lenPad,
                                             AscendC::LocalTensor<F32>& dst)
    {
        for (uint32_t i = 0; i < lenPad; ++i) dst.SetValue(i, branchIn.GetValue(offset + i));
        for (uint32_t i = lenPad; i < tileD; ++i) dst.SetValue(i, (F32)0.0f);
    }

    __aicore__ inline void SoftmaxS(const AscendC::GlobalTensor<F32>& logitsGm, F32 out[16])
    {
        F32 mx = logitsGm.GetValue(0);
        for (uint32_t i = 1; i < S; ++i) {
            const F32 v = logitsGm.GetValue((uint64_t)i);
            if (v > mx) mx = v;
        }
        F32 sum = (F32)0.0f;
        for (uint32_t i = 0; i < S; ++i) {
            const F32 e = ExpApprox(logitsGm.GetValue((uint64_t)i) - mx);
            out[i] = e;
            sum += e;
        }
        if (sum < (F32)1e-20f) sum = (F32)1e-20f;
        const F32 inv = (F32)1.0f / sum;
        for (uint32_t i = 0; i < S; ++i) out[i] = out[i] * inv;
        for (uint32_t i = S; i < 16u; ++i) out[i] = (F32)0.0f;
    }

    __aicore__ inline void BuildHresSinkhornInUb(F32 invTau, F32 logm, I32 iters,
                                                AscendC::LocalTensor<F32>& hres)
    {
        // hres holds z initially in 16x16 layout (row-major, stride 16)
        for (uint32_t i = 0; i < 256u; ++i) hres.SetValue(i, (F32)-1e30f);

        for (uint32_t r = 0; r < S; ++r) {
            const uint64_t rowBase = (uint64_t)r * (uint64_t)S;
            const uint32_t ubBase = r * 16u;
            for (uint32_t c = 0; c < S; ++c) {
                const F32 z = hResLogitsGm.GetValue(rowBase + (uint64_t)c) * invTau;
                hres.SetValue(ubBase + c, z);
            }
        }

        // u and v in UB
        AscendC::LocalTensor<F32> u = hVec0Ub.Get<F32>();
        AscendC::LocalTensor<F32> v = hVec1Ub.Get<F32>();
        for (uint32_t i = 0; i < 16u; ++i) { u.SetValue(i, (F32)0.0f); v.SetValue(i, (F32)0.0f); }

        // Sinkhorn in log domain
        for (I32 it = 0; it < iters; ++it) {
            // update u (rows)
            for (uint32_t r = 0; r < S; ++r) {
                const uint32_t base = r * 16u;
                F32 mx = (F32)-1e30f;
                for (uint32_t c = 0; c < S; ++c) {
                    const F32 val = hres.GetValue(base + c) + v.GetValue(c);
                    if (val > mx) mx = val;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t c = 0; c < S; ++c) {
                    sumExp += ExpApprox((hres.GetValue(base + c) + v.GetValue(c)) - mx);
                }
                const F32 lse = mx + LogApprox(sumExp);
                u.SetValue(r, logm - lse);
            }
            // update v (cols)
            for (uint32_t c = 0; c < S; ++c) {
                F32 mx = (F32)-1e30f;
                for (uint32_t r = 0; r < S; ++r) {
                    const F32 val = hres.GetValue(r * 16u + c) + u.GetValue(r);
                    if (val > mx) mx = val;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t r = 0; r < S; ++r) {
                    sumExp += ExpApprox((hres.GetValue(r * 16u + c) + u.GetValue(r)) - mx);
                }
                const F32 lse = mx + LogApprox(sumExp);
                v.SetValue(c, logm - lse);
            }
        }

        // materialize h_res = exp(z + u + v) * S
        const F32 scaleS = (F32)S;
        for (uint32_t r = 0; r < S; ++r) {
            const uint32_t base = r * 16u;
            for (uint32_t c = 0; c < S; ++c) {
                const F32 val = hres.GetValue(base + c) + u.GetValue(r) + v.GetValue(c);
                hres.SetValue(base + c, ExpApprox(val) * scaleS);
            }
            for (uint32_t c = S; c < 16u; ++c) hres.SetValue(base + c, (F32)0.0f);
        }
        for (uint32_t r = S; r < 16u; ++r) {
            const uint32_t base = r * 16u;
            for (uint32_t c = 0; c < 16u; ++c) hres.SetValue(base + c, (F32)0.0f);
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<F32> residualsGm;
    AscendC::GlobalTensor<F32> hResLogitsGm, hPreLogitsGm, hPostLogitsGm;
    AscendC::GlobalTensor<F32> wGm;

    AscendC::GlobalTensor<I32> itersGm;
    AscendC::GlobalTensor<F32> tauGm, logSGm;

    AscendC::GlobalTensor<F32> outGm;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> rTileUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> accTileUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> brUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> boUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wTileUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> hVec0Ub;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> hVec1Ub;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> hMatUb;

    uint32_t B{}, S{}, D{}, Dpad{}, tileD{};
    uint32_t bStart{}, bCount{};
};

extern "C" __global__ __aicore__ void static_mhc_hyper_connections_custom(
    GM_ADDR residuals,
    GM_ADDR h_res_logits,
    GM_ADDR h_pre_logits,
    GM_ADDR h_post_logits,
    GM_ADDR branch_weight,
    GM_ADDR sinkhorn_iters,
    GM_ADDR tau,
    GM_ADDR log_s,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelStaticMhcHyperConnectionsCustom op;
    op.Init(residuals, h_res_logits, h_pre_logits, h_post_logits, branch_weight,
            sinkhorn_iters, tau, log_s,
            out,
            tiling_data.B, tiling_data.S, tiling_data.D, tiling_data.Dpad, tiling_data.tileD);
    op.Process();
}
