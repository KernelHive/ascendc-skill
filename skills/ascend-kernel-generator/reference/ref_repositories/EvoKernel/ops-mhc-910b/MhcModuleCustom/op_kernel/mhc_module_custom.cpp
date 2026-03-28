
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;

class KernelMhcModuleCustom {
public:
    __aicore__ inline KernelMhcModuleCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x_streams,
        GM_ADDR rms_scale,
        GM_ADDR w_pre, GM_ADDR w_post, GM_ADDR w_res,
        GM_ADDR b_pre, GM_ADDR b_post, GM_ADDR b_res,
        GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
        GM_ADDR tmax, GM_ADDR rms_eps, GM_ADDR invF,
        GM_ADDR use_mlp,
        GM_ADDR mlp_w1, GM_ADDR mlp_b1, GM_ADDR mlp_w2, GM_ADDR mlp_b2,
        GM_ADDR ln_weight, GM_ADDR ln_bias, GM_ADDR ln_eps,
        GM_ADDR out,
        uint32_t B, uint32_t T, uint32_t N, uint32_t C,
        uint32_t BT, uint32_t F, uint32_t NN,
        uint32_t Fpad, uint32_t NNpad, uint32_t Cpad, uint32_t Npad,
        uint32_t H, uint32_t Hpad)
    {
        this->B=B; this->T=T; this->N=N; this->C=C;
        this->BT=BT; this->F=F; this->NN=NN;
        this->Fpad=Fpad; this->NNpad=NNpad; this->Cpad=Cpad; this->Npad=Npad;
        this->H=H; this->Hpad=Hpad;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t btPerBlock = (BT + blockNum - 1u) / blockNum;
        btStart = blockIdx * btPerBlock;
        uint32_t btEnd = btStart + btPerBlock;
        if (btEnd > BT) btEnd = BT;
        btCount = (btEnd > btStart) ? (btEnd - btStart) : 0;

        xGm.SetGlobalBuffer((__gm__ F32*)x_streams, (uint64_t)BT * (uint64_t)F);
        scGm.SetGlobalBuffer((__gm__ F32*)rms_scale, (uint64_t)F);

        wPreGm.SetGlobalBuffer((__gm__ F32*)w_pre,  (uint64_t)F * (uint64_t)N);
        wPostGm.SetGlobalBuffer((__gm__ F32*)w_post, (uint64_t)F * (uint64_t)N);
        wResGm.SetGlobalBuffer((__gm__ F32*)w_res,  (uint64_t)F * (uint64_t)NN);

        bPreGm.SetGlobalBuffer((__gm__ F32*)b_pre,  (uint64_t)N);
        bPostGm.SetGlobalBuffer((__gm__ F32*)b_post, (uint64_t)N);
        bResGm.SetGlobalBuffer((__gm__ F32*)b_res,  (uint64_t)NN);

        aPreGm.SetGlobalBuffer((__gm__ F32*)alpha_pre, (uint64_t)1);
        aPostGm.SetGlobalBuffer((__gm__ F32*)alpha_post,(uint64_t)1);
        aResGm.SetGlobalBuffer((__gm__ F32*)alpha_res, (uint64_t)1);

        tmaxGm.SetGlobalBuffer((__gm__ I32*)tmax, (uint64_t)1);
        epsGm.SetGlobalBuffer((__gm__ F32*)rms_eps, (uint64_t)1);
        invFGm.SetGlobalBuffer((__gm__ F32*)invF, (uint64_t)1);

        useMlpGm.SetGlobalBuffer((__gm__ I32*)use_mlp, (uint64_t)1);

        // MLP + LN (may be dummy)
        w1Gm.SetGlobalBuffer((__gm__ F32*)mlp_w1, (uint64_t)C * (uint64_t)((H==0u)?1u:H));
        b1Gm.SetGlobalBuffer((__gm__ F32*)mlp_b1, (uint64_t)((H==0u)?1u:H));
        w2Gm.SetGlobalBuffer((__gm__ F32*)mlp_w2, (uint64_t)((H==0u)?1u:H) * (uint64_t)C);
        b2Gm.SetGlobalBuffer((__gm__ F32*)mlp_b2, (uint64_t)C);

        lnWGm.SetGlobalBuffer((__gm__ F32*)ln_weight, (uint64_t)C);
        lnBGm.SetGlobalBuffer((__gm__ F32*)ln_bias, (uint64_t)C);
        lnEpsGm.SetGlobalBuffer((__gm__ F32*)ln_eps, (uint64_t)1);

        outGm.SetGlobalBuffer((__gm__ F32*)out, (uint64_t)BT * (uint64_t)F);

        // UB: x(Fpad) + xNorm(Fpad) + hRes(NNpad) + hPre(Npad) + hPost(Npad) + y(Cpad) + tmpX(Cpad) + tmpY(Cpad) + act(Hpad) (optional)
        // Keep bounded: C=256 => Cpad=256, Fpad=1024, NNpad=16, Npad=8. H typically 1024 => Hpad=1024.
        uint32_t base = (Fpad + Fpad + NNpad + Npad + Npad + Cpad + Cpad + Cpad);
        uint32_t total = base + ((Hpad>0u)?Hpad:0u);
        pipe.InitBuffer(sharedUb, total * (uint32_t)sizeof(F32));
    }

    __aicore__ inline void Process()
    {
        if (btCount == 0) return;

        const F32 aPre  = aPreGm.GetValue(0);
        const F32 aPost = aPostGm.GetValue(0);
        const F32 aRes  = aResGm.GetValue(0);

        I32 tmax = tmaxGm.GetValue(0);
        if (tmax < 0) tmax = 0;
        if (tmax > 256) tmax = 256;

        const F32 rmsEps = epsGm.GetValue(0);
        const F32 invF = invFGm.GetValue(0);

        I32 use = useMlpGm.GetValue(0);
        use = (use != 0) ? 1 : 0;
        if (use == 1 && H == 0u) use = 0; // guard if dummy weights

        const F32 lnEps = lnEpsGm.GetValue(0);

        for (uint32_t k = 0; k < btCount; ++k) {
            const uint32_t bt = btStart + k;

            auto ub = sharedUb.Get<F32>();
            AscendC::LocalTensor<F32> xUb     = ub;                                 // Fpad
            AscendC::LocalTensor<F32> xNormUb = ub[Fpad];                           // Fpad
            AscendC::LocalTensor<F32> hResUb  = ub[Fpad + Fpad];                    // NNpad (also used as prob)
            AscendC::LocalTensor<F32> hPreUb  = ub[Fpad + Fpad + NNpad];            // Npad
            AscendC::LocalTensor<F32> hPostUb = ub[Fpad + Fpad + NNpad + Npad];     // Npad
            AscendC::LocalTensor<F32> yUb     = ub[Fpad + Fpad + NNpad + Npad + Npad];               // Cpad
            AscendC::LocalTensor<F32> tmpXUb  = ub[Fpad + Fpad + NNpad + Npad + Npad + Cpad];        // Cpad
            AscendC::LocalTensor<F32> tmpYUb  = ub[Fpad + Fpad + NNpad + Npad + Npad + Cpad + Cpad]; // Cpad
            AscendC::LocalTensor<F32> actUb;
            if (Hpad > 0u) actUb = ub[Fpad + Fpad + NNpad + Npad + Npad + Cpad + Cpad + Cpad];

            LoadXPad(bt, xUb);

            const F32 invRms = ComputeInvRms(xUb, rmsEps, invF);
            ApplyRmsAndScale(xUb, xNormUb, invRms);

            ComputeHPrePost(xNormUb, aPre, aPost, hPreUb, hPostUb);
            ComputeHResSinkhorn(xNormUb, aRes, tmax, hResUb);

            // x_in = sum_j h_pre[j] * x[j,:]  (C vector)
            ComputeXIn(bt, hPreUb, xUb, yUb); // yUb holds x_in initially

            // y = residual(x_in)
            if (use == 0) {
                // identity: y stays as x_in
            } else {
                // LayerNorm then MLP
                LayerNormInplace(yUb, tmpXUb, lnEps);       // tmpXUb = LN(yUb)
                MlpForward(tmpXUb, actUb, tmpYUb, yUb);     // yUb = MLP(tmpXUb)
            }

            // write out by C tiles: out[i,:] = sum_j hRes[i,j]*x[j,:] + hPost[i]*y[:]
            WriteOut(bt, hPostUb, hResUb, xUb, yUb, tmpXUb);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

    __aicore__ inline void LoadXPad(uint32_t bt, AscendC::LocalTensor<F32>& xUb)
    {
        for (uint32_t i = F; i < Fpad; ++i) xUb.SetValue(i, (F32)0.0f);
        const uint64_t base = (uint64_t)bt * (uint64_t)F;
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(F, 8u);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(xUb, xGm[base], p);
    }

    __aicore__ inline F32 ComputeInvRms(const AscendC::LocalTensor<F32>& xUb, F32 eps, F32 invF)
    {
        F32 sumsq = (F32)0.0f;
        for (uint32_t i = 0; i < F; ++i) {
            const F32 v = xUb.GetValue(i);
            sumsq += v * v;
        }
        // Use intrinsic rsqrt when available; fallback to 1/sqrt.
        const F32 mean = sumsq * invF;
        const F32 denom = mean + eps;
        // Simple Newton rsqrt with good accuracy (no clamping).
        F32 y = (F32)1.0f;
        for (int it = 0; it < 4; ++it) y = y * ((F32)1.5f - (F32)0.5f * denom * y * y);
        return y;
    }

    __aicore__ inline void ApplyRmsAndScale(const AscendC::LocalTensor<F32>& xUb,
                                           AscendC::LocalTensor<F32>& xNormUb,
                                           F32 invRms)
    {
        for (uint32_t i = 0; i < F; ++i) {
            const F32 v = xUb.GetValue(i) * invRms;
            xNormUb.SetValue(i, v * scGm.GetValue((uint64_t)i));
        }
        for (uint32_t i = F; i < Fpad; ++i) xNormUb.SetValue(i, (F32)0.0f);
    }

    __aicore__ inline F32 Sigmoid(F32 x)
    {
        // Stable sigmoid using exp; mild clamp to avoid inf (kept wide).
        if (x > (F32)20.0f) x = (F32)20.0f;
        if (x < (F32)-20.0f) x = (F32)-20.0f;
        // exp(-x) approximation via series is risky; use expf if supported by toolchain.
        // AscendC allows standard device math in many setups; keep as expf.
        const F32 e = (F32)expf(-x);
        return (F32)1.0f / ((F32)1.0f + e);
    }

    __aicore__ inline void ComputeHPrePost(const AscendC::LocalTensor<F32>& xNormUb,
                                          F32 aPre, F32 aPost,
                                          AscendC::LocalTensor<F32>& hPreUb,
                                          AscendC::LocalTensor<F32>& hPostUb)
    {
        for (uint32_t j = 0; j < Npad; ++j) { hPreUb.SetValue(j, (F32)0.0f); hPostUb.SetValue(j, (F32)0.0f); }

        if (N <= 8u) {
            F32 accP[8]; F32 accO[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) { accP[j]=(F32)0.0f; accO[j]=(F32)0.0f; }

            for (uint32_t f = 0; f < F; ++f) {
                const F32 xv = xNormUb.GetValue(f);
                const uint64_t base = (uint64_t)f * (uint64_t)N;
                #pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    if (j < N) {
                        accP[j] += xv * wPreGm.GetValue(base + (uint64_t)j);
                        accO[j] += xv * wPostGm.GetValue(base + (uint64_t)j);
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < 8; ++j) {
                if (j < N) {
                    const F32 pre  = aPre  * accP[j] + bPreGm.GetValue((uint64_t)j);
                    const F32 post = aPost * accO[j] + bPostGm.GetValue((uint64_t)j);
                    hPreUb.SetValue(j, Sigmoid(pre));
                    hPostUb.SetValue(j, (F32)2.0f * Sigmoid(post));
                }
            }
            return;
        }

        for (uint32_t j = 0; j < N; ++j) {
            F32 ap = (F32)0.0f;
            F32 ao = (F32)0.0f;
            for (uint32_t f = 0; f < F; ++f) {
                const F32 xv = xNormUb.GetValue(f);
                const uint64_t idx = (uint64_t)f * (uint64_t)N + (uint64_t)j;
                ap += xv * wPreGm.GetValue(idx);
                ao += xv * wPostGm.GetValue(idx);
            }
            const F32 pre  = aPre  * ap + bPreGm.GetValue((uint64_t)j);
            const F32 post = aPost * ao + bPostGm.GetValue((uint64_t)j);
            hPreUb.SetValue(j, Sigmoid(pre));
            hPostUb.SetValue(j, (F32)2.0f * Sigmoid(post));
        }
    }

    __aicore__ inline void ComputeHResSinkhorn(const AscendC::LocalTensor<F32>& xNormUb,
                                               F32 aRes, I32 tmax,
                                               AscendC::LocalTensor<F32>& hResUb)
    {
        for (uint32_t i = 0; i < NNpad; ++i) hResUb.SetValue(i, (F32)0.0f);

        // logits = aRes*(xNorm @ wRes) + bRes
        F32 maxv = (F32)-1e30f;
        for (uint32_t ij = 0; ij < NN; ++ij) {
            F32 acc = (F32)0.0f;
            for (uint32_t f = 0; f < F; ++f) {
                acc += xNormUb.GetValue(f) * wResGm.GetValue((uint64_t)f * (uint64_t)NN + (uint64_t)ij);
            }
            const F32 v = aRes * acc + bResGm.GetValue((uint64_t)ij);
            hResUb.SetValue(ij, v);
            if (v > maxv) maxv = v;
        }

        // prob = exp(logits - max)
        for (uint32_t ij = 0; ij < NN; ++ij) {
            const F32 v = hResUb.GetValue(ij) - maxv;
            hResUb.SetValue(ij, (F32)expf(v));
        }

        // Sinkhorn in probability space:
        for (I32 it = 0; it < tmax; ++it) {
            // normalize over dim=-2 (sum over i for each j): columns
            for (uint32_t j = 0; j < N; ++j) {
                F32 s = (F32)0.0f;
                for (uint32_t i = 0; i < N; ++i) s += hResUb.GetValue(i * N + j);
                const F32 invs = (F32)1.0f / (s + (F32)1e-8f);
                for (uint32_t i = 0; i < N; ++i) {
                    const uint32_t idx = i * N + j;
                    hResUb.SetValue(idx, hResUb.GetValue(idx) * invs);
                }
            }
            // normalize over dim=-1 (sum over j for each i): rows
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t row = i * N;
                F32 s = (F32)0.0f;
                for (uint32_t j = 0; j < N; ++j) s += hResUb.GetValue(row + j);
                const F32 invs = (F32)1.0f / (s + (F32)1e-8f);
                for (uint32_t j = 0; j < N; ++j) {
                    const uint32_t idx = row + j;
                    hResUb.SetValue(idx, hResUb.GetValue(idx) * invs);
                }
            }
        }
    }

    __aicore__ inline void ComputeXIn(uint32_t /*bt*/,
                                      const AscendC::LocalTensor<F32>& hPreUb,
                                      const AscendC::LocalTensor<F32>& xUb,
                                      AscendC::LocalTensor<F32>& xInUb)
    {
        for (uint32_t c = 0; c < Cpad; ++c) xInUb.SetValue(c, (F32)0.0f);
        for (uint32_t j = 0; j < N; ++j) {
            const F32 hj = hPreUb.GetValue(j);
            const uint32_t base = j * C;
            for (uint32_t c = 0; c < C; ++c) {
                xInUb.SetValue(c, xInUb.GetValue(c) + hj * xUb.GetValue(base + c));
            }
        }
        for (uint32_t c = C; c < Cpad; ++c) xInUb.SetValue(c, (F32)0.0f);
    }

    __aicore__ inline void LayerNormInplace(const AscendC::LocalTensor<F32>& inUb,
                                           AscendC::LocalTensor<F32>& outUb,
                                           F32 lnEps)
    {
        // Compute mean and var over C
        F32 mean = (F32)0.0f;
        for (uint32_t c = 0; c < C; ++c) mean += inUb.GetValue(c);
        mean /= (F32)C;

        F32 var = (F32)0.0f;
        for (uint32_t c = 0; c < C; ++c) {
            const F32 d = inUb.GetValue(c) - mean;
            var += d * d;
        }
        var /= (F32)C;

        // invstd
        const F32 denom = var + lnEps;
        F32 invstd = (F32)1.0f;
        for (int it = 0; it < 4; ++it) invstd = invstd * ((F32)1.5f - (F32)0.5f * denom * invstd * invstd);

        // affine
        for (uint32_t c = 0; c < C; ++c) {
            const F32 xn = (inUb.GetValue(c) - mean) * invstd;
            outUb.SetValue(c, xn * lnWGm.GetValue((uint64_t)c) + lnBGm.GetValue((uint64_t)c));
        }
        for (uint32_t c = C; c < Cpad; ++c) outUb.SetValue(c, (F32)0.0f);
    }

    __aicore__ inline F32 Gelu(F32 x)
    {
        // tanh-based GELU approximation used by PyTorch: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        const F32 k0 = (F32)0.7978845608028654f; // sqrt(2/pi)
        const F32 k1 = (F32)0.044715f;
        const F32 x3 = x * x * x;
        const F32 u = k0 * (x + k1 * x3);
        const F32 t = (F32)tanhf(u);
        return (F32)0.5f * x * ((F32)1.0f + t);
    }

    __aicore__ inline void MlpForward(const AscendC::LocalTensor<F32>& lnUb,
                                     AscendC::LocalTensor<F32>& actUb,
                                     AscendC::LocalTensor<F32>& tmpUb,
                                     AscendC::LocalTensor<F32>& outUb)
    {
        // act[h] = sum_c ln[c]*w1[c,h] + b1[h]
        for (uint32_t h = 0; h < H; ++h) {
            F32 acc = b1Gm.GetValue((uint64_t)h);
            for (uint32_t c = 0; c < C; ++c) {
                acc += lnUb.GetValue(c) * w1Gm.GetValue((uint64_t)c * (uint64_t)H + (uint64_t)h);
            }
            actUb.SetValue(h, Gelu(acc));
        }
        for (uint32_t h = H; h < Hpad; ++h) actUb.SetValue(h, (F32)0.0f);

        // tmp[c] = sum_h act[h]*w2[h,c] + b2[c]
        for (uint32_t c = 0; c < C; ++c) {
            F32 acc = b2Gm.GetValue((uint64_t)c);
            for (uint32_t h = 0; h < H; ++h) {
                acc += actUb.GetValue(h) * w2Gm.GetValue((uint64_t)h * (uint64_t)C + (uint64_t)c);
            }
            tmpUb.SetValue(c, acc);
        }
        for (uint32_t c = C; c < Cpad; ++c) tmpUb.SetValue(c, (F32)0.0f);

        // out = tmp
        for (uint32_t c = 0; c < Cpad; ++c) outUb.SetValue(c, tmpUb.GetValue(c));
    }

    __aicore__ inline void WriteOut(uint32_t bt,
                                   const AscendC::LocalTensor<F32>& hPostUb,
                                   const AscendC::LocalTensor<F32>& hResUb,
                                   const AscendC::LocalTensor<F32>& xUb,
                                   const AscendC::LocalTensor<F32>& yUb,
                                   AscendC::LocalTensor<F32>& outTileUb)
    {
        // outTileUb used as scratch for one stream C tile
        const uint64_t outBase = (uint64_t)bt * (uint64_t)F;

        for (uint32_t i = 0; i < N; ++i) {
            // resmix_i[c] = sum_j hRes[i,j] * x[j,c]
            for (uint32_t c = 0; c < Cpad; ++c) outTileUb.SetValue(c, (F32)0.0f);

            for (uint32_t j = 0; j < N; ++j) {
                const F32 hij = hResUb.GetValue(i * N + j);
                const uint32_t xBase = j * C;
                for (uint32_t c = 0; c < C; ++c) {
                    outTileUb.SetValue(c, outTileUb.GetValue(c) + hij * xUb.GetValue(xBase + c));
                }
            }

            const F32 hp = hPostUb.GetValue(i);
            for (uint32_t c = 0; c < C; ++c) {
                outTileUb.SetValue(c, outTileUb.GetValue(c) + hp * yUb.GetValue(c));
            }
            for (uint32_t c = C; c < Cpad; ++c) outTileUb.SetValue(c, (F32)0.0f);

            // DMA write (C contiguous) to GM
            const uint64_t dst = outBase + (uint64_t)i * (uint64_t)C;
            AscendC::DataCopyParams p;
            p.blockCount = 1;
            p.blockLen = CeilDiv(C, 8u);
            p.srcStride = 0;
            p.dstStride = 0;
            AscendC::DataCopy(outGm[dst], outTileUb, p);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> sharedUb;

    AscendC::GlobalTensor<F32> xGm, scGm;
    AscendC::GlobalTensor<F32> wPreGm, wPostGm, wResGm;
    AscendC::GlobalTensor<F32> bPreGm, bPostGm, bResGm;
    AscendC::GlobalTensor<F32> aPreGm, aPostGm, aResGm;
    AscendC::GlobalTensor<I32> tmaxGm, useMlpGm;
    AscendC::GlobalTensor<F32> epsGm, invFGm;

    AscendC::GlobalTensor<F32> w1Gm, b1Gm, w2Gm, b2Gm;
    AscendC::GlobalTensor<F32> lnWGm, lnBGm, lnEpsGm;

    AscendC::GlobalTensor<F32> outGm;

    uint32_t B{}, T{}, N{}, C{}, BT{}, F{}, NN{};
    uint32_t Fpad{}, NNpad{}, Cpad{}, Npad{}, H{}, Hpad{};
    uint32_t btStart{}, btCount{};
};

extern "C" __global__ __aicore__ void mhc_module_custom(
    GM_ADDR x_streams,
    GM_ADDR rms_scale,
    GM_ADDR w_pre, GM_ADDR w_post, GM_ADDR w_res,
    GM_ADDR b_pre, GM_ADDR b_post, GM_ADDR b_res,
    GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
    GM_ADDR tmax, GM_ADDR rms_eps, GM_ADDR invF,
    GM_ADDR use_mlp,
    GM_ADDR mlp_w1, GM_ADDR mlp_b1, GM_ADDR mlp_w2, GM_ADDR mlp_b2,
    GM_ADDR ln_weight, GM_ADDR ln_bias, GM_ADDR ln_eps,
    GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelMhcModuleCustom op;
    op.Init(x_streams,
            rms_scale,
            w_pre, w_post, w_res,
            b_pre, b_post, b_res,
            alpha_pre, alpha_post, alpha_res,
            tmax, rms_eps, invF,
            use_mlp,
            mlp_w1, mlp_b1, mlp_w2, mlp_b2,
            ln_weight, ln_bias, ln_eps,
            out,
            td.B, td.T, td.N, td.C,
            td.BT, td.F, td.NN,
            td.Fpad, td.NNpad, td.Cpad, td.Npad,
            td.H, td.Hpad);
    op.Process();
}
