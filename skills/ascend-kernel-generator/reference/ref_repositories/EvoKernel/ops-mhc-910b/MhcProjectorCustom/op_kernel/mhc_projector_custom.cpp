
#include "kernel_operator.h"

using F32 = float;
using I32 = int32_t;

class KernelMhcProjectorCustom {
public:
    __aicore__ inline KernelMhcProjectorCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x_stream,
        GM_ADDR phi_pre, GM_ADDR phi_post, GM_ADDR phi_res,
        GM_ADDR b_pre, GM_ADDR b_post, GM_ADDR b_res,
        GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
        GM_ADDR tmax, GM_ADDR rmsnorm_eps,
        GM_ADDR invF,
        GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR h_res,
        uint32_t B, uint32_t T, uint32_t N, uint32_t C,
        uint32_t BT, uint32_t F, uint32_t NN, uint32_t Fpad,
        uint32_t Npad, uint32_t NNpad)
    {
        this->B = B; this->T = T; this->N = N; this->C = C;
        this->BT = BT; this->F = F; this->NN = NN;
        this->Fpad = Fpad; this->Npad = Npad; this->NNpad = NNpad;

        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t btPerBlock = (BT + blockNum - 1u) / blockNum;
        this->btStart = blockIdx * btPerBlock;
        uint32_t btEnd = this->btStart + btPerBlock;
        if (btEnd > BT) btEnd = BT;
        this->btCount = (btEnd > this->btStart) ? (btEnd - this->btStart) : 0;

        xGm.SetGlobalBuffer((__gm__ F32*)x_stream, (uint64_t)BT * (uint64_t)F);

        phiPreGm.SetGlobalBuffer((__gm__ F32*)phi_pre,  (uint64_t)F * (uint64_t)N);
        phiPostGm.SetGlobalBuffer((__gm__ F32*)phi_post,(uint64_t)F * (uint64_t)N);
        phiResGm.SetGlobalBuffer((__gm__ F32*)phi_res,  (uint64_t)F * (uint64_t)NN);

        bPreGm.SetGlobalBuffer((__gm__ F32*)b_pre,  (uint64_t)N);
        bPostGm.SetGlobalBuffer((__gm__ F32*)b_post,(uint64_t)N);
        bResGm.SetGlobalBuffer((__gm__ F32*)b_res, (uint64_t)NN);

        aPreGm.SetGlobalBuffer((__gm__ F32*)alpha_pre,  (uint64_t)1);
        aPostGm.SetGlobalBuffer((__gm__ F32*)alpha_post, (uint64_t)1);
        aResGm.SetGlobalBuffer((__gm__ F32*)alpha_res,  (uint64_t)1);
        tmaxGm.SetGlobalBuffer((__gm__ I32*)tmax, (uint64_t)1);
        epsGm.SetGlobalBuffer((__gm__ F32*)rmsnorm_eps, (uint64_t)1);
        invFGm.SetGlobalBuffer((__gm__ F32*)invF, (uint64_t)1);

        outPreGm.SetGlobalBuffer((__gm__ F32*)h_pre,  (uint64_t)BT * (uint64_t)N);
        outPostGm.SetGlobalBuffer((__gm__ F32*)h_post, (uint64_t)BT * (uint64_t)N);
        outResGm.SetGlobalBuffer((__gm__ F32*)h_res,  (uint64_t)BT * (uint64_t)NN);

        // Double-buffer x tiles
        pipe.InitBuffer(xInQ, 2, (uint32_t)(Fpad * sizeof(F32)));

        // Bias UB
        pipe.InitBuffer(bPreUb,  (uint32_t)(Npad  * sizeof(F32)));
        pipe.InitBuffer(bPostUb, (uint32_t)(Npad  * sizeof(F32)));
        pipe.InitBuffer(bResUb,  (uint32_t)(NNpad * sizeof(F32)));

        // temp UB for rmsnorm and generic path
        pipe.InitBuffer(tmpUb,   (uint32_t)(Fpad * sizeof(F32)));
        pipe.InitBuffer(resUb,   (uint32_t)(NNpad * sizeof(F32)));

        // N=4 weights staged once per core (fast path)
        if (N == 4u) {
            pipe.InitBuffer(phiPreUb,  (uint32_t)(Fpad * 4u  * sizeof(F32)));   // F x 4
            pipe.InitBuffer(phiPostUb, (uint32_t)(Fpad * 4u  * sizeof(F32)));   // F x 4
            pipe.InitBuffer(phiResUb,  (uint32_t)(Fpad * 16u * sizeof(F32)));   // F x 16
        }
    }

    __aicore__ inline void Process()
    {
        if (btCount == 0) return;

        const F32 aPre  = aPreGm.GetValue(0);
        const F32 aPost = aPostGm.GetValue(0);
        const F32 aRes  = aResGm.GetValue(0);
        I32 tmax = tmaxGm.GetValue(0);
        if (tmax < 0) tmax = 0;
        const F32 eps = epsGm.GetValue(0);
        const F32 invF = invFGm.GetValue(0);

        StageBiasesToUb();

        const bool n4 = (N == 4u);
        if (n4) {
            StagePhiN4ToUb();
        }

        // Prefetch first x
        uint32_t next = 0;
        if (next < btCount) {
            CopyInX(btStart + next);
            next++;
        }

        for (uint32_t k = 0; k < btCount; ++k) {
            if (next < btCount) {
                CopyInX(btStart + next);
                next++;
            }

            AscendC::LocalTensor<F32> xLocal = xInQ.DeQue<F32>();

            // RMSNorm in-place in UB
            (void)RmsnormInplaceUb(xLocal, eps, invF);

            if (n4) {
                ComputePrePostN4_Staged(btStart + k, xLocal, aPre, aPost);
                ComputeResSinkhornN4_Staged(btStart + k, xLocal, aRes, tmax);
            } else {
                const F32 invRms = ComputeInvRmsFromUb(xLocal, eps, invF); // recompute invRms for generic
                // scale now for generic too, to avoid repeated xv*invRms inside loops
                AscendC::Muls(xLocal, xLocal, invRms, Fpad);
                ComputePrePostGenericScaled(btStart + k, xLocal, aPre, aPost);
                ComputeResSinkhornGenericScaled(btStart + k, xLocal, aRes, tmax);
            }

            xInQ.FreeTensor(xLocal);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

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

    __aicore__ inline void StageBiasesToUb()
    {
        AscendC::LocalTensor<F32> bp = bPreUb.Get<F32>();
        AscendC::LocalTensor<F32> bo = bPostUb.Get<F32>();
        AscendC::LocalTensor<F32> br = bResUb.Get<F32>();

        for (uint32_t i = N; i < Npad; ++i) { bp.SetValue(i, (F32)0.0f); bo.SetValue(i, (F32)0.0f); }
        for (uint32_t i = NN; i < NNpad; ++i) { br.SetValue(i, (F32)0.0f); }

        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(N, 8u);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(bp, bPreGm[0], p);
        AscendC::DataCopy(bo, bPostGm[0], p);

        AscendC::DataCopyParams p2;
        p2.blockCount = 1;
        p2.blockLen = CeilDiv(NN, 8u);
        p2.srcStride = 0;
        p2.dstStride = 0;
        AscendC::DataCopy(br, bResGm[0], p2);
    }

    __aicore__ inline void CopyInX(uint32_t bt)
    {
        AscendC::LocalTensor<F32> xLocal = xInQ.AllocTensor<F32>();
        for (uint32_t i = F; i < Fpad; ++i) xLocal.SetValue(i, (F32)0.0f);

        const uint64_t xBase = (uint64_t)bt * (uint64_t)F;
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = CeilDiv(F, 8u);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(xLocal, xGm[xBase], p);
        xInQ.EnQue(xLocal);
    }

    __aicore__ inline F32 ComputeInvRmsFromUb(const AscendC::LocalTensor<F32>& xLocal, F32 eps, F32 invF)
    {
        F32 sumsq = (F32)0.0f;
        for (uint32_t i = 0; i < F; ++i) {
            const F32 v = xLocal.GetValue(i);
            sumsq += v * v;
        }
        const F32 mean = sumsq * invF;
        return RsqrtApprox(mean + eps);
    }

    __aicore__ inline F32 RmsnormInplaceUb(AscendC::LocalTensor<F32>& xLocal, F32 eps, F32 invF)
    {
        AscendC::LocalTensor<F32> tmp = tmpUb.Get<F32>();
        AscendC::Mul(tmp, xLocal, xLocal, Fpad);
        F32 sumsq = (F32)0.0f;
        for (uint32_t i = 0; i < F; ++i) sumsq += tmp.GetValue(i);
        const F32 mean = sumsq * invF;
        const F32 invRms = RsqrtApprox(mean + eps);
        AscendC::Muls(xLocal, xLocal, invRms, Fpad);
        return invRms;
    }

    __aicore__ inline void StagePhiN4ToUb()
    {
        AscendC::LocalTensor<F32> pp = phiPreUb.Get<F32>();
        AscendC::LocalTensor<F32> po = phiPostUb.Get<F32>();
        AscendC::LocalTensor<F32> pr = phiResUb.Get<F32>();

        // Copy contiguous [F,4], [F,4], [F,16] (all are multiples of 8 for F=1024; for safety, use CeilDiv).
        AscendC::DataCopyParams p4;
        p4.blockCount = 1;
        p4.blockLen = CeilDiv(F * 4u, 8u);
        p4.srcStride = 0;
        p4.dstStride = 0;
        AscendC::DataCopy(pp, phiPreGm[0], p4);
        AscendC::DataCopy(po, phiPostGm[0], p4);

        AscendC::DataCopyParams p16;
        p16.blockCount = 1;
        p16.blockLen = CeilDiv(F * 16u, 8u);
        p16.srcStride = 0;
        p16.dstStride = 0;
        AscendC::DataCopy(pr, phiResGm[0], p16);
    }

    __aicore__ inline void ComputePrePostN4_Staged(uint32_t bt, const AscendC::LocalTensor<F32>& xNorm, F32 aPre, F32 aPost)
    {
        AscendC::LocalTensor<F32> pp = phiPreUb.Get<F32>();
        AscendC::LocalTensor<F32> po = phiPostUb.Get<F32>();
        AscendC::LocalTensor<F32> bp = bPreUb.Get<F32>();
        AscendC::LocalTensor<F32> bo = bPostUb.Get<F32>();

        F32 pre0=0, pre1=0, pre2=0, pre3=0;
        F32 post0=0, post1=0, post2=0, post3=0;

        // pp/po laid out row-major: f*4 + j
        for (uint32_t f = 0; f < F; ++f) {
            const F32 xv = xNorm.GetValue(f);
            const uint32_t base = f * 4u;

            pre0  += xv * pp.GetValue(base + 0u);
            pre1  += xv * pp.GetValue(base + 1u);
            pre2  += xv * pp.GetValue(base + 2u);
            pre3  += xv * pp.GetValue(base + 3u);

            post0 += xv * po.GetValue(base + 0u);
            post1 += xv * po.GetValue(base + 1u);
            post2 += xv * po.GetValue(base + 2u);
            post3 += xv * po.GetValue(base + 3u);
        }

        pre0 = SigmoidApprox(aPre * pre0 + bp.GetValue(0));
        pre1 = SigmoidApprox(aPre * pre1 + bp.GetValue(1));
        pre2 = SigmoidApprox(aPre * pre2 + bp.GetValue(2));
        pre3 = SigmoidApprox(aPre * pre3 + bp.GetValue(3));

        post0 = (F32)2.0f * SigmoidApprox(aPost * post0 + bo.GetValue(0));
        post1 = (F32)2.0f * SigmoidApprox(aPost * post1 + bo.GetValue(1));
        post2 = (F32)2.0f * SigmoidApprox(aPost * post2 + bo.GetValue(2));
        post3 = (F32)2.0f * SigmoidApprox(aPost * post3 + bo.GetValue(3));

        const uint64_t oBase = (uint64_t)bt * 4ull;
        outPreGm.SetValue(oBase + 0, pre0);
        outPreGm.SetValue(oBase + 1, pre1);
        outPreGm.SetValue(oBase + 2, pre2);
        outPreGm.SetValue(oBase + 3, pre3);

        outPostGm.SetValue(oBase + 0, post0);
        outPostGm.SetValue(oBase + 1, post1);
        outPostGm.SetValue(oBase + 2, post2);
        outPostGm.SetValue(oBase + 3, post3);
    }

    __aicore__ inline void RowNorm4(F32 &a, F32 &b, F32 &c, F32 &d)
    {
        F32 rmax = a;
        if (b > rmax) rmax = b;
        if (c > rmax) rmax = c;
        if (d > rmax) rmax = d;
        const F32 sum = ExpApprox(a - rmax) + ExpApprox(b - rmax) + ExpApprox(c - rmax) + ExpApprox(d - rmax);
        const F32 lse = rmax + LogApprox(sum);
        a -= lse; b -= lse; c -= lse; d -= lse;
    }

    __aicore__ inline void ColNorm4(F32 &a, F32 &b, F32 &c, F32 &d)
    {
        F32 cmax = a;
        if (b > cmax) cmax = b;
        if (c > cmax) cmax = c;
        if (d > cmax) cmax = d;
        const F32 sum = ExpApprox(a - cmax) + ExpApprox(b - cmax) + ExpApprox(c - cmax) + ExpApprox(d - cmax);
        const F32 lse = cmax + LogApprox(sum);
        a -= lse; b -= lse; c -= lse; d -= lse;
    }

    __aicore__ inline void SinkhornN4InRegs(F32 &m00,F32 &m01,F32 &m02,F32 &m03,
                                           F32 &m10,F32 &m11,F32 &m12,F32 &m13,
                                           F32 &m20,F32 &m21,F32 &m22,F32 &m23,
                                           F32 &m30,F32 &m31,F32 &m32,F32 &m33,
                                           I32 tmax)
    {
        // subtract global max
        F32 maxv = m00;
        #define UPD_MAX(v) if ((v) > maxv) maxv = (v)
        UPD_MAX(m01);UPD_MAX(m02);UPD_MAX(m03);
        UPD_MAX(m10);UPD_MAX(m11);UPD_MAX(m12);UPD_MAX(m13);
        UPD_MAX(m20);UPD_MAX(m21);UPD_MAX(m22);UPD_MAX(m23);
        UPD_MAX(m30);UPD_MAX(m31);UPD_MAX(m32);UPD_MAX(m33);
        #undef UPD_MAX
        m00-=maxv;m01-=maxv;m02-=maxv;m03-=maxv;
        m10-=maxv;m11-=maxv;m12-=maxv;m13-=maxv;
        m20-=maxv;m21-=maxv;m22-=maxv;m23-=maxv;
        m30-=maxv;m31-=maxv;m32-=maxv;m33-=maxv;

        for (I32 it = 0; it < tmax; ++it) {
            RowNorm4(m00,m01,m02,m03);
            RowNorm4(m10,m11,m12,m13);
            RowNorm4(m20,m21,m22,m23);
            RowNorm4(m30,m31,m32,m33);

            ColNorm4(m00,m10,m20,m30);
            ColNorm4(m01,m11,m21,m31);
            ColNorm4(m02,m12,m22,m32);
            ColNorm4(m03,m13,m23,m33);
        }
    }

    __aicore__ inline void ComputeResSinkhornN4_Staged(uint32_t bt, const AscendC::LocalTensor<F32>& xNorm, F32 aRes, I32 tmax)
    {
        AscendC::LocalTensor<F32> pr = phiResUb.Get<F32>();
        AscendC::LocalTensor<F32> br = bResUb.Get<F32>();

        F32 m00=0,m01=0,m02=0,m03=0;
        F32 m10=0,m11=0,m12=0,m13=0;
        F32 m20=0,m21=0,m22=0,m23=0;
        F32 m30=0,m31=0,m32=0,m33=0;

        // pr row-major: f*16 + ij
        for (uint32_t f = 0; f < F; ++f) {
            const F32 xv = xNorm.GetValue(f);
            const uint32_t base = f * 16u;

            m00 += xv * pr.GetValue(base + 0u);  m01 += xv * pr.GetValue(base + 1u);
            m02 += xv * pr.GetValue(base + 2u);  m03 += xv * pr.GetValue(base + 3u);
            m10 += xv * pr.GetValue(base + 4u);  m11 += xv * pr.GetValue(base + 5u);
            m12 += xv * pr.GetValue(base + 6u);  m13 += xv * pr.GetValue(base + 7u);
            m20 += xv * pr.GetValue(base + 8u);  m21 += xv * pr.GetValue(base + 9u);
            m22 += xv * pr.GetValue(base + 10u); m23 += xv * pr.GetValue(base + 11u);
            m30 += xv * pr.GetValue(base + 12u); m31 += xv * pr.GetValue(base + 13u);
            m32 += xv * pr.GetValue(base + 14u); m33 += xv * pr.GetValue(base + 15u);
        }

        // affine + bias
        m00 = aRes*m00 + br.GetValue(0);  m01 = aRes*m01 + br.GetValue(1);
        m02 = aRes*m02 + br.GetValue(2);  m03 = aRes*m03 + br.GetValue(3);
        m10 = aRes*m10 + br.GetValue(4);  m11 = aRes*m11 + br.GetValue(5);
        m12 = aRes*m12 + br.GetValue(6);  m13 = aRes*m13 + br.GetValue(7);
        m20 = aRes*m20 + br.GetValue(8);  m21 = aRes*m21 + br.GetValue(9);
        m22 = aRes*m22 + br.GetValue(10); m23 = aRes*m23 + br.GetValue(11);
        m30 = aRes*m30 + br.GetValue(12); m31 = aRes*m31 + br.GetValue(13);
        m32 = aRes*m32 + br.GetValue(14); m33 = aRes*m33 + br.GetValue(15);

        SinkhornN4InRegs(m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33,tmax);

        const uint64_t outBase = (uint64_t)bt * 16ull;
        outResGm.SetValue(outBase + 0,  ExpApprox(m00));
        outResGm.SetValue(outBase + 1,  ExpApprox(m01));
        outResGm.SetValue(outBase + 2,  ExpApprox(m02));
        outResGm.SetValue(outBase + 3,  ExpApprox(m03));
        outResGm.SetValue(outBase + 4,  ExpApprox(m10));
        outResGm.SetValue(outBase + 5,  ExpApprox(m11));
        outResGm.SetValue(outBase + 6,  ExpApprox(m12));
        outResGm.SetValue(outBase + 7,  ExpApprox(m13));
        outResGm.SetValue(outBase + 8,  ExpApprox(m20));
        outResGm.SetValue(outBase + 9,  ExpApprox(m21));
        outResGm.SetValue(outBase + 10, ExpApprox(m22));
        outResGm.SetValue(outBase + 11, ExpApprox(m23));
        outResGm.SetValue(outBase + 12, ExpApprox(m30));
        outResGm.SetValue(outBase + 13, ExpApprox(m31));
        outResGm.SetValue(outBase + 14, ExpApprox(m32));
        outResGm.SetValue(outBase + 15, ExpApprox(m33));
    }

    __aicore__ inline void ComputePrePostGenericScaled(uint32_t bt,
                                                       const AscendC::LocalTensor<F32>& xNorm,
                                                       F32 aPre, F32 aPost)
    {
        AscendC::LocalTensor<F32> bp = bPreUb.Get<F32>();
        AscendC::LocalTensor<F32> bo = bPostUb.Get<F32>();
        const uint64_t oBase = (uint64_t)bt * (uint64_t)N;

        for (uint32_t j = 0; j < N; ++j) {
            F32 ap = (F32)0.0f;
            F32 ao = (F32)0.0f;
            for (uint32_t f = 0; f < F; ++f) {
                const F32 xv = xNorm.GetValue(f);
                const uint64_t idx = (uint64_t)f * (uint64_t)N + (uint64_t)j;
                ap += xv * phiPreGm.GetValue(idx);
                ao += xv * phiPostGm.GetValue(idx);
            }
            F32 pre  = SigmoidApprox(aPre  * ap + bp.GetValue(j));
            F32 post = (F32)2.0f * SigmoidApprox(aPost * ao + bo.GetValue(j));
            outPreGm.SetValue(oBase + (uint64_t)j, pre);
            outPostGm.SetValue(oBase + (uint64_t)j, post);
        }
    }

    __aicore__ inline F32 DotXScaledPhi(uint32_t col,
                                       const AscendC::LocalTensor<F32>& xNorm,
                                       const AscendC::GlobalTensor<F32>& phi,
                                       uint32_t outDim)
    {
        F32 acc = (F32)0.0f;
        for (uint32_t f = 0; f < F; ++f) {
            const F32 x = xNorm.GetValue(f);
            const uint64_t pIdx = (uint64_t)f * (uint64_t)outDim + (uint64_t)col;
            acc += x * phi.GetValue(pIdx);
        }
        return acc;
    }

    __aicore__ inline void ComputeResSinkhornGenericScaled(uint32_t bt,
                                                          const AscendC::LocalTensor<F32>& xNorm,
                                                          F32 aRes, I32 tmax)
    {
        AscendC::LocalTensor<F32> m = resUb.Get<F32>();
        AscendC::LocalTensor<F32> br = bResUb.Get<F32>();

        F32 maxv = (F32)-1e30f;
        for (uint32_t ij = 0; ij < NN; ++ij) {
            const F32 dot = DotXScaledPhi(ij, xNorm, phiResGm, NN);
            const F32 v = aRes * dot + br.GetValue(ij);
            m.SetValue(ij, v);
            if (v > maxv) maxv = v;
        }
        for (uint32_t ij = NN; ij < NNpad; ++ij) m.SetValue(ij, (F32)0.0f);

        for (uint32_t ij = 0; ij < NN; ++ij) m.SetValue(ij, m.GetValue(ij) - maxv);

        for (I32 it = 0; it < tmax; ++it) {
            for (uint32_t r = 0; r < N; ++r) {
                const uint32_t rowBase = r * N;
                F32 rmax = m.GetValue(rowBase);
                for (uint32_t c = 1; c < N; ++c) {
                    const F32 v = m.GetValue(rowBase + c);
                    if (v > rmax) rmax = v;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t c = 0; c < N; ++c) sumExp += ExpApprox(m.GetValue(rowBase + c) - rmax);
                const F32 lse = rmax + LogApprox(sumExp);
                for (uint32_t c = 0; c < N; ++c) m.SetValue(rowBase + c, m.GetValue(rowBase + c) - lse);
            }

            for (uint32_t c = 0; c < N; ++c) {
                F32 cmax = m.GetValue(c);
                for (uint32_t r = 1; r < N; ++r) {
                    const F32 v = m.GetValue(r * N + c);
                    if (v > cmax) cmax = v;
                }
                F32 sumExp = (F32)0.0f;
                for (uint32_t r = 0; r < N; ++r) sumExp += ExpApprox(m.GetValue(r * N + c) - cmax);
                const F32 lse = cmax + LogApprox(sumExp);
                for (uint32_t r = 0; r < N; ++r) {
                    const uint32_t idx = r * N + c;
                    m.SetValue(idx, m.GetValue(idx) - lse);
                }
            }
        }

        const uint64_t outBase = (uint64_t)bt * (uint64_t)NN;
        for (uint32_t ij = 0; ij < NN; ++ij) outResGm.SetValue(outBase + (uint64_t)ij, ExpApprox(m.GetValue(ij)));
    }

private:
    AscendC::TPipe pipe;

    AscendC::GlobalTensor<F32> xGm;
    AscendC::GlobalTensor<F32> phiPreGm, phiPostGm, phiResGm;
    AscendC::GlobalTensor<F32> bPreGm, bPostGm, bResGm;
    AscendC::GlobalTensor<F32> aPreGm, aPostGm, aResGm;
    AscendC::GlobalTensor<I32> tmaxGm;
    AscendC::GlobalTensor<F32> epsGm;
    AscendC::GlobalTensor<F32> invFGm;

    AscendC::GlobalTensor<F32> outPreGm, outPostGm, outResGm;

    AscendC::TQue<AscendC::QuePosition::VECIN, 2> xInQ;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> bPreUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bPostUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bResUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> resUb;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> phiPreUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> phiPostUb;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> phiResUb;

    uint32_t B{}, T{}, N{}, C{}, BT{}, F{}, NN{}, Fpad{}, Npad{}, NNpad{};
    uint32_t btStart{}, btCount{};
};

extern "C" __global__ __aicore__ void mhc_projector_custom(
    GM_ADDR x_stream,
    GM_ADDR phi_pre, GM_ADDR phi_post, GM_ADDR phi_res,
    GM_ADDR b_pre, GM_ADDR b_post, GM_ADDR b_res,
    GM_ADDR alpha_pre, GM_ADDR alpha_post, GM_ADDR alpha_res,
    GM_ADDR tmax, GM_ADDR rmsnorm_eps,
    GM_ADDR invF,
    GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR h_res,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMhcProjectorCustom op;
    op.Init(x_stream,
            phi_pre, phi_post, phi_res,
            b_pre, b_post, b_res,
            alpha_pre, alpha_post, alpha_res,
            tmax, rmsnorm_eps,
            invF,
            h_pre, h_post, h_res,
            tiling_data.B, tiling_data.T, tiling_data.N, tiling_data.C,
            tiling_data.BT, tiling_data.F, tiling_data.NN, tiling_data.Fpad,
            tiling_data.Npad, tiling_data.NNpad);
    op.Process();
}
