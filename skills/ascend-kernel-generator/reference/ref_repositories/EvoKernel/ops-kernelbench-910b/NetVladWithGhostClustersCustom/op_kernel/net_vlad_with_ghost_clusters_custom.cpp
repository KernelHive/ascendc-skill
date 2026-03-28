
#include "kernel_operator.h"

static constexpr uint32_t kB = 2048;
static constexpr uint32_t kN = 100;
static constexpr uint32_t kD = 512;
static constexpr uint32_t kK = 32;
static constexpr uint32_t kG = 16;
static constexpr uint32_t kKall = kK + kG; // 48
static constexpr uint32_t kDK = kD * kK;   // 16384

// Kall tiling for matmul/softmax: 48 = 3 * 16
static constexpr uint32_t kKtile = 16;
static constexpr uint32_t kNt = kKall / kKtile; // 3

class KernelNetVladWithGhostClustersCustom {
public:
    __aicore__ inline KernelNetVladWithGhostClustersCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clusters, GM_ADDR clusters2,
        GM_ADDR bn_weight, GM_ADDR bn_bias, GM_ADDR bn_mean, GM_ADDR bn_var,
        GM_ADDR y,
        uint32_t /*totalX*/, uint32_t /*totalClusters*/, uint32_t /*totalClusters2*/,
        uint32_t /*totalBnW*/, uint32_t /*totalBnB*/, uint32_t /*totalBnM*/, uint32_t /*totalBnV*/,
        uint32_t /*totalY*/,
        uint32_t B, uint32_t N, uint32_t D, uint32_t K, uint32_t Kall,
        float bnEps, float l2Eps)
    {
        xGm.SetGlobalBuffer((__gm__ float*)x);
        clustersGm.SetGlobalBuffer((__gm__ float*)clusters);
        clusters2Gm.SetGlobalBuffer((__gm__ float*)clusters2);
        bnWGm.SetGlobalBuffer((__gm__ float*)bn_weight);
        bnBGm.SetGlobalBuffer((__gm__ float*)bn_bias);
        bnMeanGm.SetGlobalBuffer((__gm__ float*)bn_mean);
        bnVarGm.SetGlobalBuffer((__gm__ float*)bn_var);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        B_ = B; N_ = N; D_ = D; K_ = K; Kall_ = Kall;
        bnEps_ = bnEps;
        l2Eps_ = l2Eps;

        // UB:
        // xVec: 512
        // wTile: D*Ktile = 512*16 = 8192
        // logits: 48, exp: 48
        // tmp: 64 (for rsqrt scalar, reductions)
        // vlad: 16384
        // bnPack: 4*Kall = 192
        pipe.InitBuffer(bufXVec_,   kD * sizeof(float));
        pipe.InitBuffer(bufWTile_,  (kD * kKtile) * sizeof(float));
        pipe.InitBuffer(bufLogits_, kKall * sizeof(float));
        pipe.InitBuffer(bufExp_,    kKall * sizeof(float));
        pipe.InitBuffer(bufTmp_,    64 * sizeof(float));
        pipe.InitBuffer(bufVlad_,   kDK * sizeof(float));
        pipe.InitBuffer(bufBnPack_, (4 * kKall) * sizeof(float));
    }

    __aicore__ inline float RsqrtSafeScalar(float v, float eps)
    {
        AscendC::LocalTensor<float> tmp = bufTmp_.Get<float>();
        tmp.SetValue(0, v + eps);
        AscendC::Sqrt(tmp, tmp, 1);
        AscendC::Reciprocal(tmp, tmp, 1);
        return tmp.GetValue(0);
    }

    __aicore__ inline void Softmax48Chunk16(AscendC::LocalTensor<float>& logits)
    {
        // Computes softmax over 48 by: (1) compute per-chunk max, (2) global max,
        // (3) exp in each chunk, (4) global sum, (5) scale.
        AscendC::LocalTensor<float> expv = bufExp_.Get<float>();

        float maxChunk[kNt];
#pragma unroll
        for (uint32_t t = 0; t < kNt; ++t) {
            float mv = -3.402823466e+38f;
#pragma unroll
            for (uint32_t i = 0; i < kKtile; ++i) {
                float v = logits.GetValue((int)(t * kKtile + i));
                if (v > mv) mv = v;
            }
            maxChunk[t] = mv;
        }
        float maxAll = maxChunk[0];
#pragma unroll
        for (uint32_t t = 1; t < kNt; ++t) if (maxChunk[t] > maxAll) maxAll = maxChunk[t];

        // exp per chunk using 16-wide vector exp
        AscendC::LocalTensor<float> tmp16 = bufTmp_.Get<float>();
#pragma unroll
        for (uint32_t t = 0; t < kNt; ++t) {
#pragma unroll
            for (uint32_t i = 0; i < kKtile; ++i) {
                tmp16.SetValue((int)i, logits.GetValue((int)(t * kKtile + i)) - maxAll);
            }
            AscendC::Exp(tmp16, tmp16, kKtile);
#pragma unroll
            for (uint32_t i = 0; i < kKtile; ++i) {
                expv.SetValue((int)(t * kKtile + i), tmp16.GetValue((int)i));
            }
        }

        float sumAll = 0.0f;
#pragma unroll
        for (uint32_t t = 0; t < kNt; ++t) {
#pragma unroll
            for (uint32_t i = 0; i < kKtile; ++i) {
                sumAll += expv.GetValue((int)(t * kKtile + i));
            }
        }
        float invSum = 1.0f / (sumAll + l2Eps_);

#pragma unroll
        for (uint32_t k = 0; k < kKall; ++k) {
            logits.SetValue((int)k, expv.GetValue((int)k) * invSum);
        }
    }

    __aicore__ inline void ProcessOneBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> xVec   = bufXVec_.Get<float>();
        AscendC::LocalTensor<float> wTile  = bufWTile_.Get<float>();
        AscendC::LocalTensor<float> logits = bufLogits_.Get<float>();
        AscendC::LocalTensor<float> vlad   = bufVlad_.Get<float>();
        AscendC::LocalTensor<float> bnPack = bufBnPack_.Get<float>();

        // BN pack in UB once per batch (small and reused across N)
        AscendC::DataCopy(bnPack,              bnWGm[0],    kKall);
        AscendC::DataCopy(bnPack[kKall],       bnBGm[0],    kKall);
        AscendC::DataCopy(bnPack[2 * kKall],   bnMeanGm[0], kKall);
        AscendC::DataCopy(bnPack[3 * kKall],   bnVarGm[0],  kKall);

        float alpha[kKall], beta[kKall];
#pragma unroll
        for (uint32_t k = 0; k < kKall; ++k) {
            float w  = bnPack.GetValue((int)k);
            float bb = bnPack.GetValue((int)(kKall + k));
            float m  = bnPack.GetValue((int)(2 * kKall + k));
            float v  = bnPack.GetValue((int)(3 * kKall + k));
            float invStd = RsqrtSafeScalar(v, bnEps_);
            float a = w * invStd;
            alpha[k] = a;
            beta[k]  = bb - m * a;
        }

        // zero vlad
#pragma unroll
        for (uint32_t i = 0; i < kDK; ++i) vlad.SetValue((int)i, 0.0f);

        float aSum[kK];
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) aSum[k] = 0.0f;

        const uint64_t xBatchBase = (uint64_t)b * (uint64_t)kN * (uint64_t)kD;

        for (uint32_t n = 0; n < kN; ++n) {
            const uint64_t xBase = xBatchBase + (uint64_t)n * (uint64_t)kD;
            AscendC::DataCopy(xVec, xGm[(int)xBase], kD);

            // compute logits in Ktile chunks; each chunk loads [D,Ktile] once.
#pragma unroll
            for (uint32_t k = 0; k < kKall; ++k) logits.SetValue((int)k, 0.0f);

#pragma unroll
            for (uint32_t t = 0; t < kNt; ++t) {
                uint32_t k0 = t * kKtile;

                // Load clusters[:, k0:k0+16] into UB as a contiguous 2D slab
                // clusters layout is [D, Kall], row-major contiguous, so each row has Kall contiguous.
                // We'll DataCopy row-by-row into wTile; 512 copies of 16 floats (still much fewer
                // calls than prior 512 copies of 48 within inner loop), and importantly each copy
                // is small but we only do it 3 times per (b,n) *per row*; to reduce call overhead,
                // we instead copy the whole slab with one 8192-float copy since [D*Ktile] is contiguous
                // in GM when taking a full Ktile for every row? It is not contiguous across rows due to stride Kall.
                // Therefore, we do row-by-row copies but only for Ktile=16 (smaller traffic) and 3 chunks.
#pragma unroll
                for (uint32_t d = 0; d < kD; ++d) {
                    uint64_t cBase = (uint64_t)d * (uint64_t)kKall + (uint64_t)k0;
                    AscendC::DataCopy(wTile[(int)(d * kKtile)], clustersGm[(int)cBase], kKtile);
                }

                float acc16[kKtile];
#pragma unroll
                for (uint32_t i = 0; i < kKtile; ++i) acc16[i] = 0.0f;

                // Dot: accumulate 16 logits with one pass over D
#pragma unroll
                for (uint32_t d = 0; d < kD; ++d) {
                    float xv = xVec.GetValue((int)d);
                    uint32_t wrow = d * kKtile;
#pragma unroll
                    for (uint32_t i = 0; i < kKtile; ++i) {
                        acc16[i] += xv * wTile.GetValue((int)(wrow + i));
                    }
                }

#pragma unroll
                for (uint32_t i = 0; i < kKtile; ++i) {
                    uint32_t kidx = k0 + i;
                    float v = acc16[i] * alpha[kidx] + beta[kidx];
                    logits.SetValue((int)kidx, v);
                }
            }

            Softmax48Chunk16(logits);

            float assignK[kK];
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                float pk = logits.GetValue((int)k);
                assignK[k] = pk;
                aSum[k] += pk;
            }

            // vlad[d,k] += assignK[k] * x[d]
            for (uint32_t d = 0; d < kD; ++d) {
                float xv = xVec.GetValue((int)d);
                uint32_t row = d * kK;
#pragma unroll
                for (uint32_t k = 0; k < kK; ++k) {
                    uint32_t idx = row + k;
                    float cur = vlad.GetValue((int)idx);
                    cur += assignK[k] * xv;
                    vlad.SetValue((int)idx, cur);
                }
            }
        }

        // center: vlad -= aSum[k] * clusters2[d,k]
        for (uint32_t d = 0; d < kD; ++d) {
            uint32_t row = d * kK;
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                uint32_t idx = row + k;
                float v = vlad.GetValue((int)idx);
                float ck = clusters2Gm.GetValue((int)idx);
                v -= aSum[k] * ck;
                vlad.SetValue((int)idx, v);
            }
        }

        // intra L2 norm per cluster
        float invIntra[kK];
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) invIntra[k] = 0.0f;

        for (uint32_t d = 0; d < kD; ++d) {
            uint32_t row = d * kK;
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                float v = vlad.GetValue((int)(row + k));
                invIntra[k] += v * v;
            }
        }
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) invIntra[k] = RsqrtSafeScalar(invIntra[k], l2Eps_);

        for (uint32_t d = 0; d < kD; ++d) {
            uint32_t row = d * kK;
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                uint32_t idx = row + k;
                float v = vlad.GetValue((int)idx) * invIntra[k];
                vlad.SetValue((int)idx, v);
            }
        }

        // final L2 norm over DK
        float s2all = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kDK; ++i) {
            float v = vlad.GetValue((int)i);
            s2all += v * v;
        }
        float invAll = RsqrtSafeScalar(s2all, l2Eps_);

        const uint64_t yBatchBase = (uint64_t)b * (uint64_t)kDK;
#pragma unroll
        for (uint32_t i = 0; i < kDK; ++i) {
            float v = vlad.GetValue((int)i) * invAll;
            yGm.SetValue((int)(yBatchBase + i), v);
        }
    }

    __aicore__ inline void Process()
    {
        if (B_ != kB || N_ != kN || D_ != kD || K_ != kK || Kall_ != kKall) return;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = (uint32_t)AscendC::GetBlockNum();

        for (uint32_t b = bid; b < B_; b += bdim) {
            ProcessOneBatch(b);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXVec_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufWTile_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufLogits_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufExp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVlad_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufBnPack_;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> clustersGm;
    AscendC::GlobalTensor<float> clusters2Gm;
    AscendC::GlobalTensor<float> bnWGm;
    AscendC::GlobalTensor<float> bnBGm;
    AscendC::GlobalTensor<float> bnMeanGm;
    AscendC::GlobalTensor<float> bnVarGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B_{0}, N_{0}, D_{0}, K_{0}, Kall_{0};
    float bnEps_{1.0e-5f};
    float l2Eps_{1.0e-12f};
};

extern "C" __global__ __aicore__ void net_vlad_with_ghost_clusters_custom(
    GM_ADDR x,
    GM_ADDR clusters,
    GM_ADDR clusters2,
    GM_ADDR bn_weight,
    GM_ADDR bn_bias,
    GM_ADDR bn_mean,
    GM_ADDR bn_var,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelNetVladWithGhostClustersCustom op;
    op.Init(x, clusters, clusters2,
            bn_weight, bn_bias, bn_mean, bn_var,
            y,
            tiling_data.totalX, tiling_data.totalClusters, tiling_data.totalClusters2,
            tiling_data.totalBnW, tiling_data.totalBnB, tiling_data.totalBnM, tiling_data.totalBnV,
            tiling_data.totalY,
            tiling_data.B, tiling_data.N, tiling_data.D, tiling_data.K, tiling_data.Kall,
            tiling_data.bnEps, tiling_data.l2Eps);
    op.Process();
}
