
#include "kernel_operator.h"

static constexpr uint32_t kB = 2048;
static constexpr uint32_t kN = 100;
static constexpr uint32_t kD = 512;
static constexpr uint32_t kK = 32;
static constexpr uint32_t kDK = kD * kK; // 16384

class KernelNetVladNoGhostClustersCustom {
public:
    __aicore__ inline KernelNetVladNoGhostClustersCustom() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR clusters, GM_ADDR clusters2,
        GM_ADDR bn_weight, GM_ADDR bn_bias, GM_ADDR bn_mean, GM_ADDR bn_var,
        GM_ADDR y,
        uint32_t /*totalX*/, uint32_t /*totalClusters*/, uint32_t /*totalClusters2*/,
        uint32_t /*totalBnW*/, uint32_t /*totalBnB*/, uint32_t /*totalBnM*/, uint32_t /*totalBnV*/,
        uint32_t /*totalY*/,
        uint32_t B, uint32_t N, uint32_t D, uint32_t K,
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

        B_ = B; N_ = N; D_ = D; K_ = K;
        bnEps_ = bnEps;
        l2Eps_ = l2Eps;

        // UB budget kept modest to avoid runtime instability:
        // xVec(512) + logits(32) + tmp(64) + vlad(16384) + cRow(32) + bnPack(128)
        pipe.InitBuffer(bufXVec_,  kD * sizeof(float));
        pipe.InitBuffer(bufLogits_,kK * sizeof(float));
        pipe.InitBuffer(bufTmp_,   64 * sizeof(float));
        pipe.InitBuffer(bufVlad_,  kDK * sizeof(float));
        pipe.InitBuffer(bufCRow_,  kK * sizeof(float));
        pipe.InitBuffer(bufBnPack_,128 * sizeof(float)); // 4*K
    }

    __aicore__ inline float RsqrtSafeScalar(float v, float eps)
    {
        AscendC::LocalTensor<float> tmp = bufTmp_.Get<float>();
        tmp.SetValue(0, v + eps);
        AscendC::Sqrt(tmp, tmp, 1);
        AscendC::Reciprocal(tmp, tmp, 1);
        return tmp.GetValue(0);
    }

    __aicore__ inline void SoftmaxK32(AscendC::LocalTensor<float>& logits)
    {
        float maxv = -3.402823466e+38f;
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) {
            float v = logits.GetValue((int)k);
            if (v > maxv) maxv = v;
        }

        AscendC::LocalTensor<float> tmp = bufTmp_.Get<float>();
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) {
            tmp.SetValue((int)k, logits.GetValue((int)k) - maxv);
        }
        AscendC::Exp(tmp, tmp, kK);

        float sumExp = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) sumExp += tmp.GetValue((int)k);

        float invSum = 1.0f / (sumExp + l2Eps_);
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) {
            logits.SetValue((int)k, tmp.GetValue((int)k) * invSum);
        }
    }

    __aicore__ inline void ProcessOneBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> xVec   = bufXVec_.Get<float>();
        AscendC::LocalTensor<float> logits = bufLogits_.Get<float>();
        AscendC::LocalTensor<float> vlad   = bufVlad_.Get<float>();
        AscendC::LocalTensor<float> cRow   = bufCRow_.Get<float>();
        AscendC::LocalTensor<float> bnPack = bufBnPack_.Get<float>();

        // Pack BN params into UB once per batch: [w(32), b(32), mean(32), var(32)]
        AscendC::DataCopy(bnPack, bnWGm[0], kK);
        AscendC::DataCopy(bnPack[kK], bnBGm[0], kK);
        AscendC::DataCopy(bnPack[2 * kK], bnMeanGm[0], kK);
        AscendC::DataCopy(bnPack[3 * kK], bnVarGm[0], kK);

        // Precompute BN fold alpha/beta in registers
        float alpha[kK], beta[kK];
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) {
            float w = bnPack.GetValue((int)k);
            float bb = bnPack.GetValue((int)(kK + k));
            float m = bnPack.GetValue((int)(2 * kK + k));
            float v = bnPack.GetValue((int)(3 * kK + k));
            float invStd = RsqrtSafeScalar(v, bnEps_);
            float a = w * invStd;
            alpha[k] = a;
            beta[k] = bb - m * a;
        }

        // vlad UB zero
#pragma unroll
        for (uint32_t i = 0; i < kDK; ++i) vlad.SetValue((int)i, 0.0f);

        float aSum[kK];
#pragma unroll
        for (uint32_t k = 0; k < kK; ++k) aSum[k] = 0.0f;

        const uint64_t xBatchBase = (uint64_t)b * (uint64_t)kN * (uint64_t)kD;

        for (uint32_t n = 0; n < kN; ++n) {
            const uint64_t xBase = xBatchBase + (uint64_t)n * (uint64_t)kD;
            AscendC::DataCopy(xVec, xGm[(int)xBase], kD);

            // logits init
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) logits.SetValue((int)k, 0.0f);

            // Accumulate dot product: for each d, burst-load clusters[d,:] into UB
#pragma unroll
            for (uint32_t d = 0; d < kD; ++d) {
                float xv = xVec.GetValue((int)d);
                uint64_t cBase = (uint64_t)d * (uint64_t)kK;
                AscendC::DataCopy(cRow, clustersGm[(int)cBase], kK);
#pragma unroll
                for (uint32_t k = 0; k < kK; ++k) {
                    float cur = logits.GetValue((int)k);
                    cur += xv * cRow.GetValue((int)k);
                    logits.SetValue((int)k, cur);
                }
            }

            // BN fold
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                float v = logits.GetValue((int)k);
                logits.SetValue((int)k, v * alpha[k] + beta[k]);
            }

            SoftmaxK32(logits); // assignment in logits

#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) aSum[k] += logits.GetValue((int)k);

            // vlad[d,k] += a[k] * x[d]
            for (uint32_t d = 0; d < kD; ++d) {
                float xv = xVec.GetValue((int)d);
                uint32_t row = d * kK;
#pragma unroll
                for (uint32_t k = 0; k < kK; ++k) {
                    uint32_t idx = row + k;
                    float cur = vlad.GetValue((int)idx);
                    cur += logits.GetValue((int)k) * xv;
                    vlad.SetValue((int)idx, cur);
                }
            }
        }

        // vlad = vlad - aSum[k] * clusters2[d,k] (stream clusters2 from GM; only DK reads)
        for (uint32_t d = 0; d < kD; ++d) {
            uint32_t row = d * kK;
#pragma unroll
            for (uint32_t k = 0; k < kK; ++k) {
                uint32_t idx = row + k;
                float v = vlad.GetValue((int)idx);
                float ck = clusters2Gm.GetValue((int)idx); // [1,D,K] contiguous
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
        if (B_ != kB || N_ != kN || D_ != kD || K_ != kK) return;

        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bdim = (uint32_t)AscendC::GetBlockNum();

        for (uint32_t b = bid; b < B_; b += bdim) {
            ProcessOneBatch(b);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufXVec_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufLogits_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmp_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVlad_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCRow_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufBnPack_;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> clustersGm;
    AscendC::GlobalTensor<float> clusters2Gm;
    AscendC::GlobalTensor<float> bnWGm;
    AscendC::GlobalTensor<float> bnBGm;
    AscendC::GlobalTensor<float> bnMeanGm;
    AscendC::GlobalTensor<float> bnVarGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B_{0}, N_{0}, D_{0}, K_{0};
    float bnEps_{1.0e-5f};
    float l2Eps_{1.0e-12f};
};

extern "C" __global__ __aicore__ void net_vlad_no_ghost_clusters_custom(
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

    KernelNetVladNoGhostClustersCustom op;
    op.Init(x, clusters, clusters2,
            bn_weight, bn_bias, bn_mean, bn_var,
            y,
            tiling_data.totalX, tiling_data.totalClusters, tiling_data.totalClusters2,
            tiling_data.totalBnW, tiling_data.totalBnB, tiling_data.totalBnM, tiling_data.totalBnV,
            tiling_data.totalY,
            tiling_data.B, tiling_data.N, tiling_data.D, tiling_data.K,
            tiling_data.bnEps, tiling_data.l2Eps);
    op.Process();
}
