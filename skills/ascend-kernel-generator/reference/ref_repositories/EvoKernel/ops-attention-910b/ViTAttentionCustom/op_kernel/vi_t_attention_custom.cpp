
#include "kernel_operator.h"

class KernelViTAttentionCustom {
public:
    __aicore__ inline KernelViTAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                               GM_ADDR scale,
                               GM_ADDR proj_w, GM_ADDR proj_b,
                               GM_ADDR y,
                               uint32_t bs, uint32_t heads, uint32_t nq, uint32_t d, uint32_t c,
                               uint32_t ocTile, uint32_t icTile, uint32_t kvStage)
    {
        this->bs = bs;
        this->heads = heads;
        this->nq = nq;
        this->d = d;
        this->c = c;
        this->ocTile = ocTile;
        this->icTile = icTile;
        this->kvStage = kvStage;

        this->coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        this->coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());

        qGm.SetGlobalBuffer((__gm__ float*)q, static_cast<uint64_t>(bs) * heads * nq * d);
        kGm.SetGlobalBuffer((__gm__ float*)k, static_cast<uint64_t>(bs) * heads * nq * d);
        vGm.SetGlobalBuffer((__gm__ float*)v, static_cast<uint64_t>(bs) * heads * nq * d);

        scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);
        wGm.SetGlobalBuffer((__gm__ float*)proj_w, static_cast<uint64_t>(c) * c);
        bGm.SetGlobalBuffer((__gm__ float*)proj_b, static_cast<uint64_t>(c));
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(bs) * nq * c);

        pipe.InitBuffer(qBuf, d * sizeof(float));
        pipe.InitBuffer(vAccBuf, d * sizeof(float));
        pipe.InitBuffer(expBuf, 1 * sizeof(float));

        // Stage K+V per head: 2*nq*d floats (only enabled when nq<=64)
        pipe.InitBuffer(kvBuf, 2u * nq * d * sizeof(float));

        pipe.InitBuffer(outTokBuf, c * sizeof(float));

        // projection buffers
        pipe.InitBuffer(inTileBuf, icTile * sizeof(float));
        pipe.InitBuffer(wTileBuf, ocTile * icTile * sizeof(float));
        pipe.InitBuffer(accTileBuf, ocTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t total = bs * nq;
        for (uint32_t linear = coreIdx; linear < total; linear += coreNum) {
            uint32_t b = linear / nq;
            uint32_t qIdx = linear - b * nq;
            ComputeTokenFused(b, qIdx);
        }
    }

private:
    __aicore__ inline uint64_t QKVIDX(uint32_t b, uint32_t h, uint32_t n, uint32_t j) const
    {
        return ((static_cast<uint64_t>(b) * heads + h) * nq + n) * d + j;
    }
    __aicore__ inline uint64_t YIDX(uint32_t b, uint32_t n, uint32_t j) const
    {
        return (static_cast<uint64_t>(b) * nq + n) * c + j;
    }

    __aicore__ inline void StageKV(uint32_t b, uint32_t h, AscendC::LocalTensor<float>& kv)
    {
        // kv layout: [0 : nq*d) => K, [nq*d : 2*nq*d) => V
        uint32_t kd = nq * d;
        for (uint32_t n = 0; n < nq; ++n) {
            AscendC::DataCopy(kv[n * d], kGm[QKVIDX(b, h, n, 0)], d);
            AscendC::DataCopy(kv[kd + n * d], vGm[QKVIDX(b, h, n, 0)], d);
        }
    }

    __aicore__ inline float DotQK_Staged(const AscendC::LocalTensor<float>& qLocal,
                                        const AscendC::LocalTensor<float>& kv,
                                        uint32_t kIdx) const
    {
        float acc = 0.0f;
        uint32_t base = kIdx * d;
        // small manual unroll for d=64 typical; works for general d too
        uint32_t j = 0;
        for (; j + 3 < d; j += 4) {
            acc += qLocal.GetValue(j)     * kv.GetValue(base + j);
            acc += qLocal.GetValue(j + 1) * kv.GetValue(base + j + 1);
            acc += qLocal.GetValue(j + 2) * kv.GetValue(base + j + 2);
            acc += qLocal.GetValue(j + 3) * kv.GetValue(base + j + 3);
        }
        for (; j < d; ++j) {
            acc += qLocal.GetValue(j) * kv.GetValue(base + j);
        }
        return acc;
    }

    __aicore__ inline void AccumWeightedV_Staged(float w,
                                                const AscendC::LocalTensor<float>& kv,
                                                uint32_t kIdx,
                                                AscendC::LocalTensor<float>& vAcc)
    {
        uint32_t kd = nq * d;
        uint32_t base = kd + kIdx * d;
        // reduce temporaries: vAcc[j] += w * v[j]
        uint32_t j = 0;
        for (; j + 3 < d; j += 4) {
            vAcc.SetValue(j,     vAcc.GetValue(j)     + w * kv.GetValue(base + j));
            vAcc.SetValue(j + 1, vAcc.GetValue(j + 1) + w * kv.GetValue(base + j + 1));
            vAcc.SetValue(j + 2, vAcc.GetValue(j + 2) + w * kv.GetValue(base + j + 2));
            vAcc.SetValue(j + 3, vAcc.GetValue(j + 3) + w * kv.GetValue(base + j + 3));
        }
        for (; j < d; ++j) {
            vAcc.SetValue(j, vAcc.GetValue(j) + w * kv.GetValue(base + j));
        }
    }

    __aicore__ inline void ComputeTokenFused(uint32_t b, uint32_t qIdx)
    {
        AscendC::LocalTensor<float> qLocal = qBuf.Get<float>();
        AscendC::LocalTensor<float> vAcc = vAccBuf.Get<float>();
        AscendC::LocalTensor<float> outTok = outTokBuf.Get<float>();
        AscendC::LocalTensor<float> kv = kvBuf.Get<float>();
        AscendC::LocalTensor<float> exp1 = expBuf.Get<float>();

        float scale = scaleGm.GetValue(0);

        // build outTok (C) by iterating heads
        for (uint32_t h = 0; h < heads; ++h) {
            bool staged = (kvStage != 0);
            if (staged) {
                StageKV(b, h, kv);
            }

            AscendC::DataCopy(qLocal, qGm[QKVIDX(b, h, qIdx, 0)], d);

            // pass1: max
            float maxv = -3.402823466e+38f;
            for (uint32_t kIdx = 0; kIdx < nq; ++kIdx) {
                float s;
                if (staged) {
                    s = DotQK_Staged(qLocal, kv, kIdx) * scale;
                } else {
                    // fallback scalar GM reads (rare when nq>64)
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < d; ++j) {
                        acc += qLocal.GetValue(j) * kGm.GetValue(QKVIDX(b, h, kIdx, j));
                    }
                    s = acc * scale;
                }
                if (s > maxv) maxv = s;
            }

            // pass2: sumexp + accumulate V
            AscendC::Duplicate(vAcc, 0.0f, static_cast<int32_t>(d));
            float sumexp = 0.0f;
            for (uint32_t kIdx = 0; kIdx < nq; ++kIdx) {
                float s;
                if (staged) {
                    s = DotQK_Staged(qLocal, kv, kIdx) * scale;
                } else {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < d; ++j) {
                        acc += qLocal.GetValue(j) * kGm.GetValue(QKVIDX(b, h, kIdx, j));
                    }
                    s = acc * scale;
                }

                float x = s - maxv;
                exp1.SetValue(0, x);
                AscendC::Exp(exp1, exp1, 1);
                float e = exp1.GetValue(0);
                sumexp += e;

                if (staged) {
                    AccumWeightedV_Staged(e, kv, kIdx, vAcc);
                } else {
                    // fallback: copy V row then mul+add
                    AscendC::LocalTensor<float> tmp = qLocal; // reuse qLocal as temp buffer (q will be reloaded next head anyway)
                    AscendC::DataCopy(tmp, vGm[QKVIDX(b, h, kIdx, 0)], d);
                    AscendC::Muls(tmp, tmp, e, static_cast<int32_t>(d));
                    AscendC::Add(vAcc, vAcc, tmp, static_cast<int32_t>(d));
                }
            }

            float invSum = (sumexp > 0.0f) ? (1.0f / sumexp) : 0.0f;
            AscendC::Muls(vAcc, vAcc, invSum, static_cast<int32_t>(d));

            uint32_t off = h * d;
            AscendC::DataCopy(outTok[off], vAcc, d);
        }

        ProjectToken(b, qIdx, outTok);
    }

    __aicore__ inline void ProjectToken(uint32_t b, uint32_t qIdx, const AscendC::LocalTensor<float>& outTok)
    {
        AscendC::LocalTensor<float> inTile = inTileBuf.Get<float>();
        AscendC::LocalTensor<float> wTile = wTileBuf.Get<float>();
        AscendC::LocalTensor<float> accTile = accTileBuf.Get<float>();

        uint32_t ocTiles = (c + ocTile - 1) / ocTile;
        uint32_t icTiles = (c + icTile - 1) / icTile;

        for (uint32_t oct = 0; oct < ocTiles; ++oct) {
            uint32_t oc0 = oct * ocTile;
            uint32_t ocLen = (oc0 + ocTile <= c) ? ocTile : (c - oc0);

            for (uint32_t o = 0; o < ocLen; ++o) {
                accTile.SetValue(o, bGm.GetValue(oc0 + o));
            }

            for (uint32_t ict = 0; ict < icTiles; ++ict) {
                uint32_t ic0 = ict * icTile;
                uint32_t icLen = (ic0 + icTile <= c) ? icTile : (c - ic0);

                for (uint32_t i = 0; i < icLen; ++i) {
                    inTile.SetValue(i, outTok.GetValue(ic0 + i));
                }

                // preload weight block [ocLen, icLen] contiguously per row to maximize MTE use
                for (uint32_t o = 0; o < ocLen; ++o) {
                    uint64_t wBase = static_cast<uint64_t>(oc0 + o) * c + ic0;
                    AscendC::DataCopy(wTile[o * icTile], wGm[wBase], icLen);
                }

                for (uint32_t o = 0; o < ocLen; ++o) {
                    float acc = accTile.GetValue(o);
                    uint32_t wOff = o * icTile;
                    uint32_t i = 0;
                    for (; i + 3 < icLen; i += 4) {
                        acc += inTile.GetValue(i)     * wTile.GetValue(wOff + i);
                        acc += inTile.GetValue(i + 1) * wTile.GetValue(wOff + i + 1);
                        acc += inTile.GetValue(i + 2) * wTile.GetValue(wOff + i + 2);
                        acc += inTile.GetValue(i + 3) * wTile.GetValue(wOff + i + 3);
                    }
                    for (; i < icLen; ++i) {
                        acc += inTile.GetValue(i) * wTile.GetValue(wOff + i);
                    }
                    accTile.SetValue(o, acc);
                }
            }

            for (uint32_t o = 0; o < ocLen; ++o) {
                yGm.SetValue(YIDX(b, qIdx, oc0 + o), accTile.GetValue(o));
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> qBuf, vAccBuf, expBuf;
    AscendC::TBuf<> kvBuf;
    AscendC::TBuf<> outTokBuf;
    AscendC::TBuf<> inTileBuf, wTileBuf, accTileBuf;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, scaleGm, wGm, bGm, yGm;

    uint32_t bs, heads, nq, d, c;
    uint32_t ocTile, icTile, kvStage;
    uint32_t coreIdx, coreNum;
};

extern "C" __global__ __aicore__ void vi_t_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR scale,
    GM_ADDR proj_weight, GM_ADDR proj_bias,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelViTAttentionCustom op;
    op.Init(q, k, v, scale, proj_weight, proj_bias, y,
            tiling_data.bs, tiling_data.heads, tiling_data.nq, tiling_data.d, tiling_data.c,
            tiling_data.ocTile, tiling_data.icTile, tiling_data.kvStage);
    op.Process();
}
