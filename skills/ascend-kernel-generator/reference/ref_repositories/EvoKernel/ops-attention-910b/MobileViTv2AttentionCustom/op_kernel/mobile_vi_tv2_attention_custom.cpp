
#include "kernel_operator.h"

class KernelMobileViTv2AttentionCustom {
public:
    __aicore__ inline KernelMobileViTv2AttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR i, GM_ADDR k, GM_ADDR v, GM_ADDR w_o, GM_ADDR b_o, GM_ADDR y,
                               uint32_t bs, uint32_t nq, uint32_t d,
                               uint32_t qTile, uint32_t dTile, uint32_t doTile)
    {
        this->bs = bs;
        this->nq = nq;
        this->d = d;
        this->qTile = qTile;
        this->dTile = dTile;
        this->doTile = doTile;

        iGm.SetGlobalBuffer((__gm__ float*)i, static_cast<uint64_t>(bs) * nq);          // i: [bs*nq]
        kGm.SetGlobalBuffer((__gm__ float*)k, static_cast<uint64_t>(bs) * nq * d);
        vGm.SetGlobalBuffer((__gm__ float*)v, static_cast<uint64_t>(bs) * nq * d);
        wOGm.SetGlobalBuffer((__gm__ float*)w_o, static_cast<uint64_t>(d) * d);
        bOGm.SetGlobalBuffer((__gm__ float*)b_o, static_cast<uint64_t>(d));
        yGm.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(bs) * nq * d);

        // Block mapping: each block owns exactly one (b, doGroup)
        uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t doGroups = (d + doTile - 1) / doTile;
        uint32_t work = bs * doGroups;

        if (blk >= work) {
            valid = false;
            bIdx = 0; gIdx = 0;
            return;
        }
        valid = true;
        bIdx = blk / doGroups;
        gIdx = blk - bIdx * doGroups;

        // UB: cache full i[b,:] and softmax weights (nq small)
        pipe.InitBuffer(iBuf, nq * sizeof(float));
        pipe.InitBuffer(wBuf, nq * sizeof(float));
        pipe.InitBuffer(expBuf, nq * sizeof(float));

        // context vector
        pipe.InitBuffer(ctxBuf, d * sizeof(float));

        // tiles for K/V and ctx slice
        pipe.InitBuffer(kTileBuf, dTile * sizeof(float));
        pipe.InitBuffer(vTileBuf, dTile * sizeof(float));
        pipe.InitBuffer(ctxTileBuf, dTile * sizeof(float));

        // W panel for this doGroup: doTile*dTile and scaled variant
        pipe.InitBuffer(wPanelBuf, doTile * dTile * sizeof(float));
        pipe.InitBuffer(wsPanelBuf, doTile * dTile * sizeof(float));

        // output panel: qTile*doTile (qTile is nq by default)
        pipe.InitBuffer(outPanelBuf, qTile * doTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!valid) return;
        ComputeSoftmaxAndCtx();     // per (bIdx) but recomputed per doGroup (no GM workspace races)
        ComputeOutGroupAndStore();  // per (bIdx, gIdx)
    }

private:
    __aicore__ inline void ComputeSoftmaxAndCtx()
    {
        AscendC::LocalTensor<float> iLocal = iBuf.Get<float>();
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> eLocal = expBuf.Get<float>();
        AscendC::LocalTensor<float> ctxLocal = ctxBuf.Get<float>();

        // Load i[b,:] once
        uint64_t iBase = static_cast<uint64_t>(bIdx) * nq;
        AscendC::DataCopy(iLocal, iGm[iBase], nq);

        // scalar reduce max (nq=49 small)
        float maxv = -3.402823466e+38f;
        for (uint32_t q = 0; q < nq; ++q) {
            float vv = iLocal.GetValue(q);
            if (vv > maxv) maxv = vv;
        }

        // iLocal = iLocal - maxv (scalar loop, nq small)
        for (uint32_t q = 0; q < nq; ++q) {
            iLocal.SetValue(q, iLocal.GetValue(q) - maxv);
        }

        // exp and sum
        AscendC::Exp(eLocal, iLocal, static_cast<int32_t>(nq));
        float sumexp = 0.0f;
        for (uint32_t q = 0; q < nq; ++q) sumexp += eLocal.GetValue(q);
        float invSum = 1.0f / sumexp;

        // materialize weights w[q]
        for (uint32_t q = 0; q < nq; ++q) {
            wLocal.SetValue(q, eLocal.GetValue(q) * invSum);
        }

        // ctx[d] = sum_q w[q] * k[b,q,:]
        AscendC::Duplicate(ctxLocal, 0.0f, static_cast<int32_t>(d));

        AscendC::LocalTensor<float> kTile = kTileBuf.Get<float>();
        uint32_t dTiles = (d + dTile - 1) / dTile;

        for (uint32_t q = 0; q < nq; ++q) {
            float wq = wLocal.GetValue(q);
            uint64_t kRowBase = (static_cast<uint64_t>(bIdx) * nq + q) * d;
            for (uint32_t dt = 0; dt < dTiles; ++dt) {
                uint32_t d0 = dt * dTile;
                uint32_t dlen = (d0 + dTile <= d) ? dTile : (d - d0);

                AscendC::DataCopy(kTile, kGm[kRowBase + d0], dlen);
                // ctxLocal[d0:d0+dlen] += wq * kTile (keep minimal UB: scalar FMA loop)
                for (uint32_t j = 0; j < dlen; ++j) {
                    float prev = ctxLocal.GetValue(d0 + j);
                    ctxLocal.SetValue(d0 + j, prev + wq * kTile.GetValue(j));
                }
            }
        }
    }

    __aicore__ inline void ComputeOutGroupAndStore()
    {
        AscendC::LocalTensor<float> ctxLocal = ctxBuf.Get<float>();
        AscendC::LocalTensor<float> vTile = vTileBuf.Get<float>();
        AscendC::LocalTensor<float> ctxTile = ctxTileBuf.Get<float>();
        AscendC::LocalTensor<float> wPanel = wPanelBuf.Get<float>();
        AscendC::LocalTensor<float> wsPanel = wsPanelBuf.Get<float>();
        AscendC::LocalTensor<float> outPanel = outPanelBuf.Get<float>();

        uint32_t do0 = gIdx * doTile;
        uint32_t dolen = (do0 + doTile <= d) ? doTile : (d - do0);

        // Initialize outPanel with bias: out[q,oo]=bias[do0+oo]
        for (uint32_t oo = 0; oo < dolen; ++oo) {
            float bias = bOGm.GetValue(do0 + oo); // small scalar read
            for (uint32_t q = 0; q < nq; ++q) {
                outPanel.SetValue(q * doTile + oo, bias);
            }
        }
        // zero-fill tail oo for alignment safety (avoid stale UB values)
        for (uint32_t oo = dolen; oo < doTile; ++oo) {
            for (uint32_t q = 0; q < nq; ++q) {
                outPanel.SetValue(q * doTile + oo, 0.0f);
            }
        }

        uint32_t dTiles = (d + dTile - 1) / dTile;

        for (uint32_t dt = 0; dt < dTiles; ++dt) {
            uint32_t d0 = dt * dTile;
            uint32_t dlen = (d0 + dTile <= d) ? dTile : (d - d0);

            // ctx slice into ctxTile and keep a contiguous tile for vector ops
            for (uint32_t j = 0; j < dlen; ++j) ctxTile.SetValue(j, ctxLocal.GetValue(d0 + j));
            for (uint32_t j = dlen; j < dTile; ++j) ctxTile.SetValue(j, 0.0f);

            // Load W panel rows for this doGroup: each row is contiguous in GM -> DataCopy
            for (uint32_t oo = 0; oo < dolen; ++oo) {
                uint64_t wBase = static_cast<uint64_t>(do0 + oo) * d + d0;
                AscendC::DataCopy(wPanel[oo * dTile], wOGm[wBase], dlen);
                // pad tail for safe Mul
                for (uint32_t j = dlen; j < dTile; ++j) {
                    wPanel.SetValue(oo * dTile + j, 0.0f);
                }
            }
            // pad tail rows
            for (uint32_t oo = dolen; oo < doTile; ++oo) {
                for (uint32_t j = 0; j < dTile; ++j) wPanel.SetValue(oo * dTile + j, 0.0f);
            }

            // wsPanel[oo,:] = wPanel[oo,:] * ctxTile[:] (vector Mul per row)
            for (uint32_t oo = 0; oo < doTile; ++oo) {
                AscendC::Mul(wsPanel[oo * dTile], wPanel[oo * dTile], ctxTile, static_cast<int32_t>(dTile));
            }

            // For each q: load v slice once, scale in-place by ctxTile (vector Mul), then dot with each wsPanel row
            for (uint32_t q = 0; q < nq; ++q) {
                uint64_t vBase = (static_cast<uint64_t>(bIdx) * nq + q) * d + d0;
                AscendC::DataCopy(vTile, vGm[vBase], dlen);
                for (uint32_t j = dlen; j < dTile; ++j) vTile.SetValue(j, 0.0f);

                // vTile *= ctxTile (elementwise)
                AscendC::Mul(vTile, vTile, ctxTile, static_cast<int32_t>(dTile));

                // accumulate: out[q,oo] += sum_j vTile[j] * w_o_row[j] (wsPanel already has w*ctx, but v is already scaled by ctx;
                // keep correctness by using original wsPanel = w*ctx and vTile = v*ctx -> would apply ctx twice.
                // Therefore use wPanel (unscaled) with vTile (scaled) OR wsPanel with unscaled v.
                // We choose: use wPanel (unscaled) with vTile (scaled).
                for (uint32_t oo = 0; oo < dolen; ++oo) {
                    float acc = outPanel.GetValue(q * doTile + oo);
                    uint32_t rowOff = oo * dTile;
                    for (uint32_t j = 0; j < dlen; ++j) {
                        acc += vTile.GetValue(j) * wPanel.GetValue(rowOff + j);
                    }
                    outPanel.SetValue(q * doTile + oo, acc);
                }
            }
        }

        // Store outPanel to GM using DataCopy per q (contiguous doTile segment)
        // First, compact each row into a temporary contiguous buffer of length dolen (in-place use wBuf as scratch since nq fits)
        AscendC::LocalTensor<float> rowTmp = wBuf.Get<float>(); // reuse wBuf UB (size nq) is enough for dolen<=16? no, nq=49.
        // Instead reuse iBuf (size nq) not enough either. So store directly with scalar for tail? Avoid.
        // Use DataCopy only when dolen==doTile (common), else scalar for tail.
        for (uint32_t q = 0; q < nq; ++q) {
            uint64_t yBase = (static_cast<uint64_t>(bIdx) * nq + q) * d + do0;
            if (dolen == doTile) {
                AscendC::DataCopy(yGm[yBase], outPanel[q * doTile], doTile);
            } else {
                for (uint32_t oo = 0; oo < dolen; ++oo) {
                    yGm.SetValue(yBase + oo, outPanel.GetValue(q * doTile + oo));
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> iBuf;
    AscendC::TBuf<> wBuf;
    AscendC::TBuf<> expBuf;
    AscendC::TBuf<> ctxBuf;

    AscendC::TBuf<> kTileBuf;
    AscendC::TBuf<> vTileBuf;
    AscendC::TBuf<> ctxTileBuf;

    AscendC::TBuf<> wPanelBuf;
    AscendC::TBuf<> wsPanelBuf;
    AscendC::TBuf<> outPanelBuf;

    AscendC::GlobalTensor<float> iGm, kGm, vGm, wOGm, bOGm, yGm;

    uint32_t bs, nq, d;
    uint32_t qTile, dTile, doTile;

    bool valid{false};
    uint32_t bIdx{0}, gIdx{0};
};

extern "C" __global__ __aicore__ void mobile_vi_tv2_attention_custom(
    GM_ADDR i, GM_ADDR k, GM_ADDR v,
    GM_ADDR w_o, GM_ADDR b_o,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMobileViTv2AttentionCustom op;
    op.Init(i, k, v, w_o, b_o, y,
            tiling_data.bs, tiling_data.nq, tiling_data.d,
            tiling_data.qTile, tiling_data.dTile, tiling_data.doTile);
    op.Process();
}
