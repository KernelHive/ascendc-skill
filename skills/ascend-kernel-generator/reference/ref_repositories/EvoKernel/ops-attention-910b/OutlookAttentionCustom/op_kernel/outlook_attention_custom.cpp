
#include "kernel_operator.h"

class KernelOutlookAttentionCustom {
public:
    __aicore__ inline KernelOutlookAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR attn, GM_ADDR v, GM_ADDR y,
                               uint32_t B, uint32_t NH, uint32_t HD, uint32_t C,
                               uint32_t H, uint32_t W, uint32_t HW,
                               uint32_t K, uint32_t P, uint32_t S, uint32_t K2,
                               uint32_t Ho, uint32_t Wo, uint32_t HWo,
                               uint32_t cTile, uint32_t tilesPerB, uint32_t totalTiles,
                               uint32_t coreNum)
    {
        this->B = B; this->NH = NH; this->HD = HD; this->C = C;
        this->H = H; this->W = W; this->HW = HW;
        this->K = K; this->P = P; this->S = S; this->K2 = K2;
        this->Ho = Ho; this->Wo = Wo; this->HWo = HWo;
        this->cTile = cTile;
        this->tilesPerB = tilesPerB;
        this->totalTiles = totalTiles;
        this->coreNum = coreNum;

        const uint64_t aSize = (uint64_t)B * (uint64_t)NH * (uint64_t)HWo * (uint64_t)K2 * (uint64_t)K2;
        const uint64_t vSize = (uint64_t)B * (uint64_t)NH * (uint64_t)HWo * (uint64_t)K2 * (uint64_t)HD;
        const uint64_t ySize = (uint64_t)B * (uint64_t)C * (uint64_t)HW; // [B,C,H,W] contiguous

        aGm.SetGlobalBuffer((__gm__ float*)attn, aSize);
        vGm.SetGlobalBuffer((__gm__ float*)v, vSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB: weights[9] + vals[9] + out[cTile]
        pipe.InitBuffer(wBuf, 9 * sizeof(float));
        pipe.InitBuffer(vBuf, 9 * sizeof(float));
        pipe.InitBuffer(outBuf, this->cTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) return;

        uint32_t per = (this->totalTiles + bnum - 1u) / bnum;
        uint32_t tStart = bid * per;
        uint32_t tEnd = tStart + per;
        if (tEnd > this->totalTiles) tEnd = this->totalTiles;

        for (uint32_t tile = tStart; tile < tEnd; ++tile) {
            uint32_t b = tile / this->tilesPerB;
            uint32_t ct = tile - b * this->tilesPerB;
            uint32_t c0 = ct * this->cTile;
            uint32_t cLen = this->C > c0 ? (this->C - c0) : 0;
            if (cLen > this->cTile) cLen = this->cTile;
            if (cLen == 0) continue;
            ComputeBCTile(b, c0, cLen);
        }
    }

private:
    __aicore__ inline void ComputeBCTile(uint32_t b, uint32_t c0, uint32_t cLen)
    {
        AscendC::LocalTensor<float> wLocal = wBuf.Get<float>();
        AscendC::LocalTensor<float> vLocal = vBuf.Get<float>();
        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();

        // For each spatial output pixel
        for (uint32_t pix = 0; pix < this->HW; ++pix) {
            // pix -> (oh, ow)
            uint32_t oh = pix / this->W;
            uint32_t ow = pix - oh * this->W;

            // each channel in tile
            for (uint32_t ci = 0; ci < cLen; ++ci) {
                outLocal.SetValue(ci, 0.0f);
            }

            // Gather-style fold mapping (matches baseline and avoids border scatter issues)
            for (uint32_t kh = 0; kh < 3; ++kh) {
                int32_t qh_i = (int32_t)oh - (int32_t)kh + (int32_t)this->P;
                if ((uint32_t)qh_i >= this->Ho) continue;
                uint32_t qh = (uint32_t)qh_i;
                for (uint32_t kw = 0; kw < 3; ++kw) {
                    int32_t qw_i = (int32_t)ow - (int32_t)kw + (int32_t)this->P;
                    if ((uint32_t)qw_i >= this->Wo) continue;
                    uint32_t qw = (uint32_t)qw_i;

                    uint32_t qpos = qh * this->Wo + qw;
                    uint32_t kk_out = kh * 3 + kw;

                    const uint64_t aRow =
                        ((((uint64_t)b * (uint64_t)this->NH + 0ull) * (uint64_t)this->HWo + (uint64_t)qpos) *
                          (uint64_t)this->K2 + (uint64_t)kk_out) * (uint64_t)this->K2;

                    const uint64_t vBase =
                        (((uint64_t)b * (uint64_t)this->NH + 0ull) * (uint64_t)this->HWo + (uint64_t)qpos) *
                        (uint64_t)this->K2 * (uint64_t)this->HD;

                    // Load 9 weights once into UB (small, reduces repeated GM reads)
                    for (uint32_t kk_in = 0; kk_in < 9; ++kk_in) {
                        wLocal.SetValue(kk_in, aGm.GetValue(aRow + (uint64_t)kk_in));
                    }

                    // For each channel in tile: load its 9 v values and do 9-FMA
                    for (uint32_t ci = 0; ci < cLen; ++ci) {
                        uint32_t d = c0 + ci; // NH==1 => C==HD
                        for (uint32_t kk_in = 0; kk_in < 9; ++kk_in) {
                            uint64_t vOff = vBase + (uint64_t)kk_in * (uint64_t)this->HD + (uint64_t)d;
                            vLocal.SetValue(kk_in, vGm.GetValue(vOff));
                        }
                        float sum = 0.0f;
                        #pragma unroll
                        for (uint32_t kk_in = 0; kk_in < 9; ++kk_in) {
                            sum += wLocal.GetValue(kk_in) * vLocal.GetValue(kk_in);
                        }
                        float prev = outLocal.GetValue(ci);
                        outLocal.SetValue(ci, prev + sum);
                    }
                }
            }

            // Store this pixel for all channels in tile (stride HW in memory)
            // y is [B,C,HW] contiguous in HW fastest, so y[b, c, pix] has stride HW in C dimension.
            for (uint32_t ci = 0; ci < cLen; ++ci) {
                uint32_t c = c0 + ci;
                uint64_t yOff = ((uint64_t)b * (uint64_t)this->C + (uint64_t)c) * (uint64_t)this->HW + (uint64_t)pix;
                yGm.SetValue(yOff, outLocal.GetValue(ci));
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> wBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> vBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outBuf;

    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B, NH, HD, C;
    uint32_t H, W, HW;
    uint32_t K, P, S, K2;
    uint32_t Ho, Wo, HWo;
    uint32_t cTile, tilesPerB, totalTiles, coreNum;
};

extern "C" __global__ __aicore__ void outlook_attention_custom(GM_ADDR attn, GM_ADDR v, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);
    KernelOutlookAttentionCustom op;
    op.Init(attn, v, y,
            t.B, t.NH, t.HD, t.C,
            t.H, t.W, t.HW,
            t.K, t.P, t.S, t.K2,
            t.Ho, t.Wo, t.HWo,
            t.cTile, t.tilesPerB, t.totalTiles,
            t.coreNum);
    op.Process();
}
