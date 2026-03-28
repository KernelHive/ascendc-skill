
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelShuffleAttentionCustom {
public:
    __aicore__ inline KernelShuffleAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1,
                               GM_ADDR gate_c, GM_ADDR s_norm,
                               GM_ADDR sweight, GM_ADDR sbias,
                               GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t G, uint32_t C2g, uint32_t HW, uint32_t yElems,
                               uint32_t coreNum, uint32_t tileNum, uint32_t tileElems, uint32_t lastTileElems)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        (void)coreNum; (void)tileNum; (void)lastTileElems;

        this->B = B;
        this->C = C;
        this->H = H;
        this->W = W;
        this->G = G;
        this->C2g = C2g;
        this->HW = HW;
        this->yElems = yElems;

        // Runtime split on flattened y [B,C,H,W]
        uint32_t bid  = (uint32_t)AscendC::GetBlockIdx();
        uint32_t bnum = (uint32_t)AscendC::GetBlockNum();
        if (bnum == 0) bnum = 1;
        uint32_t per = (yElems + bnum - 1) / bnum;
        this->coreStart = bid * per;
        uint32_t end = this->coreStart + per;
        if (end > yElems) end = yElems;
        this->coreCount = (end > this->coreStart) ? (end - this->coreStart) : 0;

        this->tileElems = tileElems / BUFFER_NUM;
        if (this->tileElems < 64) this->tileElems = 64;

        this->tileNum = (this->coreCount + this->tileElems - 1) / this->tileElems;
        if (this->tileNum == 0) this->tileNum = 1;
        this->lastTileElems = (this->coreCount > 0) ?
            (this->coreCount - (this->tileNum - 1) * this->tileElems) : 0;
        if (this->lastTileElems == 0) this->lastTileElems = (this->coreCount == 0 ? 0 : this->tileElems);

        const uint64_t BG = (uint64_t)B * (uint64_t)G;
        const uint64_t xElems = BG * (uint64_t)C2g * (uint64_t)HW; // x0/x1/s_norm
        const uint64_t gElems = BG * (uint64_t)C2g;               // gate_c
        const uint64_t pElems = (uint64_t)C2g;                    // sweight/sbias
        const uint64_t ySize  = (uint64_t)yElems;                 // y

        x0Gm.SetGlobalBuffer((__gm__ float*)x0, xElems);
        x1Gm.SetGlobalBuffer((__gm__ float*)x1, xElems);
        gateCGm.SetGlobalBuffer((__gm__ float*)gate_c, gElems);
        sNormGm.SetGlobalBuffer((__gm__ float*)s_norm, xElems);
        sWGm.SetGlobalBuffer((__gm__ float*)sweight, pElems);
        sBGm.SetGlobalBuffer((__gm__ float*)sbias, pElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB: tmp0/tmp1 (2*tileElems floats), out tile
        pipe.InitBuffer(tmpBuf, this->tileElems * 2 * sizeof(float));
        pipe.InitBuffer(qOut, BUFFER_NUM, this->tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (coreCount == 0) return;
        for (uint32_t t = 0; t < tileNum; ++t) {
            ComputeTile(t);
            CopyOut();
        }
    }

private:
    __aicore__ inline uint32_t GetTileLen(uint32_t t) const
    {
        if (coreCount == 0) return 0;
        if (t + 1 == tileNum) return lastTileElems;
        return tileElems;
    }

    // Flattened y index maps to [b, cout, hw]
    __aicore__ inline void DecodeY(uint32_t lin, uint32_t &b, uint32_t &co, uint32_t &hw) const
    {
        uint32_t cHW = C * HW;
        b = lin / cHW;
        uint32_t rem = lin - b * cHW;
        co = rem / HW;
        hw = rem - co * HW;
    }

    // channel_shuffle(groups=2) for C:
    // pre (after cat/view): x has shape [B, C, H, W]
    // shuffle = reshape(B,2,C/2,H,W) -> permute(B,C/2,2,H,W) -> reshape(B,C,H,W)
    // Thus: out_co = (co % (C/2))*2 + (co / (C/2))
    // Inverse: pre_co = (out_co/2) + (out_co%2)*(C/2)
    __aicore__ inline uint32_t InvShuffle2(uint32_t out_co) const
    {
        uint32_t half = C >> 1;
        uint32_t q = out_co >> 1;
        uint32_t r = out_co & 1u;
        return q + r * half;
    }

    // Map pre-shuffle channel index (within [0,C)) to source:
    // pre_co = g*(2*C2g) + k*C2g + j
    __aicore__ inline void PreCoToGkj(uint32_t pre_co, uint32_t &g, uint32_t &k, uint32_t &j) const
    {
        uint32_t perG = 2u * C2g;
        g = pre_co / perG;
        uint32_t rem = pre_co - g * perG;
        k = rem / C2g;     // 0: x0 branch, 1: x1 branch
        j = rem - k * C2g; // channel within branch
    }

    __aicore__ inline uint64_t XOff(uint32_t bg, uint32_t j, uint32_t hw) const
    {
        // [BG, C2g, HW]
        return ((uint64_t)bg * (uint64_t)C2g + (uint64_t)j) * (uint64_t)HW + (uint64_t)hw;
    }

    __aicore__ inline uint64_t GateCOff(uint32_t bg, uint32_t j) const
    {
        // [BG, C2g]
        return (uint64_t)bg * (uint64_t)C2g + (uint64_t)j;
    }

    __aicore__ inline void SigmoidVec(const AscendC::LocalTensor<float> &dst,
                                      const AscendC::LocalTensor<float> &src,
                                      uint32_t len)
    {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Use tmp1 as helper to avoid overlap hazards.
        AscendC::LocalTensor<float> tmp1 = tmpBuf.GetWithOffset<float>(len, len * sizeof(float));
        AscendC::Muls(dst, src, -1.0f, len);
        AscendC::Exp(dst, dst, len);
        AscendC::Adds(tmp1, dst, 1.0f, len);
        AscendC::Reciprocal(dst, tmp1, len);
    }

    __aicore__ inline void ComputeTile(uint32_t t)
    {
        uint32_t len = GetTileLen(t);
        if (len == 0) { curLen = 0; return; }

        uint32_t start = coreStart + t * tileElems;
        if (start >= yElems) { curLen = 0; return; }
        if (start + len > yElems) len = yElems - start;
        if (len == 0) { curLen = 0; return; }

        AscendC::LocalTensor<float> outLocal = qOut.AllocTensor<float>();
        AscendC::LocalTensor<float> affine = tmpBuf.Get<float>(); // len
        // tmp1 segment used in SigmoidVec

        // Build affine only for k==1 entries; others set 0.
        for (uint32_t i = 0; i < len; ++i) {
            uint32_t b, out_co, hw;
            DecodeY(start + i, b, out_co, hw);

            uint32_t pre_co = InvShuffle2(out_co);
            uint32_t g, k, j;
            PreCoToGkj(pre_co, g, k, j);

            if (k == 1) {
                uint32_t bg = b * G + g;
                float sn = sNormGm.GetValue(XOff(bg, j, hw));
                float sw = sWGm.GetValue((uint64_t)j);
                float sb = sBGm.GetValue((uint64_t)j);
                affine.SetValue(i, sw * sn + sb);
            } else {
                affine.SetValue(i, 0.0f);
            }
        }

        // gate_s = sigmoid(affine) in-place into affine
        SigmoidVec(affine, affine, len);

        // Final y for this tile
        for (uint32_t i = 0; i < len; ++i) {
            uint32_t b, out_co, hw;
            DecodeY(start + i, b, out_co, hw);

            uint32_t pre_co = InvShuffle2(out_co);
            uint32_t g, k, j;
            PreCoToGkj(pre_co, g, k, j);
            uint32_t bg = b * G + g;

            float x = (k == 0) ? x0Gm.GetValue(XOff(bg, j, hw)) : x1Gm.GetValue(XOff(bg, j, hw));
            float gate = (k == 0) ? gateCGm.GetValue(GateCOff(bg, j)) : affine.GetValue(i);
            outLocal.SetValue(i, x * gate);
        }

        qOut.EnQue(outLocal);
        curStart = start;
        curLen = len;
    }

    __aicore__ inline void CopyOut()
    {
        if (curLen == 0) return;
        AscendC::LocalTensor<float> outLocal = qOut.DeQue<float>();
        AscendC::DataCopy(yGm[curStart], outLocal, curLen);
        qOut.FreeTensor(outLocal);
        curLen = 0;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qOut;
    AscendC::TBuf<> tmpBuf;

    AscendC::GlobalTensor<float> x0Gm, x1Gm, gateCGm, sNormGm, sWGm, sBGm, yGm;

    uint32_t B{0}, C{0}, H{0}, W{0}, G{0}, C2g{0}, HW{0};
    uint32_t yElems{0};

    uint32_t coreStart{0}, coreCount{0};
    uint32_t tileNum{0}, tileElems{0}, lastTileElems{0};

    uint32_t curStart{0}, curLen{0};
};

extern "C" __global__ __aicore__ void shuffle_attention_custom(
    GM_ADDR x0, GM_ADDR x1, GM_ADDR gate_c, GM_ADDR s_norm, GM_ADDR sweight, GM_ADDR sbias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelShuffleAttentionCustom op;
    op.Init(x0, x1, gate_c, s_norm, sweight, sbias, y,
            td.B, td.C, td.H, td.W, td.G, td.C2g, td.HW, td.yElems,
            td.coreNum, td.tileNum, td.tileElems, td.lastTileElems);
    op.Process();
}
