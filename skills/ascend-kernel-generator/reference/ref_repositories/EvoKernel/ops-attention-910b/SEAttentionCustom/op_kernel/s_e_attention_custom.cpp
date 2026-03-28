
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSEAttentionCustom {
public:
    __aicore__ inline KernelSEAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t R, uint32_t HW,
                               uint32_t cTile, uint32_t rAlign,
                               float invHW, uint32_t sigTmpBytes)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->B = B; this->C = C; this->H = H; this->W = W;
        this->R = R; this->HW = HW;
        this->cTile = cTile;
        this->rAlign = rAlign;
        this->invHW = invHW;
        this->sigTmpBytes = sigTmpBytes;

        // Batch partition across cores
        uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t blkNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t bPerCore = (B + blkNum - 1) / blkNum;
        bStart = blk * bPerCore;
        bEnd = bStart + bPerCore;
        if (bEnd > B) bEnd = B;

        uint64_t totalX = static_cast<uint64_t>(B) * C * H * W;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalX);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalX);
        w1Gm.SetGlobalBuffer((__gm__ float*)w1, static_cast<uint64_t>(R) * C);
        w2Gm.SetGlobalBuffer((__gm__ float*)w2, static_cast<uint64_t>(C) * R);

        // UB buffers:
        // avgTile: up to cTile
        // gateTile: up to cTile
        // z1: R (aligned)
        // sigmoid tmp
        pipe.InitBuffer(avgBuf, this->cTile * sizeof(float));
        pipe.InitBuffer(gateBuf, this->cTile * sizeof(float));
        pipe.InitBuffer(z1Buf, this->rAlign * sizeof(float));
        pipe.InitBuffer(sigTmpBuf, this->sigTmpBytes);
    }

    __aicore__ inline void Process()
    {
        if (bStart >= bEnd) return;
        for (uint32_t b = bStart; b < bEnd; ++b) {
            ComputeOneBatch(b);
        }
    }

private:
    __aicore__ inline void ComputeOneBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> avgTile = avgBuf.Get<float>();
        AscendC::LocalTensor<float> gateTile = gateBuf.Get<float>();
        AscendC::LocalTensor<float> z1 = z1Buf.Get<float>();
        AscendC::LocalTensor<uint8_t> sigTmp = sigTmpBuf.Get<uint8_t>();

        // z1 init to 0
        AscendC::Duplicate(z1, (float)0.0f, this->R);

        // Iterate channels in tiles; compute avg per channel and accumulate z1 via FC1 weights
        uint32_t numTiles = (C + cTile - 1) / cTile;
        for (uint32_t t = 0; t < numTiles; ++t) {
            uint32_t c0 = t * cTile;
            uint32_t curC = (t == numTiles - 1) ? (C - c0) : cTile;
            if (curC == 0) continue;

            // avg per channel in this tile
            AscendC::Duplicate(avgTile, (float)0.0f, curC);
            for (uint32_t cc = 0; cc < curC; ++cc) {
                uint32_t c = c0 + cc;
                uint64_t base = (static_cast<uint64_t>(b) * C + c) * static_cast<uint64_t>(HW);
                float sum = 0.0f;
                for (uint32_t hw = 0; hw < HW; ++hw) {
                    sum += xGm.GetValue(base + hw);
                }
                avgTile.SetValue(cc, sum * invHW); // average
            }

            // Accumulate FC1: z1[r] += sum_cc avg[cc] * w1[r, c]
            for (uint32_t r = 0; r < R; ++r) {
                float acc = z1.GetValue(r);
                uint64_t w1Base = static_cast<uint64_t>(r) * C + c0;
                for (uint32_t cc = 0; cc < curC; ++cc) {
                    acc += avgTile.GetValue(cc) * w1Gm.GetValue(w1Base + cc);
                }
                z1.SetValue(r, acc);
            }
        }

        // ReLU on z1
        AscendC::Relu(z1, z1, R);

        // FC2 + Sigmoid + Scale per channel tile and write output
        for (uint32_t t = 0; t < numTiles; ++t) {
            uint32_t c0 = t * cTile;
            uint32_t curC = (t == numTiles - 1) ? (C - c0) : cTile;
            if (curC == 0) continue;

            // gateTile[cc] = sum_r z1[r] * w2[c, r]
            for (uint32_t cc = 0; cc < curC; ++cc) {
                uint32_t c = c0 + cc;
                float acc = 0.0f;
                uint64_t w2Base = static_cast<uint64_t>(c) * R;
                for (uint32_t r = 0; r < R; ++r) {
                    acc += z1.GetValue(r) * w2Gm.GetValue(w2Base + r);
                }
                gateTile.SetValue(cc, acc);
            }

            // Sigmoid in-place
            AscendC::Sigmoid<float, true>(gateTile, gateTile, sigTmp, curC);

            // Scale x -> y for this tile
            for (uint32_t cc = 0; cc < curC; ++cc) {
                float s = gateTile.GetValue(cc);
                uint32_t c = c0 + cc;
                uint64_t base = (static_cast<uint64_t>(b) * C + c) * static_cast<uint64_t>(HW);
                for (uint32_t hw = 0; hw < HW; ++hw) {
                    float xv = xGm.GetValue(base + hw);
                    yGm.SetValue(base + hw, xv * s);
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> avgBuf;
    AscendC::TBuf<> gateBuf;
    AscendC::TBuf<> z1Buf;
    AscendC::TBuf<> sigTmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> w1Gm;
    AscendC::GlobalTensor<float> w2Gm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B, C, H, W, R, HW;
    uint32_t cTile, rAlign;
    float invHW;
    uint32_t sigTmpBytes;
    uint32_t bStart, bEnd;
};

extern "C" __global__ __aicore__ void se_attention_custom(GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSEAttentionCustom op;
    op.Init(x, w1, w2, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.R, tiling_data.HW,
            tiling_data.cTile, tiling_data.rAlign,
            tiling_data.invHW, tiling_data.sigTmpBytes);
    op.Process();
}
