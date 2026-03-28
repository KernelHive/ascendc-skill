
#include "kernel_operator.h"

class KernelECAAttentionCustom {
public:
    __aicore__ inline KernelECAAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t K, uint32_t pad,
                               float invHW, uint32_t cTile, uint32_t sigTmpBytes,
                               uint32_t totalB, uint32_t bPerBlock)
    {
        this->B = B;
        this->C = C;
        this->H = H;
        this->W = W;
        this->HW = HW;
        this->K = K;
        this->pad = pad;
        this->invHW = invHW;
        this->cTile = cTile;
        this->sigTmpBytes = sigTmpBytes;

        const uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t blkNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        (void)totalB;

        const uint32_t per = (bPerBlock == 0) ? 1U : bPerBlock;
        bStart = blk * per;
        bEnd = bStart + per;
        if (bEnd > B) bEnd = B;
        if (blkNum == 0) { bStart = 0; bEnd = 0; }

        const uint64_t totalX = static_cast<uint64_t>(B) * C * H * W;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalX);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalX);

        // weight: [1,1,K] contiguous, treat as length-K vector
        wGm.SetGlobalBuffer((__gm__ float*)weight, static_cast<uint64_t>(K));

        // UB buffers:
        // pooled: [C]
        // inTile: [curC + K - 1]
        // outTile: [curC]
        // gateTile: [curC]
        // sigmoid tmp: [sigTmpBytes] bytes
        pipe.InitBuffer(pooledBuf, static_cast<uint64_t>(C) * sizeof(float));
        pipe.InitBuffer(inTileBuf, static_cast<uint64_t>(cTile + K - 1U) * sizeof(float));
        pipe.InitBuffer(outTileBuf, static_cast<uint64_t>(cTile) * sizeof(float));
        pipe.InitBuffer(gateBuf, static_cast<uint64_t>(cTile) * sizeof(float));
        pipe.InitBuffer(sigTmpBuf, static_cast<uint64_t>(sigTmpBytes));
    }

    __aicore__ inline void Process()
    {
        if (B == 0 || C == 0 || H == 0 || W == 0 || HW == 0 || K == 0) return;
        if (bStart >= bEnd) return;

        for (uint32_t b = bStart; b < bEnd; ++b) {
            ComputeOneBatch(b);
        }
    }

private:
    __aicore__ inline uint64_t XOff(uint32_t b, uint32_t c, uint32_t hw) const
    {
        // x stored as [B][C][H][W] contiguous
        return (static_cast<uint64_t>(b) * C + c) * static_cast<uint64_t>(HW) + hw;
    }

    __aicore__ inline void ComputeOneBatch(uint32_t b)
    {
        AscendC::LocalTensor<float> pooled = pooledBuf.Get<float>();
        AscendC::LocalTensor<float> inTile = inTileBuf.Get<float>();
        AscendC::LocalTensor<float> outTile = outTileBuf.Get<float>();
        AscendC::LocalTensor<float> gate = gateBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> sigTmp = sigTmpBuf.Get<uint8_t>();

        // 1) pooled[c] = mean(x[b,c,:,:]) over HW
        for (uint32_t c = 0; c < C; ++c) {
            const uint64_t base = XOff(b, c, 0);
            float sum = 0.0f;
            for (uint32_t i = 0; i < HW; ++i) {
                sum += xGm.GetValue(base + i);
            }
            pooled(c) = sum * invHW;
        }
        AscendC::PipeBarrier<PIPE_V>();

        // 2) conv1d across channels with same padding (stride=1)
        const uint32_t numTiles = (C + cTile - 1U) / cTile;
        for (uint32_t t = 0; t < numTiles; ++t) {
            const uint32_t c0 = t * cTile;
            const uint32_t curC = (c0 + cTile <= C) ? cTile : (C - c0);
            if (curC == 0) continue;

            // Build window for this tile: inTile[j] = pooled[(c0 - pad) + j] with OOB=0
            const uint32_t winLen = curC + K - 1U;
            const int32_t startC = static_cast<int32_t>(c0) - static_cast<int32_t>(pad);
            for (uint32_t j = 0; j < winLen; ++j) {
                const int32_t srcC = startC + static_cast<int32_t>(j);
                float v = 0.0f;
                if (srcC >= 0 && srcC < static_cast<int32_t>(C)) {
                    v = pooled(static_cast<uint32_t>(srcC));
                }
                inTile(j) = v;
            }
            AscendC::PipeBarrier<PIPE_V>();

            // outTile[i] = dot(inTile[i:i+K], w[0:K])
            for (uint32_t i = 0; i < curC; ++i) {
                float acc = 0.0f;
                const uint32_t base = i;
                for (uint32_t kk = 0; kk < K; ++kk) {
                    acc += inTile(base + kk) * wGm.GetValue(kk);
                }
                outTile(i) = acc;
            }
            AscendC::PipeBarrier<PIPE_V>();

            // 3) gate = sigmoid(outTile) (compute only valid curC entries)
            AscendC::Sigmoid<float, true>(gate, outTile, sigTmp, curC);
            AscendC::PipeBarrier<PIPE_V>();

            // 4) y[b,c,:,:] = x[b,c,:,:] * gate[c]
            for (uint32_t i = 0; i < curC; ++i) {
                const uint32_t c = c0 + i;
                const float g = gate(i);
                const uint64_t base = XOff(b, c, 0);
                for (uint32_t hw = 0; hw < HW; ++hw) {
                    const float xv = xGm.GetValue(base + hw);
                    yGm.SetValue(base + hw, xv * g);
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> pooledBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outTileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigTmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> wGm;

    uint32_t B, C, H, W, HW, K, pad;
    float invHW;
    uint32_t cTile;
    uint32_t sigTmpBytes;

    uint32_t bStart, bEnd;
};

extern "C" __global__ __aicore__ void eca_attention_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelECAAttentionCustom op;
    op.Init(x, weight, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.K, tiling_data.pad,
            tiling_data.invHW, tiling_data.cTile, tiling_data.sigTmpBytes,
            tiling_data.totalB, tiling_data.bPerBlock);
    op.Process();
}
