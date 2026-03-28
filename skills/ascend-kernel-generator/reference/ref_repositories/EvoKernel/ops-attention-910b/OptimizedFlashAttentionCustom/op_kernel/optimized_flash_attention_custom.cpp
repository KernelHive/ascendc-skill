
#include "kernel_operator.h"

static constexpr uint32_t MAX_SQ = 512;
static constexpr uint32_t MAX_SK = 512;
static constexpr uint32_t MAX_D  = 64;

class KernelOptimizedFlashAttention {
public:
    __aicore__ inline KernelOptimizedFlashAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR bias, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t sq, uint32_t sk, uint32_t d,
                               float scale, uint32_t hasBias, uint32_t biasBroadcastB, uint32_t tileD)
    {
        this->b = b;
        this->h = h;
        this->sq = sq;
        this->sk = sk;
        this->d  = d;
        this->scale = scale;
        this->hasBias = (hasBias != 0);
        this->biasBroadcastB = (biasBroadcastB != 0);
        this->tileD = tileD;
        if (this->tileD == 0) this->tileD = 1;
        if (this->tileD > MAX_D) this->tileD = MAX_D;

        const uint64_t qElems  = static_cast<uint64_t>(b) * h * sq * d;
        const uint64_t kvElems = static_cast<uint64_t>(b) * h * sk * d;

        qGm.SetGlobalBuffer((__gm__ float*)q, qElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, kvElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, kvElems);
        oGm.SetGlobalBuffer((__gm__ float*)out, qElems);

        if (this->hasBias) {
            biasGm.SetGlobalBuffer((__gm__ float*)bias, 0x7FFFFFFF);
        } else {
            biasGm.SetGlobalBuffer((__gm__ float*)bias, 0);
        }

        pipe.InitBuffer(bufQRow,   MAX_D * sizeof(float));
        pipe.InitBuffer(bufOutRow, MAX_D * sizeof(float));

        // Tiles for streaming K/V from GM to UB
        pipe.InitBuffer(bufKTile,  MAX_D * sizeof(float));
        pipe.InitBuffer(bufVTile,  MAX_D * sizeof(float));
        pipe.InitBuffer(bufTmpTile, MAX_D * sizeof(float));

        // Vector exp for two scalars at once
        pipe.InitBuffer(bufExpIn,  2 * sizeof(float));
        pipe.InitBuffer(bufExpOut, 2 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (sq > MAX_SQ || sk > MAX_SK || d > MAX_D) return;

        const uint32_t idx = static_cast<uint32_t>(AscendC::GetBlockIdx()); // [0, B*H*Sq)
        const uint32_t qi = idx % sq;
        const uint32_t bh = idx / sq;
        const uint32_t head = bh % h;
        const uint32_t batch = bh / h;
        if (batch >= b) return;

        LoadQRow(batch, head, qi);

        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        AscendC::Duplicate(outRow, 0.0f, d);
        AscendC::PipeBarrier<PIPE_V>();

        float m = -3.402823466e+38f; // -FLT_MAX
        float l = 0.0f;

        for (uint32_t kj = 0; kj < sk; ++kj) {
            const float s = ComputeScoreTiled(batch, head, qi, kj); // scaled + bias

            const float mNew = (s > m) ? s : m;

            // Compute exp(m-mNew) and exp(s-mNew) together (length-2 vector Exp)
            float a, bwt;
            Scalar2Exp(m - mNew, s - mNew, a, bwt);

            AscendC::Muls(outRow, outRow, a, d);
            AscendC::PipeBarrier<PIPE_V>();

            AccumulateVTiled(outRow, batch, head, kj, bwt);

            l = l * a + bwt;
            m = mNew;
        }

        const float invL = 1.0f / l;
        AscendC::Muls(outRow, outRow, invL, d);
        AscendC::PipeBarrier<PIPE_V>();

        StoreOutRow(batch, head, qi);
    }

private:
    __aicore__ inline uint64_t QBaseOffset(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(sq) * d);
    }

    __aicore__ inline uint64_t KVBaseOffset(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(sk) * d);
    }

    __aicore__ inline uint64_t BiasOffset(uint32_t batch, uint32_t head, uint32_t qi, uint32_t kj) const
    {
        if (biasBroadcastB) {
            return (static_cast<uint64_t>(head) * static_cast<uint64_t>(sq) + qi) * sk + kj; // [H,Sq,Sk]
        }
        return ((static_cast<uint64_t>(batch) * h + head) * static_cast<uint64_t>(sq) + qi) * sk + kj; // [B,H,Sq,Sk]
    }

    __aicore__ inline void LoadQRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
        const uint64_t base = QBaseOffset(batch, head) + static_cast<uint64_t>(qi) * d;
        AscendC::DataCopy(qRow, qGm[base], d);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline float ComputeScoreTiled(uint32_t batch, uint32_t head, uint32_t qi, uint32_t kj)
    {
        AscendC::LocalTensor<float> qRow  = bufQRow.Get<float>();
        AscendC::LocalTensor<float> kTile = bufKTile.Get<float>();

        const uint64_t kvBase = KVBaseOffset(batch, head) + static_cast<uint64_t>(kj) * d;

        float acc = 0.0f;
        for (uint32_t d0 = 0; d0 < d; d0 += tileD) {
            const uint32_t n = (d0 + tileD <= d) ? tileD : (d - d0);

            AscendC::DataCopy(kTile, kGm[kvBase + d0], n);
            AscendC::PipeBarrier<PIPE_MTE2>();

            // Dot in UB (still a scalar reduction, but GM traffic is now tiled bulk copies)
            for (uint32_t i = 0; i < n; ++i) {
                acc += qRow(d0 + i) * kTile(i);
            }
        }

        float s = acc * scale;
        if (hasBias) {
            s += biasGm(BiasOffset(batch, head, qi, kj));
        }
        return s;
    }

    __aicore__ inline void AccumulateVTiled(AscendC::LocalTensor<float> &outRow,
                                           uint32_t batch, uint32_t head, uint32_t kj, float weight)
    {
        AscendC::LocalTensor<float> vTile  = bufVTile.Get<float>();
        AscendC::LocalTensor<float> tmp    = bufTmpTile.Get<float>();
        const uint64_t vBase = KVBaseOffset(batch, head) + static_cast<uint64_t>(kj) * d;

        for (uint32_t d0 = 0; d0 < d; d0 += tileD) {
            const uint32_t n = (d0 + tileD <= d) ? tileD : (d - d0);

            AscendC::DataCopy(vTile, vGm[vBase + d0], n);
            AscendC::PipeBarrier<PIPE_MTE2>();

            AscendC::Muls(tmp, vTile, weight, n);
            AscendC::PipeBarrier<PIPE_V>();

            // outRow[d0:d0+n] += tmp
            AscendC::Add(outRow[d0], outRow[d0], tmp, n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Scalar2Exp(float x0, float x1, float &y0, float &y1)
    {
        AscendC::LocalTensor<float> in  = bufExpIn.Get<float>();
        AscendC::LocalTensor<float> out = bufExpOut.Get<float>();
        in(0) = x0;
        in(1) = x1;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(out, in, 2);
        AscendC::PipeBarrier<PIPE_V>();
        y0 = out(0);
        y1 = out(1);
    }

    __aicore__ inline void StoreOutRow(uint32_t batch, uint32_t head, uint32_t qi)
    {
        AscendC::LocalTensor<float> outRow = bufOutRow.Get<float>();
        const uint64_t base = QBaseOffset(batch, head) + static_cast<uint64_t>(qi) * d;
        AscendC::DataCopy(oGm[base], outRow, d);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOutRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufTmpTile;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufExpIn;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufExpOut;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, biasGm, oGm;

    uint32_t b = 0, h = 0, sq = 0, sk = 0, d = 0;
    uint32_t tileD = 64;
    float scale = 1.0f;
    bool hasBias = false;
    bool biasBroadcastB = false;
};

extern "C" __global__ __aicore__ void optimized_flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelOptimizedFlashAttention op;
    op.Init(q, k, v, bias, out,
            tiling_data.b, tiling_data.h, tiling_data.sq, tiling_data.sk, tiling_data.d,
            tiling_data.scale, tiling_data.hasBias, tiling_data.biasBroadcastB, tiling_data.tileD);
    op.Process();
}
