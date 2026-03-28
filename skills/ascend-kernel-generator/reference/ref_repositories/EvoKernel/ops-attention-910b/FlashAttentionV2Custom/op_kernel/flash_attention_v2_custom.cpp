
#include "kernel_operator.h"

class KernelFlashAttentionV2Custom {
public:
    __aicore__ inline KernelFlashAttentionV2Custom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR scale, GM_ADDR o,
                               uint32_t B, uint32_t H, uint32_t S, uint32_t D,
                               uint32_t Ti, uint32_t Tj)
    {
        this->B = B; this->H = H; this->S = S; this->D = D;
        this->Ti = Ti; this->Tj = Tj;
        this->totalBh = B * H;

        const uint64_t totalElems = (uint64_t)B * (uint64_t)H * (uint64_t)S * (uint64_t)D;
        qGm.SetGlobalBuffer((__gm__ float*)q, totalElems);
        kGm.SetGlobalBuffer((__gm__ float*)k, totalElems);
        vGm.SetGlobalBuffer((__gm__ float*)v, totalElems);
        oGm.SetGlobalBuffer((__gm__ float*)o, totalElems);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, 1);

        // UB buffers
        pipe.InitBuffer(qBuf, (uint32_t)(Ti * D * sizeof(float)));
        pipe.InitBuffer(kBuf, (uint32_t)(Tj * D * sizeof(float)));
        pipe.InitBuffer(vBuf, (uint32_t)(Tj * D * sizeof(float)));

        // calc:
        // m[Ti], l[Ti]
        // scores[Ti*Tj], exps[Ti*Tj]
        // out[Ti*D]
        // redMul[D] + redTmp[D+16]
        // oneElem[4]
        const uint32_t stateSz = 2u * Ti;
        const uint32_t scoreSz = 2u * Ti * Tj;
        const uint32_t outSz   = Ti * D;
        const uint32_t redSz   = 2u * D + 32u;
        const uint32_t oneSz   = 8u;
        pipe.InitBuffer(calcBuf, (uint32_t)((stateSz + scoreSz + outSz + redSz + oneSz) * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        uint32_t cid = (uint32_t)AscendC::GetBlockIdx();
        uint32_t cnum = (uint32_t)AscendC::GetBlockNum();
        if (cnum == 0) cnum = 1;

        coreScale = scaleGm.GetValue(0);

        for (uint32_t bh = cid; bh < totalBh; bh += cnum) {
            uint32_t b = bh / H;
            uint32_t h = bh - b * H;
            ComputeOneHead(b, h);
        }
    }

private:
    __aicore__ inline float ExpScalarUb(float x, AscendC::LocalTensor<float> one)
    {
        one.SetValue(0, x);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(one, one, 1);
        AscendC::PipeBarrier<PIPE_V>();
        return one.GetValue(0);
    }

    __aicore__ inline void ComputeOneHead(uint32_t b, uint32_t h)
    {
        const uint64_t base = ((uint64_t)b * (uint64_t)H + (uint64_t)h) * (uint64_t)S * (uint64_t)D;

        AscendC::LocalTensor<float> all = calcBuf.Get<float>();
        AscendC::LocalTensor<float> m      = all;                               // [Ti]
        AscendC::LocalTensor<float> l      = all[Ti];                           // [Ti]
        AscendC::LocalTensor<float> scores = all[2u * Ti];                      // [Ti*Tj]
        AscendC::LocalTensor<float> exps   = all[2u * Ti + Ti * Tj];            // [Ti*Tj]
        AscendC::LocalTensor<float> out    = all[2u * Ti + 2u * Ti * Tj];       // [Ti*D]
        AscendC::LocalTensor<float> red    = all[2u * Ti + 2u * Ti * Tj + Ti * D];
        AscendC::LocalTensor<float> one    = all[2u * Ti + 2u * Ti * Tj + Ti * D + 2u * D + 32u];

        AscendC::LocalTensor<float> redMul = red;            // [D]
        AscendC::LocalTensor<float> redTmp = red[D];         // [>= D+16]

        for (uint32_t qi0 = 0; qi0 < S; qi0 += Ti) {
            uint32_t qiN = (qi0 + Ti <= S) ? Ti : (S - qi0);

            AscendC::LocalTensor<float> qTile = qBuf.Get<float>();
            // contiguous copy of Ti*D
            AscendC::DataCopy(qTile, qGm[base + (uint64_t)qi0 * D], qiN * D);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (uint32_t ii = 0; ii < qiN; ++ii) {
                m.SetValue(ii, -3.402823466e+38f);
                l.SetValue(ii, 0.0f);
                AscendC::Duplicate(out[ii * D], 0.0f, (int32_t)D);
            }
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t kj0 = 0; kj0 < S; kj0 += Tj) {
                uint32_t kjN = (kj0 + Tj <= S) ? Tj : (S - kj0);

                AscendC::LocalTensor<float> kTile = kBuf.Get<float>();
                AscendC::LocalTensor<float> vTile = vBuf.Get<float>();

                // contiguous copy of kjN*D for both K and V
                AscendC::DataCopy(kTile, kGm[base + (uint64_t)kj0 * D], kjN * D);
                AscendC::DataCopy(vTile, vGm[base + (uint64_t)kj0 * D], kjN * D);
                AscendC::PipeBarrier<PIPE_MTE2>();

                // scores[ii,jj] = dot(q,k)*scale
                for (uint32_t ii = 0; ii < qiN; ++ii) {
                    for (uint32_t jj = 0; jj < kjN; ++jj) {
                        AscendC::Mul(redMul, qTile[ii * D], kTile[jj * D], (int32_t)D);
                        // removed extra barrier here to reduce stalls; keep the one after ReduceSum
                        AscendC::ReduceSum<float>(redTmp[0], redMul, redTmp[1], (int32_t)D);
                        AscendC::PipeBarrier<PIPE_V>();
                        float dot = redTmp.GetValue(0);
                        scores.SetValue(ii * Tj + jj, dot * coreScale);
                    }
                }

                for (uint32_t ii = 0; ii < qiN; ++ii) {
                    AscendC::ReduceMax<float>(redTmp[0], scores[ii * Tj], redTmp[1], (int32_t)kjN);
                    AscendC::PipeBarrier<PIPE_V>();
                    float tileMax = redTmp.GetValue(0);

                    float mPrev = m.GetValue(ii);
                    float mNew = (mPrev > tileMax) ? mPrev : tileMax;

                    AscendC::Adds(exps[ii * Tj], scores[ii * Tj], -mNew, (int32_t)kjN);
                    AscendC::Exp(exps[ii * Tj], exps[ii * Tj], (int32_t)kjN);
                    AscendC::PipeBarrier<PIPE_V>();

                    AscendC::ReduceSum<float>(redTmp[0], exps[ii * Tj], redTmp[1], (int32_t)kjN);
                    AscendC::PipeBarrier<PIPE_V>();
                    float sumTile = redTmp.GetValue(0);

                    float alpha = ExpScalarUb(mPrev - mNew, one);

                    float lPrev = l.GetValue(ii);
                    float lNew = lPrev * alpha + sumTile;
                    float invL = 1.0f / lNew;

                    m.SetValue(ii, mNew);
                    l.SetValue(ii, lNew);

                    float outScale = (lPrev == 0.0f) ? 0.0f : (lPrev * alpha * invL);
                    AscendC::Muls(out[ii * D], out[ii * D], outScale, (int32_t)D);
                    AscendC::PipeBarrier<PIPE_V>();

                    for (uint32_t jj = 0; jj < kjN; ++jj) {
                        float w = exps.GetValue(ii * Tj + jj) * invL;
                        AscendC::Axpy(out[ii * D], vTile[jj * D], w, (int32_t)D);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            // contiguous store of qiN*D
            AscendC::DataCopy(oGm[base + (uint64_t)qi0 * D], out, qiN * D);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> qBuf;
    AscendC::TBuf<> kBuf;
    AscendC::TBuf<> vBuf;
    AscendC::TBuf<> calcBuf;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t B = 0, H = 0, S = 0, D = 0;
    uint32_t Ti = 1, Tj = 16;
    uint32_t totalBh = 0;
    float coreScale = 1.0f;
};

extern "C" __global__ __aicore__ void flash_attention_v2_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                               GM_ADDR scale, GM_ADDR o,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelFlashAttentionV2Custom op;
    op.Init(q, k, v, scale, o,
            tiling_data.B, tiling_data.H, tiling_data.S, tiling_data.D,
            tiling_data.Ti, tiling_data.Tj);
    op.Process();
}
