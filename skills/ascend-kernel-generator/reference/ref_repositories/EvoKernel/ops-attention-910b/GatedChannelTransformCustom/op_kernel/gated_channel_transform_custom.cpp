
#include "kernel_operator.h"

class KernelGatedChannelTransformCustom {
public:
    __aicore__ inline KernelGatedChannelTransformCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR alpha, GM_ADDR gamma, GM_ADDR beta, GM_ADDR epsilon, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t CHW, uint32_t totalLength,
                               uint32_t CAlign, float invC, uint32_t tanhTmpBytes)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->N = (int32_t)N;
        this->C = (int32_t)C;
        this->H = (int32_t)H;
        this->W = (int32_t)W;
        this->HW = (int32_t)HW;
        this->CHW = (int32_t)CHW;
        this->totalLength = (int32_t)totalLength;
        this->CAlign = (int32_t)CAlign;
        this->invC = invC;
        this->tanhTmpBytes = tanhTmpBytes;

        const int32_t blk = (int32_t)AscendC::GetBlockIdx();
        const int32_t blkNum = (int32_t)AscendC::GetBlockNum();
        const int32_t nPerCore = (this->N + blkNum - 1) / blkNum;
        nStart = blk * nPerCore;
        nEnd = nStart + nPerCore;
        if (nEnd > this->N) nEnd = this->N;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)this->totalLength);

        alphaGm.SetGlobalBuffer((__gm__ float*)alpha, (uint64_t)this->C);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, (uint64_t)this->C);
        betaGm.SetGlobalBuffer((__gm__ float*)beta,  (uint64_t)this->C);
        epsGm.SetGlobalBuffer((__gm__ float*)epsilon, 1ULL);

        // UB buffers
        pipe.InitBuffer(alphaBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(gammaBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(betaBuf,  (uint32_t)this->CAlign * sizeof(float));

        pipe.InitBuffer(embedBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(embed2Buf,(uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(gateBuf,  (uint32_t)this->CAlign * sizeof(float));

        // shared tmp for ReduceSum and small scalar scratch
        pipe.InitBuffer(reduceTmpBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(scalarBuf, 8U * sizeof(float));

        pipe.InitBuffer(tanhTmpBuf, this->tanhTmpBytes);

        // Cache params into UB once per core
        AscendC::LocalTensor<float> aUb = alphaBuf.Get<float>();
        AscendC::LocalTensor<float> gUb = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> bUb = betaBuf.Get<float>();

        AscendC::DataCopy(aUb, alphaGm, (uint32_t)this->C);
        AscendC::DataCopy(gUb, gammaGm, (uint32_t)this->C);
        AscendC::DataCopy(bUb, betaGm,  (uint32_t)this->C);

        if (this->CAlign > this->C) {
            AscendC::Duplicate(aUb[this->C], 1.0f, (int32_t)(this->CAlign - this->C));
            AscendC::Duplicate(gUb[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
            AscendC::Duplicate(bUb[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
        }
    }

    __aicore__ inline void Process()
    {
        if (nStart >= nEnd) return;
        if (this->C <= 0 || this->HW <= 0) return;

        const float eps = epsGm.GetValue(0);
        if (eps < 0.0f) return;

        AscendC::LocalTensor<float> embed  = embedBuf.Get<float>();
        AscendC::LocalTensor<float> embed2 = embed2Buf.Get<float>();
        AscendC::LocalTensor<float> gate   = gateBuf.Get<float>();

        AscendC::LocalTensor<float> aUb = alphaBuf.Get<float>();
        AscendC::LocalTensor<float> gUb = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> bUb = betaBuf.Get<float>();

        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> tanhTmp = tanhTmpBuf.Get<uint8_t>();

        for (int32_t n = nStart; n < nEnd; ++n) {
            // embedding[c] = sqrt(sum_{hw} x^2 + eps) * alpha[c]
            ComputeEmbeddingL2(n, eps, aUb, embed, scalar);

            // denom = sqrt(mean_c(embedding^2) + eps)
            AscendC::Mul(embed2, embed, embed, (uint32_t)this->CAlign);
            AscendC::ReduceSum(scalar, embed2, reduceTmp, (int32_t)this->CAlign);
            float sumEmb2 = scalar.GetValue(0);
            float denomArg = sumEmb2 * this->invC + eps;
            scalar.SetValue(0, denomArg);
            AscendC::Sqrt(scalar, scalar, 1);
            float denom = scalar.GetValue(0);
            if (denom == 0.0f) denom = 1.0f;
            const float invDenom = 1.0f / denom;

            // gate[c] = 1 + tanh( embedding[c] * (gamma[c]*invDenom) + beta[c] )
            // vector compute z = embedding * (gamma*invDenom) + beta
            AscendC::Mul(embed2, embed, gUb, (uint32_t)this->CAlign);     // embed2 = embedding*gamma
            AscendC::Muls(embed2, embed2, invDenom, (uint32_t)this->CAlign);
            AscendC::Add(embed2, embed2, bUb, (uint32_t)this->CAlign);    // z

            // tanh requires separate dst/src
            AscendC::Tanh<float, true>(gate, embed2, tanhTmp, (uint32_t)this->CAlign);
            AscendC::Adds(gate, gate, 1.0f, (uint32_t)this->CAlign);

            // apply y = x * gate
            ApplyGate(n, gate);
        }
    }

private:
    __aicore__ inline void ComputeEmbeddingL2(int32_t n, float eps,
                                             const AscendC::LocalTensor<float>& aUb,
                                             const AscendC::LocalTensor<float>& embed,
                                             const AscendC::LocalTensor<float>& scalar)
    {
        AscendC::Duplicate(embed, 0.0f, this->CAlign);

        const int64_t baseN = (int64_t)n * (int64_t)this->CHW;
        for (int32_t c = 0; c < this->C; ++c) {
            const int64_t off = baseN + (int64_t)c * (int64_t)this->HW;
            float sumsq = 0.0f;
            for (int32_t i = 0; i < this->HW; ++i) {
                const uint64_t idx = (uint64_t)(off + (int64_t)i);
                const float xv = xGm.GetValue(idx);
                sumsq += xv * xv;
            }
            scalar.SetValue(0, sumsq + eps);
            AscendC::Sqrt(scalar, scalar, 1);
            const float emb = scalar.GetValue(0) * aUb.GetValue((uint32_t)c);
            embed.SetValue((uint32_t)c, emb);
        }

        if (this->CAlign > this->C) {
            AscendC::Duplicate(embed[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
        }
    }

    __aicore__ inline void ApplyGate(int32_t n, const AscendC::LocalTensor<float>& gate)
    {
        const int64_t baseN = (int64_t)n * (int64_t)this->CHW;
        for (int32_t c = 0; c < this->C; ++c) {
            const float s = gate.GetValue((uint32_t)c);
            const int64_t off = baseN + (int64_t)c * (int64_t)this->HW;
            for (int32_t i = 0; i < this->HW; ++i) {
                const uint64_t idx = (uint64_t)(off + (int64_t)i);
                const float xv = xGm.GetValue(idx);
                yGm.SetValue(idx, xv * s);
            }
        }
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<> alphaBuf;
    AscendC::TBuf<> gammaBuf;
    AscendC::TBuf<> betaBuf;

    AscendC::TBuf<> embedBuf;
    AscendC::TBuf<> embed2Buf;
    AscendC::TBuf<> gateBuf;

    AscendC::TBuf<> reduceTmpBuf;
    AscendC::TBuf<> scalarBuf;
    AscendC::TBuf<> tanhTmpBuf;

    AscendC::GlobalTensor<float> xGm, yGm;
    AscendC::GlobalTensor<float> alphaGm, gammaGm, betaGm;
    AscendC::GlobalTensor<float> epsGm;

    int32_t N, C, H, W, HW, CHW, totalLength, CAlign;
    float invC;
    uint32_t tanhTmpBytes;
    int32_t nStart, nEnd;
};

extern "C" __global__ __aicore__ void gated_channel_transform_custom(GM_ADDR x,
                                                                    GM_ADDR alpha,
                                                                    GM_ADDR gamma,
                                                                    GM_ADDR beta,
                                                                    GM_ADDR epsilon,
                                                                    GM_ADDR y,
                                                                    GM_ADDR workspace,
                                                                    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelGatedChannelTransformCustom op;
    op.Init(x, alpha, gamma, beta, epsilon, y,
            tiling_data.N, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.CHW, tiling_data.totalLength,
            tiling_data.CAlign, tiling_data.invC, tiling_data.tanhTmpBytes);
    op.Process();
}
