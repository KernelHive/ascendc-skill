
#include "kernel_operator.h"

class KernelLCTCustom {
public:
    __aicore__ inline KernelLCTCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR groups, GM_ADDR eps, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t CHW, uint32_t totalLength,
                               uint32_t CAlign, float invHW, uint32_t sigTmpBytes)
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
        this->invHW = invHW;
        this->sigTmpBytes = sigTmpBytes;

        int32_t blk = (int32_t)AscendC::GetBlockIdx();
        int32_t blkNum = (int32_t)AscendC::GetBlockNum();
        int32_t nPerCore = (this->N + blkNum - 1) / blkNum;
        nStart = blk * nPerCore;
        nEnd = nStart + nPerCore;
        if (nEnd > this->N) nEnd = this->N;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)this->totalLength);
        wGm.SetGlobalBuffer((__gm__ float*)w, (uint64_t)this->C);
        bGm.SetGlobalBuffer((__gm__ float*)b, (uint64_t)this->C);
        gGm.SetGlobalBuffer((__gm__ int32_t*)groups, 1ULL);
        epsGm.SetGlobalBuffer((__gm__ float*)eps, 1ULL);

        // UB: pooled[CAlign], gate[CAlign], w[CAlign], b[CAlign], sigmoid tmp, scratch
        pipe.InitBuffer(pooledBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(gateBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(wBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(bBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(sigTmpBuf, this->sigTmpBytes);
        pipe.InitBuffer(scratchBuf, 8U * sizeof(float));

        AscendC::LocalTensor<float> wUb = wBuf.Get<float>();
        AscendC::LocalTensor<float> bUb = bBuf.Get<float>();
        AscendC::DataCopy(wUb, wGm, (uint32_t)this->C);
        AscendC::DataCopy(bUb, bGm, (uint32_t)this->C);
        if (this->CAlign > this->C) {
            AscendC::Duplicate(wUb[this->C], 1.0f, (int32_t)(this->CAlign - this->C));
            AscendC::Duplicate(bUb[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
        }
    }

    __aicore__ inline void Process()
    {
        if (nStart >= nEnd) return;

        const int32_t groups = gGm.GetValue(0);
        const float eps = epsGm.GetValue(0);

        if (groups <= 0) return;
        if (this->C <= 0) return;
        if ((this->C % groups) != 0) return;
        if (eps < 0.0f) return;

        const int32_t Cg = this->C / groups;

        AscendC::LocalTensor<float> pooled = pooledBuf.Get<float>();
        AscendC::LocalTensor<float> gate = gateBuf.Get<float>();
        AscendC::LocalTensor<float> wUb = wBuf.Get<float>();
        AscendC::LocalTensor<float> bUb = bBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> sigTmp = sigTmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> scratch = scratchBuf.Get<float>();

        for (int32_t n = nStart; n < nEnd; ++n) {
            ComputePooled(n, pooled);

            // Build pre-sigmoid gate scalarly from UB pooled/w/b to avoid any misaligned UB slice vector ops.
            for (int32_t g = 0; g < groups; ++g) {
                const int32_t cBase = g * Cg;

                float sum = 0.0f;
                float sum2 = 0.0f;
                for (int32_t i = 0; i < Cg; ++i) {
                    float v = pooled.GetValue(cBase + i);
                    sum += v;
                    sum2 += v * v;
                }

                const float invCg = 1.0f / (float)Cg;
                const float mean = sum * invCg;
                const float mean2 = sum2 * invCg;
                float var = mean2 - mean * mean;
                if (var < 0.0f) var = 0.0f;

                scratch.SetValue(0, var + eps);
                AscendC::Sqrt(scratch, scratch, 1);
                float denom = scratch.GetValue(0);
                if (denom == 0.0f) denom = 1.0f;
                const float invStd = 1.0f / denom;

                for (int32_t i = 0; i < Cg; ++i) {
                    const int32_t c = cBase + i;
                    float v = pooled.GetValue(c);
                    float norm = (v - mean) * invStd;
                    float pre = norm * wUb.GetValue(c) + bUb.GetValue(c);
                    gate.SetValue(c, pre);
                }
            }

            if (this->CAlign > this->C) {
                AscendC::Duplicate(gate[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
            }

            // Sigmoid stays vectorized over the whole aligned buffer base (safe alignment).
            AscendC::Sigmoid<float, true>(pooled, gate, sigTmp, (uint32_t)this->CAlign); // reuse pooled as dst
            AscendC::DataCopy(gate, pooled, (uint32_t)this->CAlign);

            ApplyGate(n, gate);
        }
    }

private:
    __aicore__ inline void ComputePooled(int32_t n, const AscendC::LocalTensor<float>& pooled)
    {
        AscendC::Duplicate(pooled, 0.0f, this->CAlign);

        const int64_t base = (int64_t)n * (int64_t)this->CHW;
        for (int32_t c = 0; c < this->C; ++c) {
            const int64_t off = base + (int64_t)c * (int64_t)this->HW;
            float sum = 0.0f;
            for (int32_t hw = 0; hw < this->HW; ++hw) {
                sum += xGm.GetValue((uint64_t)(off + (int64_t)hw));
            }
            pooled.SetValue(c, sum * this->invHW);
        }

        if (this->CAlign > this->C) {
            AscendC::Duplicate(pooled[this->C], 0.0f, (int32_t)(this->CAlign - this->C));
        }
    }

    __aicore__ inline void ApplyGate(int32_t n, const AscendC::LocalTensor<float>& gate)
    {
        const int64_t base = (int64_t)n * (int64_t)this->CHW;
        for (int32_t c = 0; c < this->C; ++c) {
            const float s = gate.GetValue(c);
            const int64_t off = base + (int64_t)c * (int64_t)this->HW;
            for (int32_t hw = 0; hw < this->HW; ++hw) {
                const uint64_t idx = (uint64_t)(off + (int64_t)hw);
                const float xv = xGm.GetValue(idx);
                yGm.SetValue(idx, xv * s);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> pooledBuf;
    AscendC::TBuf<> gateBuf;
    AscendC::TBuf<> wBuf;
    AscendC::TBuf<> bBuf;
    AscendC::TBuf<> sigTmpBuf;
    AscendC::TBuf<> scratchBuf;

    AscendC::GlobalTensor<float> xGm, yGm, wGm, bGm;
    AscendC::GlobalTensor<int32_t> gGm;
    AscendC::GlobalTensor<float> epsGm;

    int32_t N, C, H, W, HW, CHW, totalLength, CAlign;
    float invHW;
    uint32_t sigTmpBytes;
    int32_t nStart, nEnd;
};

extern "C" __global__ __aicore__ void lct_custom(GM_ADDR x, GM_ADDR w, GM_ADDR b, GM_ADDR groups, GM_ADDR eps,
                                                GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelLCTCustom op;
    op.Init(x, w, b, groups, eps, y,
            tiling_data.N, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.CHW, tiling_data.totalLength,
            tiling_data.CAlign, tiling_data.invHW, tiling_data.sigTmpBytes);
    op.Process();
}
