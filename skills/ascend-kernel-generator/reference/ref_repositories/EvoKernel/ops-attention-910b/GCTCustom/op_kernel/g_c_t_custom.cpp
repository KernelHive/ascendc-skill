
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGCTCustom {
public:
    __aicore__ inline KernelGCTCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR c, GM_ADDR eps, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t CHW, uint32_t totalLength,
                               uint32_t CAlign, float invC)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        // keep indices signed in device code
        this->N = (int32_t)N;
        this->C = (int32_t)C;
        this->H = (int32_t)H;
        this->W = (int32_t)W;
        this->HW = (int32_t)HW;
        this->CHW = (int32_t)CHW;
        this->totalLength = (int32_t)totalLength;
        this->CAlign = (int32_t)CAlign;
        this->invC = invC;

        int32_t blk = (int32_t)AscendC::GetBlockIdx();
        int32_t blkNum = (int32_t)AscendC::GetBlockNum();
        int32_t nPerCore = (this->N + blkNum - 1) / blkNum;
        nStart = blk * nPerCore;
        nEnd = nStart + nPerCore;
        if (nEnd > this->N) nEnd = this->N;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)this->totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)this->totalLength);

        cGm.SetGlobalBuffer((__gm__ float*)c, 1ULL);
        epsGm.SetGlobalBuffer((__gm__ float*)eps, 1ULL);

        // UB: pooled[CAlign], tmp[CAlign], work[CAlign], tanh/exp tmp (sharedTmp for Exp)
        pipe.InitBuffer(pooledBuf, (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(tmpBuf,    (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(workBuf,   (uint32_t)this->CAlign * sizeof(float));
        pipe.InitBuffer(expTmpBuf, 16 * 1024);
    }

    __aicore__ inline void Process()
    {
        if (nStart >= nEnd) return;

        const float cVal = cGm.GetValue(0);
        const float epsVal = epsGm.GetValue(0);

        AscendC::LocalTensor<float> pooled = pooledBuf.Get<float>();
        AscendC::LocalTensor<float> tmp    = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> work   = workBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> expTmp = expTmpBuf.Get<uint8_t>();

        for (int32_t n = nStart; n < nEnd; ++n) {
            // pooled[c] = mean_hw(x[n,c,:,:])
            ComputePooled(n, pooled);

            // mean = mean_c(pooled)
            AscendC::ReduceSum<float>(work, pooled, tmp, (int32_t)CAlign);
            float mean = work.GetValue(0) * invC;

            // mean_x2 = mean_c(pooled^2)
            AscendC::Mul(tmp, pooled, pooled, (int32_t)CAlign);
            AscendC::ReduceSum<float>(work, tmp, pooled, (int32_t)CAlign);
            float mean_x2 = work.GetValue(0) * invC;

            // var = mean_x2 - mean^2
            float var = mean_x2 - mean * mean;

            // inv_std = 1 / sqrt(var + eps)
            tmp.SetValue(0, var + epsVal);
            AscendC::Sqrt(tmp, tmp, 1);
            float inv_std = 1.0f / tmp.GetValue(0);

            // y_norm = (pooled - mean) * inv_std
            AscendC::Duplicate(tmp, mean, (int32_t)CAlign);
            AscendC::Sub(tmp, pooled, tmp, (int32_t)CAlign);
            AscendC::Muls(tmp, tmp, inv_std, (int32_t)CAlign);

            // y_transform = exp(-(y_norm^2 / 2 * c))
            // tmp = y_norm^2
            AscendC::Mul(tmp, tmp, tmp, (int32_t)CAlign);
            AscendC::Muls(tmp, tmp, (-0.5f * cVal), (int32_t)CAlign);
            AscendC::Exp<float>(tmp, tmp, (int32_t)CAlign); // exp uses internal tmp; okay for float path
            // apply to x
            ApplyTransform(n, tmp);
        }
        (void)expTmp; // reserved buffer in case Exp variant needs shared tmp on some toolchains
    }

private:
    __aicore__ inline void ComputePooled(int32_t n, const AscendC::LocalTensor<float>& pooled)
    {
        AscendC::Duplicate(pooled, 0.0f, (int32_t)CAlign);

        int64_t baseN = (int64_t)n * (int64_t)CHW;
        const float invHW = 1.0f / (float)HW;

        for (int32_t c = 0; c < C; ++c) {
            int64_t off = baseN + (int64_t)c * (int64_t)HW;
            float sum = 0.0f;
            for (int32_t hw = 0; hw < HW; ++hw) {
                float xv = xGm.GetValue((uint64_t)(off + hw));
                sum += xv;
            }
            pooled.SetValue((uint32_t)c, sum * invHW);
        }

        if (CAlign > C) {
            AscendC::Duplicate(pooled[(uint32_t)C], 0.0f, (int32_t)(CAlign - C));
        }
    }

    __aicore__ inline void ApplyTransform(int32_t n, const AscendC::LocalTensor<float>& t)
    {
        int64_t baseN = (int64_t)n * (int64_t)CHW;

        for (int32_t c = 0; c < C; ++c) {
            float s = t.GetValue((uint32_t)c);
            int64_t off = baseN + (int64_t)c * (int64_t)HW;
            for (int32_t hw = 0; hw < HW; ++hw) {
                float xv = xGm.GetValue((uint64_t)(off + hw));
                yGm.SetValue((uint64_t)(off + hw), xv * s);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> pooledBuf;
    AscendC::TBuf<> tmpBuf;
    AscendC::TBuf<> workBuf;
    AscendC::TBuf<> expTmpBuf;

    AscendC::GlobalTensor<float> xGm, yGm;
    AscendC::GlobalTensor<float> cGm, epsGm;

    int32_t N, C, H, W, HW, CHW, totalLength, CAlign;
    float invC;
    int32_t nStart, nEnd;
};

extern "C" __global__ __aicore__ void gct_custom(GM_ADDR x,
                                                GM_ADDR c,
                                                GM_ADDR eps,
                                                GM_ADDR y,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGCTCustom op;
    op.Init(x, c, eps, y,
            tiling_data.N, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.CHW, tiling_data.totalLength,
            tiling_data.CAlign, tiling_data.invC);
    op.Process();
}
