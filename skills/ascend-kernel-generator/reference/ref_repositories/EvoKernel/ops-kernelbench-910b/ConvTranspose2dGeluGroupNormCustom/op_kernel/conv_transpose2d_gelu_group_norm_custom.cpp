
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelConvTranspose2dGeluGroupNormCustom {
public:
    __aicore__ inline KernelConvTranspose2dGeluGroupNormCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                               uint32_t N, uint32_t C, uint32_t HW,
                               uint32_t G, uint32_t groupSize, uint32_t groupsTotal,
                               float invReduce, float eps)
    {
        this->N = N;
        this->C = C;
        this->HW = HW;
        this->G = G;
        this->groupSize = groupSize;
        this->groupsTotal = groupsTotal;
        this->invReduce = invReduce;
        this->eps = eps;

        xBase = reinterpret_cast<__gm__ float*>(x);
        yBase = reinterpret_cast<__gm__ float*>(y);
        gammaBase = reinterpret_cast<__gm__ float*>(gamma);
        betaBase = reinterpret_cast<__gm__ float*>(beta);

        const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());

        const uint32_t perCore = (this->groupsTotal + blockNum - 1u) / blockNum;
        const uint32_t start = blockIdx * perCore;

        uint32_t count = 0;
        if (start < this->groupsTotal) {
            const uint32_t remain = this->groupsTotal - start;
            count = (remain < perCore) ? remain : perCore;
        }
        groupStart = start;
        groupCount = count;

        // UB scalar scratch for Exp/Rsqrt on single element.
        pipe.InitBuffer(qSc, BUFFER_NUM, 8 * sizeof(float));
        sc = qSc.AllocTensor<float>();
    }

    __aicore__ inline void Process()
    {
        if (groupCount == 0) { qSc.FreeTensor(sc); return; }
        if (N == 0 || C == 0 || HW == 0 || G == 0 || groupSize == 0) { qSc.FreeTensor(sc); return; }
        if ((C % G) != 0) { qSc.FreeTensor(sc); return; }

        for (uint32_t gi = 0; gi < groupCount; ++gi) {
            const uint32_t gg = groupStart + gi;
            ComputeOneGroup(gg);
        }

        qSc.FreeTensor(sc);
    }

private:
    __aicore__ inline float ExpFast(float v)
    {
        sc.SetValue(0, v);
        AscendC::Exp<float>(sc, sc, 1);
        return sc.GetValue(0);
    }

    __aicore__ inline float RsqrtFast(float v)
    {
        sc.SetValue(0, v);
        AscendC::Rsqrt<float>(sc, sc, 1);
        return sc.GetValue(0);
    }

    __aicore__ inline float TanhApprox(float x)
    {
        const float e2x = ExpFast(2.0f * x);
        return (e2x - 1.0f) / (e2x + 1.0f);
    }

    __aicore__ inline float GeluApprox(float x)
    {
        // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        const float k0 = 0.7978845608028654f;
        const float k1 = 0.044715f;
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float t = k0 * (x + k1 * x3);
        const float th = TanhApprox(t);
        return 0.5f * x * (1.0f + th);
    }

    __aicore__ inline void ComputeOneGroup(uint32_t globalGroup)
    {
        const uint32_t n = globalGroup / G;
        const uint32_t g = globalGroup - n * G;
        const uint32_t cStart = g * groupSize;

        const uint64_t CHW = static_cast<uint64_t>(C) * static_cast<uint64_t>(HW);
        const uint64_t nBase = static_cast<uint64_t>(n) * CHW;

        // Pass1: compute mean/var of GELU(x) over channels in this group and HW.
        float sum = 0.0f;
        float sumsq = 0.0f;

        for (uint32_t ci = 0; ci < groupSize; ++ci) {
            const uint32_t c = cStart + ci;
            const uint64_t cBase = nBase + static_cast<uint64_t>(c) * static_cast<uint64_t>(HW);
            for (uint32_t h = 0; h < HW; ++h) {
                const float v = GeluApprox(xBase[cBase + h]);
                sum += v;
                sumsq += v * v;
            }
        }

        const float mean = sum * invReduce;
        float var = sumsq * invReduce - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const float invStd = RsqrtFast(var + eps);

        // Pass2: normalize and apply affine (per-channel gamma/beta).
        for (uint32_t ci = 0; ci < groupSize; ++ci) {
            const uint32_t c = cStart + ci;
            const float gmma = gammaBase[c];
            const float bta = betaBase[c];

            const uint64_t cBase = nBase + static_cast<uint64_t>(c) * static_cast<uint64_t>(HW);
            for (uint32_t h = 0; h < HW; ++h) {
                const float v = GeluApprox(xBase[cBase + h]);
                const float nv = (v - mean) * invStd;
                yBase[cBase + h] = nv * gmma + bta;
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> qSc;
    AscendC::LocalTensor<float> sc;

    __gm__ float* xBase = nullptr;
    __gm__ float* yBase = nullptr;
    __gm__ float* gammaBase = nullptr;
    __gm__ float* betaBase = nullptr;

    uint32_t N = 0, C = 0, HW = 0;
    uint32_t G = 0, groupSize = 0, groupsTotal = 0;
    uint32_t groupStart = 0, groupCount = 0;
    float invReduce = 0.0f;
    float eps = 1e-5f;
};

extern "C" __global__ __aicore__ void conv_transpose2d_gelu_group_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(t, tiling);

    KernelConvTranspose2dGeluGroupNormCustom op;
    op.Init(x, gamma, beta, y,
            t.N, t.C, t.HW,
            t.G, t.groupSize, t.groupsTotal,
            t.invReduce, t.eps);
    op.Process();
}
