
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGroupNormCustom {
public:
    __aicore__ inline KernelGroupNormCustom() {}

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
        this->groupStart = start;
        this->groupCount = count;

        // Reusable 1-element UB buffer for Rsqrt (allocated once per core).
        pipe.InitBuffer(qSc, BUFFER_NUM, 8 * sizeof(float));
        sc = qSc.AllocTensor<float>();
    }

    __aicore__ inline void Process()
    {
        if (groupCount == 0) return;
        if (N == 0 || C == 0 || HW == 0 || G == 0 || groupSize == 0) return;
        if ((C % G) != 0) return;

        for (uint32_t gi = 0; gi < groupCount; ++gi) {
            const uint32_t globalGroup = groupStart + gi; // [0, N*G)
            ComputeOneGroup(globalGroup);
        }

        // Free the reusable UB tensor once.
        qSc.FreeTensor(sc);
    }

private:
    __aicore__ inline float InvSqrtFast(float v)
    {
        // Use vector Rsqrt on a single element (no extra queues, no Sqrt+Div).
        sc.SetValue(0, v);
        AscendC::Rsqrt<float>(sc, sc, 1);
        return sc.GetValue(0);
    }

    __aicore__ inline void ComputeOneGroup(uint32_t globalGroup)
    {
        const uint32_t n = globalGroup / G;
        const uint32_t g = globalGroup - n * G;
        const uint32_t cStart = g * groupSize;

        float sum = 0.0f;
        float sumsq = 0.0f;

        // Pass 1: accumulate sum and sumsq over groupSize*HW
        // Unroll HW loop by 4 to reduce loop/control and address arithmetic overhead.
        for (uint32_t ci = 0; ci < groupSize; ++ci) {
            const uint32_t c = cStart + ci;
            const uint64_t base = (static_cast<uint64_t>(n) * static_cast<uint64_t>(C) + static_cast<uint64_t>(c)) * static_cast<uint64_t>(HW);

            uint32_t i = 0;
            const uint32_t HW4 = HW & ~3u;
            for (; i < HW4; i += 4) {
                const float v0 = xBase[base + i + 0];
                const float v1 = xBase[base + i + 1];
                const float v2 = xBase[base + i + 2];
                const float v3 = xBase[base + i + 3];
                sum += (v0 + v1) + (v2 + v3);
                sumsq += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
            }
            for (; i < HW; ++i) {
                const float v = xBase[base + i];
                sum += v;
                sumsq += v * v;
            }
        }

        const float mean = sum * invReduce;
        float var = sumsq * invReduce - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const float invStd = InvSqrtFast(var + eps);

        // Pass 2: normalize and affine per channel (also unroll by 4).
        for (uint32_t ci = 0; ci < groupSize; ++ci) {
            const uint32_t c = cStart + ci;
            const float gmma = gammaBase[c];
            const float bta = betaBase[c];
            const uint64_t base = (static_cast<uint64_t>(n) * static_cast<uint64_t>(C) + static_cast<uint64_t>(c)) * static_cast<uint64_t>(HW);

            uint32_t i = 0;
            const uint32_t HW4 = HW & ~3u;
            for (; i < HW4; i += 4) {
                float v0 = xBase[base + i + 0];
                float v1 = xBase[base + i + 1];
                float v2 = xBase[base + i + 2];
                float v3 = xBase[base + i + 3];

                v0 = (v0 - mean) * invStd;
                v1 = (v1 - mean) * invStd;
                v2 = (v2 - mean) * invStd;
                v3 = (v3 - mean) * invStd;

                yBase[base + i + 0] = v0 * gmma + bta;
                yBase[base + i + 1] = v1 * gmma + bta;
                yBase[base + i + 2] = v2 * gmma + bta;
                yBase[base + i + 3] = v3 * gmma + bta;
            }
            for (; i < HW; ++i) {
                const float v = xBase[base + i];
                const float nv = (v - mean) * invStd;
                yBase[base + i] = nv * gmma + bta;
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

extern "C" __global__ __aicore__ void group_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelGroupNormCustom op;
    op.Init(x, gamma, beta, y,
            tiling_data.N, tiling_data.C, tiling_data.HW,
            tiling_data.G, tiling_data.groupSize, tiling_data.groupsTotal,
            tiling_data.invReduce, tiling_data.eps);
    op.Process();
}
