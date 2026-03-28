
#include "kernel_operator.h"

class KernelDoubleAttentionCustom {
public:
    __aicore__ inline KernelDoubleAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR v, GM_ADDR z,
                               uint32_t B, uint32_t Cm, uint32_t Cn, uint32_t HW,
                               uint32_t batchesPerCore)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->B = B;
        this->Cm = Cm;
        this->Cn = Cn;
        this->HW = HW;
        this->batchesPerCore = batchesPerCore;

        const uint64_t aSize = (uint64_t)B * (uint64_t)Cm * (uint64_t)HW;
        const uint64_t bSize = (uint64_t)B * (uint64_t)Cn * (uint64_t)HW;
        const uint64_t vSize = (uint64_t)B * (uint64_t)Cn * (uint64_t)HW;
        const uint64_t zSize = (uint64_t)B * (uint64_t)Cm * (uint64_t)HW;

        aGm.SetGlobalBuffer((__gm__ float*)a, aSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        vGm.SetGlobalBuffer((__gm__ float*)v, vSize);
        zGm.SetGlobalBuffer((__gm__ float*)z, zSize);

        // UB buffers (kept modest; no full HWxCn or Cm x Cn storage):
        // rowHW: exp values for a single Cn row over HW
        // tmpSca: scalar exp helper (1 element is enough, allocate a few)
        // descCm: descriptor vector length Cm (per Cn channel)
        pipe.InitBuffer(rowHWBuf, (uint64_t)HW * sizeof(float));
        pipe.InitBuffer(tmpScaBuf, 8 * sizeof(float));
        pipe.InitBuffer(descCmBuf, (uint64_t)Cm * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blk = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t startB = blk * batchesPerCore;
        uint32_t endB = startB + batchesPerCore;
        if (endB > B) endB = B;
        if (startB >= endB) return;

        AscendC::LocalTensor<float> rowHW  = rowHWBuf.Get<float>();
        AscendC::LocalTensor<float> tmpSca = tmpScaBuf.Get<float>();
        AscendC::LocalTensor<float> descCm = descCmBuf.Get<float>();

        const uint64_t aStride = (uint64_t)Cm * (uint64_t)HW;
        const uint64_t bStride = (uint64_t)Cn * (uint64_t)HW;
        const uint64_t zStride = (uint64_t)Cm * (uint64_t)HW;

        for (uint32_t n = startB; n < endB; ++n) {
            const uint64_t aBase = (uint64_t)n * aStride;
            const uint64_t bBase = (uint64_t)n * bStride;
            const uint64_t vBase = (uint64_t)n * bStride;
            const uint64_t zBase = (uint64_t)n * zStride;

            // Initialize output Z to 0 for this batch.
            for (uint64_t idx = 0; idx < zStride; ++idx) {
                zGm.SetValue(zBase + idx, 0.0f);
            }

            for (uint32_t j = 0; j < Cn; ++j) {
                // softmax over HW for B[j,:]
                float bMax = bGm.GetValue(bBase + (uint64_t)j * (uint64_t)HW);
                for (uint32_t k = 1; k < HW; ++k) {
                    float bv = bGm.GetValue(bBase + (uint64_t)j * (uint64_t)HW + (uint64_t)k);
                    if (bv > bMax) bMax = bv;
                }
                float bSum = 0.0f;
                for (uint32_t k = 0; k < HW; ++k) {
                    float bv = bGm.GetValue(bBase + (uint64_t)j * (uint64_t)HW + (uint64_t)k) - bMax;
                    tmpSca.SetValue(0, bv);
                    AscendC::Exp(tmpSca, tmpSca, 1);
                    float ev = tmpSca.GetValue(0);
                    rowHW.SetValue(k, ev);
                    bSum += ev;
                }
                float bInvSum = 1.0f / bSum;

                // descCm[i] = sum_k A[i,k] * softmaxB[k]
                for (uint32_t i = 0; i < Cm; ++i) {
                    float acc = 0.0f;
                    const uint64_t aRowBase = aBase + (uint64_t)i * (uint64_t)HW;
                    for (uint32_t k = 0; k < HW; ++k) {
                        float aVal = aGm.GetValue(aRowBase + (uint64_t)k);
                        float p = rowHW.GetValue(k) * bInvSum;
                        acc += aVal * p;
                    }
                    descCm.SetValue(i, acc);
                }

                // softmax over HW for V[j,:]
                float vMax = vGm.GetValue(vBase + (uint64_t)j * (uint64_t)HW);
                for (uint32_t k = 1; k < HW; ++k) {
                    float vv = vGm.GetValue(vBase + (uint64_t)j * (uint64_t)HW + (uint64_t)k);
                    if (vv > vMax) vMax = vv;
                }
                float vSum = 0.0f;
                for (uint32_t k = 0; k < HW; ++k) {
                    float vv = vGm.GetValue(vBase + (uint64_t)j * (uint64_t)HW + (uint64_t)k) - vMax;
                    tmpSca.SetValue(0, vv);
                    AscendC::Exp(tmpSca, tmpSca, 1);
                    float ev = tmpSca.GetValue(0);
                    rowHW.SetValue(k, ev);
                    vSum += ev;
                }
                float vInvSum = 1.0f / vSum;

                // Z[i,k] += descCm[i] * softmaxV[k]
                for (uint32_t k = 0; k < HW; ++k) {
                    float q = rowHW.GetValue(k) * vInvSum;
                    const uint64_t zColBase = zBase + (uint64_t)k;
                    for (uint32_t i = 0; i < Cm; ++i) {
                        const uint64_t zIdx = zColBase + (uint64_t)i * (uint64_t)HW;
                        float prev = zGm.GetValue(zIdx);
                        float addv = descCm.GetValue(i) * q;
                        zGm.SetValue(zIdx, prev + addv);
                    }
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> rowHWBuf;
    AscendC::TBuf<> tmpScaBuf;
    AscendC::TBuf<> descCmBuf;

    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> zGm;

    uint32_t B {0}, Cm {0}, Cn {0}, HW {0};
    uint32_t batchesPerCore {1};
};

extern "C" __global__ __aicore__
void double_attention_custom(GM_ADDR a, GM_ADDR b, GM_ADDR v, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelDoubleAttentionCustom op;
    op.Init(a, b, v, z,
            tiling_data.B, tiling_data.Cm, tiling_data.Cn, tiling_data.HW,
            tiling_data.batchesPerCore);
    op.Process();
}
