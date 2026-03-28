
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGCModuleCustom {
public:
    __aicore__ inline KernelGCModuleCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_bc11, GM_ADDR out,
                               uint32_t N, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t HW, uint32_t totalElems,
                               uint32_t blockElems, uint32_t tileElems)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->N = (int32_t)N;
        this->C = (int32_t)C;
        this->H = (int32_t)H;
        this->W = (int32_t)W;
        this->HW = (int32_t)HW;
        this->totalElems = (int32_t)totalElems;
        this->blockElems = (int32_t)blockElems;
        this->tileElems = (int32_t)tileElems;

        int32_t blk = (int32_t)AscendC::GetBlockIdx();
        int32_t blkNum = (int32_t)AscendC::GetBlockNum();

        int32_t coreOffset = blk * this->blockElems;
        this->coreOffset = coreOffset;

        int32_t valid = 0;
        if (coreOffset < this->totalElems) {
            int32_t remain = this->totalElems - coreOffset;
            valid = remain < this->blockElems ? remain : this->blockElems;
        }
        this->validElems = valid;

        xGm.SetGlobalBuffer((__gm__ float*)x + (uint64_t)coreOffset, (uint64_t)this->blockElems);
        outGm.SetGlobalBuffer((__gm__ float*)out + (uint64_t)coreOffset, (uint64_t)this->blockElems);

        // y is [N,C,1,1] contiguous => flatten length N*C
        yBase = (__gm__ float*)y_bc11;

        pipe.InitBuffer(xBuf, (uint32_t)this->tileElems * sizeof(float));
        pipe.InitBuffer(outBuf, (uint32_t)this->tileElems * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (validElems <= 0) return;

        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();

        // loop tiles inside this core slice
        for (int32_t base = 0; base < validElems; base += tileElems) {
            int32_t len = validElems - base;
            if (len > tileElems) len = tileElems;
            if (len <= 0) break;

            AscendC::DataCopy(xLocal, xGm[base], (uint32_t)len);

            // scalar loop for mapping idx->(n,c): nc = globalIdx / HW
            int32_t gBase = coreOffset + base;
            for (int32_t i = 0; i < len; ++i) {
                int32_t g = gBase + i;
                int32_t nc = g / HW; // in [0, N*C)
                float yv = yBase[(uint64_t)nc];
                float xv = xLocal.GetValue(i);
                outLocal.SetValue(i, xv + yv);
            }

            AscendC::DataCopy(outGm[base], outLocal, (uint32_t)len);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> xBuf;
    AscendC::TBuf<> outBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> outGm;
    __gm__ float* yBase = nullptr;

    int32_t N = 0, C = 0, H = 0, W = 0;
    int32_t HW = 0;
    int32_t totalElems = 0;
    int32_t blockElems = 0;
    int32_t tileElems = 0;
    int32_t coreOffset = 0;
    int32_t validElems = 0;
};

extern "C" __global__ __aicore__ void gc_module_custom(GM_ADDR x, GM_ADDR y_bc11,
                                                       GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGCModuleCustom op;
    op.Init(x, y_bc11, out,
            tiling_data.N, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.HW, tiling_data.totalElems,
            tiling_data.blockElems, tiling_data.tileElems);
    op.Process();
}
