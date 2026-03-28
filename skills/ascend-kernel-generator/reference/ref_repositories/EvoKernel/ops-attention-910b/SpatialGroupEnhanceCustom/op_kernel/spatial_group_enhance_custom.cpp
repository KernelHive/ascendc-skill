
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSpatialGroupEnhanceCustom {
public:
    __aicore__ inline KernelSpatialGroupEnhanceCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                               uint32_t B, uint32_t C, uint32_t H, uint32_t W,
                               uint32_t G, uint32_t Cg, uint32_t HW, uint32_t groupsTotal,
                               uint32_t hwAlign, float invHW, float invCg, float epsilon, uint32_t sigTmpBytes)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        this->B = B; this->C = C; this->H = H; this->W = W;
        this->G = G; this->Cg = Cg; this->HW = HW; this->groupsTotal = groupsTotal;
        this->hwAlign = hwAlign;
        this->invHW = invHW;
        this->invCg = invCg;
        this->epsilon = epsilon;
        this->sigTmpBytes = sigTmpBytes;

        uint64_t xSize = static_cast<uint64_t>(B) * C * HW;
        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, xSize);

        // weight/bias: [1,G,1,1] contiguous length G
        wGm.SetGlobalBuffer((__gm__ float*)weight, static_cast<uint64_t>(G));
        bGm.SetGlobalBuffer((__gm__ float*)bias, static_cast<uint64_t>(G));

        // UB buffers:
        // gate: HW floats (allocated hwAlign)
        // tmp:  HW floats
        // red:  HW floats (ReduceSum work buffer; safe upper bound)
        // sigTmp bytes
        pipe.InitBuffer(gateBuf, this->hwAlign * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->hwAlign * sizeof(float));
        pipe.InitBuffer(redBuf, this->hwAlign * sizeof(float));
        pipe.InitBuffer(sigTmpBuf, this->sigTmpBytes);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = static_cast<uint32_t>(AscendC::GetBlockIdx());
        uint32_t coreNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        if (coreNum == 0) return;

        for (uint32_t groupIndex = coreId; groupIndex < this->groupsTotal; groupIndex += coreNum) {
            ComputeOneGroup(groupIndex);
        }
    }

private:
    __aicore__ inline void ComputeOneGroup(uint32_t groupIndex)
    {
        // groupIndex in [0, B*G)
        uint32_t b = groupIndex / this->G;
        uint32_t g = groupIndex - b * this->G;

        AscendC::LocalTensor<float> gate = gateBuf.Get<float>();   // length HW valid
        AscendC::LocalTensor<float> tmp  = tmpBuf.Get<float>();    // scratch
        AscendC::LocalTensor<float> red  = redBuf.Get<float>();    // ReduceSum work
        AscendC::LocalTensor<uint8_t> sigTmp = sigTmpBuf.Get<uint8_t>();

        // Base offsets for NCHW contiguous: ((b*C + c)*HW + hw)
        uint64_t batchBase = static_cast<uint64_t>(b) * this->C * this->HW;
        uint64_t groupBase = batchBase + static_cast<uint64_t>(g) * this->Cg * this->HW;

        // gate[hw] = sum_{ci}( x(ci,hw) * mean_hw(x(ci,*)) ) where mean over HW for each channel
        AscendC::Duplicate(gate, 0.0f, static_cast<int32_t>(this->HW));

        for (uint32_t ci = 0; ci < this->Cg; ++ci) {
            uint64_t chBase = groupBase + static_cast<uint64_t>(ci) * this->HW;

            // compute sum over HW for channel
            float sum = 0.0f;
            for (uint32_t hw = 0; hw < this->HW; ++hw) {
                sum += xGm.GetValue(chBase + hw);
            }
            float mean = sum * this->invHW;

            // gate += x * mean
            for (uint32_t hw = 0; hw < this->HW; ++hw) {
                float v = xGm.GetValue(chBase + hw);
                float acc = gate.GetValue(hw);
                gate.SetValue(hw, acc + v * mean);
            }
        }

        // Reduce across HW: meanT
        AscendC::ReduceSum<float>(tmp, gate, red, static_cast<int32_t>(this->HW));
        float meanT = tmp.GetValue(0) * this->invHW;

        // gate = gate - meanT
        AscendC::Duplicate(tmp, meanT, static_cast<int32_t>(this->HW));
        AscendC::Sub(gate, gate, tmp, static_cast<int32_t>(this->HW));

        // var = mean(gate^2)
        AscendC::Mul(tmp, gate, gate, static_cast<int32_t>(this->HW));
        AscendC::ReduceSum<float>(tmp, tmp, red, static_cast<int32_t>(this->HW));
        float var = tmp.GetValue(0) * this->invHW;

        // std = sqrt(var + eps)
        tmp.SetValue(0, var + this->epsilon);
        AscendC::Sqrt(tmp, tmp, 1);
        float stdv = tmp.GetValue(0);

        // gate = gate / std
        AscendC::Duplicate(tmp, stdv, static_cast<int32_t>(this->HW));
        AscendC::Div(gate, gate, tmp, static_cast<int32_t>(this->HW));

        // affine per group (broadcast to HW): gate = gate * w[g] + b[g]
        float wg = wGm.GetValue(g);
        float bg = bGm.GetValue(g);
        AscendC::Muls(gate, gate, wg, static_cast<int32_t>(this->HW));
        AscendC::Adds(gate, gate, bg, static_cast<int32_t>(this->HW));

        // sigmoid(gate) in-place
        AscendC::Sigmoid<float, true>(gate, gate, sigTmp, this->HW);

        // scale original x for all channels in group and write to y
        for (uint32_t ci = 0; ci < this->Cg; ++ci) {
            uint64_t chBase = groupBase + static_cast<uint64_t>(ci) * this->HW;
            for (uint32_t hw = 0; hw < this->HW; ++hw) {
                float xv = xGm.GetValue(chBase + hw);
                float s = gate.GetValue(hw);
                yGm.SetValue(chBase + hw, xv * s);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> gateBuf;
    AscendC::TBuf<> tmpBuf;
    AscendC::TBuf<> redBuf;
    AscendC::TBuf<> sigTmpBuf;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;

    uint32_t B, C, H, W;
    uint32_t G, Cg, HW, groupsTotal;
    uint32_t hwAlign;
    float invHW;
    float invCg;
    float epsilon;
    uint32_t sigTmpBytes;
};

extern "C" __global__ __aicore__ void spatial_group_enhance_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSpatialGroupEnhanceCustom op;
    op.Init(x, weight, bias, y,
            tiling_data.B, tiling_data.C, tiling_data.H, tiling_data.W,
            tiling_data.G, tiling_data.Cg, tiling_data.HW, tiling_data.groupsTotal,
            tiling_data.hwAlign, tiling_data.invHW, tiling_data.invCg, tiling_data.epsilon, tiling_data.sigTmpBytes);
    op.Process();
}
