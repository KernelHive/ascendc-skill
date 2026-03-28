
#include "kernel_operator.h"

static constexpr uint32_t MAX_TOPK = 8;
static constexpr uint32_t MAX_K    = 4096;
static constexpr uint32_t K_TILE   = 1024; // must match host tiling

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

// Still uses safe scalar bf16<->f32 conversion, but eliminates scalar GM loads by
// bulk-copying expert_output tiles into UB (MTE2) and computing from UB.
class KernelUnPermuteCustom {
public:
    __aicore__ inline KernelUnPermuteCustom() {}

    __aicore__ inline void Init(GM_ADDR expert_output,
                               GM_ADDR topk_vals,
                               GM_ADDR inv_perm,
                               GM_ADDR out,
                               uint32_t m, uint32_t k, uint32_t topk,
                               uint32_t kTile,
                               AscendC::TPipe* pipe)
    {
        this->m = m;
        this->k = k;
        this->topk = topk;
        this->kTile = kTile;
        this->pipe = pipe;

        expertGm.SetGlobalBuffer((__gm__ uint16_t*)expert_output,
                                 (uint64_t)m /*not used*/);
        topkGm.SetGlobalBuffer((__gm__ uint16_t*)topk_vals,
                               (uint64_t)m * (uint64_t)topk);
        invGm.SetGlobalBuffer((__gm__ int64_t*)inv_perm,
                              (uint64_t)m * (uint64_t)topk);
        outGm.SetGlobalBuffer((__gm__ uint16_t*)out,
                              (uint64_t)m * (uint64_t)k);

        // UB buffers:
        // - acc: FP32 accumulate
        // - in : BF16 tile from expert_output row
        // - out: BF16 tile to write
        pipe->InitBuffer(accBuf, K_TILE * (uint32_t)sizeof(float));
        pipe->InitBuffer(inBuf,  K_TILE * (uint32_t)sizeof(uint16_t));
        pipe->InitBuffer(outBuf, K_TILE * (uint32_t)sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();
        const uint32_t coreIdx = (uint32_t)AscendC::GetBlockIdx();
        const uint32_t cores = (coreNum == 0 ? 1u : coreNum);

        const uint32_t perCore = CeilDivU32(m, cores);
        const uint32_t mStart = coreIdx * perCore;
        uint32_t mEnd = mStart + perCore;
        if (mEnd > m) mEnd = m;

        AscendC::LocalTensor<float> accTile = accBuf.Get<float>();
        AscendC::LocalTensor<uint16_t> inTile = inBuf.Get<uint16_t>();
        AscendC::LocalTensor<uint16_t> outTile = outBuf.Get<uint16_t>();

        for (uint32_t mi = mStart; mi < mEnd; ++mi) {
            const uint64_t invBase = (uint64_t)mi * (uint64_t)topk;
            const uint64_t wBase   = (uint64_t)mi * (uint64_t)topk;
            const uint64_t outBase = (uint64_t)mi * (uint64_t)k;

            for (uint32_t k0 = 0; k0 < k; k0 += kTile) {
                const uint32_t curK = (k0 + kTile <= k) ? kTile : (k - k0);

                AscendC::Duplicate<float>(accTile, 0.0f, (int32_t)curK);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t t = 0; t < topk; ++t) {
                    const uint64_t row = (uint64_t)invGm.GetValue(invBase + t);
                    const float w = Bf16ToF32(topkGm.GetValue(wBase + t));

                    const uint64_t expOff = row * (uint64_t)k + (uint64_t)k0;

                    // Bulk load BF16 tile from GM into UB.
                    AscendC::DataCopy(inTile, expertGm[expOff], curK);
                    AscendC::PipeBarrier<PIPE_MTE2>();

                    // UB compute: acc += bf16_to_f32(inTile) * w
                    for (uint32_t di = 0; di < curK; ++di) {
                        accTile(di) = accTile(di) + Bf16ToF32(inTile(di)) * w;
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }

                // Convert to bf16 tile and store.
                for (uint32_t di = 0; di < curK; ++di) {
                    outTile(di) = F32ToBf16(accTile(di));
                }
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::DataCopy(outGm[outBase + (uint64_t)k0], outTile, curK);
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        }
    }

private:
    __aicore__ inline float Bf16ToF32(uint16_t v) const
    {
        union { uint32_t u; float f; } x;
        x.u = ((uint32_t)v) << 16;
        return x.f;
    }

    __aicore__ inline uint16_t F32ToBf16(float f) const
    {
        union { uint32_t u; float f; } x;
        x.f = f;
        const uint32_t lsb = (x.u >> 16) & 1u;
        const uint32_t bias = 0x7FFFu + lsb;
        const uint32_t rounded = x.u + bias;
        return (uint16_t)(rounded >> 16);
    }

private:
    AscendC::TPipe* pipe = nullptr;

    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;

    AscendC::GlobalTensor<uint16_t> expertGm; // bf16 bits
    AscendC::GlobalTensor<uint16_t> topkGm;   // bf16 bits
    AscendC::GlobalTensor<int64_t>  invGm;
    AscendC::GlobalTensor<uint16_t> outGm;    // bf16 bits

    uint32_t m = 0, k = 0, topk = 0, kTile = 0;
};

extern "C" __global__ __aicore__ void un_permute_custom(GM_ADDR expert_output,
                                                       GM_ADDR topk_vals,
                                                       GM_ADDR inv_perm,
                                                       GM_ADDR out,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);

    AscendC::TPipe pipe;
    KernelUnPermuteCustom op;
    op.Init(expert_output, topk_vals, inv_perm, out,
            td.m, td.k, td.topk, td.kTile,
            &pipe);
    op.Process();
}
