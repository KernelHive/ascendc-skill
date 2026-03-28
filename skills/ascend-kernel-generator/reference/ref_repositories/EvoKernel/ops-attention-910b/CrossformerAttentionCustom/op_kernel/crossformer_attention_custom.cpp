
#include "kernel_operator.h"
#include <cfloat>

class KernelCrossformerAttentionCustom {
public:
    __aicore__ inline KernelCrossformerAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR attn, GM_ADDR v, GM_ADDR w, GM_ADDR b, GM_ADDR y,
                               uint32_t B, uint32_t H, uint32_t N, uint32_t Dh, uint32_t C,
                               uint32_t totalRows, uint32_t blockRows, uint32_t dhTile)
    {
        this->B = B; this->H = H; this->N = N; this->Dh = Dh; this->C = C;
        this->totalRows = totalRows;
        this->blockRows = blockRows;
        this->dhTile = dhTile;

        const uint64_t attnSize = (uint64_t)B * (uint64_t)H * (uint64_t)N * (uint64_t)N;
        const uint64_t vSize    = (uint64_t)B * (uint64_t)H * (uint64_t)N * (uint64_t)Dh;
        const uint64_t wSize    = (uint64_t)C * (uint64_t)C;
        const uint64_t bSize    = (uint64_t)C;
        const uint64_t ySize    = (uint64_t)B * (uint64_t)N * (uint64_t)C;

        attnGm.SetGlobalBuffer((__gm__ float*)attn, attnSize);
        vGm.SetGlobalBuffer((__gm__ float*)v, vSize);
        wGm.SetGlobalBuffer((__gm__ float*)w, wSize);
        bGm.SetGlobalBuffer((__gm__ float*)b, bSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        // UB: per head we need 3*dhTile floats: attnRow(N) streamed from GM (scalar),
        // vTile(dhTile) load per m, ctxTile(dhTile) accum.
        // Additionally we keep one vTile buffer and one ctx buffer. Keep small and deterministic.
        pipe.InitBuffer(ubuf, (uint64_t)(2 * this->dhTile) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId  = (uint32_t)AscendC::GetBlockIdx();
        uint32_t coreNum = (uint32_t)AscendC::GetBlockNum();
        if (coreNum == 0) return;

        uint32_t startRow = coreId * this->blockRows;
        uint32_t endRow = startRow + this->blockRows;
        if (endRow > this->totalRows) endRow = this->totalRows;
        if (startRow >= endRow) return;

        for (uint32_t row = startRow; row < endRow; ++row) {
            ComputeRow(row);
        }
    }

private:
    __aicore__ inline void ComputeRow(uint32_t rowBN)
    {
        // rowBN maps to (b,n)
        uint32_t b = rowBN / this->N;
        uint32_t n = rowBN - b * this->N;

        // Output y row base [C]
        const uint64_t yBase = ((uint64_t)b * (uint64_t)this->N + (uint64_t)n) * (uint64_t)this->C;

        // We compute ctx[b,n,c_in] for c_in=0..C-1 (in head-major order),
        // then y[c_out] = sum_{c_in} ctx[c_in] * w[c_out, c_in] + bias[c_out]
        // with w layout same as PyTorch Linear weight [C_out, C_in].

        // Because C=512 in the benchmark, correctness-first scalar loops are acceptable.
        // For better performance later: replace projection with Matmul API, and keep ctx in UB/GM.
        for (uint32_t c_out = 0; c_out < this->C; ++c_out) {
            float accOut = bGm.GetValue(c_out);

            for (uint32_t c_in = 0; c_in < this->C; ++c_in) {
                // decode head and dh
                uint32_t h  = c_in / this->Dh;
                uint32_t dh = c_in - h * this->Dh;

                // ctx = sum_m attn[b,h,n,m] * v[b,h,m,dh]
                // We compute ctx in dh-tiled manner; here we only need one dh scalar, so do scalar over m.
                const uint64_t attnRowBase = (((uint64_t)b * (uint64_t)this->H + (uint64_t)h) * (uint64_t)this->N + (uint64_t)n) * (uint64_t)this->N;
                const uint64_t vHeadBase   = (((uint64_t)b * (uint64_t)this->H + (uint64_t)h) * (uint64_t)this->N) * (uint64_t)this->Dh;

                float ctx = 0.0f;
                for (uint32_t m = 0; m < this->N; ++m) {
                    float a = attnGm.GetValue(attnRowBase + (uint64_t)m);
                    float vv = vGm.GetValue(vHeadBase + (uint64_t)m * (uint64_t)this->Dh + (uint64_t)dh);
                    ctx += a * vv;
                }

                float wv = wGm.GetValue((uint64_t)c_out * (uint64_t)this->C + (uint64_t)c_in);
                accOut += ctx * wv;
            }

            yGm.SetValue(yBase + (uint64_t)c_out, accOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> ubuf;

    AscendC::GlobalTensor<float> attnGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t B{0}, H{0}, N{0}, Dh{0}, C{0};
    uint32_t totalRows{0}, blockRows{0}, dhTile{0};
};

extern "C" __global__ __aicore__ void crossformer_attention_custom(
    GM_ADDR attn, GM_ADDR v, GM_ADDR proj_weight, GM_ADDR proj_bias,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossformerAttentionCustom op;
    op.Init(attn, v, proj_weight, proj_bias, y,
            tiling_data.B, tiling_data.H, tiling_data.N, tiling_data.Dh, tiling_data.C,
            tiling_data.totalRows, tiling_data.blockRows, tiling_data.dhTile);
    op.Process();
}
