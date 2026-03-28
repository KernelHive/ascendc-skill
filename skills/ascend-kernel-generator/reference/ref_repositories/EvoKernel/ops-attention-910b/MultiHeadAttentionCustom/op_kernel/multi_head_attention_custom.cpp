
#include "kernel_operator.h"

static constexpr uint32_t MAX_D = 128;
static constexpr uint32_t MAX_S = 512;

class KernelMultiHeadAttentionCustom {
public:
    __aicore__ inline KernelMultiHeadAttentionCustom() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                               uint32_t b, uint32_t h, uint32_t s, uint32_t d,
                               float scale)
    {
        this->b = b;
        this->h = h;
        this->s = s;
        this->d = d;
        this->scale = scale;

        const uint64_t total = static_cast<uint64_t>(b) * h * s * d;
        qGm.SetGlobalBuffer((__gm__ float*)q, total);
        kGm.SetGlobalBuffer((__gm__ float*)k, total);
        vGm.SetGlobalBuffer((__gm__ float*)v, total);
        oGm.SetGlobalBuffer((__gm__ float*)out, total);

        pipe.InitBuffer(bufQRow, MAX_D * sizeof(float));
        pipe.InitBuffer(bufKRow, MAX_D * sizeof(float));
        pipe.InitBuffer(bufVRow, MAX_D * sizeof(float));
        pipe.InitBuffer(bufOut,  MAX_D * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (s == 0 || d == 0) return;
        if (d > MAX_D || s > MAX_S) return;

        const uint64_t totalRows = static_cast<uint64_t>(b) * h * s;
        const uint32_t grid = static_cast<uint32_t>(AscendC::GetBlockNum());
        const uint32_t bid  = static_cast<uint32_t>(AscendC::GetBlockIdx());

        for (uint64_t linear = bid; linear < totalRows; linear += grid) {
            uint64_t t = linear;
            const uint32_t qi = static_cast<uint32_t>(t % s);
            t /= s;
            const uint32_t head = static_cast<uint32_t>(t % h);
            const uint32_t batch = static_cast<uint32_t>(t / h);

            const uint64_t bhBase = (static_cast<uint64_t>(batch) * h + head) * (static_cast<uint64_t>(s) * d);
            const uint64_t qOff = bhBase + static_cast<uint64_t>(qi) * d;

            // Load Q row once
            AscendC::LocalTensor<float> qRow = bufQRow.Get<float>();
            AscendC::DataCopy(qRow, qGm[qOff], d);
            AscendC::PipeBarrier<PIPE_MTE2>();

            AscendC::LocalTensor<float> out = bufOut.Get<float>();
            if (d == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; ++di) out(di) = 0.0f;
            } else {
                for (uint32_t di = 0; di < d; ++di) out(di) = 0.0f;
            }

            float m = -3.402823466e+38f; // -FLT_MAX
            float denom = 0.0f;

            AscendC::LocalTensor<float> kRow = bufKRow.Get<float>();
            AscendC::LocalTensor<float> vRow = bufVRow.Get<float>();

            // Hoist base pointer arithmetic
            const uint64_t rowStride = static_cast<uint64_t>(d);

            for (uint32_t j = 0; j < s; ++j) {
                const uint64_t kvOff = bhBase + static_cast<uint64_t>(j) * rowStride;

                // Paired loads then a single MTE2 barrier
                AscendC::DataCopy(kRow, kGm[kvOff], d);
                AscendC::DataCopy(vRow, vGm[kvOff], d);
                AscendC::PipeBarrier<PIPE_MTE2>();

                const float score = DotQK(qRow, kRow);

                // Online softmax update, but keep the D==64 update fully unrolled.
                if (score > m) {
                    const float alpha = (m == -3.402823466e+38f) ? 0.0f : FastExpApprox(m - score);
                    denom = denom * alpha + 1.0f;

                    if (d == 64u) {
                        #pragma unroll
                        for (uint32_t di = 0; di < 64u; ++di) {
                            out(di) = out(di) * alpha + vRow(di);
                        }
                    } else {
                        for (uint32_t di = 0; di < d; ++di) {
                            out(di) = out(di) * alpha + vRow(di);
                        }
                    }
                    m = score;
                } else {
                    const float p = FastExpApprox(score - m);
                    denom += p;

                    if (d == 64u) {
                        #pragma unroll
                        for (uint32_t di = 0; di < 64u; ++di) {
                            out(di) = out(di) + p * vRow(di);
                        }
                    } else {
                        for (uint32_t di = 0; di < d; ++di) {
                            out(di) = out(di) + p * vRow(di);
                        }
                    }
                }
                // No PIPE_V barrier here: we already ordered MTE2 and then scalar compute.
            }

            const float invDen = (denom == 0.0f) ? 0.0f : (1.0f / denom);
            if (d == 64u) {
                #pragma unroll
                for (uint32_t di = 0; di < 64u; ++di) out(di) = out(di) * invDen;
            } else {
                for (uint32_t di = 0; di < d; ++di) out(di) = out(di) * invDen;
            }

            AscendC::DataCopy(oGm[qOff], out, d);
            AscendC::PipeBarrier<PIPE_MTE3>();
        }
    }

private:
    __aicore__ inline float DotQK(const AscendC::LocalTensor<float>& qRow,
                                 const AscendC::LocalTensor<float>& kRow) const
    {
        float acc = 0.0f;
        if (d == 64u) {
            #pragma unroll
            for (uint32_t di = 0; di < 64u; di += 16u) {
                acc += qRow(di + 0u)  * kRow(di + 0u);
                acc += qRow(di + 1u)  * kRow(di + 1u);
                acc += qRow(di + 2u)  * kRow(di + 2u);
                acc += qRow(di + 3u)  * kRow(di + 3u);
                acc += qRow(di + 4u)  * kRow(di + 4u);
                acc += qRow(di + 5u)  * kRow(di + 5u);
                acc += qRow(di + 6u)  * kRow(di + 6u);
                acc += qRow(di + 7u)  * kRow(di + 7u);
                acc += qRow(di + 8u)  * kRow(di + 8u);
                acc += qRow(di + 9u)  * kRow(di + 9u);
                acc += qRow(di + 10u) * kRow(di + 10u);
                acc += qRow(di + 11u) * kRow(di + 11u);
                acc += qRow(di + 12u) * kRow(di + 12u);
                acc += qRow(di + 13u) * kRow(di + 13u);
                acc += qRow(di + 14u) * kRow(di + 14u);
                acc += qRow(di + 15u) * kRow(di + 15u);
            }
        } else {
            for (uint32_t di = 0; di < d; ++di) acc += qRow(di) * kRow(di);
        }
        return acc * scale;
    }

    __aicore__ inline float FastExpApprox(float x) const
    {
        if (x < -20.0f) x = -20.0f;
        if (x >  0.0f)  x =  0.0f;

        const float inv_ln2 = 1.4426950408889634f;
        float y = x * inv_ln2;

        int32_t n = (int32_t)(y);
        float f = y - (float)n; // [0,1)

        const float c1 = 0.69314718056f;
        const float c2 = 0.24022650695f;
        const float c3 = 0.05550410866f;
        float p = 1.0f + f * (c1 + f * (c2 + f * c3));

        int32_t e = n + 127;
        if (e <= 0) return 0.0f;
        if (e >= 255) e = 255;
        uint32_t bits = ((uint32_t)e) << 23;
        union { uint32_t u; float f; } u;
        u.u = bits;
        return u.f * p;
    }

private:
    AscendC::TPipe pipe;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufQRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufKRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufVRow;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufOut;

    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> oGm;

    uint32_t b, h, s, d;
    float scale;
};

extern "C" __global__ __aicore__ void multi_head_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiHeadAttentionCustom op;
    op.Init(q, k, v, out,
            tiling_data.b, tiling_data.h, tiling_data.s, tiling_data.d,
            tiling_data.scale);
    op.Process();
}
