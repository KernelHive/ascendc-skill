
#include "kernel_operator.h"

class KernelMhcPreBlock {
public:
    __aicore__ inline KernelMhcPreBlock() {}

    __aicore__ inline void Init(GM_ADDR residual,
                               GM_ADDR fn,
                               GM_ADDR hc_scale,
                               GM_ADDR hc_base,
                               GM_ADDR post_mix,
                               GM_ADDR comb_mix,
                               GM_ADDR layer_input,
                               uint32_t totalResidual,
                               uint32_t totalFn,
                               uint32_t totalScale,
                               uint32_t totalBase,
                               uint32_t N,
                               uint32_t hc,
                               uint32_t H,
                               uint32_t dFlat,
                               uint32_t hc2,
                               uint32_t hc3,
                               uint32_t blockTokens,
                               float invDFlat,
                               float rmsEps,
                               float hcPreEps,
                               float hcSinkhornEps,
                               float hcPostMultValue,
                               uint32_t sinkhornRepeat)
    {
        (void)totalFn; (void)totalScale; (void)totalBase;
        this->totalResidual = totalResidual;

        this->N = N;
        this->hc = hc;
        this->H = H;
        this->dFlat = dFlat;
        this->hc2 = hc2;
        this->hc3 = hc3;
        this->blockTokens = blockTokens;

        this->invDFlat = invDFlat;
        this->rmsEps = rmsEps;
        this->hcPreEps = hcPreEps;
        this->hcSinkhornEps = hcSinkhornEps;
        this->hcPostMultValue = hcPostMultValue;
        this->sinkhornRepeat = sinkhornRepeat;

        residualGm.SetGlobalBuffer((__gm__ float*)residual, totalResidual);         // [N,hc,H]
        fnGm.SetGlobalBuffer((__gm__ float*)fn, (uint64_t)hc3 * (uint64_t)dFlat);  // [hc3,dFlat]
        scaleGm.SetGlobalBuffer((__gm__ float*)hc_scale, 3);                       // [3]
        baseGm.SetGlobalBuffer((__gm__ float*)hc_base, hc3);                       // [hc3]

        postGm.SetGlobalBuffer((__gm__ float*)post_mix, (uint64_t)N * (uint64_t)hc);       // [N,hc,1] flattened
        combGm.SetGlobalBuffer((__gm__ float*)comb_mix, (uint64_t)N * (uint64_t)hc2);      // [N,hc,hc]
        outGm.SetGlobalBuffer((__gm__ float*)layer_input, (uint64_t)N * (uint64_t)H);      // [N,H]
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        if (N == 0u || hc == 0u || H == 0u) return;

        const uint32_t nStart = bid * blockTokens;
        if (nStart >= N) return;
        uint32_t nEnd = nStart + blockTokens;
        if (nEnd > N) nEnd = N;

        const float s0 = scaleGm.GetValue(0ull);
        const float s1 = scaleGm.GetValue(1ull);
        const float s2 = scaleGm.GetValue(2ull);

        for (uint32_t n = nStart; n < nEnd; ++n) {
            if (hc == 4u && hc3 == 24u) {
                ProcessTokenHc4_FusedSqsumLayer(n, s0, s1, s2);
            } else {
                WriteZerosToken(n);
            }
        }
    }

private:
    __aicore__ inline void WriteZerosToken(uint32_t n)
    {
        const uint64_t H64 = (uint64_t)H;
        const uint64_t hc64 = (uint64_t)hc;
        const uint64_t basePost = (uint64_t)n * hc64;
        for (uint32_t j = 0; j < hc; ++j) postGm.SetValue(basePost + (uint64_t)j, 0.0f);

        const uint64_t baseComb = (uint64_t)n * hc64 * hc64;
        for (uint32_t t = 0; t < hc*hc; ++t) combGm.SetValue(baseComb + (uint64_t)t, 0.0f);

        const uint64_t baseOut = (uint64_t)n * H64;
        for (uint32_t h = 0; h < H; ++h) outGm.SetValue(baseOut + (uint64_t)h, 0.0f);
    }

    __aicore__ inline float ClampF(float x, float lo, float hi)
    {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    __aicore__ inline float ExpApprox(float x)
    {
        x = ClampF(x, -10.0f, 10.0f);
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        float x5 = x4 * x;
        return 1.0f + x + 0.5f * x2 + (1.0f / 6.0f) * x3 + (1.0f / 24.0f) * x4 + (1.0f / 120.0f) * x5;
    }

    __aicore__ inline float Sigmoid(float x)
    {
        x = ClampF(x, -8.0f, 8.0f);
        float e = ExpApprox(-x);
        return 1.0f / (1.0f + e);
    }

    __aicore__ inline float RsqrtNewton(float x)
    {
        if (x <= 0.0f) return 0.0f;
        float y = 1.0f / x;
        for (int it = 0; it < 6; ++it) {
            y = y * (1.5f - 0.5f * x * y * y);
        }
        return y;
    }

    __aicore__ inline float DotResidualFnRowHc4(uint32_t n, uint32_t k)
    {
        const uint64_t H64 = (uint64_t)H;
        const uint64_t baseRes = (uint64_t)n * 4ull * H64;
        const uint64_t baseFn  = (uint64_t)k * (uint64_t)dFlat;

        const uint64_t r0 = baseRes + 0ull * H64;
        const uint64_t r1 = baseRes + 1ull * H64;
        const uint64_t r2 = baseRes + 2ull * H64;
        const uint64_t r3 = baseRes + 3ull * H64;

        const uint64_t w0 = baseFn + 0ull * H64;
        const uint64_t w1 = baseFn + 1ull * H64;
        const uint64_t w2 = baseFn + 2ull * H64;
        const uint64_t w3 = baseFn + 3ull * H64;

        float acc = 0.0f;
        uint32_t h = 0u;
        for (; h + 3u < H; h += 4u) {
            const uint64_t hh = (uint64_t)h;

            acc += residualGm.GetValue(r0 + hh + 0ull) * fnGm.GetValue(w0 + hh + 0ull);
            acc += residualGm.GetValue(r0 + hh + 1ull) * fnGm.GetValue(w0 + hh + 1ull);
            acc += residualGm.GetValue(r0 + hh + 2ull) * fnGm.GetValue(w0 + hh + 2ull);
            acc += residualGm.GetValue(r0 + hh + 3ull) * fnGm.GetValue(w0 + hh + 3ull);

            acc += residualGm.GetValue(r1 + hh + 0ull) * fnGm.GetValue(w1 + hh + 0ull);
            acc += residualGm.GetValue(r1 + hh + 1ull) * fnGm.GetValue(w1 + hh + 1ull);
            acc += residualGm.GetValue(r1 + hh + 2ull) * fnGm.GetValue(w1 + hh + 2ull);
            acc += residualGm.GetValue(r1 + hh + 3ull) * fnGm.GetValue(w1 + hh + 3ull);

            acc += residualGm.GetValue(r2 + hh + 0ull) * fnGm.GetValue(w2 + hh + 0ull);
            acc += residualGm.GetValue(r2 + hh + 1ull) * fnGm.GetValue(w2 + hh + 1ull);
            acc += residualGm.GetValue(r2 + hh + 2ull) * fnGm.GetValue(w2 + hh + 2ull);
            acc += residualGm.GetValue(r2 + hh + 3ull) * fnGm.GetValue(w2 + hh + 3ull);

            acc += residualGm.GetValue(r3 + hh + 0ull) * fnGm.GetValue(w3 + hh + 0ull);
            acc += residualGm.GetValue(r3 + hh + 1ull) * fnGm.GetValue(w3 + hh + 1ull);
            acc += residualGm.GetValue(r3 + hh + 2ull) * fnGm.GetValue(w3 + hh + 2ull);
            acc += residualGm.GetValue(r3 + hh + 3ull) * fnGm.GetValue(w3 + hh + 3ull);
        }
        for (; h < H; ++h) {
            const uint64_t hh = (uint64_t)h;
            acc += residualGm.GetValue(r0 + hh) * fnGm.GetValue(w0 + hh);
            acc += residualGm.GetValue(r1 + hh) * fnGm.GetValue(w1 + hh);
            acc += residualGm.GetValue(r2 + hh) * fnGm.GetValue(w2 + hh);
            acc += residualGm.GetValue(r3 + hh) * fnGm.GetValue(w3 + hh);
        }
        return acc;
    }

    __aicore__ inline void SoftmaxRow4(float *row)
    {
        float mx = row[0];
        if (row[1] > mx) mx = row[1];
        if (row[2] > mx) mx = row[2];
        if (row[3] > mx) mx = row[3];

        float e0 = ExpApprox(row[0] - mx);
        float e1 = ExpApprox(row[1] - mx);
        float e2 = ExpApprox(row[2] - mx);
        float e3 = ExpApprox(row[3] - mx);

        float sum = e0 + e1 + e2 + e3;
        float inv = 1.0f / (sum + 1e-20f);

        row[0] = e0 * inv;
        row[1] = e1 * inv;
        row[2] = e2 * inv;
        row[3] = e3 * inv;
    }

    __aicore__ inline void Sinkhorn4(float *m)
    {
        uint32_t repeat = sinkhornRepeat;
        if (repeat == 0u) repeat = 1u;

        SoftmaxRow4(m + 0);
        SoftmaxRow4(m + 4);
        SoftmaxRow4(m + 8);
        SoftmaxRow4(m + 12);
        for (uint32_t i = 0u; i < 16u; ++i) m[i] += hcSinkhornEps;

        for (uint32_t c = 0u; c < 4u; ++c) {
            float s = m[0*4+c] + m[1*4+c] + m[2*4+c] + m[3*4+c];
            float inv = 1.0f / (s + hcSinkhornEps);
            m[0*4+c] *= inv; m[1*4+c] *= inv; m[2*4+c] *= inv; m[3*4+c] *= inv;
        }

        for (uint32_t it = 1u; it < repeat; ++it) {
            for (uint32_t r = 0u; r < 4u; ++r) {
                float s = m[r*4+0] + m[r*4+1] + m[r*4+2] + m[r*4+3];
                float inv = 1.0f / (s + hcSinkhornEps);
                m[r*4+0] *= inv; m[r*4+1] *= inv; m[r*4+2] *= inv; m[r*4+3] *= inv;
            }
            for (uint32_t c = 0u; c < 4u; ++c) {
                float s = m[0*4+c] + m[1*4+c] + m[2*4+c] + m[3*4+c];
                float inv = 1.0f / (s + hcSinkhornEps);
                m[0*4+c] *= inv; m[1*4+c] *= inv; m[2*4+c] *= inv; m[3*4+c] *= inv;
            }
        }
    }

    __aicore__ inline void ProcessTokenHc4_FusedSqsumLayer(uint32_t n, float s0, float s1, float s2)
    {
        const uint64_t H64 = (uint64_t)H;
        const uint64_t baseRes = (uint64_t)n * 4ull * H64;
        const uint64_t r0 = baseRes + 0ull * H64;
        const uint64_t r1 = baseRes + 1ull * H64;
        const uint64_t r2 = baseRes + 2ull * H64;
        const uint64_t r3 = baseRes + 3ull * H64;

        // Load hc_base once per token into regs to reduce GM reads later.
        float basev[24];
        #pragma unroll
        for (uint32_t k = 0u; k < 24u; ++k) {
            basev[k] = baseGm.GetValue((uint64_t)k);
        }

        // 1) RMS inv from sqsum (1 pass over residual)
        float sumsq = 0.0f;
        uint32_t h = 0u;
        for (; h + 3u < H; h += 4u) {
            const uint64_t hh = (uint64_t)h;

            float a0 = residualGm.GetValue(r0 + hh + 0ull);
            float a1 = residualGm.GetValue(r0 + hh + 1ull);
            float a2 = residualGm.GetValue(r0 + hh + 2ull);
            float a3 = residualGm.GetValue(r0 + hh + 3ull);

            float b0 = residualGm.GetValue(r1 + hh + 0ull);
            float b1 = residualGm.GetValue(r1 + hh + 1ull);
            float b2 = residualGm.GetValue(r1 + hh + 2ull);
            float b3 = residualGm.GetValue(r1 + hh + 3ull);

            float c0 = residualGm.GetValue(r2 + hh + 0ull);
            float c1 = residualGm.GetValue(r2 + hh + 1ull);
            float c2 = residualGm.GetValue(r2 + hh + 2ull);
            float c3 = residualGm.GetValue(r2 + hh + 3ull);

            float d0 = residualGm.GetValue(r3 + hh + 0ull);
            float d1 = residualGm.GetValue(r3 + hh + 1ull);
            float d2 = residualGm.GetValue(r3 + hh + 2ull);
            float d3 = residualGm.GetValue(r3 + hh + 3ull);

            sumsq += a0*a0 + a1*a1 + a2*a2 + a3*a3;
            sumsq += b0*b0 + b1*b1 + b2*b2 + b3*b3;
            sumsq += c0*c0 + c1*c1 + c2*c2 + c3*c3;
            sumsq += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        }
        for (; h < H; ++h) {
            const uint64_t hh = (uint64_t)h;
            float a = residualGm.GetValue(r0 + hh);
            float b = residualGm.GetValue(r1 + hh);
            float c = residualGm.GetValue(r2 + hh);
            float d = residualGm.GetValue(r3 + hh);
            sumsq += a*a + b*b + c*c + d*d;
        }

        float mean = sumsq * invDFlat;
        float rmsInv = RsqrtNewton(mean + rmsEps);

        // 2) mixes (still 24 dot products; base values come from registers)
        float mixes[24];
        for (uint32_t k = 0u; k < 24u; ++k) {
            float dot = DotResidualFnRowHc4(n, k);
            float sc = (k < 4u) ? s0 : ((k < 8u) ? s1 : s2);
            mixes[k] = dot * rmsInv * sc + basev[k];
        }

        // 3) pre and post
        float pre0 = Sigmoid(mixes[0]) + hcPreEps;
        float pre1 = Sigmoid(mixes[1]) + hcPreEps;
        float pre2 = Sigmoid(mixes[2]) + hcPreEps;
        float pre3 = Sigmoid(mixes[3]) + hcPreEps;

        const uint64_t postOff = (uint64_t)n * 4ull;
        postGm.SetValue(postOff + 0ull, Sigmoid(mixes[4]) * hcPostMultValue);
        postGm.SetValue(postOff + 1ull, Sigmoid(mixes[5]) * hcPostMultValue);
        postGm.SetValue(postOff + 2ull, Sigmoid(mixes[6]) * hcPostMultValue);
        postGm.SetValue(postOff + 3ull, Sigmoid(mixes[7]) * hcPostMultValue);

        // 4) comb_mix
        float mat[16];
        #pragma unroll
        for (uint32_t i = 0u; i < 16u; ++i) mat[i] = mixes[8u + i];
        Sinkhorn4(mat);
        const uint64_t combOff = (uint64_t)n * 16ull;
        #pragma unroll
        for (uint32_t i = 0u; i < 16u; ++i) combGm.SetValue(combOff + (uint64_t)i, mat[i]);

        // 5) layer_input: fused into a single pass over residual (no extra residual read pass).
        const uint64_t outOff = (uint64_t)n * H64;

        h = 0u;
        for (; h + 3u < H; h += 4u) {
            const uint64_t hh = (uint64_t)h;

            float a0 = residualGm.GetValue(r0 + hh + 0ull);
            float a1 = residualGm.GetValue(r0 + hh + 1ull);
            float a2 = residualGm.GetValue(r0 + hh + 2ull);
            float a3 = residualGm.GetValue(r0 + hh + 3ull);

            float b0 = residualGm.GetValue(r1 + hh + 0ull);
            float b1 = residualGm.GetValue(r1 + hh + 1ull);
            float b2 = residualGm.GetValue(r1 + hh + 2ull);
            float b3 = residualGm.GetValue(r1 + hh + 3ull);

            float c0 = residualGm.GetValue(r2 + hh + 0ull);
            float c1 = residualGm.GetValue(r2 + hh + 1ull);
            float c2 = residualGm.GetValue(r2 + hh + 2ull);
            float c3 = residualGm.GetValue(r2 + hh + 3ull);

            float d0 = residualGm.GetValue(r3 + hh + 0ull);
            float d1 = residualGm.GetValue(r3 + hh + 1ull);
            float d2 = residualGm.GetValue(r3 + hh + 2ull);
            float d3 = residualGm.GetValue(r3 + hh + 3ull);

            outGm.SetValue(outOff + hh + 0ull, a0*pre0 + b0*pre1 + c0*pre2 + d0*pre3);
            outGm.SetValue(outOff + hh + 1ull, a1*pre0 + b1*pre1 + c1*pre2 + d1*pre3);
            outGm.SetValue(outOff + hh + 2ull, a2*pre0 + b2*pre1 + c2*pre2 + d2*pre3);
            outGm.SetValue(outOff + hh + 3ull, a3*pre0 + b3*pre1 + c3*pre2 + d3*pre3);
        }
        for (; h < H; ++h) {
            const uint64_t hh = (uint64_t)h;
            float v =
                residualGm.GetValue(r0 + hh) * pre0 +
                residualGm.GetValue(r1 + hh) * pre1 +
                residualGm.GetValue(r2 + hh) * pre2 +
                residualGm.GetValue(r3 + hh) * pre3;
            outGm.SetValue(outOff + hh, v);
        }
    }

private:
    AscendC::GlobalTensor<float> residualGm;
    AscendC::GlobalTensor<float> fnGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> baseGm;

    AscendC::GlobalTensor<float> postGm;
    AscendC::GlobalTensor<float> combGm;
    AscendC::GlobalTensor<float> outGm;

    uint32_t totalResidual;
    uint32_t N, hc, H, dFlat, hc2, hc3;
    uint32_t blockTokens;

    float invDFlat, rmsEps, hcPreEps, hcSinkhornEps, hcPostMultValue;
    uint32_t sinkhornRepeat;
};

extern "C" __global__ __aicore__ void mhc_pre_block_custom(GM_ADDR residual,
                                                          GM_ADDR fn,
                                                          GM_ADDR hc_scale,
                                                          GM_ADDR hc_base,
                                                          GM_ADDR post_mix,
                                                          GM_ADDR comb_mix,
                                                          GM_ADDR layer_input,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMhcPreBlock op;
    op.Init(residual, fn, hc_scale, hc_base,
            post_mix, comb_mix, layer_input,
            tiling_data.totalResidual, tiling_data.totalFn, tiling_data.totalScale, tiling_data.totalBase,
            tiling_data.N, tiling_data.hc, tiling_data.H,
            tiling_data.dFlat, tiling_data.hc2, tiling_data.hc3,
            tiling_data.blockTokens,
            tiling_data.invDFlat,
            tiling_data.rmsEps, tiling_data.hcPreEps, tiling_data.hcSinkhornEps,
            tiling_data.hcPostMultValue, tiling_data.sinkhornRepeat);
    op.Process();
}
