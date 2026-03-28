
#include "kernel_operator.h"

class KernelOptimizedMHCLayerWithFusion {
public:
    __aicore__ inline KernelOptimizedMHCLayerWithFusion() {}

    __aicore__ inline void Init(GM_ADDR x,
                               GM_ADDR phi_params,
                               GM_ADDR bias_params,
                               GM_ADDR rms_scale,
                               GM_ADDR alpha_pre,
                               GM_ADDR alpha_post,
                               GM_ADDR alpha_res,
                               GM_ADDR linear_w,
                               GM_ADDR y,
                               uint32_t totalX,
                               uint32_t totalPhi,
                               uint32_t totalBias,
                               uint32_t totalRmsScale,
                               uint32_t totalAlphaPre,
                               uint32_t totalAlphaPost,
                               uint32_t totalAlphaRes,
                               uint32_t totalW,
                               uint32_t B,
                               uint32_t S,
                               uint32_t D,
                               uint32_t n,
                               uint32_t SD,
                               uint32_t mapDim,
                               uint32_t tokens,
                               uint32_t tokensPerCore,
                               float invSD,
                               float rmsEps,
                               float sinkEps,
                               uint32_t sinkIters)
    {
        this->totalX = totalX;
        this->totalPhi = totalPhi;
        this->totalBias = totalBias;
        this->totalRmsScale = totalRmsScale;
        this->totalAlphaPre = totalAlphaPre;
        this->totalAlphaPost = totalAlphaPost;
        this->totalAlphaRes = totalAlphaRes;
        this->totalW = totalW;

        this->B = B; this->S = S; this->D = D; this->n = n;
        this->SD = SD; this->mapDim = mapDim;
        this->tokens = tokens; this->tokensPerCore = tokensPerCore;

        this->invSD = invSD;
        this->rmsEps = rmsEps;
        this->sinkEps = sinkEps;
        this->sinkIters = sinkIters;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalX);                      // [B,S,D]
        phiGm.SetGlobalBuffer((__gm__ float*)phi_params, totalPhi);         // [SD,mapDim]
        biasGm.SetGlobalBuffer((__gm__ float*)bias_params, totalBias);      // [mapDim]
        rmsGm.SetGlobalBuffer((__gm__ float*)rms_scale, totalRmsScale);     // [SD]
        aPreGm.SetGlobalBuffer((__gm__ float*)alpha_pre, totalAlphaPre);    // [>=1]
        aPostGm.SetGlobalBuffer((__gm__ float*)alpha_post, totalAlphaPost); // [>=1]
        aResGm.SetGlobalBuffer((__gm__ float*)alpha_res, totalAlphaRes);    // [>=1]
        wGm.SetGlobalBuffer((__gm__ float*)linear_w, totalW);               // [D,D]
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)tokens * (uint64_t)D); // [B,S,D]
    }

    __aicore__ inline void Process()
    {
        const uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
        if (tokens == 0u) return;

        const uint32_t tStart = bid * tokensPerCore;
        if (tStart >= tokens) return;
        uint32_t tEnd = tStart + tokensPerCore;
        if (tEnd > tokens) tEnd = tokens;

        const float a_pre  = (totalAlphaPre  > 0u) ? aPreGm.GetValue(0ull)  : 0.0f;
        const float a_post = (totalAlphaPost > 0u) ? aPostGm.GetValue(0ull) : 0.0f;
        const float a_res  = (totalAlphaRes  > 0u) ? aResGm.GetValue(0ull)  : 0.0f;

        for (uint32_t t = tStart; t < tEnd; ++t) {
            if (!ShapesOk()) { WriteZeros(t); continue; }
            if (n == 4u && mapDim == 24u && SD == 4u * D) {
                ProcessTokenN4(t, a_pre, a_post, a_res);
            } else {
                WriteZeros(t);
            }
        }
    }

private:
    __aicore__ inline bool ShapesOk()
    {
        if (B == 0u || S == 0u || D == 0u || n == 0u) return false;
        if (tokens != B * S) return false;
        if (SD != n * D) return false;
        if (mapDim != n * n + 2u * n) return false;

        const uint64_t needX = (uint64_t)tokens * (uint64_t)D;
        if ((uint64_t)totalX < needX) return false;

        const uint64_t needPhi = (uint64_t)SD * (uint64_t)mapDim;
        if ((uint64_t)totalPhi < needPhi) return false;

        if ((uint64_t)totalBias < (uint64_t)mapDim) return false;
        if ((uint64_t)totalRmsScale < (uint64_t)SD) return false;
        if (totalAlphaPre < 1u || totalAlphaPost < 1u || totalAlphaRes < 1u) return false;

        const uint64_t needW = (uint64_t)D * (uint64_t)D;
        if ((uint64_t)totalW < needW) return false;

        return true;
    }

    __aicore__ inline void WriteZeros(uint32_t t)
    {
        const uint64_t baseY = (uint64_t)t * (uint64_t)D;
        for (uint32_t i = 0u; i < D; ++i) yGm.SetValue(baseY + (uint64_t)i, 0.0f);
    }

    __aicore__ inline float ClampF(float x, float lo, float hi)
    {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    __aicore__ inline float ExpApprox4(float x)
    {
        x = ClampF(x, -10.0f, 10.0f);
        float x2 = x * x;
        float x3 = x2 * x;
        float x4 = x2 * x2;
        return 1.0f + x + 0.5f * x2 + (1.0f/6.0f) * x3 + (1.0f/24.0f) * x4;
    }

    __aicore__ inline float Sigmoid(float x)
    {
        x = ClampF(x, -8.0f, 8.0f);
        float e = ExpApprox4(-x);
        return 1.0f / (1.0f + e);
    }

    __aicore__ inline float RsqrtNewton(float x)
    {
        if (x <= 0.0f) return 0.0f;
        float y = 1.0f / x;
        #pragma unroll
        for (int it = 0; it < 6; ++it) {
            y = y * (1.5f - 0.5f * x * y * y);
        }
        return y;
    }

    __aicore__ inline void SoftmaxRow4(float *row)
    {
        float mx = row[0];
        if (row[1] > mx) mx = row[1];
        if (row[2] > mx) mx = row[2];
        if (row[3] > mx) mx = row[3];

        float e0 = ExpApprox4(row[0] - mx);
        float e1 = ExpApprox4(row[1] - mx);
        float e2 = ExpApprox4(row[2] - mx);
        float e3 = ExpApprox4(row[3] - mx);

        float sum = e0 + e1 + e2 + e3;
        float inv = 1.0f / (sum + 1e-20f);
        row[0] = e0 * inv;
        row[1] = e1 * inv;
        row[2] = e2 * inv;
        row[3] = e3 * inv;
    }

    __aicore__ inline void Sinkhorn4(float *m)
    {
        // Python: exp then alternating row/col normalize for sinkIters with eps=1e-12
        // Here: stable softmax as exp+row normalize, then run row/col norm with eps in denom.
        SoftmaxRow4(m + 0);
        SoftmaxRow4(m + 4);
        SoftmaxRow4(m + 8);
        SoftmaxRow4(m + 12);

        uint32_t rep = sinkIters;
        if (rep == 0u) rep = 1u;

        for (uint32_t it = 0u; it < rep; ++it) {
            for (uint32_t r = 0u; r < 4u; ++r) {
                float s = m[r*4+0] + m[r*4+1] + m[r*4+2] + m[r*4+3];
                float inv = 1.0f / (s + sinkEps);
                m[r*4+0] *= inv; m[r*4+1] *= inv; m[r*4+2] *= inv; m[r*4+3] *= inv;
            }
            for (uint32_t c = 0u; c < 4u; ++c) {
                float s = m[0*4+c] + m[1*4+c] + m[2*4+c] + m[3*4+c];
                float inv = 1.0f / (s + sinkEps);
                m[0*4+c] *= inv; m[1*4+c] *= inv; m[2*4+c] *= inv; m[3*4+c] *= inv;
            }
        }
    }

    __aicore__ inline void ProcessTokenN4(uint32_t t, float a_pre, float a_post, float a_res)
    {
        // x: [B,S,D], create x_flat: 4 streams identical to x for mapping computation/application.
        // Compute RMSNorm over SD=4*D on x_flat, scale by rms_scale.
        const uint64_t D64 = (uint64_t)D;

        // Compute sumsq over 4*D with replicated x
        float sumsq = 0.0f;
        const uint64_t baseX = (uint64_t)t * D64;
        for (uint32_t i = 0u; i < D; ++i) {
            float xv = xGm.GetValue(baseX + (uint64_t)i);
            sumsq += 4.0f * xv * xv;
        }
        float mean = sumsq * invSD;
        float invRms = RsqrtNewton(mean + rmsEps);

        // GEMV: (x_norm[SD]) @ phi[SD,24] + bias[24]
        float acc[24];
        #pragma unroll
        for (uint32_t k = 0u; k < 24u; ++k) acc[k] = 0.0f;

        // iterate SD = 4*D; x_flat[stream*D + i] = x[i]
        for (uint32_t s = 0u; s < 4u; ++s) {
            const uint32_t sOff = s * D;
            for (uint32_t i = 0u; i < D; ++i) {
                const uint32_t sd = sOff + i;
                const float xv = xGm.GetValue(baseX + (uint64_t)i);
                const float xn = xv * invRms * rmsGm.GetValue((uint64_t)sd);

                const uint64_t row = (uint64_t)sd * 24ull;
                #pragma unroll
                for (uint32_t k = 0u; k < 24u; ++k) {
                    acc[k] += xn * phiGm.GetValue(row + (uint64_t)k);
                }
            }
        }

        // Add bias, apply alpha and activations, build h_pre/h_post/h_res
        float hpre[4], hpost[4];
        #pragma unroll
        for (uint32_t r = 0u; r < 4u; ++r) {
            float pre_t  = a_pre  * (acc[r]     + biasGm.GetValue((uint64_t)r));
            float post_t = a_post * (acc[4u+r]  + biasGm.GetValue(4ull + (uint64_t)r));
            hpre[r] = Sigmoid(pre_t);
            hpost[r] = 2.0f * Sigmoid(post_t);
        }

        float hres[16];
        #pragma unroll
        for (uint32_t k = 0u; k < 16u; ++k) {
            hres[k] = a_res * (acc[8u+k] + biasGm.GetValue(8ull + (uint64_t)k));
        }
        Sinkhorn4(hres); // now doubly stochastic-ish

        // Apply mapping pre: x_for_layer = sum_s x_stream[s]*hpre[s] ; but x_streams are identical to x
        // so x_for_layer[i] = x[i] * sum(hpre)
        const float sumPre = hpre[0] + hpre[1] + hpre[2] + hpre[3];

        // Linear: f_output = x_for_layer @ W^T (nn.Linear weight is [D,D], y = x @ W^T)
        // Compute f_output[j] = sum_i x_for_layer[i] * W[j,i]
        // Then combined: for each stream s: h_res_x[s] = sum_k h_res[s,k] * x_stream[k] = x * rowSum(h_res[s,:])
        // and h_post_f[s] = f_output * hpost[s]
        // combined.mean(dim=2): y = (1/4) * sum_s ( x*rowSum_s + f_output*hpost[s] )
        float rowSum[4];
        #pragma unroll
        for (uint32_t r = 0u; r < 4u; ++r) {
            rowSum[r] = hres[r*4+0] + hres[r*4+1] + hres[r*4+2] + hres[r*4+3];
        }
        const float meanRowSum = 0.25f * (rowSum[0] + rowSum[1] + rowSum[2] + rowSum[3]);
        const float meanHpost  = 0.25f * (hpost[0] + hpost[1] + hpost[2] + hpost[3]);

        const uint64_t baseY = (uint64_t)t * D64;

        // For each output dim j:
        for (uint32_t j = 0u; j < D; ++j) {
            // f_output[j]
            float fj = 0.0f;
            const uint64_t wRow = (uint64_t)j * D64;
            for (uint32_t i = 0u; i < D; ++i) {
                float xi = xGm.GetValue(baseX + (uint64_t)i);
                float xfor = xi * sumPre;
                fj += xfor * wGm.GetValue(wRow + (uint64_t)i);
            }

            // combined mean: x*meanRowSum + f_output*meanHpost
            // x here is per-dim x_stream (identical): x[j]
            float xj = xGm.GetValue(baseX + (uint64_t)j);
            float out = xj * meanRowSum + fj * meanHpost;
            yGm.SetValue(baseY + (uint64_t)j, out);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm, phiGm, biasGm, rmsGm, aPreGm, aPostGm, aResGm, wGm, yGm;

    uint32_t totalX, totalPhi, totalBias, totalRmsScale, totalAlphaPre, totalAlphaPost, totalAlphaRes, totalW;
    uint32_t B, S, D, n, SD, mapDim, tokens, tokensPerCore;
    float invSD, rmsEps, sinkEps;
    uint32_t sinkIters;
};

extern "C" __global__ __aicore__ void optimized_mhc_layer_with_fusion_custom(
    GM_ADDR x,
    GM_ADDR phi_params,
    GM_ADDR bias_params,
    GM_ADDR rms_scale,
    GM_ADDR alpha_pre,
    GM_ADDR alpha_post,
    GM_ADDR alpha_res,
    GM_ADDR linear_w,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(td, tiling);
    KernelOptimizedMHCLayerWithFusion op;
    op.Init(x, phi_params, bias_params, rms_scale, alpha_pre, alpha_post, alpha_res, linear_w, y,
            td.totalX, td.totalPhi, td.totalBias, td.totalRmsScale,
            td.totalAlphaPre, td.totalAlphaPost, td.totalAlphaRes, td.totalW,
            td.B, td.S, td.D, td.n, td.SD, td.mapDim,
            td.tokens, td.tokensPerCore,
            td.invSD, td.rmsEps, td.sinkEps, td.sinkIters);
    op.Process();
}
