
#include "kernel_operator.h"

// Specialized AvgPool3d:
// kernel=3, stride=2, padding=1, count_include_pad=True (divide by 27).
//
// This revision reduces scalar/control cost on the hot interior path by:
// - computing 2 output width points per loop iteration
// - reusing the overlapping input column between adjacent stride-2 windows
//   (windows share 1 column: [w0,w1,w2] and next [w2,w3,w4])
// - hoisting base pointers and minimizing repeated index math

class KernelAveragePooling3dCustom {
public:
    __aicore__ inline KernelAveragePooling3dCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t n, uint32_t c,
                               uint32_t d_in, uint32_t h_in, uint32_t w_in,
                               uint32_t d_out, uint32_t h_out, uint32_t w_out,
                               uint32_t rows, uint32_t rowsPerBlock)
    {
        this->n = n;
        this->c = c;
        this->d_in = d_in;
        this->h_in = h_in;
        this->w_in = w_in;
        this->d_out = d_out;
        this->h_out = h_out;
        this->w_out = w_out;
        this->rows = rows;
        this->rowsPerBlock = rowsPerBlock;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline float Sum9(const uint64_t p0, const uint64_t p1, const uint64_t p2,
                                 uint32_t w0, uint32_t w1, uint32_t w2)
    {
        float acc = 0.0f;
        acc += xGm.GetValue(p0 + w0); acc += xGm.GetValue(p0 + w1); acc += xGm.GetValue(p0 + w2);
        acc += xGm.GetValue(p1 + w0); acc += xGm.GetValue(p1 + w1); acc += xGm.GetValue(p1 + w2);
        acc += xGm.GetValue(p2 + w0); acc += xGm.GetValue(p2 + w1); acc += xGm.GetValue(p2 + w2);
        return acc;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t K = 3;
        constexpr int32_t S = 2;
        constexpr int32_t P = 1;
        const float invK3 = 1.0f / 27.0f;

        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint32_t startRow = coreId * rowsPerBlock;
        if (startRow >= rows) return;
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > rows) endRow = rows;

        uint32_t row = startRow;
        uint32_t oh = row % h_out;
        uint32_t t0 = row / h_out;
        uint32_t od = t0 % d_out;
        uint32_t t1 = t0 / d_out;
        uint32_t ci = t1 % c;
        uint32_t ni = t1 / c;

        const uint64_t hw_in = static_cast<uint64_t>(h_in) * static_cast<uint64_t>(w_in);
        const uint64_t dhw_in = static_cast<uint64_t>(d_in) * hw_in;
        const uint64_t hw_out = static_cast<uint64_t>(h_out) * static_cast<uint64_t>(w_out);
        const uint64_t dhw_out = static_cast<uint64_t>(d_out) * hw_out;

        uint64_t xBaseNC = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(c) + ci) * dhw_in;
        uint64_t yBaseNC = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(c) + ci) * dhw_out;

        for (; row < endRow; ++row) {
            const uint64_t yRowBase = yBaseNC + (static_cast<uint64_t>(od) * hw_out +
                                                static_cast<uint64_t>(oh) * static_cast<uint64_t>(w_out));

            const int32_t dstart = static_cast<int32_t>(od) * S - P;
            const int32_t hstart = static_cast<int32_t>(oh) * S - P;

            const bool dInterior = (dstart >= 0) && (dstart + K <= static_cast<int32_t>(d_in));
            const bool hInterior = (hstart >= 0) && (hstart + K <= static_cast<int32_t>(h_in));

            const uint32_t Ow = w_out;
            if (Ow != 0) {
                const uint32_t last = Ow - 1;

                if (dInterior && hInterior) {
                    const uint32_t di = static_cast<uint32_t>(dstart);
                    const uint32_t hi = static_cast<uint32_t>(hstart);

                    const uint64_t d0 = xBaseNC + static_cast<uint64_t>(di + 0) * hw_in;
                    const uint64_t d1 = xBaseNC + static_cast<uint64_t>(di + 1) * hw_in;
                    const uint64_t d2 = xBaseNC + static_cast<uint64_t>(di + 2) * hw_in;

                    const uint64_t p00 = d0 + static_cast<uint64_t>(hi + 0) * w_in;
                    const uint64_t p01 = d0 + static_cast<uint64_t>(hi + 1) * w_in;
                    const uint64_t p02 = d0 + static_cast<uint64_t>(hi + 2) * w_in;

                    const uint64_t p10 = d1 + static_cast<uint64_t>(hi + 0) * w_in;
                    const uint64_t p11 = d1 + static_cast<uint64_t>(hi + 1) * w_in;
                    const uint64_t p12 = d1 + static_cast<uint64_t>(hi + 2) * w_in;

                    const uint64_t p20 = d2 + static_cast<uint64_t>(hi + 0) * w_in;
                    const uint64_t p21 = d2 + static_cast<uint64_t>(hi + 1) * w_in;
                    const uint64_t p22 = d2 + static_cast<uint64_t>(hi + 2) * w_in;

                    // Left border ow=0 (wstart=-1) => valid wi are 0..1
                    {
                        float acc = 0.0f;
                        acc += xGm.GetValue(p00 + 0); acc += xGm.GetValue(p00 + 1);
                        acc += xGm.GetValue(p01 + 0); acc += xGm.GetValue(p01 + 1);
                        acc += xGm.GetValue(p02 + 0); acc += xGm.GetValue(p02 + 1);

                        acc += xGm.GetValue(p10 + 0); acc += xGm.GetValue(p10 + 1);
                        acc += xGm.GetValue(p11 + 0); acc += xGm.GetValue(p11 + 1);
                        acc += xGm.GetValue(p12 + 0); acc += xGm.GetValue(p12 + 1);

                        acc += xGm.GetValue(p20 + 0); acc += xGm.GetValue(p20 + 1);
                        acc += xGm.GetValue(p21 + 0); acc += xGm.GetValue(p21 + 1);
                        acc += xGm.GetValue(p22 + 0); acc += xGm.GetValue(p22 + 1);
                        yGm.SetValue(yRowBase + 0, acc * invK3);
                    }

                    // Interior ow: process two outputs per iteration and reuse overlap
                    if (Ow > 2) {
                        uint32_t ow = 1;
                        for (; ow + 1 < last; ow += 2) {
                            // For ow: w0,w1,w2; next ow+1 starts at w2 (overlap)
                            const uint32_t w0 = ow * 2 - 1;
                            const uint32_t w1 = w0 + 1;
                            const uint32_t w2 = w0 + 2;
                            const uint32_t w3 = w2 + 1;
                            const uint32_t w4 = w2 + 2;

                            float acc0 = 0.0f;
                            float acc1 = 0.0f;

                            // depth 0
                            acc0 += Sum9(p00, p01, p02, w0, w1, w2);
                            acc1 += Sum9(p00, p01, p02, w2, w3, w4);
                            // depth 1
                            acc0 += Sum9(p10, p11, p12, w0, w1, w2);
                            acc1 += Sum9(p10, p11, p12, w2, w3, w4);
                            // depth 2
                            acc0 += Sum9(p20, p21, p22, w0, w1, w2);
                            acc1 += Sum9(p20, p21, p22, w2, w3, w4);

                            yGm.SetValue(yRowBase + static_cast<uint64_t>(ow),     acc0 * invK3);
                            yGm.SetValue(yRowBase + static_cast<uint64_t>(ow + 1), acc1 * invK3);
                        }
                        // tail one interior point if needed
                        for (; ow < last; ++ow) {
                            const uint32_t w0 = ow * 2 - 1;
                            const uint32_t w1 = w0 + 1;
                            const uint32_t w2 = w0 + 2;

                            float acc = 0.0f;
                            acc += Sum9(p00, p01, p02, w0, w1, w2);
                            acc += Sum9(p10, p11, p12, w0, w1, w2);
                            acc += Sum9(p20, p21, p22, w0, w1, w2);
                            yGm.SetValue(yRowBase + static_cast<uint64_t>(ow), acc * invK3);
                        }
                    }

                    // Right border ow=last
                    {
                        const int32_t wstart = static_cast<int32_t>(last) * S - P;
                        int32_t wi0 = wstart;
                        int32_t wi1 = wstart + K;
                        if (wi0 < 0) wi0 = 0;
                        if (wi1 > static_cast<int32_t>(w_in)) wi1 = static_cast<int32_t>(w_in);

                        float acc = 0.0f;
                        for (int32_t wi = wi0; wi < wi1; ++wi) {
                            const uint64_t uw = static_cast<uint64_t>(wi);
                            acc += xGm.GetValue(p00 + uw); acc += xGm.GetValue(p01 + uw); acc += xGm.GetValue(p02 + uw);
                            acc += xGm.GetValue(p10 + uw); acc += xGm.GetValue(p11 + uw); acc += xGm.GetValue(p12 + uw);
                            acc += xGm.GetValue(p20 + uw); acc += xGm.GetValue(p21 + uw); acc += xGm.GetValue(p22 + uw);
                        }
                        yGm.SetValue(yRowBase + static_cast<uint64_t>(last), acc * invK3);
                    }
                } else {
                    // General border path for D/H
                    int32_t di0 = dstart;
                    int32_t di1 = dstart + K;
                    if (di0 < 0) di0 = 0;
                    if (di1 > static_cast<int32_t>(d_in)) di1 = static_cast<int32_t>(d_in);

                    int32_t hi0 = hstart;
                    int32_t hi1 = hstart + K;
                    if (hi0 < 0) hi0 = 0;
                    if (hi1 > static_cast<int32_t>(h_in)) hi1 = static_cast<int32_t>(h_in);

                    // Left border ow=0
                    {
                        const int32_t wstart0 = -P;
                        int32_t wi0 = wstart0;
                        int32_t wi1 = wstart0 + K;
                        if (wi0 < 0) wi0 = 0;
                        if (wi1 > static_cast<int32_t>(w_in)) wi1 = static_cast<int32_t>(w_in);

                        float acc = 0.0f;
                        for (int32_t di = di0; di < di1; ++di) {
                            const uint64_t dOff = xBaseNC + static_cast<uint64_t>(di) * hw_in;
                            for (int32_t hi = hi0; hi < hi1; ++hi) {
                                const uint64_t line = dOff + static_cast<uint64_t>(hi) * w_in;
                                for (int32_t wi = wi0; wi < wi1; ++wi) {
                                    acc += xGm.GetValue(line + static_cast<uint64_t>(wi));
                                }
                            }
                        }
                        yGm.SetValue(yRowBase + 0, acc * invK3);
                    }

                    // Interior ow
                    if (Ow > 2) {
                        for (uint32_t ow = 1; ow < last; ++ow) {
                            const uint32_t w0 = ow * 2 - 1;
                            const uint32_t w1 = w0 + 1;
                            const uint32_t w2 = w0 + 2;

                            float acc = 0.0f;
                            for (int32_t di = di0; di < di1; ++di) {
                                const uint64_t dOff = xBaseNC + static_cast<uint64_t>(di) * hw_in;
                                for (int32_t hi = hi0; hi < hi1; ++hi) {
                                    const uint64_t line = dOff + static_cast<uint64_t>(hi) * w_in;
                                    acc += xGm.GetValue(line + w0);
                                    acc += xGm.GetValue(line + w1);
                                    acc += xGm.GetValue(line + w2);
                                }
                            }
                            yGm.SetValue(yRowBase + static_cast<uint64_t>(ow), acc * invK3);
                        }
                    }

                    // Right border ow=last
                    {
                        const int32_t wstart = static_cast<int32_t>(last) * S - P;
                        int32_t wi0 = wstart;
                        int32_t wi1 = wstart + K;
                        if (wi0 < 0) wi0 = 0;
                        if (wi1 > static_cast<int32_t>(w_in)) wi1 = static_cast<int32_t>(w_in);

                        float acc = 0.0f;
                        for (int32_t di = di0; di < di1; ++di) {
                            const uint64_t dOff = xBaseNC + static_cast<uint64_t>(di) * hw_in;
                            for (int32_t hi = hi0; hi < hi1; ++hi) {
                                const uint64_t line = dOff + static_cast<uint64_t>(hi) * w_in;
                                for (int32_t wi = wi0; wi < wi1; ++wi) {
                                    acc += xGm.GetValue(line + static_cast<uint64_t>(wi));
                                }
                            }
                        }
                        yGm.SetValue(yRowBase + static_cast<uint64_t>(last), acc * invK3);
                    }
                }
            }

            ++oh;
            if (oh == h_out) {
                oh = 0;
                ++od;
                if (od == d_out) {
                    od = 0;
                    ++ci;
                    if (ci == c) {
                        ci = 0;
                        ++ni;
                    }
                    xBaseNC = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(c) + ci) * dhw_in;
                    yBaseNC = (static_cast<uint64_t>(ni) * static_cast<uint64_t>(c) + ci) * dhw_out;
                }
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t n, c, d_in, h_in, w_in, d_out, h_out, w_out, rows, rowsPerBlock;
};

extern "C" __global__ __aicore__ void average_pooling3d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelAveragePooling3dCustom op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c,
            tiling_data.d_in, tiling_data.h_in, tiling_data.w_in,
            tiling_data.d_out, tiling_data.h_out, tiling_data.w_out,
            tiling_data.rows, tiling_data.rowsPerBlock);
    op.Process();
}
