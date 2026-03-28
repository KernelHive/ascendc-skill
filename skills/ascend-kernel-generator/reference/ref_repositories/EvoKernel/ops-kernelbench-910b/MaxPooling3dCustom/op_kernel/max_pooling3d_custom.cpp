
#include "kernel_operator.h"

// MaxPool3d specialized for kernel_size=3, stride=2, padding=1, dilation=1.
// Key optimization: interior width handled in a quad-output unrolled loop.
// For 4 consecutive outputs (ow, ow+1, ow+2, ow+3), windows start at iw = iw0 + 0,2,4,6
// and each needs 3 elements => total slab width = 9 contiguous values per (d,h) row.
// We load 9 values once and reduce into 4 maxima, amortizing scalar overhead.

class KernelMaxPooling3dCustom {
public:
    __aicore__ inline KernelMaxPooling3dCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                               uint32_t n, uint32_t c,
                               uint32_t d_in, uint32_t h_in, uint32_t w_in,
                               uint32_t d_out, uint32_t h_out, uint32_t w_out,
                               uint32_t totalRows, uint32_t blockDim, uint32_t rowsPerBlock)
    {
        this->n = n;
        this->c = c;
        this->d_in = d_in;
        this->h_in = h_in;
        this->w_in = w_in;

        this->d_out = d_out;
        this->h_out = h_out;
        this->w_out = w_out;

        this->totalRows = totalRows;
        this->blockDim = blockDim;
        this->rowsPerBlock = rowsPerBlock;

        hw_in  = static_cast<uint64_t>(h_in) * w_in;
        dhw_in = static_cast<uint64_t>(d_in) * hw_in;

        hw_out  = static_cast<uint64_t>(h_out) * w_out;
        dhw_out = static_cast<uint64_t>(d_out) * hw_out;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        yGm.SetGlobalBuffer((__gm__ float*)y);
    }

    __aicore__ inline float Max3(float a, float b, float c0) const
    {
        float m = a;
        m = (b > m) ? b : m;
        m = (c0 > m) ? c0 : m;
        return m;
    }

    __aicore__ inline void Max27Interior1(uint64_t base, float &out0) const
    {
        const float ninf = -3.402823466e+38f;
        float m = ninf;

        uint64_t b0 = base;
        // d0
        float v00 = xGm.GetValue(b0 + 0);
        float v01 = xGm.GetValue(b0 + 1);
        float v02 = xGm.GetValue(b0 + 2);
        float v10 = xGm.GetValue(b0 + w_in + 0);
        float v11 = xGm.GetValue(b0 + w_in + 1);
        float v12 = xGm.GetValue(b0 + w_in + 2);
        float v20 = xGm.GetValue(b0 + 2ULL * w_in + 0);
        float v21 = xGm.GetValue(b0 + 2ULL * w_in + 1);
        float v22 = xGm.GetValue(b0 + 2ULL * w_in + 2);

        m = (v00 > m) ? v00 : m; m = (v01 > m) ? v01 : m; m = (v02 > m) ? v02 : m;
        m = (v10 > m) ? v10 : m; m = (v11 > m) ? v11 : m; m = (v12 > m) ? v12 : m;
        m = (v20 > m) ? v20 : m; m = (v21 > m) ? v21 : m; m = (v22 > m) ? v22 : m;

        // d1
        uint64_t b1 = base + hw_in;
        float w00 = xGm.GetValue(b1 + 0);
        float w01 = xGm.GetValue(b1 + 1);
        float w02 = xGm.GetValue(b1 + 2);
        float w10 = xGm.GetValue(b1 + w_in + 0);
        float w11 = xGm.GetValue(b1 + w_in + 1);
        float w12 = xGm.GetValue(b1 + w_in + 2);
        float w20 = xGm.GetValue(b1 + 2ULL * w_in + 0);
        float w21 = xGm.GetValue(b1 + 2ULL * w_in + 1);
        float w22 = xGm.GetValue(b1 + 2ULL * w_in + 2);

        m = (w00 > m) ? w00 : m; m = (w01 > m) ? w01 : m; m = (w02 > m) ? w02 : m;
        m = (w10 > m) ? w10 : m; m = (w11 > m) ? w11 : m; m = (w12 > m) ? w12 : m;
        m = (w20 > m) ? w20 : m; m = (w21 > m) ? w21 : m; m = (w22 > m) ? w22 : m;

        // d2
        uint64_t b2 = base + 2ULL * hw_in;
        float u00 = xGm.GetValue(b2 + 0);
        float u01 = xGm.GetValue(b2 + 1);
        float u02 = xGm.GetValue(b2 + 2);
        float u10 = xGm.GetValue(b2 + w_in + 0);
        float u11 = xGm.GetValue(b2 + w_in + 1);
        float u12 = xGm.GetValue(b2 + w_in + 2);
        float u20 = xGm.GetValue(b2 + 2ULL * w_in + 0);
        float u21 = xGm.GetValue(b2 + 2ULL * w_in + 1);
        float u22 = xGm.GetValue(b2 + 2ULL * w_in + 2);

        m = (u00 > m) ? u00 : m; m = (u01 > m) ? u01 : m; m = (u02 > m) ? u02 : m;
        m = (u10 > m) ? u10 : m; m = (u11 > m) ? u11 : m; m = (u12 > m) ? u12 : m;
        m = (u20 > m) ? u20 : m; m = (u21 > m) ? u21 : m; m = (u22 > m) ? u22 : m;

        out0 = m;
    }

    __aicore__ inline void ProcessInteriorQuad(uint64_t base, float &o0, float &o1, float &o2, float &o3) const
    {
        // base points to in-bounds (id_start, ih_start, iw_start) for ow being the first output in the quad.
        // Need 9 contiguous columns [0..8] per (d,h) row.
        const float ninf = -3.402823466e+38f;
        float m0 = ninf, m1 = ninf, m2 = ninf, m3 = ninf;

        #define UPDATE9(ptr) do {                         \
            float a0 = xGm.GetValue((ptr) + 0);           \
            float a1 = xGm.GetValue((ptr) + 1);           \
            float a2 = xGm.GetValue((ptr) + 2);           \
            float a3 = xGm.GetValue((ptr) + 3);           \
            float a4 = xGm.GetValue((ptr) + 4);           \
            float a5 = xGm.GetValue((ptr) + 5);           \
            float a6 = xGm.GetValue((ptr) + 6);           \
            float a7 = xGm.GetValue((ptr) + 7);           \
            float a8 = xGm.GetValue((ptr) + 8);           \
            float t0 = Max3(a0,a1,a2);                    \
            float t1 = Max3(a2,a3,a4);                    \
            float t2 = Max3(a4,a5,a6);                    \
            float t3 = Max3(a6,a7,a8);                    \
            m0 = (t0 > m0) ? t0 : m0;                     \
            m1 = (t1 > m1) ? t1 : m1;                     \
            m2 = (t2 > m2) ? t2 : m2;                     \
            m3 = (t3 > m3) ? t3 : m3;                     \
        } while (0)

        uint64_t b0 = base;
        // d0
        UPDATE9(b0);
        UPDATE9(b0 + w_in);
        UPDATE9(b0 + 2ULL * w_in);
        // d1
        uint64_t b1 = base + hw_in;
        UPDATE9(b1);
        UPDATE9(b1 + w_in);
        UPDATE9(b1 + 2ULL * w_in);
        // d2
        uint64_t b2 = base + 2ULL * hw_in;
        UPDATE9(b2);
        UPDATE9(b2 + w_in);
        UPDATE9(b2 + 2ULL * w_in);

        #undef UPDATE9

        o0 = m0; o1 = m1; o2 = m2; o3 = m3;
    }

    __aicore__ inline void Process()
    {
        constexpr int32_t K = 3;
        constexpr int32_t S = 2;
        constexpr int32_t P = 1;

        const uint32_t bid = static_cast<uint32_t>(AscendC::GetBlockIdx());
        const float ninf = -3.402823466e+38f;

        uint32_t startRow = bid * rowsPerBlock;
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > totalRows) endRow = totalRows;

        for (uint32_t row = startRow; row < endRow; ++row) {
            uint32_t t = row;
            const uint32_t oh = t % h_out; t /= h_out;
            const uint32_t od = t % d_out; t /= d_out;
            const uint32_t ci = t % c;     t /= c;
            const uint32_t ni = t;

            const int32_t id_start = static_cast<int32_t>(od) * S - P;
            const int32_t ih_start = static_cast<int32_t>(oh) * S - P;

            const int32_t id0 = (id_start < 0) ? 0 : id_start;
            const int32_t ih0 = (ih_start < 0) ? 0 : ih_start;
            const int32_t id1 = (id_start + K > static_cast<int32_t>(d_in)) ? static_cast<int32_t>(d_in) : (id_start + K);
            const int32_t ih1 = (ih_start + K > static_cast<int32_t>(h_in)) ? static_cast<int32_t>(h_in) : (ih_start + K);

            const bool dh_interior =
                (id_start >= 0) && (id_start + (K - 1) < static_cast<int32_t>(d_in)) &&
                (ih_start >= 0) && (ih_start + (K - 1) < static_cast<int32_t>(h_in));

            const uint64_t xNCBase = (static_cast<uint64_t>(ni) * c + ci) * dhw_in;
            const uint64_t yRowBase = (static_cast<uint64_t>(ni) * c + ci) * dhw_out +
                                      (static_cast<uint64_t>(od) * h_out + oh) * w_out;

            // Left boundary ow=0 (iw_start=-1) always boundary for P=1.
            {
                const int32_t iw_start = -P;
                float maxv = ninf;

                const int32_t iw0 = (iw_start < 0) ? 0 : iw_start;
                const int32_t iw1 = (iw_start + K > static_cast<int32_t>(w_in)) ? static_cast<int32_t>(w_in) : (iw_start + K);

                for (int32_t id = id0; id < id1; ++id) {
                    const uint64_t xDBase = xNCBase + static_cast<uint64_t>(id) * hw_in;
                    for (int32_t ih = ih0; ih < ih1; ++ih) {
                        const uint64_t xHBase = xDBase + static_cast<uint64_t>(ih) * w_in;
                        for (int32_t iw = iw0; iw < iw1; ++iw) {
                            float v = xGm.GetValue(xHBase + static_cast<uint64_t>(iw));
                            maxv = (v > maxv) ? v : maxv;
                        }
                    }
                }
                yGm.SetValue(yRowBase + 0, maxv);
            }

            // Interior region starts from ow=1 (iw_start=1).
            uint32_t ow = 1;
            if (dh_interior && w_in >= 9) {
                // For quad starting at ow, need iw_start = 2*ow - 1 >= 0 and iw_start + 8 < w_in
                // => 2*ow - 1 <= w_in - 9 => ow <= (w_in - 8)/2.
                const uint32_t owQuadEnd = (static_cast<uint32_t>(w_in) - 8U) >> 1;
                while (ow + 3 < w_out && ow <= owQuadEnd) {
                    const int32_t iw_start = static_cast<int32_t>(ow) * S - P; // = 2*ow -1
                    const uint64_t base =
                        xNCBase +
                        static_cast<uint64_t>(id_start) * hw_in +
                        static_cast<uint64_t>(ih_start) * w_in +
                        static_cast<uint64_t>(iw_start);

                    float o0, o1, o2, o3;
                    ProcessInteriorQuad(base, o0, o1, o2, o3);
                    yGm.SetValue(yRowBase + ow + 0, o0);
                    yGm.SetValue(yRowBase + ow + 1, o1);
                    yGm.SetValue(yRowBase + ow + 2, o2);
                    yGm.SetValue(yRowBase + ow + 3, o3);
                    ow += 4;
                }
            }

            // Remaining outputs: use interior single-output fast path when possible, else boundary triple loop.
            for (; ow < w_out; ++ow) {
                const int32_t iw_start = static_cast<int32_t>(ow) * S - P;
                float maxv = ninf;

                const bool interior =
                    dh_interior &&
                    (iw_start >= 0) && (iw_start + (K - 1) < static_cast<int32_t>(w_in));

                if (interior) {
                    const uint64_t base =
                        xNCBase +
                        static_cast<uint64_t>(id_start) * hw_in +
                        static_cast<uint64_t>(ih_start) * w_in +
                        static_cast<uint64_t>(iw_start);
                    Max27Interior1(base, maxv);
                } else {
                    const int32_t iw0 = (iw_start < 0) ? 0 : iw_start;
                    const int32_t iw1 = (iw_start + K > static_cast<int32_t>(w_in)) ? static_cast<int32_t>(w_in) : (iw_start + K);

                    for (int32_t id = id0; id < id1; ++id) {
                        const uint64_t xDBase = xNCBase + static_cast<uint64_t>(id) * hw_in;
                        for (int32_t ih = ih0; ih < ih1; ++ih) {
                            const uint64_t xHBase = xDBase + static_cast<uint64_t>(ih) * w_in;
                            for (int32_t iw = iw0; iw < iw1; ++iw) {
                                float v = xGm.GetValue(xHBase + static_cast<uint64_t>(iw));
                                maxv = (v > maxv) ? v : maxv;
                            }
                        }
                    }
                }

                yGm.SetValue(yRowBase + ow, maxv);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t n, c, d_in, h_in, w_in;
    uint32_t d_out, h_out, w_out;
    uint32_t totalRows, blockDim, rowsPerBlock;

    uint64_t hw_in, dhw_in;
    uint64_t hw_out, dhw_out;
};

extern "C" __global__ __aicore__ void max_pooling3d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tiling_data, tiling);

    KernelMaxPooling3dCustom op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c,
            tiling_data.d_in, tiling_data.h_in, tiling_data.w_in,
            tiling_data.d_out, tiling_data.h_out, tiling_data.w_out,
            tiling_data.totalRows, tiling_data.blockDim, tiling_data.rowsPerBlock);
    op.Process();
}
