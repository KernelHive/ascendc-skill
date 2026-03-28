#### Format

```cpp
enum Format {
    FORMAT_NCHW = 0,                    // NCHW
    FORMAT_NHWC,                        // NHWC
    FORMAT_ND,                          // Nd Tensor
    FORMAT_NC1HWC0,                     // NC1HWC0
    FORMAT_FRACTAL_Z,                   // FRACTAL_Z
    FORMAT_NC1C0HWPAD = 5,              // NC1C0HWPAD
    FORMAT_NHWC1C0,                     // NHWC1C0
    FORMAT_FSR_NCHW,                    // FSR_NCHW
    FORMAT_FRACTAL_DECONV,              // FRACTAL_DECONV
    FORMAT_C1HWNC0,                     // C1HWNC0
    FORMAT_FRACTAL_DECONV_TRANSPOSE = 10, // FRACTAL_DECONV_TRANSPOSE
    FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, // FRACTAL_DECONV_SP_STRIDE_TRANS
    FORMAT_NC1HWC0_C04,                 // NC1HWC0, C0 is 4
    FORMAT_FRACTAL_Z_C04,               // FRACTAL_Z, C0 is 4
    FORMAT_CHWN,                        // CHWN
    FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15, // FRACTAL_DECONV_SP_STRIDE8_TRANS
    FORMAT_HWCN,                        // HWCN
    FORMAT_NC1KHKWHWC0,                 // KH,KW kernel h& kernel w maxpooling max output format
    FORMAT_BN_WEIGHT,                   // BN_WEIGHT
    FORMAT_FILTER_HWCK,                 // filter input tensor format
    FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20, // HASHTABLE_LOOKUP_LOOKUPS
    FORMAT_HASHTABLE_LOOKUP_KEYS,       // HASHTABLE_LOOKUP_KEYS
    FORMAT_HASHTABLE_LOOKUP_VALUE,      // HASHTABLE_LOOKUP_VALUE
    FORMAT_HASHTABLE_LOOKUP_OUTPUT,     // HASHTABLE_LOOKUP_OUTPUT
    FORMAT_HASHTABLE_LOOKUP_HITS,       // HASHTABLE_LOOKUP_HITS
    FORMAT_C1HWNCoC0 = 25,              // C1HWNCoC0
    FORMAT_MD,                          // MD
    FORMAT_NDHWC,                       // NDHWC
    FORMAT_FRACTAL_ZZ,                  // FRACTAL_ZZ
    FORMAT_FRACTAL_NZ,                  // FRACTAL_NZ
    FORMAT_NCDHW = 30,                  // NCDHW
    FORMAT_DHWCN,                       // 3D filter input tensor format
    FORMAT_NDC1HWC0,                    // NDC1HWC0
    FORMAT_FRACTAL_Z_3D,                // FRACTAL_Z_3D
    FORMAT_CN,                          // CN
    FORMAT_NC = 35,                     // NC
    FORMAT_DHWNC,                       // DHWNC
    FORMAT_FRACTAL_Z_3D_TRANSPOSE,      // 3D filter(transpose) input tensor format
    FORMAT_FRACTAL_ZN_LSTM,             // FRACTAL_ZN_LSTM
    FORMAT_FRACTAL_Z_G,                 // FRACTAL_Z_G
    FORMAT_RESERVED = 40,               // RESERVED
    FORMAT_ALL,                         // ALL
    FORMAT_NULL,                        // NULL
    FORMAT_ND_RNN_BIAS,                 // ND_RNN_BIAS
    FORMAT_FRACTAL_ZN_RNN,              // FRACTAL_ZN_RNN
    FORMAT_NYUV = 45,                   // NYUV
    FORMAT_NYUV_A,                      // NYUV_A
    FORMAT_NCL,                         // NCL
    FORMAT_FRACTAL_Z_WINO,              // FRACTAL_Z_WINO
    FORMAT_C1HWC0,                      // C1HWC0
    FORMAT_FRACTAL_NZ_C0_16,            // 当前版本不支持该类型
    FORMAT_FRACTAL_NZ_C0_32,            // 当前版本不支持该类型
    // Add new formats definition here
    FORMAT_END,
    // FORMAT_MAX defines the max value of Format.
    // Any Format should not exceed the value of FORMAT_MAX.
    // ** Attention ** : FORMAT_MAX stands for the SPEC of enum Format and almost SHOULD NOT be used in code.
    // If you want to judge the range of Format, you can use FORMAT_END.
    FORMAT_MAX = 0xff
};
```

## Format Enum Values

The Format enum values correspond to sequential numbers starting from 0 and incrementing.

## IR Graph Construction Unsupported Formats

IR graph construction does not support the following FORMAT inputs:

- FORMAT_NC1HWC0
- FORMAT_FRACTAL_Z
- FORMAT_NC1C0HWPAD
- FORMAT_NHWC1C0
- FORMAT_FRACTAL_DECONV
- FORMAT_C1HWNC0
- FORMAT_FRACTAL_DECONV_TRANSPOSE
- FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS
- FORMAT_NC1HWC0_C04
- FORMAT_FRACTAL_Z_C04
- FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS
- FORMAT_NC1KHKWHWC0
- FORMAT_C1HWNCoC0
- FORMAT_FRACTAL_ZZ
- FORMAT_FRACTAL_NZ
- FORMAT_NDC1HWC0
- FORMAT_FRACTAL_Z_3D
- FORMAT_FRACTAL_Z_3D_TRANSPOSE
- FORMAT_FRACTAL_ZN_LSTM
- FORMAT_FRACTAL_Z_G
- FORMAT_ND_RNN_BIAS
- FORMAT_FRACTAL_ZN_RNN
- FORMAT_NYUV
- FORMAT_NYUV_A
