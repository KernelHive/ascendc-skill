###### CompareScalar

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

逐元素比较一个 tensor 中的元素和另一个 Scalar 的大小，如果比较后的结果为真，则输出结果的对应比特位为 1，否则为 0。

支持多种比较模式：

- **LT**：小于（less than）
- **GT**：大于（greater than）
- **GE**：大于或等于（greater than or equal to）
- **EQ**：等于（equal to）
- **NE**：不等于（not equal to）
- **LE**：小于或等于（less than or equal to）

## 函数原型

### tensor 前 n 个数据计算

```cpp
template <typename T, typename U>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, uint32_t count)
```

### tensor 高维切分计算

#### mask 逐 bit 模式

```cpp
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

#### mask 连续模式

```cpp
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 源操作数数据类型 |
| U | 目的操作数数据类型 |
| isSetMask | 是否在接口内部设置 mask<br>• true：表示在接口内部设置 mask<br>• false：表示在接口外部设置 mask，开发者需要使用 SetVectorMask 接口设置 mask 值。这种模式下，本接口入参中的 mask 值必须设置为占位符 MASK_PLACEHOLDER |

### 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dst | 输出 | 目的操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>LocalTensor 的起始地址需要 32 字节对齐<br>dst 用于存储比较结果，将 dst 中 uint8_t 类型的数据按照 bit 位展开，由左至右依次表征对应位置的 src0 和 src1 的比较结果，如果比较后的结果为真，则对应比特位为 1，否则为 0<br>支持的数据类型：<br>• Atlas 推理系列产品 AI Core：uint8_t<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品：uint8_t<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品：uint8_t<br>• Atlas 200I/500 A2 推理产品：uint8_t |
| src0 | 输入 | 源操作数<br>类型为 LocalTensor，支持的 TPosition 为 VECIN/VECCALC/VECOUT<br>LocalTensor 的起始地址需要 32 字节对齐<br>支持的数据类型：<br>• Atlas 推理系列产品 AI Core：half/float<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品：half/float（所有 CMPMODE 都支持），int32_t（只支持 CMPMODE::EQ）<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品：half/float（所有 CMPMODE 都支持），int32_t（只支持 CMPMODE::EQ）<br>• Atlas 200I/500 A2 推理产品：int16_t/uint16_t/half/float/int32_t/uint32_t |
| src1Scalar | 输入 | 源操作数，Scalar 标量。数据类型和 src0 保持一致 |
| cmpMode | 输入 | CMPMODE 类型，表示比较模式，包括 EQ，NE，GE，LE，GT，LT<br>• LT：src0 小于（less than）src1<br>• GT：src0 大于（greater than）src1<br>• GE：src0 大于或等于（greater than or equal to）src1<br>• EQ：src0 等于（equal to）src1<br>• NE：src0 不等于（not equal to）src1<br>• LE：src0 小于或等于（less than or equal to）src1 |
| mask/mask[] | 输入 | mask 用于控制每次迭代内参与计算的元素<br>• Atlas 推理系列产品 AI Core：保留参数，设置无效<br>• Atlas A2 训练系列产品/Atlas A2 推理系列产品：保留参数，设置无效<br>• Atlas A3 训练系列产品/Atlas A3 推理系列产品：保留参数，设置无效<br>• Atlas 200I/500 A2 推理产品：设置有效<br>&nbsp;&nbsp;• 连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为 16 位时，mask∈[1, 128]；当操作数为 32 位时，mask∈[1, 64]<br>&nbsp;&nbsp;• 逐 bit 模式：可以按位控制哪些元素参与计算，bit 位的值为 1 表示参与计算，0 表示不参与。参数类型为长度为 2 或者 4 的 uint64_t 类型数组<br>&nbsp;&nbsp;&nbsp;&nbsp;例如，mask=[8, 0]，8=0b1000，表示仅第 4 个元素参与计算<br>&nbsp;&nbsp;&nbsp;&nbsp;参数取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为 16 位时，mask[0]、mask[1]∈[0, 2^64-1] 并且不同时为 0；当操作数为 32 位时，mask[1] 为 0，mask[0]∈(0, 2^64-1] |
| repeatTime | 输入 | 重复迭代次数。矢量计算单元，每次读取连续的 256Bytes 数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime 表示迭代的次数<br>关于该参数的具体描述请参考 12.3 如何使用 Tensor 高维切分计算 API |
| repeatParams | 输入 | 控制操作数地址步长的参数<br>UnaryRepeatParams 类型，包含操作数相邻迭代间相同 DataBlock 的地址步长，操作数同一迭代内不同 DataBlock 的地址步长等参数<br>相邻迭代间的地址步长参数说明请参考 repeatStride；同一迭代内 DataBlock 的地址步长参数说明请参考 dataBlockStride |
| count | 输入 | 参与计算的元素个数。设置 count 时，需要保证 count 个元素所占空间 256 字节对齐 |

## 返回值说明

无

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束
- dst 按照小端顺序排序成二进制结果，对应 src 中相应位置的数据比较结果
- 使用 tensor 前 n 个数据参与计算的接口，设置 count 时，需要保证 count 个元素所占空间 256 字节对齐

## 调用示例

本样例中，源操作数 src0Local 存储了 256 个 float 类型的数据。样例实现的功能为，对 src0Local 中的元素和 src1Local.GetValue(0) 中的数据进行比较，如果 src0Local 中的元素小于 src1Local.GetValue(0) 中的元素，dstLocal 结果中对应的比特位置 1；反之，则置 0。dst 结果使用 uint8_t 类型数据存储。

### tensor 前 n 个数据计算接口样例

```cpp
AscendC::CompareScalar(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, srcDataSize);
```

### tensor 高维切分计算 - mask 连续模式

```cpp
uint64_t mask = 256 / sizeof(float); // 256 为每个迭代处理的字节数
int repeat = 4;
AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
// repeat = 4, 64 elements one repeat, 256 elements total
// dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::CompareScalar(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
```

### tensor 高维切分计算 - mask 逐 bit 模式

```cpp
uint64_t mask[2] = { UINT64_MAX, 0};
int repeat = 4;
AscendC::UnaryRepeatParams repeatParams = { 1, 1, 8, 8 };
// repeat = 4, 64 elements one repeat, 256 elements total
// srcBlkStride, = 1, no gap between blocks in one repeat
// dstRepStride, srcRepStride = 8, no gap between repeats
AscendC::CompareScalar(dstLocal, src0Local, src1Scalar, AscendC::CMPMODE::LT, mask, repeat, repeatParams);
```

## 结果示例

输入数据 (src0_gm):

```
[ 16.604824 45.069473 65.108345 -59.68792 21.043684 75.90726 -27.046307 -40.10546 -5.933778 83.56574 58.87062 -12.77814 28.17882 62.549377 -22.310246 -67.69001 81.06072 69.988945 69.10082 -6.667376 96.20256 18.532446 -66.56364 -32.531246 49.980835 35.668995 -16.847628 1.3236234 10.0143795 43.878166 26.628105 31.774637 47.9279 79.7291 -54.09651 95.49459 -18.404795 -86.84594 9.406091 -79.54437 0.49116692 -48.151714 -12.97062 -99.89055 23.475513 -27.366564 -69.229675 83.613304 52.14729 40.98426 -23.422009 -53.386215 1.6576616 -62.36946 54.693733 66.2058 -4.0042257 -25.351263 1.0000885 -6.458584 25.447659 71.647316 82.31162 -7.7359715 28.107353 -79.22045 20.292479 67.7434 -76.054085 -7.754251 38.632687 -4.8460293 -69.791954 -57.574455 -99.96178 -73.29611 -68.57477 98.200035 -55.30482 -55.590027 79.53274 -1.862139 -37.60953 -12.225406 -35.2875 -24.047668 -66.07609 21.9362 80.603516 28.928387 26.579298 97.6649 78.94723 -89.86824 73.29788 18.957182 -73.87053 -23.508097 -51.02931 39.158726 -96.61422 -41.192455 54.973663 47.58695 -3.9818003 -81.05088 -67.62415 -17.491713 -34.916042 -95.993744 -3.4719822 -55.956417 6.223455 12.240832 15.055512 94.70584 -13.33949 -50.46866 54.612816 -28.521824 -87.63997 59.53054 41.000504 -31.266075 -31.419422 -32.940186 53.449913 50.012768 -13.663364 40.931725 -68.80396 -86.63726 76.866585 -83.76385 3.7227867 58.443035 -74.333046 -92.52674 24.249512 -7.935491 24.197245 -34.85033 67.854645 72.65312 13.622443 -70.94266 15.401667 -9.332295 -86.61463 72.659676 -83.63352 9.279887 81.037964 46.285606 -12.967846 -48.72901 69.07614 -40.355286 -94.257034 -45.514374 24.966864 -9.657219 61.803864 -83.09603 77.769035 -97.44226 -89.71987 -53.969315 43.892918 73.88798 67.23104 36.65282 -93.70069 -87.48934 -27.679005 -36.825226 -30.117033 -41.579655 -97.325325 77.1972 -49.883194 33.061394 -63.844925 89.74327 64.549416 80.16943 73.26347 -87.307175 -96.62777 81.8532 7.5365276 28.357092 59.896378 -15.95738 -77.42723 0.03529428 -20.263502 45.59324 -90.160835 89.478004 57.608685 60.71819 45.8125 39.94484 -48.77375 -56.897358 5.2580256 -6.937905 -49.80309 -42.527523 72.91772 89.53271 -62.181187 18.490683 -69.40782 6.141204 13.938042 75.312515 21.766457 -8.157599 55.53147 -30.789118 -12.087165 82.435684 23.4884 82.73172 -2.026827 -8.124383 -10.707488 -74.32759 -54.702602 14.209252 93.73145 98.93554 52.803623 32.200726 41.823833 90.193756 -34.512424 -85.64022 97.47763 33.353424 94.84875 23.03139 99.97347 -72.47978 19.51753 -88.28579 -88.70721 -18.659292 -79.5277 62.90431 21.837631 45.989056 -9.62086 11.4855795 ]
```

输入数据 (src1_gm):

```
[-95.16087 -71.4676 51.817818 -12.358237 96.60704 -12.0067835 -44.128048 7.5811195 84.61196 -60.303513 21.470125 98.96244 18.262054 80.014244 48.37233 -75.03457 ]
```

输出数据 (dst_gm):

```
[ 0 0 0 0 0 8 0 0 0 4 0 0 16 32 0 0 0 0 0 0 32 0 4 16 0 0 0 0 0 0 0 0 ]
```

## 样例模板

```cpp
#include "kernel_operator.h"

template <typename T> 
class KernelCmp {
public:
    __aicore__ inline KernelCmp() {}
    
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm,
                               uint32_t dataSize, AscendC::CMPMODE mode)
    {
        srcDataSize = dataSize;
        dstDataSize = srcDataSize / AscendC::AscendCUtils::GetBitSize(sizeof(uint8_t));
        cmpMode = mode;
        src0Global.SetGlobalBuffer((__gm__ T*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ T*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ uint8_t*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, srcDataSize * sizeof(T));
        pipe.InitBuffer(inQueueSrc1, 1, 16 * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(uint8_t));
    }
    
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.AllocTensor<T>();
        AscendC::DataCopy(src0Local, src0Global, srcDataSize);
        AscendC::DataCopy(src1Local, src1Global, 16);
        inQueueSrc0.EnQue(src0Local);
        inQueueSrc1.EnQue(src1Local);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0Local = inQueueSrc0.DeQue<T>();
        AscendC::LocalTensor<T> src1Local = inQueueSrc1.DeQue<T>();
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.AllocTensor<uint8_t>();
        AscendC::PipeBarrier<PIPE_ALL>();
        T src1Scalar = src1Local.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::CompareScalar(dstLocal, src0Local, static_cast<T>(src1Scalar), cmpMode, srcDataSize);
        outQueueDst.EnQue<uint8_t>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueSrc1.FreeTensor(src1Local);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<uint8_t> dstLocal = outQueueDst.DeQue<uint8_t>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<T> src0Global, src1Global;
    AscendC::GlobalTensor<uint8_t> dstGlobal;
    uint32_t srcDataSize = 0;
    uint32_t dstDataSize = 0;
    AscendC::CMPMODE cmpMode;
};

template <typename T>
__aicore__ void main_cpu_cmp_sel_demo(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm, 
                                     uint32_t dataSize, AscendC::CMPMODE mode)
{
    KernelCmp<T> op;
    op.Init(src0Gm, src1Gm, dstGm, dataSize, mode);
    op.Process();
}

extern "C" __global__ __aicore__ void kernel_vec_compare_scalar_256_LT_float(GM_ADDR src0_gm,
                                                                            GM_ADDR src1_gm, 
                                                                            GM_ADDR dst_gm)
{
    main_cpu_cmp_sel_demo<float>(src0_gm, src1_gm, dst_gm, 256, AscendC::CMPMODE::LT);
}
```
