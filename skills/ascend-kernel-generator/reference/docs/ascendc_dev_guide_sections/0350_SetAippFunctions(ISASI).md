##### SetAippFunctions(ISASI)

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

设置图片预处理（AIPP，AI core pre-process）相关参数。与 `LoadImageToLocal(ISASI)` 接口配合使用。设置后，调用 `LoadImageToLocal(ISASI)` 接口可在搬运过程中完成图像预处理操作，包括：数据填充、通道交换、单行读取、数据类型转换、通道填充、色域转换。

调用 `SetAippFunctions` 接口时需传入源图片在 Global Memory 上的矩阵、源图片的图片格式。

### 数据填充

在图片 HW 方向上 padding。分为如下几种模式：

- **模式0：常量填充模式** - padding 区域各位置填充为常数，支持设置每个通道填充的常数。该模式下仅支持左右 padding，不支持上下 padding。
- **模式1：行列填充模式** - padding 区域各位置填充行/列上最邻近源图片位置的数据。
- **模式2：块填充模式** - 按照 padding 的宽高，从源图片拷贝数据块进行 padding 区域填充。
- **模式3：镜像块填充模式** - 按照 padding 的宽高，从源图片拷贝数据块的镜像进行 padding 区域填充。

### 通道交换

将图片通道进行交换：

- 对于 RGB888 格式，支持交换 R 和 B 通道
- 对于 YUV420SP 格式，支持交换 U 和 V 通道
- 对于 XRGB8888 格式，支持 X 通道后移（XRGB→RGBX）、支持交换 R 和 B 通道

### 单行读取

源图片中仅读取一行。

> **说明**
> 调用数据搬运接口时，开启单行读取后设置的目的图片高度参数无效，如 `LoadImageToLocal(ISASI)` 接口的 `loadImageToLocalParams.vertSize`。

### 数据类型转换

转换像素的数据类型，支持 `uint8_t` 转换为 `int8_t` 或 `half`。当 `uint8_t` 转换成 `int8_t` 的时候，输出数据范围限制在 `[-128, 127]`。

```cpp
// 例1：实现 uint8_t -> int8_t 的类型转换，同时实现零均值化
output[i][j][k] = input[i][j][k] - mean[k]

// 例2：实现 uint8_t -> fp16 的类型转换，同时实现归一化
uint8_t -> fp16: output[i][j][k] = (input[i][j][k] - mean[k] - min[k]) * var[k]
```

> **说明**
> 转换后的数据类型是由模板参数 U 配置，U 为 `uint8_t` 时数据类型转换功能不生效。
> 调用数据搬运接口时，目的 Tensor 的数据类型需要与本接口输出数据类型保持一致，如 `LoadImageToLocal(ISASI)` 的 `dstLocal` 参数的数据类型。

### 通道填充

在图片通道方向上 padding。默认为模式0：

- **模式0**：将通道 padding 至 32Bytes。即输出数据类型为 `uint8_t`/`int8_t` 时，padding 至 32 通道；输出数据类型为 `fp16` 时，padding 至 16 通道
- **模式1**：将通道 padding 至 4 通道

### 色域转换

RGB 格式转换为 YUV 格式，或 YUV 模式转换为 RGB 格式。

## 函数原型

### 输入图片格式为 YUV400、RGB888、XRGB8888

```cpp
template<typename T, typename U>
void SetAippFunctions(const GlobalTensor<T>& src0, AippInputFormat format, AippParams<U> config)
```

### 输入图片格式为 YUV420 Semi-Planar

```cpp
template<typename T, typename U>
void SetAippFunctions(const GlobalTensor<T>& src0, const GlobalTensor<T>& src1, AippInputFormat format, AippParams<U> config)
```

## 参数说明

### 模板参数说明

| 参数名称 | 含义 |
|----------|------|
| T | 输入的数据类型，需要与 format 中设置的数据类型保持一致 |
| U | 输出的数据类型，需要在搬运接口配置同样的数据类型，如 `LoadImageToLocal(ISASI)` 的 `dstLocal` 参数数据类型。<br>• 如果不使能数据类型转换功能，需要与输入类型保持一致<br>• 如果使能数据类型转换功能，需要与期望转换后的类型保持一致 |

### 参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| src0 | 输入 | 源图片在 Global Memory 上的矩阵。<br>源图片格式为 YUV420SP 时，表示 Y 维度在 Global Memory 上的矩阵 |
| src1 | 输入 | 源图片格式为 YUV420SP 时，表示 UV 维度在 Global Memory 上的矩阵<br>源图片格式为其他格式时，该参数无效 |
| format | 输入 | 源图片的图片格式。AippInputFormat 为枚举类型，取值为：<br>`AippInputFormat::YUV420SP_U8`：图片格式为 YUV420 Semi-Planar，数据类型为 `uint8_t`<br>`AippInputFormat::XRGB8888_U8`：图片格式为 XRGB8888，数据类型为 `uint8_t`<br>`AippInputFormat::RGB888_U8`：图片格式为 RGB888，数据类型为 `uint8_t`<br>`AippInputFormat::YUV400_U8`：图片格式为 YUV400，数据类型为 `uint8_t`<br><br>```cpp<br>enum class AippInputFormat : uint8_t {<br>  YUV420SP_U8 = 0,<br>  XRGB8888_U8 = 1,<br>  RGB888_U8 = 4,<br>  YUV400_U8 = 9,<br>};<br>``` |
| config | 输入 | 图片预处理的相关参数，类型为 AippParams，结构体具体定义为：<br><br>```cpp<br>template <typename U><br>struct AippParams {<br>  AippPaddingParams<U> paddingParams;<br>  AippSwapParams swapParams;<br>  AippSingleLineParams singleLineParams;<br>  AippDataTypeConvParams dtcParams;<br>  AippChannelPaddingParams<U> cPaddingParams;<br>  AippColorSpaceConvParams cscParams;<br>};<br>``` |

### AippParams 结构体详细定义

#### 数据填充功能相关参数

```cpp
template <typename U>
struct AippPaddingParams {
  uint32_t paddingMode;
  U paddingValueCh0;
  U paddingValueCh1;
  U paddingValueCh2;
  U paddingValueCh3;
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| paddingMode | 输入 | padding 的模式，取值范围 [0, 3]，默认值为 0<br>0：常数填充模式，此模式仅支持左右填充<br>1：行列拷贝模式<br>2：块拷贝模式<br>3：镜像块拷贝模式 |
| paddingValueCh0 | 输入 | padding 区域中 channel0 填充的数据，仅常数填充模式有效，数据类型为 U，默认值为 0 |
| paddingValueCh1 | 输入 | padding 区域中 channel1 填充的数据，仅常数填充模式有效，数据类型为 U，默认值为 0 |
| paddingValueCh2 | 输入 | padding 区域中 channel2 填充的数据，仅常数填充模式有效，数据类型为 U，默认值为 0 |
| paddingValueCh3 | 输入 | padding 区域中 channel3 填充的数据，仅常数填充模式有效，数据类型为 U，默认值为 0 |

#### 通道交换功能相关参数

```cpp
struct AippSwapParams {
  bool isSwapRB;
  bool isSwapUV;
  bool isSwapAX;
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| isSwapRB | 输入 | 对于 RGB888、XRGB8888 格式，是否交换 R 和 B 通道。默认值为 false |
| isSwapUV | 输入 | 对于 YUV420SP 格式，是否交换 U 和 V 通道。默认值为 false |
| isSwapAX | 输入 | 对于 XRGB8888 格式，是否将 X 通道后移，即 XRGB→RGBX。默认值为 false |

#### 单行读取功能相关参数

```cpp
struct AippSingleLineParams {
  bool isSingleLineCopy;
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| isSingleLineCopy | 输入 | 是否开启单行读取模式。开启后，仅从源图片读取一行。默认值为 false |

#### 数据类型转换功能相关参数

```cpp
struct AippDataTypeConvParams {
  uint8_t dtcMeanCh0{ 0 };
  uint8_t dtcMeanCh1{ 0 };
  uint8_t dtcMeanCh2{ 0 };
  half dtcMinCh0{ 0 };
  half dtcMinCh1{ 0 };
  half dtcMinCh2{ 0 };
  half dtcVarCh0{ 1.0 };
  half dtcVarCh1{ 1.0 };
  half dtcVarCh2{ 1.0 };
  uint32_t dtcRoundMode{ 0 };
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dtcMeanCh0 | 输入 | 计算公式内的 mean 值，channel0，数据类型为 `uint8_t`，默认值为 0 |
| dtcMeanCh1 | 输入 | 计算公式内的 mean 值，channel1，数据类型为 `uint8_t`，默认值为 0 |
| dtcMeanCh2 | 输入 | 计算公式内的 mean 值，channel2，数据类型为 `uint8_t`，默认值为 0 |
| dtcMinCh0 | 输入 | 计算公式内的 min 值，channel0，数据类型为 `half`，默认值为 0<br>Atlas 200I/500 A2 推理产品不支持配置该参数 |
| dtcMinCh1 | 输入 | 计算公式内的 min 值，channel1，数据类型为 `half`，默认值为 0<br>Atlas 200I/500 A2 推理产品不支持配置该参数 |
| dtcMinCh2 | 输入 | 计算公式内的 min 值，channel2，数据类型为 `half`，默认值为 0<br>Atlas 200I/500 A2 推理产品不支持配置该参数 |
| dtcVarCh0 | 输入 | 计算公式内的 var 值，channel0，数据类型为 `half`，默认值为 1.0 |
| dtcVarCh1 | 输入 | 计算公式内的 var 值，channel1，数据类型为 `half`，默认值为 1.0 |
| dtcVarCh2 | 输入 | 计算公式内的 var 值，channel2，数据类型为 `half`，默认值为 1.0 |
| dtcRoundMode | 输入 | 控制 dtc 做数据类型转换的模式，数据类型为 `uint32_t`，默认值为 0<br>0：四舍五入到最接近的整数值（C语言 round）<br>1：四舍五入到最接近的偶数（C语言 rint）<br>仅 Atlas 200I/500 A2 推理产品支持配置该参数 |

#### 通道填充功能相关参数

```cpp
template <typename U>
struct AippChannelPaddingParams {
  uint32_t cPaddingMode;
  U cPaddingValue;
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| cPaddingMode | 输入 | channel padding 的类型，取值范围为 [0, 1]，默认值为 0<br>0：填充到 32B。即输出数据类型 U 为 `uint8_t`/`int8_t` 时填充到 32 通道，为 `half` 时填充到 16 通道<br>1：填充到 4 通道 |
| cPaddingValue | 输入 | channel padding 填充的值，数据类型为 U，默认值为 0 |

#### 色域转换功能相关参数

```cpp
struct AippColorSpaceConvParams {
  bool isEnableCsc;
  int16_t cscMatrixR0C0;
  int16_t cscMatrixR0C1;
  int16_t cscMatrixR0C2;
  int16_t cscMatrixR1C0;
  int16_t cscMatrixR1C1;
  int16_t cscMatrixR1C2;
  int16_t cscMatrixR2C0;
  int16_t cscMatrixR2C1;
  int16_t cscMatrixR2C2;
  uint8_t cscBiasIn0;
  uint8_t cscBiasIn1;
  uint8_t cscBiasIn2;
  uint8_t cscBiasOut0;
  uint8_t cscBiasOut1;
  uint8_t cscBiasOut2;
};
```

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| isEnableCsc | 输入 | 是否开启色域转换功能，默认值为 false |
| cscMatrixR0C0 | 输入 | 色域转换矩阵 cscMatrix[0][0] |
| cscMatrixR0C1 | 输入 | 色域转换矩阵 cscMatrix[0][1] |
| cscMatrixR0C2 | 输入 | 色域转换矩阵 cscMatrix[0][2] |
| cscMatrixR1C0 | 输入 | 色域转换矩阵 cscMatrix[1][0] |
| cscMatrixR1C1 | 输入 | 色域转换矩阵 cscMatrix[1][1] |
| cscMatrixR1C2 | 输入 | 色域转换矩阵 cscMatrix[1][2] |
| cscMatrixR2C0 | 输入 | 色域转换矩阵 cscMatrix[2][0] |
| cscMatrixR2C1 | 输入 | 色域转换矩阵 cscMatrix[2][1] |
| cscMatrixR2C2 | 输入 | 色域转换矩阵 cscMatrix[2][2] |
| cscBiasIn0 | 输入 | RGB 转 YUV 偏置 cscBiasIn[0]。YUV 转 RGB 时无效 |
| cscBiasIn1 | 输入 | RGB 转 YUV 偏置 cscBiasIn[1]。YUV 转 RGB 时无效 |
| cscBiasIn2 | 输入 | RGB 转 YUV 偏置 cscBiasIn[2]。YUV 转 RGB 时无效 |
| cscBiasOut0 | 输入 | YUV 转 RGB 偏置 cscBiasOut0[0]。RGB 转 YUV 时无效 |
| cscBiasOut1 | 输入 | YUV 转 RGB 偏置 cscBiasOut1[1]。RGB 转 YUV 时无效 |
| cscBiasOut2 | 输入 | YUV 转 RGB 偏置 cscBiasOut2[2]。RGB 转 YUV 时无效 |

## 约束说明

- src0、src1 在 Global Memory 上的地址对齐要求如下：

| 图片格式 | src0 | src1 |
|----------|------|------|
| YUV420SP | 必须 2Bytes 对齐 | 必须 2Bytes 对齐 |
| XRGB8888 | 必须 4Bytes 对齐 | - |
| RGB888 | 无对齐要求 | - |
| YUV400 | 无对齐要求 | - |

- 对于 XRGB 输入格式的数据，芯片在处理的时候会默认丢弃掉第四个通道的数据输出 RGB 格式的数据，所以如果是 X 在 channel0 的场景下，那么为了达成上述目的，X 通道后移的功能必须使能，将输入的通道转换为 RGBX；反之如果是 X 在 channel3 的场景下，X 通道后移的功能必须不使能以输出 RGB 格式的数据。

## 返回值说明

无

## 调用示例

该调用示例支持的运行平台为 Atlas 推理系列产品 AI Core，示例图片格式为 YUV420SP。

```cpp
#include "kernel_operator.h"

class KernelLoadImage {
public:
  __aicore__ inline KernelLoadImage()
  {
    // YUV420SP 图片中，Y 维度的 size
    gmSrc0Size = srcHorizSize * srcVertSize;
    // YUV420SP 图片中，UV 维度的 size
    gmSrc1Size = (srcHorizSize / 2) * (srcVertSize / 2) * 2;
    dstSize = dstHorizSize * dstVertSize * cSize;
  }
  
  __aicore__ inline void Init(__gm__ uint8_t *fmGm, __gm__ uint8_t *dstGm)
  {
    fmGlobal.SetGlobalBuffer((__gm__ uint8_t *)fmGm);
    dstGlobal.SetGlobalBuffer((__gm__ int8_t *)dstGm);
    pipe.InitBuffer(inQueueA1, 1, (gmSrc0Size + gmSrc1Size) * sizeof(int8_t));
    pipe.InitBuffer(outQueueUB, 1, dstSize * sizeof(int8_t));
  }
  
  __aicore__ inline void Process()
  {
    CopyIn();
    CopyToUB();
    CopyOut();
  }
  
private:
  __aicore__ inline void CopyIn()
  {
    AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.AllocTensor<int8_t>();
    uint64_t fm_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmGlobal.GetPhyAddr()));
    
    // aipp config
    AscendC::AippParams<int8_t> aippConfig;
    aippConfig.cPaddingParams.cPaddingMode = cPadMode;
    aippConfig.cPaddingParams.cPaddingValue = cPaddingValue;
    
    // fmGlobal 为整张输入图片，src1 参数处填入图片 UV 维度的起始地址
    AscendC::SetAippFunctions(fmGlobal, fmGlobal[gmSrc0Size], inputFormat, aippConfig);
    AscendC::LoadImageToLocal(featureMapA1, { horizSize, vertSize, horizStartPos, vertStartPos,
      srcHorizSize, topPadSize, botPadSize, leftPadSize, rightPadSize });
    inQueueA1.EnQue(featureMapA1);
  }
  
  __aicore__ inline void CopyToUB()
  {
    AscendC::LocalTensor<int8_t> featureMapA1 = inQueueA1.DeQue<int8_t>();
    AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.AllocTensor<int8_t>();
    AscendC::DataCopy(featureMapUB, featureMapA1, dstSize);
    
    event_t eventIdMTE1ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE1_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
    
    outQueueUB.EnQue<int8_t>(featureMapUB);
    inQueueA1.FreeTensor(featureMapA1);
  }
  
  __aicore__ inline void CopyOut()
  {
    AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.DeQue<int8_t>();
    AscendC::DataCopy(dstGlobal, featureMapUB, dstSize);
    outQueueUB.FreeTensor(featureMapUB);
  }
  
private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
  AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;
  
  AscendC::GlobalTensor<uint8_t> fmGlobal;
  AscendC::GlobalTensor<int8_t> dstGlobal;
  
  uint16_t horizSize = 32, vertSize = 32, horizStartPos = 0, vertStartPos = 0, srcHorizSize = 32,
    srcVertSize = 32, leftPadSize = 0, rightPadSize = 0;
  uint32_t dstHorizSize = 32, dstVertSize = 32, cSize = 32;
  uint8_t topPadSize = 0, botPadSize = 0;
  uint32_t gmSrc0Size = 0, gmSrc1Size = 0, dstSize = 0;
  AscendC::AippInputFormat inputFormat = AscendC::AippInputFormat::YUV420SP_U8;
  uint32_t cPadMode = 0;
  int8_t cPaddingValue = 0;
};

extern "C" __global__ __aicore__ void load_image_simple_kernel(__gm__ uint8_t *fmGm, __gm__ uint8_t *dstGm)
{
  KernelLoadImage op;
  op.Init(fmGm, dstGm);
  op.Process();
}
```
