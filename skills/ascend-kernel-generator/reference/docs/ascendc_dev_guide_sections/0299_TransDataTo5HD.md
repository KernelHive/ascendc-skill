###### TransDataTo5HD

```markdown
## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 AI Core | √ |
| Atlas 推理系列产品 Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

数据格式转换，一般用于将 NCHW 格式转换成 NC1HWC0 格式。特别的，也可以用于二维矩阵数据块的转置。

完成转置功能时，相比于 Transpose 接口：
- Transpose 仅支持 16×16 大小的矩阵转置
- 本接口单次 repeat 内可处理 512Byte 的数据（16 个 datablock）
- 根据数据类型不同，支持不同 shape 的矩阵转置（比如数据类型为 half 时，单次 repeat 可完成 16×16 大小的矩阵转置）
- 同时还可以支持多次 repeat 操作

### 单次 repeat 内转换规则

#### 输入数据类型位宽为 16 位时
- 每个 datablock 中包含 16 个数
- 指令内部会循环 16 次
- 每次循环都会分别从指定的 16 个 datablock 中的对应位置取值，组成一个新的 datablock 单元放入目的地址中
- 图中的 srcList[0]-srcList[15] 代表源操作数的 16 个 datablock

图 15-9 输入数据类型位宽为 16 位时的转换规则

#### 输入数据类型位宽为 32 位时
- 每个 datablock 包含 8 个数
- 指令内部会循环 8 次
- 每次循环都会分别从指定的 16 个 datablock 中的对应位置取值，组成 2 个新的 datablock 放入目的地址中

图 15-10 输入数据类型位宽为 32 位时的转换规则

#### 输入数据类型位宽为 8 位时
- 每个 datablock 包含 32 个数
- 指令内部会循环 16 次
- 每次循环都会分别从指定的 16 个 datablock 中的对应位置取值，组成半个 datablock 放入目的地址中
- 读取和存放是在 datablock 的高半部还是低半部由参数 srcHighHalf 和 dstHighHalf 决定

图 15-11 输入数据类型位宽为 8 位时的转换规则

基于以上的转换规则，使用该接口进行 NC1HWC0 格式转换或者矩阵转置。

NC1HWC0 格式转换相对复杂，这里给出其具体的转换方法：
NCHW 格式转换成 NC1HWC0 格式时：
- 如果数据类型的位宽为 32 位或者 16 位，则 C0=16
- 如果数据类型的位宽为 8 位，则 C0=32

下图以 C0=16 为例进行介绍：

## 函数原型

### 原型 1：dstList 与 srcList 类型为 LocalTensor 的数组

```cpp
// NCHW_CONV_ADDR_LIST_SIZE 值为 16
template <typename T>
__aicore__ inline void TransDataTo5HD(
    const LocalTensor<T> (&dstList)[NCHW_CONV_ADDR_LIST_SIZE],
    const LocalTensor<T> (&srcList)[NCHW_CONV_ADDR_LIST_SIZE],
    const TransDataTo5HDParams& nchwconvParams
)
```

### 原型 2：dstList 与 srcList 类型为 uint64_t 的数组

```cpp
// NCHW_CONV_ADDR_LIST_SIZE 值为 16
template<typename T>
__aicore__ inline void TransDataTo5HD(
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE],
    const TransDataTo5HDParams& nchwconvParams
)
```

### 原型 3：dst 与 src 类型为 uint64_t 的 LocalTensor

```cpp
template <typename T>
__aicore__ inline void TransDataTo5HD(
    const LocalTensor<uint64_t>& dst,
    const LocalTensor<uint64_t>& src,
    const TransDataTo5HDParams& nchwconvParams
)
```

## 参数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数数据类型。<br>Atlas 训练系列产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half<br>Atlas 推理系列产品 AI Core，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas 200I/500 A2 推理产品，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/int32_t/uint32_t/float |

### 参数列表

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dstList | 输出 | 目的操作数地址序列。<br>类型为 LocalTensor 或者 LocalTensor 的地址值，LocalTensor 支持的 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32B 对齐。支持的数据类型参考模板参数 T 说明。 |
| srcList | 输入 | 源操作数地址序列。<br>类型为 LocalTensor 或者 LocalTensor 的地址值，LocalTensor 支持的 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32B 对齐。支持的数据类型参考模板参数 T 说明。<br>数据类型需要与 dstList 保持一致。 |
| dst | 输出 | 目的操作数。<br>类型为 LocalTensor，连续存储对应 LocalTensor 的地址值。LocalTensor 支持的 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32B 对齐。 |
| src | 输入 | 源操作数。<br>类型为 LocalTensor，连续存储对应 LocalTensor 的地址值。LocalTensor 支持的 TPosition 为 VECIN/VECCALC/VECOUT。LocalTensor 的起始地址需要 32B 对齐。 |
| nchwconvParams | 输入 | 控制 TransdataTo5HD 的数据结构。结构体内包含：读取和写入位置的控制参数，迭代次数，相邻迭代间的地址步长等参数。<br>具体定义请参考 `${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_transpose.h`，`${INSTALL_DIR}` 请替换为 CANN 软件安装后文件存储路径。<br>参数说明请参考表 15-214。 |

### TransDataTo5HDParams 结构体内参数说明

| 参数名称 | 类型 | 说明 |
|----------|------|------|
| dstHighHalf | 输入 | 指定每个 dstList 地址中的数据存储到 datablock 的高半部还是低半部，该配置只支持 int8_t/uint8_t 的数据类型。<br>支持的数据类型为 bool，有以下两种取值：<br>● True：表示存储于 datablock 的高半部<br>● False：表示存储于 datablock 的低半部 |
| srcHighHalf | 输入 | 指定每个 srcList 地址中的数据从 datablock 的高半部还是低半部读取，该配置只支持 int8_t/uint8_t 的数据类型。<br>支持的数据类型为 bool，有以下两种取值：<br>● True：表示从 datablock 的高半部读取<br>● False：表示从 datablock 的低半部读取 |
| repeatTimes | 输入 | 重复迭代次数，repeatTimes ∈ [0,255]。<br>关于该参数的具体描述请参考 12.3 如何使用 Tensor 高维切分计算 API。<br>注意事项：<br>● 当 repeatTimes 为 1 时，目的操作数/源操作数的有效起始位置为 dstList/srcList 序列输入的起始位置加上 dstRepStride/srcRepStride；repeatTimes 为 1，如果要让目的操作数/源操作数的有效起始位置为 dstList/srcList 序列输入的起始位置，需要将 dstRepStride/srcRepStride 置为 0。<br>● 当 repeatTimes 大于 1 时，第一次 repeat 中目的操作数/源操作数的有效起始位置为 dstList/srcList 序列输入的起始位置，第二次需要加上 dstRepStride/srcRepStride。以此类推。 |
| dstRepStride | 输入 | 相邻迭代间，目的操作数相同 datablock 地址 stride，单位：datablock。<br>相邻迭代间相同 datablock 的地址步长参数的详细说明请参考 repeatStride。 |
| srcRepStride | 输入 | 相邻迭代间，源操作数相同 datablock 地址 stride，单位：datablock。<br>相邻迭代间相同 datablock 的地址步长参数的详细说明请参考 repeatStride。 |

## 约束说明

- 操作数地址对齐要求请参见通用地址对齐约束。
- 操作数地址重叠约束请参考通用地址重叠约束。
- 进行 NCHW 格式到 NC1HWC0 格式的转换时，一般用法是将 srcList/dstList 中的每个元素配置为每个 HW 平面的起点。
- 为了性能更优，数据类型位宽为 8 位时建议先固定 dstHighHalf、srcHighHalf，在 HW 方向 repeat 后，再改变 dstHighHalf、srcHighHalf。
- dst 与 src 中的地址需要连续存放，详见调用示例。

## 返回值说明

无

## 调用示例

### NCHW 格式转换成 NC1HWC0 格式调用示例

输入数据为 half 类型，输入 NCHW 格式为 (2, 32, 16, 16)，目标格式 NC1HWC0 为 (2, 2, 16, 16, 16)。

```cpp
#include "kernel_operator.h"

class KernelTransDataTo5HD {
public:
    __aicore__ inline KernelTransDataTo5HD() {}
    __aicore__ inline void Init(__gm__ uint8_t *src, __gm__ uint8_t *dstGm)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half *)src);
        dstGlobal.SetGlobalBuffer((__gm__ half *)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
        pipe.InitBuffer(workQueueSrc1, 1, 16 * sizeof(uint64_t));
        pipe.InitBuffer(workQueueSrc2, 1, 16 * sizeof(uint64_t));
        pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
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
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
        inQueueSrc.EnQue(srcLocal);
    }
    
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
        AscendC::TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = 16;
        transDataParams.dstRepStride = 16;
        transDataParams.srcRepStride = 1;
        
        for(int j = 0; j < 4; j++) {
            // // 入参类型是 LocalTensor 的调用方式
            // AscendC::LocalTensor<half> dstLocalList[16];
            // for (int i = 0; i < 16; i++) {
            //     dstLocalList[i] = dstLocal[j * c0size * height * width + width * i];
            // }
            // AscendC::LocalTensor<half> srcLocalList[16];
            // for (int i = 0; i < 16; i++) {
            //     srcLocalList[i] = srcLocal[j * c0size * height * width + height * width * i];
            // }
            // AscendC::TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
            
            // 入参类型是 LocalTensor 地址值的调用方式，推荐使用
            uint64_t dstLocalList[16];
            for (int i = 0; i < 16; i++) {
                dstLocalList[i] = (uint64_t)(dstLocal[j * c0size * height * width + width * i].GetPhyAddr());
            }
            uint64_t srcLocalList[16];
            for (int i = 0; i < 16; i++) {
                srcLocalList[i] = (uint64_t)(srcLocal[j * c0size * height * width + height * width * i].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
            
            // // 入参类型是地址 LocalTensor 的调用方式
            // AscendC::LocalTensor<uint64_t> dst = workQueueSrc1.AllocTensor<uint64_t>();
            // for (int i = 0; i < 16; i++) {
            //     dst.SetValue(i, (uint64_t)(dstLocal[j * c0size * height * width + width * i].GetPhyAddr()));
            // }
            // AscendC::LocalTensor<uint64_t> src = workQueueSrc2.AllocTensor<uint64_t>();
            // for (int i = 0; i < 16; i++) {
            //     src.SetValue(i, (uint64_t)(srcLocal[j * c0size * height * width + height * width * i].GetPhyAddr()));
            // }
            // AscendC::TransDataTo5HD<half>(dst, src, transDataParams);
            // workQueueSrc1.FreeTensor(dst);
            // workQueueSrc2.FreeTensor(src);
        }
        
        outQueueDst.EnQue<half>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
        outQueueDst.FreeTensor(dstLocal);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueueSrc2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    int srcDataSize = 16384;
    int dstDataSize = 16384;
    int width = 16; // H
    int height = 16; // W
    int c0size = 16; // C0
};

extern "C" __global__ __aicore__ void vec_transdata5hd_b16_nchw2nc1hwc0(__gm__ uint8_t *src, __gm__ uint8_t *dstGm)
{
    KernelTransDataTo5HD op;
    op.Init(src, dstGm);
    op.Process();
}
```

**输入数据：**
```
[[[[ 0. 0. 0. ... 0. 0. 0.]
   [ 0. 0. 0. ... 0. 0. 0.]
   [ 0. 0. 0. ... 0. 0. 0.]
   ...
   [ 0. 0. 0. ... 0. 0. 0.]
   [ 0. 0. 0. ... 0. 0. 0.]
   [ 0. 0. 0. ... 0. 0. 0.]]

  [[ 1. 1. 1. ... 1. 1. 1.]
   [ 1. 1. 1. ... 1. 1. 1.]
   [ 1. 1. 1. ... 1. 1. 1.]
   ...
   [ 1. 1. 1. ... 1. 1. 1.]
   [ 1. 1. 1. ... 1. 1. 1.]
   [ 1. 1. 1. ... 1. 1. 1.]]

  [[ 2. 2. 2. ... 2. 2. 2.]
   [ 2. 2. 2. ... 2. 2. 2.]
   [ 2. 2. 2. ... 2. 2. 2.]
   ...
   [ 2. 2. 2. ... 2. 2. 2.]
   [ 2. 2. 2. ... 2. 2. 2.]
   [ 2. 2. 2. ... 2. 2. 2.]]

  ...

  [[29. 29. 29. ... 29. 29. 29.]
   [29. 29. 29. ... 29. 29. 29.]
   [29. 29. 29. ... 29. 29. 29.]
   ...
   [29. 29. 29. ... 29. 29. 29.]
   [29. 29. 29. ... 29. 29. 29.]
   [29. 29. 29. ... 29. 29. 29.]]

  [[30. 30. 30. ... 30. 30. 30.]
   [30. 30. 30. ... 30. 30. 30.]
   [30. 30. 30. ... 30. 30. 30.]
   ...
   [30. 30. 30. ... 30. 30. 30.]
   [30. 30. 30. ... 30. 30. 30.]
   [30. 30. 30. ... 30. 30. 30.]]

  [[31. 31. 31. ... 31. 31. 31.]
   [31. 31. 31. ... 31. 31. 31.]
   [31. 31. 31. ... 31. 31. 31.]
   ...
   [31. 31. 31. ... 31. 31. 31.]
   [31. 31. 31. ... 31. 31. 31.]
   [31. 31. 31. ... 31. 31. 31.]]]

 [[[32. 32. 32. ... 32. 32. 32.]
   [32. 32. 32. ... 32. 32. 32.]
   [32. 32. 32. ... 32. 32. 32.]
   ...
   [32. 32. 32. ... 32. 32. 32.]
   [32. 32. 32. ... 32. 32. 32.]
   [32. 32. 32. ... 32. 32. 32.]]

  [[33. 33. 33. ... 33. 33. 33.]
   [33. 33. 33. ... 33. 33. 33.]
   [33. 33. 33. ... 33. 33. 33.]
   ...
   [33. 33. 33. ... 33. 33. 33.]
   [33. 33. 33. ... 33. 33. 33.]
   [33. 33. 33. ... 33. 33. 33.]]

  [[34. 34. 34. ... 34. 34. 34.]
   [34. 34. 34. ... 34. 34. 34.]
   [34. 34. 34. ... 34. 34. 34.]
   ...
   [34. 34. 34. ... 34. 34. 34.]
   [34. 34. 34. ... 34. 34. 34.]
   [34. 34. 34. ... 34. 34. 34.]]

  ...

  [[61. 61. 61. ... 61. 61. 61.]
   [61. 61. 61. ... 61. 61. 61.]
   [61. 61. 61. ... 61. 61. 61.]
   ...
   [61. 61. 61. ... 61. 61. 61.]
   [61. 61. 61. ... 61. 61. 61.]
   [61. 61. 61. ... 61. 61. 61.]]

  [[62. 62. 62. ... 62. 62. 62.]
   [62. 62. 62. ... 62. 62. 62.]
   [62. 62. 62. ... 62. 62. 62.]
   ...
   [62. 62. 62. ... 62. 62. 62.]
   [62. 62. 62. ... 62. 62. 62.]
   [62. 62. 62. ... 62. 62. 62.]]

  [[63. 63. 63. ... 63. 63. 63.]
   [63. 63. 63. ... 63. 63. 63.]
   [63. 63. 63. ... 63. 63. 63.]
   ...
   [63. 63. 63. ... 63. 63. 63.]
   [63. 63. 63. ... 63. 63. 63.]
   [63. 63. 63. ... 63. 63. 63.]]]]
```

**输出数据：**
```
[[[[[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]

   [[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]

   [[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]

   ...

   [[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]

   [[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]

   [[ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    ...
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]
    [ 0. 1. 2. ... 13. 14. 15.]]]

  [[[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]

   [[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]

   [[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]

   ...

   [[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]

   [[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]

   [[16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    ...
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]
    [16. 17. 18. ... 29. 30. 31.]]]]]

 [[[[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]

   [[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]

   [[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]

   ...

   [[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]

   [[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]

   [[32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    ...
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]
    [32. 33. 34. ... 45. 46. 47.]]]

  [[[48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    ...
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]]

   [[48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    ...
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]]

   [[48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    ...
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]]

   ...

   [[48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    ...
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]
    [48. 49. 50. ... 61. 62. 63.]]

   [[48. 49. 50. ... 61. 62.
