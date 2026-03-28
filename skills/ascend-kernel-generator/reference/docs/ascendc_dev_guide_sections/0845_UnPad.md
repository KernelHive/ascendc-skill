##### UnPad

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

对 `height * width` 的二维 Tensor 在 width 方向上进行 unpad。如果 Tensor 的 width 非 32B 对齐，则不支持调用本接口 unpad。

本接口具体功能场景如下：Tensor 的 width 已 32B 对齐，以 half 为例，如 16*16，进行 UnPad，变成 16*15。

## 函数原型

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小 BufferSize 的获取方法：通过 15.1.5.8.4 UnPad Tiling 中提供的 GetUnPadMaxMinTmpSize 接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持接口框架申请和开发者通过 sharedTmpBuffer 入参传入两种方式，因此 UnPad 接口的函数原型有两种：

- **通过 sharedTmpBuffer 入参传入临时空间**

```cpp
template <typename T>
__aicore__ inline void UnPad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
UnPadParams &unPadParams, LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
```

该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

- **接口框架申请临时空间**

```cpp
template <typename T>
__aicore__ inline void UnPad(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
UnPadParams &unPadParams, UnPadTiling &tiling)
```

该方式下开发者无需申请，但是需要预留临时空间的大小。

## 参数说明

### 表 15-846 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float |

### 表 15-847 接口参数说明

| 参数名称 | 输入/输出 | 含义 |
|----------|-----------|------|
| dstTensor | 输出 | 目的操作数，shape为二维，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数，shape为二维，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| UnPadParams | 输入 | UnPad详细参数，UnPadParams数据类型，具体结构体参数说明如下：<br>● leftPad，左边unpad的数据量。leftPad要求小于32B。单位：列。当前暂不生效。<br>● rightPad，右边unpad的数据量。rightPad要求小于32B，大于0。单位：列。当前只支持在右边进行unpad。<br>UnPadParams结构体的定义如下：<br>```cpp<br>struct UnPadParams {<br>    uint16_t leftPad = 0;<br>    uint16_t rightPad = 0;<br>};<br>``` |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考15.1.5.8.4 UnPad Tiling。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| tiling | 输入 | 计算所需tiling信息，Tiling信息的获取请参考相关章节。 |
