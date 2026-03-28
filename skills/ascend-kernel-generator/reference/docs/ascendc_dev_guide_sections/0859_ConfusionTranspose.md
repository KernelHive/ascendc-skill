##### ConfusionTranspose

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ✓ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | ✓ |
| Atlas 200I/500 A2 推理产品 | ✗ |
| Atlas 推理系列产品AI Core | ✓ |
| Atlas 推理系列产品Vector Core | ✗ |
| Atlas 训练系列产品 | ✗ |

## 功能说明

对输入数据进行数据排布及Reshape操作，具体功能如下：

### 场景1：NZ2ND，1、2轴互换

**输入Tensor**
```json
{
  "shape": "[B, N, H/N/16, S/16, 16, 16]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, S, N, H/N]",
  "origin_shape": "[B, S, N, H/N]",
  "format": "ND",
  "origin_format": "ND"
}
```

*图 15-84 场景 1 数据排布变换*

### 场景2：NZ2NZ，1、2轴互换

**输入Tensor**
```json
{
  "shape": "[B, N, H/N/16, S/16, 16, 16]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, S, H/N/16, N/16, 16, 16]",
  "origin_shape": "[B, S, N, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

*图 15-85 场景 2 数据排布变换*

### 场景3：NZ2NZ，尾轴切分

**输入Tensor**
```json
{
  "shape": "[B, H/16, S/16, 16, 16]",
  "origin_shape": "[B, S, H]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, N, H/N/16, S/16, 16, 16]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

*图 15-86 场景 3 数据排布变换*

### 场景4：NZ2ND，尾轴切分

**输入Tensor**
```json
{
  "shape": "[B, H/16, S/16, 16, 16]",
  "origin_shape": "[B, S, H]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, N, S, H/N]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "ND",
  "origin_format": "ND"
}
```

*图 15-87 场景 4 数据排布变换*

### 场景5：NZ2ND，尾轴合并

**输入Tensor**
```json
{
  "shape": "[B, N, H/N/16, S/16, 16, 16]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, S, H]",
  "origin_shape": "[B, S, H]",
  "format": "ND",
  "origin_format": "ND"
}
```

*图 15-88 场景 5 数据排布变换*

### 场景6：NZ2NZ，尾轴合并

**输入Tensor**
```json
{
  "shape": "[B, N, H/N/16, S/16, 16, 16]",
  "origin_shape": "[B, N, S, H/N]",
  "format": "NZ",
  "origin_format": "ND"
}
```

**输出Tensor**
```json
{
  "shape": "[B, H/16, S/16, 16, 16]",
  "origin_shape": "[B, S, H]",
  "format": "NZ",
  "origin_format": "ND"
}
```

*图 15-89 场景 6 数据排布变换*

### 场景7：二维转置

支持在UB上对二维Tensor进行转置，其中srcShape中的H、W均是16的整倍。

*图 15-90 场景 7 数据排布变换*

## 实现原理

对应ConfusionTranspose的7种功能场景，每种功能场景的算法框图如图所示。

*图 15-91 场景 1：NZ2ND，1、2 轴互换*

计算过程分为如下几步：

先后沿H/N方向，N方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：沿S方向转置S/16个连续的16*16的方形到temp中，在temp中每个方形与方形之间连续存储；
2. **第2次TransDataTo5HD步骤**：将temp中S/16个16*16的方形转置到dst中，在dst中是ND格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移(H/N)*N个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移16个元素。

*图 15-92 场景 2：NZ2NZ，1、2 轴互换*

计算过程分为如下几步：

先后沿H/N方向，N方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：沿S方向分别取S/16个连续的16*16的方形到temp中，在temp中每个方形与方形之间连续存储；
2. **第2次TransDataTo5HD步骤**：将temp中S/16个16*16的方形转置到dst中，在dst中是NZ格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移(H/N)*N个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移N*16个元素。

*图 15-93 场景 3：NZ2NZ，尾轴切分*

计算过程分为如下几步：

先后沿H方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：每次转置S/16个连续的16*16的方形到temp1中；
2. **DataCopy步骤**：
   - 当H/N<=16时，每次搬运H/N*S个元素到temp2中；
   - 当H/N>16时，前H/N/16次搬运16*S个元素到temp2中，最后一次搬运H/N%16*S个元素到temp2中；
3. **第2次TransDataTo5HD步骤**：将temp2中的16*S的方形转置到dst中，在dst中是NZ格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移16个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移S*16个元素。

*图 15-94 场景 4：NZ2ND，尾轴切分*

计算过程分为如下几步：

先后沿H方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：每次转置S/16个连续的16*16的方形到temp1中；
2. **DataCopy步骤**：
   - 当H/N<=16时，每次搬运H/N*S个元素到temp2中；
   - 当H/N>16时，前H/N/16次搬运16*S个元素到temp2中，最后一次搬运H/N%16*S个元素到tmp2中；
3. **第2次TransDataTo5HD步骤**：将temp2中的数据转置到dst中，在dst中是ND格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移(H/N+16-1)/16*16个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移(H/N+16-1)/16*16*S个元素。

*图 15-95 场景 5：NZ2ND，尾轴合并*

计算过程分为如下几步：

先后沿H方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：每次转置一个S*16的方形到temp1中；
2. **DataCopy步骤**：
   - 当H/N<=16时，每次搬运H/N*S个元素到temp2中；
   - 当H/N>16时，前H/N/16次搬运16*S个元素到temp2中，最后一次搬运H/N%16*S个元素到tmp2中；
3. **第2次TransDataTo5HD步骤**：将temp2中的16*S的方形转置到dst中，在dst中是ND格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移(H+16-1)/16*16个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移H/N*S个元素。

*图 15-96 场景 6：NZ2NZ，尾轴合并*

计算过程分为如下几步：

先后沿H方向，B方向循环处理：

1. **第1次TransDataTo5HD步骤**：每次转置一个S*16的方形到temp1中；
2. **DataCopy步骤**：
   - 当H/N<=16时，每次搬运H/N*S个元素到temp2中；
   - 当H/N>16时，前H/N/16次搬运16*S个元素到temp2中，最后一次搬运H/N%16*S个元素到tmp2中；
3. **第2次TransDataTo5HD步骤**：将temp2中的16*S的方形转置到dst中，在dst中是NZ格式，来自同一个方形的连续2行数据在目的操作数上的地址偏移16个元素，沿H方向的每2个方形的同一行数据在目的操作数上的地址偏移S*16个元素。

*图 15-97 场景 7：二维转置*

计算过程如下：

1. 调用TransDataTo5HD，通过设置不同的源操作数地址序列和目的操作数地址序列，将[H, W]转置为[W, H]，src和dst均是ND格式。

## 函数原型

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过15.1.5.12.2 ConfusionTranspose Tiling中提供的GetConfusionTransposeMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持接口框架申请和开发者通过sharedTmpBuffer入参传入两种方式，因此ConfusionTranspose接口的函数原型有两种：

### 通过sharedTmpBuffer入参传入临时空间

```cpp
template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T>& dstTensor, 
                                         const LocalTensor<T>& srcTensor, 
                                         const LocalTensor<uint8_t> &sharedTmpBuffer, 
                                         TransposeType transposeType, 
                                         ConfusionTransposeTiling& tiling)
```

该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

### 接口框架申请临时空间

```cpp
template <typename T>
__aicore__ inline void ConfusionTranspose(const LocalTensor<T>& dstTensor, 
                                         const LocalTensor<T>& srcTensor, 
                                         TransposeType transposeType, 
                                         ConfusionTransposeTiling& tiling)
```

该方式下开发者无需申请，但是需要预留临时空间的大小。

## 参数说明

### 表 15-862 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 操作数的数据类型。<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float<br>Atlas 推理系列产品AI Core，支持的数据类型为：int16_t/uint16_t/half/int32_t/uint32_t/float |

### 表 15-863 接口参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dstTensor | 输出 | 目的操作数，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数，LocalTensor数据结构的定义请参考15.1.3.1 LocalTensor。<br>类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考相关文档。 |
