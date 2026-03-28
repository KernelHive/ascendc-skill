###### ReadSpmBuffer

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | √ |

## 功能说明

从SPM Buffer读回到local数据中。

SPM Buffer的具体介绍和完整使用样例请参考12.8 如何使用SPM Buffer。

## 函数原型

- **适用于连续和不连续的数据回读：**
  ```cpp
  template <typename T>
  __aicore__ inline void ReadSpmBuffer(const LocalTensor<T>& readBuffer, const DataCopyParams& copyParams, int32_t readOffset = 0)
  ```

- **适用于连续的数据回读：**
  ```cpp
  template <typename T>
  __aicore__ inline void ReadSpmBuffer(const LocalTensor<T>& readBuffer, const int32_t readSize, int32_t readOffset = 0)
  ```

## 参数说明

### 表15-346 接口参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| readBuffer | 输入 | 读回的目标local内存 |
| copyParams | 输入 | 搬运参数，DataCopyParams类型，DataCopyParams结构定义请参考表15-347 |
| readSize | 输入 | 读回的元素个数 |
| readOffset | 输入 | SPM Buffer的偏移，单位为字节 |

### 表15-347 DataCopyParams 结构体参数定义

| 参数名称 | 含义 |
|----------|------|
| blockCount | 待搬运的连续传输数据块个数。uint16_t类型，取值范围：blockCount ∈ [1, 4095] |
| blockLen | 待搬运的每个连续传输数据块长度，单位为DataBlock（32字节）。uint16_t类型，取值范围：blockLen ∈ [1, 65535]。特别地，当dst位于C2PIPE2GM时，单位为128B；当dst位于C2时，表示源操作数的连续传输数据块长度，单位为64B |
| srcGap | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，srcGap不要超出该数据类型的取值范围 |
| dstGap | 目的操作数相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，dstGap不要超出该数据类型的取值范围。特别地，当dstLocal位于C2PIPE2GM时，单位为128B；当dstLocal位于C2时，单位为64B |

## 约束说明

无

## 返回值说明

无

## 调用示例

**示例1：使用copyParams参数**
```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;

int dataSize = 32; // 假设T为half类型，从UB上申请一块内存32 * sizeof(half)字节
int offset = 32; // 读回时在spmBuffer上偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half));
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
AscendC::DataCopyParams copyParams{1, 2, 0, 0}; // 搬运一个连续传输数据块，连续传输数据块的长度为2个datablock，一个datablock32字节
pipe.ReadSpmBuffer(writeLocal, copyParams, offset);
```

**示例2：使用readSize参数**
```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;

int dataSize = 64; // 从UB上申请一块内存64*sizeof(half)字节
int offset = 32; // 读回时在spmBuffer上偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half));
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
pipe.ReadSpmBuffer(writeLocal, dataSize, offset);
```
