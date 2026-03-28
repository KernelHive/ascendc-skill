###### InitBuffer

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

调用 `TBufPool::InitBuffer` 接口为 `TQue`/`TBuf` 进行内存分配。

## 函数原型

```cpp
template <class T> 
__aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)

template <TPosition pos> 
__aicore__ inline bool InitBuffer(TBuf<pos>& buf, uint32_t len)
```

## 参数说明

### 模板参数说明

| 参数名 | 说明 |
|--------|------|
| T | que参数的类型 |
| pos | Buffer逻辑位置，可以为 VECIN、VECOUT、VECCALC、A1、B1、C1。关于 TPosition 的具体介绍请参考 15.1.4.4.12 TPosition |

### InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| que | 输入 | 需要分配内存的 TQue 对象 |
| num | 输入 | 分配内存块的个数 |
| len | 输入 | 每个内存块的大小，单位为 Bytes，非 32Bytes 对齐会自动向上补齐至 32Bytes 对齐 |

### InitBuffer(TBuf<pos>& buf, uint32_t len) 原型定义参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| buf | 输入 | 需要分配内存的 TBuf 对象 |
| len | 输入 | 为 TBuf 分配的内存大小，单位为 Bytes，非 32Bytes 对齐会自动向上补齐至 32Bytes 对齐 |

## 约束说明

声明 `TBufPool` 时，可以通过 `bufIDSize` 指定可分配 Buffer 的最大数量，默认上限为 4，最大为 16。`TQue` 或 `TBuf` 的物理内存需要和 `TBufPool` 一致。

## 返回值说明

无

## 调用示例

参考 `InitBufPool`
