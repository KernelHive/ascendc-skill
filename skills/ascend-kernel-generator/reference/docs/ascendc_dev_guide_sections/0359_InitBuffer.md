###### InitBuffer

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

## 功能说明

用于为TQue等队列和TBuf分配内存。

## 函数原型

### 为TQue等队列分配内存

```cpp
template <class T>
__aicore__ inline bool InitBuffer(T& que, uint8_t num, uint32_t len)
```

### 为TBuf分配内存

```cpp
template <TPosition bufPos>
__aicore__ inline bool InitBuffer(TBuf<bufPos>& buf, uint32_t len)
```

## 参数说明

### bool InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义模板参数说明

| 参数名 | 含义 |
|--------|------|
| T | 队列的类型，支持取值TQue、TQueBind |

### bool InitBuffer(T& que, uint8_t num, uint32_t len) 原型定义参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| que | 输入 | 需要分配内存的TQue等对象 |
| num | 输入 | 分配内存块的个数。double buffer功能通过该参数开启：num设置为1，表示不开启double buffer；num设置为2，表示开启double buffer |
| len | 输入 | 每个内存块的大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考6.2.7 非对齐场景 |

### InitBuffer(TBuf<bufPos>& buf, uint32_t len) 原型定义模板参数说明

| 参数名 | 含义 |
|--------|------|
| bufPos | TBuf所在的逻辑位置，TPosition类型 |

### InitBuffer(TBuf<bufPos>& buf, uint32_t len) 原型定义参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| buf | 输入 | 需要分配内存的TBuf对象 |
| len | 输入 | 为TBuf分配的内存大小，单位为字节。当传入的len不满足32字节对齐时，API内部会自动向上补齐至32字节对齐，后续的数据搬运过程会涉及非对齐处理，具体内容请参考6.2.7 非对齐场景 |

## 约束说明

- InitBuffer申请的内存会在TPipe对象销毁时通过析构函数自动释放，无需手动释放
- 如果需要重新分配InitBuffer申请的内存，可以调用Reset，再调用InitBuffer接口
- 一个kernel中所有使用的Buffer数量之和不能超过64

## 返回值说明

返回Buffer初始化的结果。

## 调用示例

```cpp
// 为TQue分配内存，分配内存块数为2，每块大小为128字节
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; // 输出数据队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);

// 为TBuf分配内存，分配长度为128字节
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::A1> buf; // 输出数据管理对象，TPosition为A1
uint32_t len = 128;
pipe.InitBuffer(buf, len);
```
