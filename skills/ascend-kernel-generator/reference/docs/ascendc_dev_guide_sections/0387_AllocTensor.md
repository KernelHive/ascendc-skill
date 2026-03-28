###### AllocTensor

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

从 Queue 中分配 Tensor，Tensor 所占大小为 InitBuffer 时设置的每块内存长度。

## 函数原型

### non-inplace 接口

构造新的 Tensor 作为内存管理的对象。

```cpp
template <typename T>
__aicore__ inline LocalTensor<T> AllocTensor()
```

### inplace 接口

直接使用传入的 Tensor 作为内存管理的对象，可以减少 Tensor 反复创建的开销。

```cpp
template <typename T>
__aicore__ inline void AllocTensor(LocalTensor<T>& tensor)
```

具体使用指导可参考 "12.9 如何使用 Tensor 原地操作提升算子性能"。

## 参数说明

### 模板参数说明

| 参数名 | 说明 |
|--------|------|
| T | Tensor 的数据类型 |

### 参数说明

| 参数名 | 输入/输出 | 含义 |
|--------|-----------|------|
| tensor | 输入 | inplace 接口需要传入 LocalTensor 作为内存管理的对象 |

## 约束说明

- 同一个 TPosition 上的所有 Queue，连续调用 AllocTensor 接口申请的 Tensor 数量，根据 AI 处理器型号的不同，有数量约束。申请 Buffer 时，需要满足该约束：
  - Atlas 训练系列产品不超过 4 个
  - Atlas 推理系列产品 AI Core 不超过 8 个
  - Atlas 推理系列产品 Vector Core 不超过 8 个
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品不超过 8 个
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品不超过 8 个
  - Atlas 200I/500 A2 推理产品不超过 8 个

- non-inplace 接口分配的 Tensor 内容可能包含随机值
- non-inplace 接口，需要将 TQueBind 的 depth 模板参数设置为非零值；inplace 接口，需要将 TQueBind 的 depth 模板参数设置为 0

## 返回值说明

- non-inplace 接口返回值为 LocalTensor 对象
- inplace 接口没有返回值

## 调用示例

### 示例一

```cpp
// 使用 AllocTensor 分配 Tensor
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len); // InitBuffer 分配内存块数为 2，每块大小为 1024Bytes
AscendC::LocalTensor<half> tensor1 = que.AllocTensor<half>(); // AllocTensor 分配 Tensor 长度为 1024Bytes
```

### 示例二

```cpp
// 连续使用 AllocTensor 的限制场景举例如下：
AscendC::TQue<AscendC::TPosition::VECIN, 1> que0;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que1;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que2;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que3;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que4;
AscendC::TQue<AscendC::TPosition::VECIN, 1> que5;

// 不建议：
// 比如，算子有 6 个输入，需要申请 6 块 buffer
// 通过 6 个队列为其申请内存，分别为 que0~que5，每个 que 分配 1 块，申请 VECIN TPosition 上的 buffer 总数为 6
// 假设，同一个 TPosition 上连续 Alloc 的 Buffer 数量限制为 4，超出该限制后，使用 AllocTensor/FreeTensor 会出现分配资源失败
// 在 NPU 上可能体现为卡死等异常行为，在 CPU Debug 场景会出现报错提示
pipe.InitBuffer(que0, 1, len);
pipe.InitBuffer(que1, 1, len);
pipe.InitBuffer(que2, 1, len);
pipe.InitBuffer(que3, 1, len);
pipe.InitBuffer(que4, 1, len);
pipe.InitBuffer(que5, 1, len);

AscendC::LocalTensor<T> local1 = que0.AllocTensor<T>();
AscendC::LocalTensor<T> local2 = que1.AllocTensor<T>();
AscendC::LocalTensor<T> local3 = que2.AllocTensor<T>();
AscendC::LocalTensor<T> local4 = que3.AllocTensor<T>();
// 第 5 个 AllocTensor 会出现资源分配失败，同一个 TPosition 上同时 Alloc 出来的 Tensor 数量超出了 4 个的限制
AscendC::LocalTensor<T> local5 = que4.AllocTensor<T>();

// 此时建议通过以下方法解决：
// 如果确实有多块 buffer 使用，可以将多个 buffer 合并到一块 buffer，通过偏移使用
pipe.InitBuffer(que0, 1, len * 3);
pipe.InitBuffer(que1, 1, len * 3);

/*
 * 分配出 3 块内存大小的 LocalTensor，local1 的地址为 que0 中 buffer 的起始地址，
 * local2 的地址为 local1 的地址偏移 len 后的地址，local3 的地址为 local1 的地址偏移
 * len * 2 的地址
 */
int32_t offset1 = len;
int32_t offset2 = len * 2;
AscendC::LocalTensor<T> local1 = que0.AllocTensor<T>();
AscendC::LocalTensor<T> local2 = local1[offset1];
AscendC::LocalTensor<T> local3 = local1[offset2];
```

### 示例三：inplace 接口

```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::QuePosition::VECIN, 0> que;
int num = 2;
int len = 1024;
pipe.InitBuffer(que, num, len); // InitBuffer 分配内存块数为 2，每块大小为 1024Bytes
AscendC::LocalTensor<half> tensor1;
que.AllocTensor<half>(tensor1); // AllocTensor 分配 Tensor 长度为 1024Bytes
```
