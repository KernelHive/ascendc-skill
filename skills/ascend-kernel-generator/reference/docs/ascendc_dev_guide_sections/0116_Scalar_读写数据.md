### Scalar 读写数据

AI Core 中的 Scalar 计算单元负责各类型的标量数据运算和程序的流程控制。根据硬件架构设计，Scalar 仅支持对 Global Memory 和 Unified Buffer 的读写操作，而不支持对 L1 Buffer、L0A Buffer、L0B Buffer 和 L0C Buffer 等其他类型存储的访问。下文分别介绍了 Scalar 读写 Global Memory 和 Unified Buffer 的方式，以及 Scalar 读写数据时的同步机制。

## 读写 Global Memory

Scalar 读写 GM 数据时会经过 DataCache，DataCache 主要用于提高标量访存指令的执行效率，每一个 AIC/AIV 核内均有一个独立的 DataCache。下面通过一个具体示例来讲解 DataCache 的具体工作机制。

`globalTensor1` 是位于 GM 上的 Tensor：

- 执行完 `GetValue(0)` 后，`globalTensor1` 的前 8 个元素会进入 DataCache，后续 `GetValue(1)`~`GetValue(7)` 不需要再访问 GM，而可以直接从 DataCache 的 Cache Line 中读取数据，提高了标量连续访问的效率。
- 执行完 `SetValue(8, val)` 后，`globalTensor1` 的 index 为 8~15 的元素会进入 DataCache，`SetValue` 只会修改 DataCache 中的 Cache Line 数据，同时将 Cache Line 的状态设置为 Dirty，表明 Cache Line 中的数据与 GM 中的数据不一致。

```cpp
AscendC::GlobalTensor<int64_t> globalTensor1;
globalTensor1.SetGlobalBuffer((__gm__ int64_t *)input);

// 从0~7共计8个uint64_t类型，DataCache的Cache Line长度为64字节
// 执行完GetValue(0)后，GetValue(1)~GetValue(7)可以直接从Cache Line中读取，不需要再访问GM
globalTensor1.GetValue(0);
globalTensor1.GetValue(1);
globalTensor1.GetValue(2);
globalTensor1.GetValue(3);
globalTensor1.GetValue(4);
globalTensor1.GetValue(5);
globalTensor1.GetValue(6);
globalTensor1.GetValue(7);

// 执行完SetValue(8)后，不会修改GM上的数据，只会修改DataCache中Cache Line数据
// 同时Cache Line的状态置为dirty，dirty表示DataCache中Cache Line数据与GM中的数据不一致
int64_t val = 32;
globalTensor1.SetValue(8, val);
globalTensor1.GetValue(8);
```

根据上文的工作机制，多核间访问 `globalTensor1` 会出现数据不一致的情况，如果其余核需要获取 GM 数据的变化，则需要开发者手动调用 `DataCacheCleanAndInvalid` 来保证数据的一致性。

## 读写 Unified Buffer

Scalar 读写 Unified Buffer 时，可以使用 LocalTensor 的 `SetValue` 和 `GetValue` 接口。示例如下：

```cpp
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对inputLocal中第i个位置进行赋值为i
}

for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal.GetValue(i); // 获取inputLocal中第i个位置的数值
}
```

## 读写数据时的同步

Scalar 读写 Global Memory 和 Unified Buffer 时属于 `PIPE_S`（Scalar 流水）操作，当用户使用 `SetValue` 或者 `GetValue` 接口，且算子工程使能自动同步时，不需要手动插入同步事件。

如果用户关闭算子工程的自动同步功能时，则需要手动插入同步事件：

```cpp
// GetValue为Scalar操作，与后续的Duplicate存在数据依赖
// 因此Vector流水需要等待Scalar操作结束
float inputVal = srcLocal.GetValue(0);
SetFlag<HardEvent::S_V>(eventID1);
WaitFlag<HardEvent::S_V>(eventID1);
AscendC::Duplicate(dstLocal, inputVal, srcDataSize);

// SetValue为Scalar操作，与后续的数据搬运操作存在数据依赖
// 因此MTE3流水需要等待Scalar操作结束
srcLocal.SetValue(0, value);
SetFlag<HardEvent::S_MTE3>(eventID2);
WaitFlag<HardEvent::S_MTE3>(eventID2);
AscendC::DataCopy(dstGlobal, srcLocal, srcDataSize);
```
