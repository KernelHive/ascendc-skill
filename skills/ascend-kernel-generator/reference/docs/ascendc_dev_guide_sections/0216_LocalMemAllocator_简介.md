##### LocalMemAllocator 简介

```markdown
LocalMemAllocator 是在使用 Ascend C 更底层编程方式时用于内存管理的类，用户无需构建 TPipe/TQue，而是直接创建 LocalTensor 对象（也可以直接通过 LocalTensor 构造函数进行构造）并开发算子，从而减少运行时的开销，实现更优的性能。

LocalMemAllocator 仅支持在 Ascend C 更底层编程方式中使用，不可以与 TPipe 等接口混用。

## 需要包含的头文件

```cpp
#include "kernel_operator.h"
```

## 原型定义

```cpp
template<Hardware hard = Hardware::UB>
class LocalMemAllocator {
public:
    __aicore__ inline LocalMemAllocator();
    __aicore__ inline uint32_t GetCurAddr() const;
    template <TPosition pos, class DataType, uint32_t tileSize>
    __aicore__ inline LocalTensor<DataType> Alloc();
    template <TPosition pos, class DataType>
    LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize);
    template <class TensorTraitType>
    LocalTensor<TensorTraitType> __aicore__ inline Alloc();
};
```

## 模板参数

| 参数名 | 描述 |
|--------|------|
| hard | 用于表示数据的物理位置，Hardware 枚举类型，定义如下，合法位置为：UB、L1、L0A、L0B、L0C、BIAS、FIXBUF。物理位置的具体说明可参考存储单元。<br><br>```cpp<br>enum class Hardware : uint8_t {<br>    GM,     // Global Memory<br>    UB,     // Unified Buffer<br>    L1,     // L1 Buffer<br>    L0A,    // L0A Buffer<br>    L0B,    // L0B Buffer<br>    L0C,    // L0C Buffer<br>    BIAS,   // BiasTable Buffer<br>    FIXBUF, // Fixpipe Buffer<br>    MAX<br>};<br>``` |

## Public 成员函数

- `__aicore__ inline LocalMemAllocator()`
- `__aicore__ inline uint32_t GetCurAddr() const`
- `template <TPosition pos, class DataType, uint32_t tileSize> __aicore__ inline LocalTensor<DataType> Alloc()`
- `template <TPosition pos, class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)`
- `template <class TensorTraitType> LocalTensor<TensorTraitType> __aicore__ inline Alloc()`
```
