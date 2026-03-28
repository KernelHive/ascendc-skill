### 通过缩减 Tensor ShapeInfo 维度，优化栈空间

## 优先级
中

## 描述
GlobalTensor 和 LocalTensor 中通过 ShapeInfo 类型的成员变量来保存 shape 信息。SetShapeInfo/GetShapeInfo 可以设置或者获取 ShapeInfo，在算子实现内部用于 shape 信息保存和传递。

默认情况下支持的最大维度为 8。在不使用上述 ShapeInfo 功能的情况下，不需要这些信息，可以通过 `K_MAX_SHAPE_DIM` 宏将其设置为 0。经实测减小 `K_MAX_SHAPE_DIM` 值，可缩减栈空间，减少 scalar 指令和 cache miss 几率，提升算子性能。

```cpp
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif

struct ShapeInfo {
public:
    uint32_t shape[K_MAX_SHAPE_DIM];
    uint32_t originalShape[K_MAX_SHAPE_DIM];
};

template <typename T>
class GlobalTensor {
private:
    ShapeInfo shapeInfo_;
};

template <typename T>
class LocalTensor {
private:
    ShapeInfo shapeInfo_;
};
```

## 反例
算子无需使用 ShapeInfo，但未对 ShapeInfo 大小进行限制（使用默认值 8），导致浪费 `K_MAX_SHAPE_DIM * sizeof(uint32_t) * 2 * 4` 字节的栈空间。其中：
- 2 表示有 shape 和 originalShape 两个数组
- 4 表示该样例中使用了 GlobalTensor 和 LocalTensor 共 4 个 Tensor

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GlobalTensor<T> dataIn;
    GlobalTensor<T> dataOut;
    LocalTensor<T> vecIn;
    LocalTensor<T> vecOut;
    // ...
}
```

## 正例
算子无需使用 ShapeInfo，设置 `#define K_MAX_SHAPE_DIM 0`，有效缩减了 `K_MAX_SHAPE_DIM * sizeof(uint32_t) * 2 * 4` 大小的栈空间。

```cpp
#define K_MAX_SHAPE_DIM 0

#include "kernel_operator.h"  // 需注意定义 K_MAX_SHAPE_DIM 宏的位置须在包含 Ascend C 相关头文件之前

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GlobalTensor<T> dataIn;
    GlobalTensor<T> dataOut;
    LocalTensor<T> vecIn;
    LocalTensor<T> vecOut;
    // ...
}
```
