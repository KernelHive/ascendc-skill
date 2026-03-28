##### Layout 简介

`Layout<Shape, Stride>` 数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

`Layout` 包含两个核心组成部分：

- **Shape**：定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。
- **Stride**：定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与 Shape 的维度信息一一对应。

例如，一个二维矩阵的 Shape 为 `(4, 2)`，Stride 为 `(4, 1)`，表示：

- 矩阵有 4 行、2 列。
- 列方向上的步长为 1，即每行中相邻元素在内存中的间隔为 1 个元素；行方向上的步长为 4，即相邻行的起始地址间隔为 4 个元素。

表 15-62 中给出了一维内存地址空间视图，表 15-63 中给出了该二维矩阵的逻辑视图。

**表 15-62 线性地址视图**

| 地址 | 0   | 1   | 2-3 | 4   | 5   | 6-7 | 8   | 9   | 10-11 | 12  | 13  |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-------|-----|-----|
| 元素 | a00 | a01 | -   | a10 | a11 | -   | a20 | a21 | -     | a30 | a31 |

**表 15-63 矩阵逻辑视图**

| 索引 | 列0（地址） | 列1（地址） |
|------|-------------|-------------|
| 行0  | a00 (0)     | a01 (1)     |
| 行1  | a10 (4)     | a11 (5)     |
| 行2  | a20 (8)     | a21 (9)     |
| 行3  | a30 (12)    | a31 (13)    |

## 需要包含的头文件

```cpp
#include "kernel_operator_layout.h"
```

## 原型定义

```cpp
template <typename ShapeType, typename StrideType>
struct Layout : private Std::tuple<ShapeType, StrideType> {
    __aicore__ inline constexpr Layout(const ShapeType& shape = {}, const StrideType& stride = {}) :
        Std::tuple<ShapeType, StrideType>(shape, stride) {}

    __aicore__ inline constexpr decltype(auto) layout() {}
    __aicore__ inline constexpr decltype(auto) layout() const {}

    __aicore__ inline constexpr decltype(auto) GetShape() {}
    __aicore__ inline constexpr decltype(auto) GetShape() const {}

    __aicore__ inline constexpr decltype(auto) GetStride() {}
    __aicore__ inline constexpr decltype(auto) GetStride() const {}

    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
};
```

## 模板参数

**表 15-64 模板参数说明**

| 参数名      | 描述                                                                                     |
|-------------|------------------------------------------------------------------------------------------|
| ShapeType   | `Std::tuple` 结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。 |
| StrideType  | `Std::tuple` 结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与 Shape 的维度信息一一对应。 |

## 成员函数

```cpp
__aicore__ inline constexpr Layout(const ShapeType& shape = {}, const StrideType& stride = {}) :
    Std::tuple<ShapeType, StrideType>(shape, stride)
```

```cpp
__aicore__ inline constexpr decltype(auto) layout()
__aicore__ inline constexpr decltype(auto) layout() const
```

```cpp
__aicore__ inline constexpr decltype(auto) GetShape()
__aicore__ inline constexpr decltype(auto) GetShape() const
```

```cpp
__aicore__ inline constexpr decltype(auto) GetStride()
__aicore__ inline constexpr decltype(auto) GetStride() const
```

```cpp
template <typename CoordType>
__aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
```

## 相关接口

```cpp
// Shape 结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)

// Stride 结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)

// Layout 结构构造方法
template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto MakeLayout(const ShapeType& shape, const StrideType& stride)

// is_layout 原型定义
template <T>
struct is_layout;
```
