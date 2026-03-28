##### Coordinate 简介

Coordinate 本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。Coordinate（坐标）和 Layout（布局）、Index（内存位置索引）之间存在紧密的关联：

- **从 Coordinate 到 Index 的转换**：Layout 定义了张量的形状和各维度的步长，根据这些信息和给定的 Coordinate，可以计算出该坐标在内存中的位置索引。
- **从 Index 到 Coordinate 的转换**：基于 Layout 中定义的形状和步长信息，对于一个已知的内存位置索引，通过相应的计算可以得到该索引对应的 Coordinate。

## 原型定义

```cpp
template <typename... Coords>
using Coord = Std::tuple<Coords...>;
```

## 模板参数

**表 模板参数说明**

| 参数名 | 描述 |
|--------|------|
| Coords | 表示输入数据类型的形参包，参数个数范围为 [0, 64]。输入的数据类型支持 `size_t` 和 `Std::Int`。 |

## 相关接口

```cpp
// Coord 结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(Ts const&... t);
```

```cpp
// Layout 输入，Coordinate 转换为内存位置索引 Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const Layout<ShapeType, StrideType>& layout);
```

```cpp
// Shape 和 Stride 输入，Coordinate 转换为内存位置索引 Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const ShapeType& shape, const StrideType& stride);
```
