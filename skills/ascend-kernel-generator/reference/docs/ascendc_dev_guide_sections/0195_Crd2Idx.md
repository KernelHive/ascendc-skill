##### Crd2Idx

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

Crd2Idx函数用于将多维坐标（Coordinate）通过布局（Layout）转换为内存位置索引（Index），这里的Layout包含了Shape和Stride信息。

对于一个布局Layout，其Shape为(d₀, d₁, ..., dₙ)，Stride为(s₀, s₁, ..., sₙ)，Coordinate为(c₀, c₁, ..., cₙ)到线性索引Index的转换公式为：

```
Index = c₀ × s₀ + c₁ × s₁ + ... + cₙ × sₙ
```

例如，对于Shape (3, 4, 5)，Stride (20, 5, 1)和Coordinate (1, 2, 3)：

- 维度0: c₀ × s₀ = 1 × 20 = 20
- 维度1: c₁ × s₁ = 2 × 5 = 10  
- 维度2: c₂ × s₂ = 3 × 1 = 3
- Index = 20 + 10 + 3 = 33

当Coordinate维度和Stride维度不相同时，可以采用去线性化（delinearize）的方法，使得Coordinate维度和Stride维度相同，再使用上述公式计算得到最终结果。

去线性化的方法介绍如下：对于一个n维数组，形状为(d₀, d₁, ..., dₙ)，线性坐标c对应的多维坐标(c₀, c₁, ..., cₙ)，可以通过以下公式进行转换：

```
c₀ = c % d₀
c₁ = (c / d₀) % d₁
c₂ = (c / d₀ / d₁) % d₂
...
cₙ = c / d₀ / d₁ / ... / dₙ₋₁
```

例如：对于Shape ((2, 4), (3, 5))，Stride ((3, 6), (1, 24))，Layout ((2, 4), (3, 5))，Coordinate（11），按照列优先原则，Crd2Idx的结果为：

```
crd2idx = delinearize(11) × stride
= ((11 % 2, (11 / 2) % 4), ((11 / 2 / 4) % 3, (11 / 2 / 4 / 3))) × ((3, 6), (1, 24))
= ((1, 1), (1, 0)) × ((3, 6), (1, 24))
= 1 × 3 + 1 × 6 + 1 × 1 + 0 × 24
= 10
```

总结上述过程，计算公式如下：

```
Index = Σ(cᵢ × sᵢ)
```

其中(d₀, d₁, ..., dₙ)为Shape，(s₀, s₁, ..., sₙ)为Stride，delinearize公式展开如下：

```
c₀ = c % d₀
c₁ = (c / d₀) % d₁
c₂ = (c / d₀ / d₁) % d₂
...
cₙ = c / d₀ / d₁ / ... / dₙ₋₁
```

## 函数原型

```cpp
// Layout输入，Coordinate转换为Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const Layout<ShapeType, StrideType>& layout)

// Shape和Stride输入，Coordinate转换为Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const ShapeType& shape, const StrideType& stride)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| coord | 输入 | Std::tuple结构类型，用于表示张量在不同维度上的坐标值。输入的数据类型支持size_t和Std::Int。 |
| layout | 输入 | 输入的Layout对象。输入的数据类型支持size_t和Std::Int。 |
| shape | 输入 | Std::tuple结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。输入的数据类型支持size_t和Std::Int。 |
| stride | 输入 | Std::tuple结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。输入的数据类型支持size_t和Std::Int。 |

## 返回值说明

返回根据Coordinate信息转换之后的索引值。

## 约束说明

输入参数需满足对应的数据类型要求。

## 调用示例

```cpp
// Layout形式入参计算索引值
constexpr int M = 11;
constexpr int N = 12;
constexpr int blockM = 13;
constexpr int blockN = 14;

auto coord = AscendC::MakeCoord(AscendC::Std::Int<20>{}, AscendC::Std::Int<30>{});
auto shape = AscendC::MakeShape(AscendC::MakeShape(AscendC::Std::Int<blockM>{}, AscendC::Std::Int<M/blockM>{}), AscendC::MakeShape(AscendC::Std::Int<blockN>{}, AscendC::Std::Int<N/blockN>{}));
auto stride = AscendC::MakeStride(AscendC::MakeStride(AscendC::Std::Int<blockN>{}, AscendC::Std::Int<blockM*blockN>{}), AscendC::MakeStride(AscendC::Std::Int<1>{}, AscendC::Std::Int<M*blockN>{}));

auto layout = AscendC::MakeLayout(shape, stride);
auto index = layout(coord); // decltype(index)::value = 590
index = AscendC::Crd2Idx(coord, layout); // decltype(index)::value = 590

// Shape和Stride形式入参计算索引值
auto blockCoordM = AscendC::Std::Int<11>{};
auto blockCoordN = AscendC::Std::Int<12>{};
auto baseShapeM = AscendC::Std::Int<13>{};
auto baseShapeN = AscendC::Std::Int<14>{};
auto basestrideM = AscendC::Std::Int<15>{};
auto basestrideN = AscendC::Std::Int<16>{};
auto coord = AscendC::MakeCoord(AscendC::Std::Int<0>{}, blockCoordN);
auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM), AscendC::MakeStride(basestrideN, basestrideN));

auto index = AscendC::Crd2Idx(coord, shape, stride); // decltype(index)::value = 192
```
