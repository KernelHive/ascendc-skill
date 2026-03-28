##### GetStride

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

返回描述内存访问步长的Stride对象，与Shape的维度信息一一对应。

## 函数原型

```cpp
__aicore__ inline constexpr decltype(auto) GetStride() {}
__aicore__ inline constexpr decltype(auto) GetStride() const {}
```

## 参数说明

无

## 返回值说明

描述内存访问步长的Stride对象，Stride结构类型（Std::tuple类型的别名），定义如下：

```cpp
template <typename... Strides>
using Stride = Std::tuple<Strides...>;
```

## 约束说明

无

## 调用示例

```cpp
// 初始化Layout数据结构，获取对应数值
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);
AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

int value = AscendC::Std::get<0>(layoutInit.GetStride()); // value = 1
value = AscendC::Std::get<1>(layoutInit.GetStride()); // value = 100
value = AscendC::Std::get<2>(layoutInit.GetStride()); // value = 200
```
