##### MakeStride

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

将传入的数据打包成Stride数据结构。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Ts... | 输入 | 表示输入类型的形参包，使用方法和约束说明同`Std::tuple` |

## 返回值说明

Stride结构类型（`Std::tuple`类型的别名），用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，与Shape的维度信息一一对应。定义如下：

```cpp
template <typename... Strides>
using Stride = Std::tuple<Strides...>;
```

## 约束说明

同`Std::tuple`。

## 调用示例

参见调用示例。
