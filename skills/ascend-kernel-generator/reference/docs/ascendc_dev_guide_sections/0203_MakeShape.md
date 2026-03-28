##### MakeShape

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

将传入的数据打包成Shape数据结构。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Ts... | 输入 | 表示输入类型的形参包，使用方法和约束说明同 `Std::tuple` |

## 返回值说明

Shape结构类型（`Std::tuple`类型的别名），用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。

定义如下：

```cpp
template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;
```

## 约束说明

同 `Std::tuple`。

## 调用示例

参见调用示例。
