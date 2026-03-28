##### MakeCoord

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

将传入的数据打包成Coord数据结构。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(Ts const&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | 表示输入数据类型的形参包，具体使用方法和约束说明同`Std::tuple`。输入的数据类型支持`size_t`和`Std::Int`。 |

## 返回值说明

Coord结构类型（`Std::tuple`类型的别名）。

## 约束说明

无

## 调用示例

```cpp
auto blockCoordM = Std::Int<11>{};
auto blockCoordN = Std::Int<12>{};
auto coord = AscendC::MakeCoord(blockCoordM, blockCoordN);
```
