##### layout

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

获取Layout实例化对象。

## 函数原型

```cpp
__aicore__ inline constexpr decltype(auto) layout() {}
__aicore__ inline constexpr decltype(auto) layout() const {}
```

## 参数说明

无

## 返回值说明

返回Layout实例化对象。

## 约束说明

构造Layout对象时传入的Shape和Stride结构，需是Std::tuple结构类型，且满足Std::tuple结构类型的使用约束。

## 调用示例

```cpp
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

// 使用layout函数获取实例化对象
constexpr auto& layout = layoutInit.layout();
```
