##### ConstructFromInputs

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

支持基于用户构造的 Operator 对象生成一个 Graph 对象。

功能与 `15.2.3.9.24 SetInputs` 一致。SetInputs 未来会逐步消亡，统一使用此接口。

## 函数原型

```cpp
static GraphPtr ConstructFromInputs(const std::vector<Operator> &inputs, const AscendString &name)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| inputs | 输入 | 整图输入的 Operator |
| name   | 输入 | Graph 的名字 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | GraphPtr | 图指针，返回新构造的图 |

## 约束说明

无

## 调用示例

```cpp
GraphPtr graph;
graph = Graph::ConstructFromInputs(inputs, graph_name);
graph->SetOutputs(outputs);
```
