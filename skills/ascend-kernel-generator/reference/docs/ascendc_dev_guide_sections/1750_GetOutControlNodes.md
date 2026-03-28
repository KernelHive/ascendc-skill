##### GetOutControlNodes

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取算子的控制输出节点。

## 函数原型

```cpp
std::vector<GNodePtr> GetOutControlNodes() const
```

## 参数说明

无

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | vector<GNodePtr> | 算子的控制输出节点列表，空表示没有控制算子。 |

## 约束说明

无
