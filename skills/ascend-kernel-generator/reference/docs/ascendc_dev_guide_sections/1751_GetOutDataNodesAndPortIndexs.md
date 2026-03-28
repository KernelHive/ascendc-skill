##### GetOutDataNodesAndPortIndexs

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取算子指定输出端口对应的对端算子以及对端算子的输入端口号。

## 函数原型

```cpp
std::vector<std::pair<GNodePtr, int32_t>> GetOutDataNodesAndPortIndexs(const int32_t index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| index | 输入 | 算子的输出端口号 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | `vector<std::pair<GNodePtr, int32_t>>` | 获取算子指定输出端口对应的对端算子以及对端算子的输入端口号 |

## 约束说明

无
