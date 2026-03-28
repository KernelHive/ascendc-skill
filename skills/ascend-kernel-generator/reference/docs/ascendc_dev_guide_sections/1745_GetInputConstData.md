##### GetInputConstData

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

获取输入为Const节点的值。

如果算子的输入是Const节点，会返回Const节点的值，否则返回失败。支持跨子图获取输入是否是Const及Const下的value。

## 函数原型

```cpp
graphStatus GetInputConstData(const int32_t index, Tensor &data) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| index | 输入 | 算子的输入端口号 |
| data | 输出 | 输入Const的值 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS表示成功，GRAPH_NODE_WITHOUT_CONST_INPUT输入非Const，其他表示失败 |

## 约束说明

无
