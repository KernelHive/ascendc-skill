##### UpdateOutputDesc

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

更新指定输出端口的 tensor 格式。

## 函数原型

```cpp
graphStatus UpdateOutputDesc(const int32_t index, const TensorDesc &tensor_desc)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| index | 输入 | 算子的输出端口号。 |
| tensor_desc | 输入 | 需要更新的 tensor 格式。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无
