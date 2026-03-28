##### LoadFromMem

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

从内存中读取序列化后 Graph。

## 函数原型

```cpp
graphStatus LoadFromMem(const GraphBuffer &graph_buffer)
graphStatus LoadFromMem(const uint8_t *data, const size_t len)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| graph_buffer | 输入 | 读取的 Graph 的 buffer。 |
| data | 输入 | 读取的 Graph 的内存起始位置。 |
| len | 输入 | 读取的 Graph 的内存长度。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStat | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

仅支持读取 air 格式的文件。
