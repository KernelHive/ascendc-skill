##### AddDataEdge

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

新增一条数据边。

## 函数原型

```cpp
graphStatus AddDataEdge(GNode &src_node, const int32_t src_port_index, GNode &dst_node, const int32_t dst_port_index)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| src_node | 输入 | 连接边的源节点。 |
| src_port_index | 输入 | 源节点的输出端口号。 |
| dst_node | 输入 | 连接边的目的节点。 |
| dst_port_index | 输入 | 目的节点的输入端口号。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无
