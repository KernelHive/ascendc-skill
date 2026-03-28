##### RemoveNode

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

删除图中的指定节点，并删除节点之间的连边。

## 函数原型

```cpp
graphStatus RemoveNode(GNode &node)
graphStatus RemoveNode(GNode &node, bool contain_subgraph)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| node | 输入 | 删除图中的指定节点，如果是子图中的节点，则需要指定 `contain_subgraph` 为 `true`。 |
| contain_subgraph | 输入 | 删除的指定节点是否在子图中。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无
