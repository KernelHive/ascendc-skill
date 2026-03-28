##### CopyFrom

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

拷贝原图生成目标图。

目标图的 `graph_name` 优先使用用户指定的，未指定则使用原图的 `graph_name`。

`graph_id`/`session` 归属，拷贝后需要用户自行添加。

## 函数原型

```cpp
graphStatus Graph::CopyFrom(const Graph &src_graph)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| src_graph | 输入 | 待拷贝的原图 |

## 返回值说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败 |

## 约束说明

无
