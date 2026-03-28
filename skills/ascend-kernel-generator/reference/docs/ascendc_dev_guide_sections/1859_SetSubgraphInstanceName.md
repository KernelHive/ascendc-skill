##### SetSubgraphInstanceName

## 功能说明

设置子图实例的名称。

## 函数原型

```c
graphStatus SetSubgraphInstanceName(const uint32_t index, const char_t *name);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| index  | 输入      | 子图索引。 |
| name   | 输入      | 子图实例的名称。 |

## 返回值说明

graphStatus 类型：成功，返回 `GRAPH_SUCCESS`，否则，返回 `GRAPH_FAILED`。

## 异常处理

无

## 约束说明

无
