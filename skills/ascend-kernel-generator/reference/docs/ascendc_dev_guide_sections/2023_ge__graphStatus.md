#### ge::graphStatus

```markdown
`graphStatus`类型即`uint32_t`类型，其不同的状态说明如下：

| 状态 | 值 | 说明 |
|------|----|------|
| `ge::GRAPH_SUCCESS` | 0 | 对应函数执行成功。 |
| `ge::GRAPH_FAILED` | 0xFFFF | 对应函数执行失败。 |
| `ge::GRAPH_PARAM_INVALID` | 50331649 | 对应函数执行失败，执行时存在参数无法通过校验的情况。 |
| `ge::GRAPH_NOT_CHANGED` | 1343242304 | 不符合常量折叠的条件时的错误码。 |
| `ge::GRAPH_NODE_WITHOUT_CONST_INPUT` | 50331648 | 检测到网络中的某个算子的输入没有const数据的场景。 |
```
