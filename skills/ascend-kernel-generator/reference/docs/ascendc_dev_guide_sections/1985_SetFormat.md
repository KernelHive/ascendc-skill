##### SetFormat

## 函数功能
设置Tensor的Format。

## 函数原型
```cpp
graphStatus SetFormat(const ge::Format &format)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| format | 输入 | 需设置的Format值。<br>关于Format类型，请参见15.2.3.59 Format。 |

## 返回值
graphStatus类型：设置成功返回`GRAPH_SUCCESS`，否则，返回`GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
