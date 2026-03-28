##### operator==

## 函数功能

判断格式是否相同。

## 函数原型

```cpp
bool operator==(const StorageFormat &other) const
```

## 参数说明

| 参数   | 输入/输出 | 说明         |
|--------|-----------|--------------|
| other  | 输入      | 另一种格式。 |

## 返回值说明

- `true`：代表格式相同
- `false`：代表格式不相同

## 约束说明

无。

## 调用示例

```cpp
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
StorageFormat another_format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC, dim_type);
bool is_same_fmt = format == another_format; // false
```
