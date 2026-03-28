#### TensorDescInfo

```cpp
struct TensorDescInfo {
    Format format_ = FORMAT_RESERVED;     /* tbe op register support format */
    DataType dataType_ = DT_UNDEFINED;    /* tbe op register support datatype */
};
```

**成员说明：**

- `format_` - tbe op register support format
  - 类型：`Format`（枚举类型）
  - 默认值：`FORMAT_RESERVED`
  - 定义参考：15.2.3.59 Format

- `dataType_` - tbe op register support datatype
  - 类型：`DataType`（枚举类型）
  - 默认值：`DT_UNDEFINED`
  - 定义参考：15.2.3.58 DataType
