##### SetDataType

## 函数功能

向 `CompileTimeTensorDesc` 中设置 Tensor 的数据类型。

## 函数原型

```cpp
void SetDataType(const ge::DataType data_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| data_type | 输入 | 需设置的 `CompileTimeTensorDesc` 所描述的 Tensor 的数据类型信息。<br>关于 `ge::DataType` 类型，请参见 15.2.3.58 DataType。 |

## 返回值说明

无。

## 约束说明

无。

## 调用示例

```cpp
auto dtype_ = ge::DataType::DT_INT32;
StorageFormat fmt_(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
ExpandDimsType type_("1001");
gert::CompileTimeTensorDesc td;

td.SetDataType(dtype_);
auto dtype = td.GetDataType();  // ge::DataType::DT_INT32;

td.SetStorageFormat(fmt_.GetStorageFormat());
auto storage_fmt = td.GetStorageFormat();  // ge::FORMAT_NCHW

td.SetOriginFormat(fmt_.GetOriginFormat());
auto origin_fmt = td.GetOriginFormat();  // ge::Format::FORMAT_NC

td.SetExpandDimsType(type_);
auto type = td.GetExpandDimsType();  // type_("1001")
```
