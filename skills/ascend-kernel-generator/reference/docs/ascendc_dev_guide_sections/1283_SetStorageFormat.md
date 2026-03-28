##### SetStorageFormat

## 函数功能

向 CompileTimeTensorDesc 中设置运行时 Tensor 的数据排布格式。

## 函数原型

```cpp
void SetStorageFormat(const ge::Format format)
```

## 参数说明

| 参数   | 输入/输出 | 说明                                       |
| ------ | --------- | ------------------------------------------ |
| format | 输入      | 需要设置的运行时数据排布格式信息。         |

> 关于 `ge::Format` 的定义，请参见 15.2.3.59 Format。

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
