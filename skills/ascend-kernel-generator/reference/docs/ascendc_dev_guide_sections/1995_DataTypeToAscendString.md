##### DataTypeToAscendString

## 函数功能

将 DataType 类型值转化为字符串表达。

## 使用说明

使用该接口需要包含 `type_utils.h` 头文件。

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static AscendString DataTypeToAscendString(const DataType &data_type)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| data_type | 输入 | 待转换的 DataType，支持的 DataType 请参考 15.2.3.58 DataType |

## 返回值说明

转换后的 DataType 字符串，AscendString 类型。

## 约束说明

无。

## 调用示例

```cpp
DataType data_type = ge::DT_UINT32;
auto type_str = ge::TypeUtils::DataTypeToAscendString(data_type); // "DT_UINT32"
const char *ptr = type_str.GetString(); // 获取 char* 指针
```
