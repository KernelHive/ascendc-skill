##### AscendStringToDataType

## 函数功能

将 DataType 字符串表达转化为 DataType 类型值。

使用该接口需要包含 `type_utils.h` 头文件。

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static DataType AscendStringToDataType(const AscendString &str)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| str  | 输入      | 待转换的 DataType 字符串形式，AscendString 类型。 |

## 返回值说明

- 输入合法时，返回转换后的 DataType enum 值，枚举定义请参考 15.2.3.58 DataType。
- 输入不合法时，返回 `DT_UNDEFINED` 并打印报错日志。

## 约束说明

无。

## 调用示例

```cpp
ge::AscendString type_str("DT_UINT32");
auto data_type = ge::TypeUtils::AscendStringToDataType(type_str); // 8
```
