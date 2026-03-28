##### GetDataTypeLength

## 功能描述
获取数据类型所占内存大小。

使用该接口需要包含 `type_utils.h` 头文件：

```cpp
#include "graph/utils/type_utils.h"
```

## 函数原型

```cpp
static bool GetDataTypeLength(const ge::DataType data_type, uint32_t &length)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|------------|------|
| data_type | 输入 | 数据类型 |
| length | 输出 | 数据类型所占内存大小，单位：字节 |

## 返回值说明
- 获取成功时返回 `true`
- `data_type` 不支持时返回 `false`

## 约束说明
无

## 调用示例

```cpp
uint32_t type_length;
ge::DataType data_type = ge::DT_INT8;
const bool ret = ge::TypeUtils::GetDataTypeLength(data_type, type_length); // type_length 1
if (!ret) {
    // ...
}
```
