##### TilingData 结构定义

## 功能说明

定义一个 `TilingData` 类，添加所需的成员变量（TilingData 字段），用于保存所需 TilingData 参数。完成该 `TilingData` 类的定义后，该类通过继承 `TilingDef` 类（用来存放、处理用户自定义 Tiling 结构体成员变量的基类）提供以下接口：

- `set_{field_name}` 接口：用于设置 `TilingData` 类的字段值，`field_name` 为定义 `TilingData` 类时添加的字段名。
- `get_{field_name}` 接口：用于获取字段名为 `field_name` 的字段值。
- `SaveToBuffer` 接口：完成 `TilingData` 的序列化和保存。
- `GetDataSize` 接口：获取 `TilingData` 的长度。
- `CheckAlignAndGenPlaceHolder`：该接口是内部关联接口，用于框架侧检查 Tiling 结构体中成员变量是否满足字节对齐要求，并对不对齐的变量进行补齐，开发者无需关注。
- `SetDataPtr` 接口：该接口为预留接口，开发者无需关注。

## 函数原型

- 定义一个 `TilingData` 类：
  ```cpp
  BEGIN_TILING_DATA_DEF(class_name)
  ```

- 添加通用数据类型的 TilingData 字段：
  ```cpp
  TILING_DATA_FIELD_DEF(data_type, field_name)
  ```

- 添加数组类型的 TilingData 字段，数组的元素数据类型为通用数据类型：
  ```cpp
  TILING_DATA_FIELD_DEF_ARR(arr_type, arr_size, field_name)
  ```

- 添加结构体类型的 TilingData 字段：
  ```cpp
  TILING_DATA_FIELD_DEF_STRUCT(struct_type, field_name)
  ```

- 定义结束：
  ```cpp
  END_TILING_DATA_DEF
  ```

## 参数说明

### BEGIN_TILING_DATA_DEF 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| class_name | 输入 | 用户定义 tiling 结构体名，与 C++ 变量命名要求一致 |

### TILING_DATA_FIELD_DEF 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| data_type | 输入 | 字段的数据类型 |
| field_name | 输入 | 字段名，与 C++ 变量命名要求一致 |

### TILING_DATA_FIELD_DEF_ARR 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| arr_type | 输入 | 数组元素数据类型 |
| arr_size | 输入 | 数组元素个数 |
| field_name | 输入 | 字段名，与 C++ 变量命名要求一致 |

### TILING_DATA_FIELD_DEF_STRUCT 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| struct_type | 输入 | 结构体类型 |
| field_name | 输入 | 字段名，与 C++ 变量命名要求一致 |

## 约束说明

- 使用时需要包含头文件 `register/tilingdata_base.h`。
- `TILING_DATA_FIELD_DEF` 和 `TILING_DATA_FIELD_DEF_ARR` 中定义的变量，仅支持 `int8_t`、`uint8_t`、`int16_t`、`uint16_t`、`int32_t`、`uint32_t`、`int64_t`、`uint64_t`、`float` 数据类型。
- `TILING_DATA_FIELD_DEF_STRUCT` 中 `struct_type` 仅支持用 `BEGIN_TILING_DATA_DEF` 等定义的 tiling 结构体，不支持直接使用 C++ 语法定义的结构体类型。
- 用户在 host 侧设置参数值和使用 tiling 数据需要使用 `set_xxx` 和 `get_xxx` 接口（`xxx` 请替换为字段名），具体使用方法见调用示例。
- tiling 数据成员需要满足字节对齐要求，即：当前数据成员 `dataVar` 位于结构体的偏移 `offset` 满足 `offset % sizeof(dataVar) == 0`。
- tiling 结构体是全局属性，需注意应通过结构体名作为全局唯一标记，不同算子若注册同名不同结构 tiling 结构体则会发生未定义行为。
- 注册中间结构体时，若中间结构体名为 `struct_name`，则第一个参数固定为 `struct_name#Op`。
- 设置 `TILING_DATA_FIELD_DEF_ARR` 定义的字段值时，需注意 `set_{field_name}` 仅传入数组指针并按照宏中定义的数组长度进行赋值，因此，需用户自行保证传入数组指针指向的数组长度不小于宏中定义的数组长度，避免越界访问的问题。

## 调用示例

```cpp
#include "register/tilingdata_base.h"

// 定义 tilingdata 类
namespace optiling {
BEGIN_TILING_DATA_DEF(Matmul)
TILING_DATA_FIELD_DEF(uint16_t, mmVar);
TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, mmArr);
END_TILING_DATA_DEF;
// 注册中间结构体，第一个参数固定为 struct_name#Op，第二个参数即 struct_name，如 struct_name 为 Matmul，第一个参数为 MatmulOp，第二个参数为 Matmul
REGISTER_TILING_DATA_CLASS(MatmulOp, Matmul) // 注册中间结构体

BEGIN_TILING_DATA_DEF(AddCustomTilingData) // 注册一个 tiling 类，以 tiling 的名字作为入参
TILING_DATA_FIELD_DEF(uint32_t, blkDim); // 添加 tiling 变量类型字段，参与计算核数
TILING_DATA_FIELD_DEF(uint32_t, totalSize); // 添加 tiling 变量类型字段，总计算数据量
TILING_DATA_FIELD_DEF(uint32_t, splitTile); // 添加 tiling 变量类型字段，每个 core 处理的数据分块计算
TILING_DATA_FIELD_DEF_ARR(uint16_t, 3, arrSample); // 添加 tiling 数组类型字段
TILING_DATA_FIELD_DEF_STRUCT(Matmul, mm); // 添加 tiling 结构体类型字段
END_TILING_DATA_DEF; // 定义结束
// 注册算子 tilingdata 类到对应的 AddCustom 算子
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}

// host 侧设置参数值和使用 tiling 参数
static void TilingAddInit(AddCustomTilingData *tiling, uint32_t blockDim)
{
    // 设置参数值
    tiling->set_blkDim(blockDim); // 置值通用数据类型变量 blockDim
    uint16_t arr[] = {10, 2, 8, 2, 3, 4, 5, 2, 1, 2, 4, 4, 5};
    tiling->set_arrSample(arr); // 置值通用数据类型数组变量 arrSample，仅会复制 arr 数据的前三个数据，与 TILING_DATA_FIELD_DEF_ARR 中 arr_size 一致
    tiling->mm.set_mmVar(1); // 置值嵌套结构体通用数据类型变量 mmVar
    tiling->mm.set_mmArr(arr); // 置值嵌套结构体通用数据类型数组 mmArr

    // 使用参数值
    uint32_t useBlockDim = tiling->get_blkDim(); // 获取通用数据类型变量 blockDim
    uint32_t* arrPoint = tiling->get_arrSample(); // 获取通用数据类型数组变量 arrSample
    useBlockDim = tiling->mm.get_mmVar(); // 获取嵌套结构体通用数据类型变量 mmVar
    arrPoint = tiling->mm.get_mmArr(); // 获取嵌套结构体通用数据类型数组 mmArr
}
```
