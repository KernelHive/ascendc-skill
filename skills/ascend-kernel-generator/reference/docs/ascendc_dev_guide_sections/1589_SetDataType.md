##### SetDataType

## 函数功能
设置 Tensor 的数据类型。

## 函数原型
```cpp
void SetDataType(const ge::DataType data_type)
```

## 参数说明
| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| data_type | 输入 | 需要设置的 Tensor 的数据类型。<br>关于 `ge::DataType` 的定义，请参见 15.2.3.58 DataType。 |

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
StorageShape sh({1, 2, 3}, {1, 2, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetDataType(ge::DT_DOUBLE);
```
