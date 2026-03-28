##### CreateFollowing

## 函数功能

创建一个指定数据类型以及大小的 Tensor，其数据在 Tensor 对象后连续排布。

## 函数原型

### 传入元素个数和数据类型，创建 Tensor

```cpp
static std::unique_ptr<uint8_t[]> CreateFollowing(const int64_t shape_size, const ge::DataType dt, size_t &total_size)
```

### 传入数据类型和 Tensor 长度，创建 Tensor

```cpp
static std::unique_ptr<uint8_t[]> CreateFollowing(const ge::DataType dt, const size_t tensor_size, size_t &total_size)
```

## 参数说明

### 表 15-1051 参数说明（传入元素个数和数据类型，创建 Tensor）

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| shape_size | 输入 | 元素个数 |
| dt | 输入 | 数据类型，15.2.3.58 DataType 类型 |
| total_size | 输出 | 创建出的 Tensor 在内存中的长度。包含 Tensor 对象的长度和 Tensor 数据的长度 |

### 表 15-1052 参数说明（传入数据类型和 Tensor 长度，创建 Tensor）

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| dt | 输入 | 数据类型，15.2.3.58 DataType 类型 |
| tensor_size | 输入 | Tensor 数据的长度。单位为字节 |
| total_size | 输出 | 创建出的 Tensor 在内存中的长度。和 tensor_size 参数不同，total_size 包含 Tensor 对象的长度和 Tensor 数据的长度。单位为字节 |

## 返回值说明

创建的 Tensor 指针。

## 约束说明

无。

## 调用示例

```cpp
size_t total_size;
auto tensor_holder = Tensor::CreateFollowing(shape_size, tensor_desc.GetDataType(), total_size);
```
