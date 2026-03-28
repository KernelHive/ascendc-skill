## aclInitTensor

## 函数功能
初始化给定 `aclTensor` 的参数，`aclTensor` 由 `aclCreateTensor` 接口创建。

当用户想复用已创建的 `aclTensor` 时，可使用该接口先重置 `aclTensor` 的各项属性。

## 函数原型
```c
aclnnStatus aclInitTensor(aclTensor *tensor,
                          const int64_t *viewDims,
                          uint64_t viewDimsNum,
                          aclDataType dataType,
                          const int64_t *stride,
                          int64_t offset,
                          aclFormat format,
                          const int64_t *storageDims,
                          uint64_t storageDimsNum,
                          void *tensorDataAddr)
```

## 参数说明

### 数据类型说明
- `aclDataType` 是框架定义的一种数据类型枚举类，具体参见《应用开发指南 (C&C++)》中“acl API参考 > 数据类型及其操作接口 > aclDataType”。
- `aclFormat` 是框架定义的一种数据格式枚举类，具体参见《应用开发指南 (C&C++)》中“acl API参考 > 数据类型及其操作接口 > aclFormat”。

### aclTensor 的 StorageShape 和 ViewShape
- **ViewShape** 表示 Tensor 的逻辑 shape，是 Tensor 在实际使用时需要用到的大小。
- **StorageShape** 表示 Tensor 的实际物理排布 shape，是 Tensor 在内存上实际存在的大小。

#### 举例说明
- StorageShape 为 `[10, 20]`：表示该 Tensor 在内存上是按照 `[10, 20]` 排布的。
- ViewShape 为 `[2, 5, 20]`：在算子使用时，表示该 Tensor 可被视为一块 `[2, 5, 20]` 的数据使用。

### 参数详情

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 待初始化参数的 aclTensor |
| viewDims | 输入 | tensor 的 ViewShape 维度值，为非负整数 |
| viewDimsNum | 输入 | tensor 的 ViewShape 维度数 |
| dataType | 输入 | tensor 的数据类型 |
| stride | 输入 | tensor 各维度元素的访问步长，为非负整数 |
| offset | 输入 | tensor 首元素相对于 storage 的偏移，为非负整数 |
| format | 输入 | tensor 的数据排布格式 |
| storageDims | 输入 | tensor 的 StorageShape 维度值，为非负整数 |
| storageDimsNum | 输入 | tensor 的 StorageShape 维度数 |
| tensorDataAddr | 输入 | tensor 在 Device 侧的存储地址 |

## 返回值说明
返回 0 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。

## 约束与限制
无

## 调用示例
关键代码示例如下，仅供参考，不支持直接拷贝运行。

```cpp
std::vector<int64_t> viewDims = {2, 4};
std::vector<int64_t> stride = {4, 1};
std::vector<int64_t> storageDims = {2, 4};
// tensor 为复用的已经创建的 aclTensor
// deviceAddr 为 tensor 在 Device 侧的存储地址
auto ret = aclInitTensor(tensor,
                         viewDims.data(),
                         viewDims.size(),
                         ACL_FLOAT16,
                         stride.data(),
                         0,
                         aclFormat::ACL_FORMAT_ND,
                         storageDims.data(),
                         storageDims.size(),
                         deviceAddr);
```
