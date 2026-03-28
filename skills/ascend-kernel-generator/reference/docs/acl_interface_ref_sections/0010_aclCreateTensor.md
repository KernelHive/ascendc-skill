## aclCreateTensor

## 函数功能

根据 Tensor 的数据类型、数据排布格式、维度、步长、偏移、Device 侧存储地址等数据，创建 aclTensor 对象，作为单算子 API 执行接口的入参。

aclTensor 是框架定义的一种用来管理和存储张量数据的结构，开发者无需关注其内部实现，直接使用即可。

## 函数原型

```c
aclTensor *aclCreateTensor(const int64_t *viewDims, 
                           uint64_t viewDimsNum,
                           aclDataType dataType, 
                           const int64_t *stride, 
                           int64_t offset, 
                           aclFormat format,
                           const int64_t *storageDims, 
                           uint64_t storageDimsNum, 
                           void *tensorData)
```

## 参数说明

### 数据类型说明

- **aclDataType**：框架定义的一种数据类型枚举类，具体参见《应用开发指南 (C&C++)》中“acl API参考 > 数据类型及其操作接口 > aclDataType”。
- **aclFormat**：框架定义的一种数据格式枚举类，具体参见《应用开发指南 (C&C++)》中“acl API参考 > 数据类型及其操作接口 > aclFormat”。

### StorageShape 和 ViewShape 说明

- **ViewShape**：表示 Tensor 的逻辑 shape，是 Tensor 在实际使用时需要用到的大小。
- **StorageShape**：表示 Tensor 的实际物理排布 shape，是 Tensor 在内存上实际存在的大小。

#### 举例说明

- **StorageShape 为 [10, 20]**：表示该 Tensor 在内存上是按照 [10, 20] 排布的。
- **ViewShape 为 [2, 5, 20]**：在算子使用时，表示该 Tensor 可被视为一块 [2, 5, 20] 的数据使用。

### 参数详情

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| viewDims | 输入 | tensor 的 ViewShape 维度值，为非负整数 |
| viewDimsNum | 输入 | tensor 的 ViewShape 维度数 |
| dataType | 输入 | tensor 的数据类型 |
| stride | 输入 | tensor 各维度元素的访问步长，为非负整数 |
| offset | 输入 | tensor 首元素相对于 storage 的偏移，为非负整数 |
| format | 输入 | tensor 的数据排布格式 |
| storageDims | 输入 | tensor 的 StorageShape 维度值，为非负整数 |
| storageDimsNum | 输入 | tensor 的 StorageShape 维度数 |
| tensorData | 输入 | tensor 在 Device 侧的存储地址 |

## 返回值说明

成功则返回创建好的 aclTensor，否则返回 nullptr。

## 约束与限制

- 本接口需与 `aclDestroyTensor` 接口配套使用，分别完成 aclTensor 的创建与销毁。
- 如需创建多个 aclTensor 对象，可调用 `aclCreateTensorList` 接口来存储张量列表。
- 调用以下接口可获取 aclTensor 的相关属性：
  - `aclGetDataType`：获取 aclTensor 的 DataType
  - `aclGetFormat`：获取 aclTensor 的 format
  - `aclGetStorageShape`：获取 aclTensor 的 StorageShape
  - `aclGetViewOffset`：获取 aclTensor 的 ViewOffset（即 ViewShape 对应的 offset）
  - `aclGetViewShape`：获取 aclTensor 的 ViewShape
  - `aclGetViewStrides`：获取 aclTensor 的 ViewStrides（即 ViewShape 对应的 stride）
  - `aclInitTensor`：初始化给定 tensor 的参数

- 调用以下接口可刷新或获取不同场景下 aclTensor 中记录的 Device 内存地址：
  - `aclSetInputTensorAddr`
  - `aclSetOutputTensorAddr`
  - `aclSetTensorAddr`
  - `aclGetRawTensorAddr`
  - `aclSetRawTensorAddr`

## 调用示例

aclTensor 的定义与 torch.Tensor 相似，由一块连续或非连续的内存地址和一系列描述信息（如 stride、offset 等）组成。Tensor 根据 shape、stride、offset 信息，可自由取出内存中的数据，也可获得非连续的内存。

### 创建 X Tensor

```c
aclTensor *CreateXTensor()
{
    std::vector<int64_t> viewDims = {2, 4};
    std::vector<int64_t> stride = {4, 1};  // 第1维步长4，第2维步长1
    std::vector<int64_t> storageDims = {2, 4};
    return aclCreateTensor(viewDims.data(), 2, ACL_FLOAT16, stride.data(), 0, 
                          ACL_FORMAT_ND, storageDims.data(), 2, nullptr);
}
```

### 创建转置 X Tensor

```c
aclTensor *CreateXTransposedTensor()
{
    std::vector<int64_t> viewDims = {4, 2};
    std::vector<int64_t> stride = {1, 4};  // 转置跨度通过 stride 表示
    std::vector<int64_t> storageDims = {2, 4};
    return aclCreateTensor(viewDims.data(), 2, ACL_FLOAT16, stride.data(), 0, 
                          ACL_FORMAT_ND, storageDims.data(), 2, nullptr);
}
```

### 使用示例

基于上述举例，将 aclTensor 作为单算子 API 执行接口入参的示例代码如下（仅供参考，不支持直接拷贝运行）：

```c
// 创建 aclTensor
aclTensor *xTensor = CreateXTensor();
aclTensor *xTransposedTensor = CreateXTransposedTensor();

// aclTensor 作为单算子 API 执行接口的入参
auto ret = aclxxXxxGetWorkspaceSize(xTensor, xTransposedTensor, ..., outTensor, ..., 
                                   &workspaceSize, &executor);
ret = aclxxXxx(...);

// 销毁 aclTensor
ret = aclDestroyTensor(xTensor);
ret = aclDestroyTensor(xTransposedTensor);
```
