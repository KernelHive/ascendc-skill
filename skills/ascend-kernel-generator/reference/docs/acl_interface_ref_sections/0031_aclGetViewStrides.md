## aclGetViewStrides

## 函数功能

获取 aclTensor 的 ViewStrides，即 ViewShape 对应的 stride。aclTensor 由 `aclCreateTensor` 接口创建。

## 函数原型

```c
aclnnStatus aclGetViewStrides(const aclTensor *tensor, int64_t **stridesValue, uint64_t *stridesNum)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 输入的 aclTensor。需提前调用 `aclCreateTensor` 接口创建 aclTensor。 |
| stridesValue | 输出 | 返回的 ViewStrides 值。 |
| stridesNum | 输出 | 返回的 ViewStrides 的 stride 值个数。 |

## 返回值说明

返回 0 表示成功，返回其他值表示失败，返回码列表参见“公共接口返回码”。

可能失败的原因：

- 返回 161001：参数 tensor 或 stridesValue 或 stridesNum 空指针。

## 约束与限制

参数 stridesValue 内存是本接口内部申请，使用完后必须手动释放。

## 调用示例

假设已有 aclTensor 对象（xTensor），获取其数据类型、数据排布格式、维度、步长、偏移等属性，再根据这些属性创建一个新的 aclTensor 对象（yTensor）。

关键代码示例如下，仅供参考，不支持直接拷贝运行。

```c
// 1. 创建 xTensor
int64_t xViewDims[] = {2, 4};
int64_t xStridesValue[] = {4, 1}; // 第1维步长4，第2维步长1
int64_t xStorageDims[] = {2, 4};
xTensor = aclCreateTensor(xViewDims, 2, ACL_FLOAT16, xStridesValue, 0, ACL_FORMAT_ND, xStorageDims, 2, nullptr);

// 2. 获取 xTensor 的各种属性值
// 获取 xTensor 的逻辑 shape，viewDims 为 {2, 4}, viewDimsNum 为 2
int64_t *viewDims = nullptr;
uint64_t viewDimsNum = 0;
auto ret = aclGetViewShape(xTensor, &viewDims, &viewDimsNum);

// 获取 xTensor 的数据类型为 ACL_FLOAT16
aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
ret = aclGetDataType(xTensor, &dataType);

// 获取 xTensor 的步长信息，stridesValue 为 {4, 1}, stridesNum 为 2
int64_t *stridesValue = nullptr;
uint64_t stridesNum = 0;
ret = aclGetViewStrides(xTensor, &stridesValue, &stridesNum);

// 获取 xTensor 的首元素对于 storage 的偏移值，offset 为 0
int64_t offset = 0;
ret = aclGetViewOffset(xTensor, &offset);

// 获取 xTensor 的数据排布格式为 ACL_FORMAT_ND
aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
ret = aclGetFormat(xTensor, &format);

// 获取 xTensor 的实际物理排布 shape，storageDims 为 {2, 4}, storageDimsNum 为 2
int64_t *storageDims = nullptr;
uint64_t storageDimsNum = 0;
ret = aclGetStorageShape(xTensor, &storageDims, &storageDimsNum);

// device 侧地址
void *deviceAddr;

// 3. 根据 xTensor 的属性创建新的 tensor
aclTensor *yTensor = aclCreateTensor(viewDims, viewDimsNum, dataType, stridesValue, offset, format, storageDims, storageDimsNum, deviceAddr);

// 4. 手动释放内存
delete[] viewDims;
delete[] stridesValue;
delete[] storageDims;
```
