## aclGetStorageShape

## 函数功能

获取aclTensor的StorageShape，aclTensor由`aclCreateTensor`接口创建。

StorageShape表示aclTensor的实际物理排布shape，是Tensor在内存上实际存在的大小。假设StorageShape为`[10, 20]`，表示该aclTensor在内存上是按照`[10, 20]`排布的。

## 函数原型

```c
aclnnStatus aclGetStorageShape(const aclTensor *tensor, 
                              int64_t **storageDims, 
                              uint64_t *storageDimsNum)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| tensor | 输入 | 输入aclTensor的地址。需提前调用`aclCreateTensor`接口创建aclTensor |
| storageDims | 输出 | 表示StorageShape的维度值 |
| storageDimsNum | 输出 | 表示StorageShape的维度数 |

## 返回值说明

- 返回`0`表示成功
- 返回其他值表示失败，返回码列表参见公共接口返回码

### 可能失败的原因

- 返回`161001`：参数`tensor`或`storageDims`或`storageDimsNum`为空指针

## 约束与限制

参数`storageDims`内存是本接口内部申请，使用完后必须手动释放。

## 调用示例

假设已有aclTensor对象（xTensor），获取其数据类型、数据排布格式、维度、步长、偏移等属性，再根据这些属性创建一个新的aclTensor对象（yTensor）。

关键代码示例如下，仅供参考，不支持直接拷贝运行：

```c
// 1. 创建xTensor
int64_t xViewDims = {2, 4};
int64_t xStridesValue = {4, 1}; // 第1维步长4，第2维步长1
int64_t xStorageDims = {2, 4};
xTensor = aclCreateTensor(xViewDims, 2, ACL_FLOAT16, xStridesValue, 0, 
                         ACL_FORMAT_ND, xStorageDims, 2, nullptr);

// 2. 获取xTensor的各种属性值
// 获取xTensor的逻辑shape，viewDims为{2, 4}, viewDimsNum为2
int64_t *viewDims = nullptr;
uint64_t viewDimsNum = 0;
auto ret = aclGetViewShape(xTensor, &viewDims, &viewDimsNum);

// 获取xTensor的数据类型为ACL_FLOAT16
aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
ret = aclGetDataType(xTensor, &dataType);

// 获取xTensor的步长信息，stridesValue为{4, 1}, stridesNum为2
int64_t *stridesValue = nullptr;
uint64_t stridesNum = 0;
ret = aclGetViewStrides(xTensor, &stridesValue, &stridesNum);

// 获取xTensor的首元素对于storage的偏移值，offset为0
int64_t offset = 0;
ret = aclGetViewOffset(xTensor, &offset);

// 获取xTensor的数据排布格式为ACL_FORMAT_ND
aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
ret = aclGetFormat(xTensor, &format);

// 获取xTensor的实际物理排布shape，storageDims为{2, 4}, storageDimsNum为2
int64_t *storageDims = nullptr;
uint64_t storageDimsNum = 0;
ret = aclGetStorageShape(xTensor, &storageDims, &storageDimsNum);

// device侧地址
void *deviceAddr;

// 3. 根据xTensor的属性创建新的tensor
aclTensor *yTensor = aclCreateTensor(viewDims, viewDimsNum, dataType, 
                                   stridesValue, offset, format,
                                   storageDims, storageDimsNum, deviceAddr);

// 4. 手动释放内存
delete[] viewDims;
delete[] stridesValue;
delete[] storageDims;
```
