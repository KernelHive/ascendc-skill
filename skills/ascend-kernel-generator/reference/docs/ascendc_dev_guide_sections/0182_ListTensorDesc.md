#### ListTensorDesc

## 功能说明

ListTensorDesc 用来解析符合特定内存排布格式的数据，并在 kernel 侧根据索引获取储存对应数据的地址及 shape 信息。

## 需要包含的头文件

```cpp
#include "kernel_operator_list_tensor_intf.h"
```

## 定义原型

```cpp
class ListTensorDesc {
    ListTensorDesc();
    ListTensorDesc(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff);
    void Init(__gm__ void* data, uint32_t length = 0xffffffff, uint32_t shapeSize = 0xffffffff);
    template<class T> void GetDesc(TensorDesc<T>& desc, uint32_t index);
    template<class T> T* GetDataPtr(uint32_t index);
    uint32_t GetSize();
}
```

## 函数说明

### 模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | Tensor 中元素的数据类型 |

### 函数及参数说明

| 函数名称 | 入参说明 | 含义 |
|----------|----------|------|
| ListTensorDesc | - | 默认构造函数，需配合 Init 函数使用 |
| ListTensorDesc | data：待解析数据的首地址<br>length：待解析内存的长度<br>shapeSize：数据指针的个数 | ListTensorDesc 类的构造函数，用于解析对应的内存排布<br>length 和 shapeSize 仅用于校验，不填写时不进行校验 |
| Init | data：待解析数据的首地址<br>length：待解析内存的长度<br>shapeSize：数据指针的个数 | 初始化函数，用于解析对应的内存排布<br>length 和 shapeSize 仅用于校验，不填写时不进行校验 |
| GetDesc | desc：出参，解析后的 Tensor 描述信息<br>index：索引值 | 根据 index 获得对应的 TensorDesc 信息<br>使用 GetDesc 前需要先调用 TensorDesc.SetShapeAddr 为 desc 指定用于储存 shape 信息的地址，调用 GetDesc 后会将 shape 信息写入该地址 |
| GetDataPtr | index：索引值 | 根据 index 获取储存对应数据的地址 |
| GetSize | - | 获取 ListTensor 中包含的数据指针的个数 |

## 支持型号

- Atlas 推理系列产品 AI Core
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品

## 调用示例

```cpp
AscendC::ListTensorDesc listTensorDesc(reinterpret_cast<__gm__ void *>(srcGm)); // srcGm 为待解析的 gm 地址
uint32_t size = listTensorDesc.GetSize(); // size = 2
auto dataPtr0 = listTensorDesc.GetDataPtr<int32_t>(0); // 获取 ptr0
auto dataPtr1 = listTensorDesc.GetDataPtr<int32_t>(1); // 获取 ptr1

uint64_t buf[100] = {0}; // 示例中 Tensor 的 dim 为 3，此处的 100 表示预留足够大的空间
AscendC::TensorDesc<int32_t> desc;
desc.SetShapeAddr(buf); // 为 desc 指定用于储存 shape 信息的地址
listTensorDesc.GetDesc(desc, 0); // 获取索引 0 的 shape 信息

uint64_t dim = desc.GetDim(); // dim = 3
uint64_t idx = desc.GetIndex(); // idx = 0
uint64_t shape[3] = {0};
for (uint32_t i = 0; i < desc.GetDim(); i++) {
    shape[i] = desc.GetShape(i); // GetShape(0) = 1, GetShape(1) = 2, GetShape(2) = 3
}
auto ptr = desc.GetDataPtr();
```
