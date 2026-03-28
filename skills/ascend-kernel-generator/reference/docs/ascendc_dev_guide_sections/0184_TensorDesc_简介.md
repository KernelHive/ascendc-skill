##### TensorDesc 简介

`TensorDesc` 用于储存 `ListTensorDesc.GetDesc()` 中根据 index 获取对应的 Tensor 描述信息。

## 原型定义

```cpp
template<class T> class TensorDesc {
    TensorDesc();
    ~TensorDesc();
    void SetShapeAddr(uint64_t* shapePtr);
    uint64_t GetDim();
    uint64_t GetIndex();
    uint64_t GetShape(uint32_t offset);
    T* GetDataPtr();
    GlobalTensor<T> GetDataObj();
}
```

## 模板参数

**表 模板参数说明**

| 参数名 | 描述           |
| ------ | -------------- |
| T      | Tensor 数据类型 |

## 成员函数

- `TensorDesc()`
- `~TensorDesc()`
- `void SetShapeAddr(uint64_t* shapePtr)`
- `uint64_t GetDim()`
- `uint64_t GetIndex()`
- `uint64_t GetShape(uint32_t offset)`
- `T* GetDataPtr()`
- `GlobalTensor<T> GetDataObj()`
