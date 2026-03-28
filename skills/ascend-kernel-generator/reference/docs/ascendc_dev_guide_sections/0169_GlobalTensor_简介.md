##### GlobalTensor 简介

`GlobalTensor` 用于存放 Global Memory（外部存储）中的全局数据。

`GlobalTensor` 的公共成员函数如下。类型 `T` 支持基础数据类型以及 `TensorTrait` 类型，但需要遵循使用此 `GlobalTensor` 的指令所支持的数据类型。

```cpp
template <typename T> 
class GlobalTensor : public BaseGlobalTensor<T> {
public:
    // PrimT 用于在 T 传入为 TensorTrait 类型时，萃取 TensorTrait 中的 LiteType 基础数据类型
    using PrimType = PrimT<T>;

    // 构造函数
    __aicore__ inline GlobalTensor<T>() {}

    // 初始化 GlobalTensor
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer, uint64_t bufferSize);
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer);

    // 获取全局数据的地址
    __aicore__ inline const __gm__ PrimType* GetPhyAddr() const;
    __aicore__ inline __gm__ PrimType* GetPhyAddr(const uint64_t offset) const;

    // 获取 GlobalTensor 的相应偏移位置的值
    __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint64_t offset) const;

    // 获取某个索引位置的元素的引用
    __aicore__ inline __inout_pipe__(S) __gm__ PrimType& operator()(const uint64_t offset) const;

    // 设置 GlobalTensor 相应偏移位置的值
    __aicore__ inline void SetValue(const uint64_t offset, PrimType value);

    // 获取 GlobalTensor 的元素个数
    __aicore__ inline uint64_t GetSize() const;

    // 返回指定偏移量的 GlobalTensor
    __aicore__ inline GlobalTensor operator[](const uint64_t offset) const;

    // 设置 GlobalTensor 的 shape 信息
    __aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo);

    // 获取 GlobalTensor 的 shape 信息
    __aicore__ inline ShapeInfo GetShapeInfo() const;

    // 设置 GlobalTensor 写入 L2 Cache 的模式
    template<CacheRwMode rwMode = CacheRwMode::RW>
    __aicore__ inline void SetL2CacheHint(CacheMode mode);

    ...
};
```
