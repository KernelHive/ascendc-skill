###### pool_allocator

> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 15-1019 接口列表**

| 接口定义 | 功能说明 |
|----------|----------|
| `MallocPtr(size_t size)` | 从内存池中申请 `size` 字节的内存。 |
| `FreePtr(void *block)` | 将 `data block` 释放回内存池。 |
| `PoolAllocator()` | `PoolAllocator` 构造函数。 |
| `PoolAllocator(const PoolAllocator<U> &)` | `PoolAllocator` 拷贝构造函数。 |
| `allocate(size_t n)` | 从内存池中申请 `n` 个 `T` 类型大小的内存。 |
| `deallocate(T *p, [[maybe_unused]] size_t n)` | 将 `p` 释放回内存池。 |
| `construct(_Up *__p, _Args &&...__args)` | 用给定的参数 `__args` 和地址 `__p` 调用 `_Up` 的构造函数。 |
| `destroy(_Up *__p)` | 调用 `__p` 的析构函数。 |
