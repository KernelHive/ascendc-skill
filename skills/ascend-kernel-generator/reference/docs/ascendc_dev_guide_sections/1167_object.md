###### object

> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

**表 15-1009 接口列表**

| 接口定义 | 功能说明 |
|----------|----------|
| `Object()` | Object的构造函数。 |
| `new(size_t size)` | Object的内存申请函数。 |
| `new(size_t size, [[maybe_unused]] const std::nothrow_t &tag)` | Object的内存申请函数。 |
| `delete(void *addr)` | Object的内存释放函数。 |
