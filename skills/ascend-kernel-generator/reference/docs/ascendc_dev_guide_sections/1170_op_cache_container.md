###### op_cache_container

```markdown
> 本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `ListHead()` | 双向链表的构造函数，用于初始化双向链表。 |
| `Add(ListHead *head)` | 将head插入当前链表的表头。 |
| `Del()` | 将当前链表节点从链表中删除。 |
| `Empty()` | 判断链表是否为空。 |
| `HlistNode()` | hash表节点的构造函数，用于初始化hash表节点。 |
| `HlistHead()` | hash表链表头的构造函数，用于初始化hash表链表头。 |
| `Add(HlistNode *node)` | 将node插入hash表链表头。 |
| `Lru()` | LRU（Least Recently Used）链表的构造函数。 |
| `Head()` | 获取LRU（Least Recently Used）链表头。 |
| `Tail()` | 获取LRU（Least Recently Used）链表尾。 |
| `Sentinel()` | 获取LRU（Least Recently Used）链表哨兵节点。 |
| `Active(ListHead &entry)` | 将entry移动至LRU（Least Recently Used）链表头。 |
| `Del(ListHead &entry)` | 将entry从LRU（Least Recently Used）链表中删除。 |
| `OpCacheContainerIterator(pointer ptr, ListHead *sentinel, bool reverse = false)` | OpCacheContainerIterator构造函数。 |
| `OpCacheContainerIterator(const OpCacheContainerIterator<KeyType, ValueType> &iter)` | OpCacheContainerIterator拷贝构造函数。 |
| `OpCacheContainer(const hasher &hash = hasher(), const key_equal &equal = key_equal())` | OpCacheContainer构造函数（需提供hash函数）。 |
| `OpCacheContainer()` | OpCacheContainer构造函数。 |
| `begin()` | 获取OpCacheContainer的首节点。 |
| `init(size_t capacity)` | OpCacheContainer初始化。 |
| `find(const key_type &key)` | 用给定的key从OpCacheContainer中查找value。 |
| `insert(reference value)` | 将value插入OpCacheContainer中。 |
| `erase(reference value)` | 将value从OpCacheContainer中删除。 |
| `rbegin()` | 获取OpCacheContainer的反向头节点。 |
| `rend()` | 获取OpCacheContainer的反向尾节点。 |
| `size()` | 获取OpCacheContainer的大小。 |
| `bucket(const KeyType &key)` | 用给定的key获取OpCacheContainer的桶。 |
| `bucket_count()` | 获取OpCacheContainer的桶个数。 |
| `value_type()` | 获取OpCacheContainer的value构造函数。 |
| `GetBucket(const key_type &key)` | 用给定的key获取OpCacheContainer的桶。 |
```
