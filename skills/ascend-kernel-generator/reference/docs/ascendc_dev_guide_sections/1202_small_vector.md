###### small_vector

```markdown
> **注意**：本章接口为预留接口，后续有可能变更或废弃，不建议开发者使用，开发者无需关注。

## 接口列表

| 接口定义 | 功能说明 |
|----------|----------|
| `PtrToValue(const void *const ptr)` | 将void*型数据转换为uint64_t型。 |
| `ValueToPtr(const uint64_t value)` | 将uint64_t型数据转换为void*型。 |
| `VPtrToValue(const std::vector<void *> v_ptr)` | 将vector容器中void*型元素转换为uint64_t型元素。 |
| `PtrToPtr(TI *const ptr)` | 任意类型指针变量（指针地址不可修改）转换为另一任意类型的指针变量。 |
| `PtrToPtr(const TI *const ptr)` | 任意类型指针变量（指针地址和所指的值不可修改）转换为另一任意类型的指针变量（指针所指的值不可修改）。 |
| `PtrAdd(T *const ptr, const size_t max_buf_len, const size_t idx)` | 当指针变量ptr不为空且idx小于max_buf_len，返回指针地址ptr增加idx长度后的指针，否则返回null。 |
| `SmallVector(const allocator_type &alloc = Alloc())` | SmallVector默认构造函数。 |
| `SmallVector(const size_type count, const T &value, const allocator_type &alloc = Alloc())` | SmallVector构造函数，初始化count个元素值为相同value。 |
| `SmallVector(const size_type count, const allocator_type &alloc = Alloc())` | SmallVector构造函数，带有count个初始元素。 |
| `SmallVector(InputIt first, const InputIt last, const allocator_type &alloc = Alloc())` | SmallVector构造函数，利用迭代器范围初始化元素。 |
| `SmallVector(const SmallVector &other)` | SmallVector拷贝构造函数，深拷贝，复制other中元素并存储在新的内存中。 |
| `SmallVector(SmallVector &&other)` | SmallVector移动构造函数，将已存在的vector对象资源移动到新创建的vector中。 |
| `SmallVector(const std::initializer_list<T> init, const allocator_type &alloc = Alloc())` | SmallVector构造函数。 |
| `clear()` | 清空整个SmallVector。 |
| `ClearElements()` | 清空整个SmallVector，并返回迭代器初始地址。 |
| `FreeStorage()` | 清空内存。 |
| `at(const size_type index)` | 返回SmallVector中index位置的元素的引用。 |
| `front()` | 返回SmallVector中第一个位置元素的引用。 |
| `begin()` | 返回一个迭代器，指向SmallVector容器中第一个元素。 |
| `back()` | 返回SmallVector中最后一个位置元素的引用。 |
| `rbegin()` | 返回一个反向迭代器，指向SmallVector容器中的最后一个元素。 |
| `data()` | 返回指向第一个元素的指针。 |
| `GetPointer()` | 返回指向第一个元素的指针。 |
| `cbegin()` | 返回一个const_iterator，指向SmallVector容器第一个元素。 |
| `cend()` | 返回一个const_iterator，指向SmallVector容器最后一个元素的下一个元素。 |
| `crbegin()` | 返回一个const_iterator，指向SmallVector容器中最后一个元素。 |
| `rend()` | 返回一个反向迭代器，指向SmallVector容器中第一个元素之前的元素，该元素被视为其反向结束。 |
| `crend()` | 返回一个const反向迭代器，指向SmallVector容器中第一个元素之前的元素，该元素被视为其反向结束。 |
| `size()` | 返回SmallVector容器中元素的个数。 |
| `reserve(const size_type new_cap)` | 当new_cap大于已分配容量，则更改容器容量，要求为SmallVector容器的元素分配的存储空间的容量至少足以容纳返回已分配存储容量的大小个元素。 |
| `capacity()` | 返回已分配存储容量的大小。 |
| `insert(const_iterator const pos, const T &value)` | 在SmallVector容器指定位置前插入元素。 |
| `insert(const_iterator const pos, T &&value)` | 在SmallVector容器指定位置前插入value元素，value是浅拷贝。 |
| `insert(const_iterator const pos, const size_type count, const T &value)` | 在SmallVector容器指定位置前插入count个值为value的元素。 |
| `insert(const_iterator const pos, const InputIt first, const InputIt last)` | 将指定元素范围[first, last)中的元素复制并插入到SmallVector容器指定位置前。 |
| `insert(const_iterator const pos, const std::initializer_list<T> value_list)` | 在SmallVector容器指定位置前插入T类型对象数组的所有元素。 |
| `emplace(const_iterator const pos, Args &&...args)` | 将一个新元素直接插入到SmallVector容器中的pos之前，该元素是通过args参数直接构造出来的。 |
| `erase(const_iterator const pos)` | 删除SmallVector容器指定位置的一个元素。 |
| `erase(const_iterator const first, const_iterator const last)` | 删除SmallVector容器指定范围[first, last)的元素。 |
| `push_back(const T &value)` | 在SmallVector容器末尾加上一个元素。 |
| `push_back(T &&value)` | 在SmallVector容器末尾加上元素value元素，value元素是浅拷贝。 |
| `emplace_back(Args &&...args)` | 在SmallVector容器末尾插入一个元素，该元素是通过args参数直接构造出来的。 |
| `pop_back()` | 删除SmallVector容器最后一个元素。 |
| `resize(const size_type count)` | 调整SmallVector容器大小为count。如果count小于当前容器大小，则取前count个元素，否则在容器后面使用默认构造函数增加相应的元素。 |
| `resize(const size_type count, const T &value)` | 调整SmallVector容器大小为count。如果count小于当前容器大小，则取前count个元素，否则将value复制到增加的元素中。 |
| `swap(SmallVector &other)` | 将SmallVector容器的内容与other SmallVector容器的内容交换。 |
| `GetPointer(const size_type idx = 0UL)` | 返回指定位置的地址。 |
| `InitStorage(const size_type size)` | 初始化整个容器内存，若size小于当前容量，初始化当前容量大小的内存，若size大于当前容量，resize容器到size大小，在初始化内存。 |
| `CopyRange(T *iter, InputIt first, const InputIt last)` | 拷贝[first, last)范围内存到迭代器iter位置。 |
| `MoveFrom(SmallVector &other)` | 从other容器拷贝内存到当前容器。 |
| `CheckOutOfRange(const size_type index)` | 检查index是否超出容器范围。 |
| `ExpandCap(const size_type range_begin, const size_type range_len)` | 先申请一块新的内存，大小为原容器容量加上range_len大小，新容量不足原容器2倍则按2倍扩容。在原SmallVector容器range_begin的位置拓展range_len大小的内存，并拷贝到新内存中，释放原容器内存。 |
| `ExpandSize(const size_type range_begin, const size_type range_len)` | 在SmallVector容器中，range_begin的位置，拓展range_len大小的内存。 |
| `Expand(const size_type range_begin, const size_type range_len)` | 拓展SmallVector容器内存，若拓展后大于容器最大容量，按ExpandCap方式拓展，否则按ExpandSize方式拓展。 |
| `Shrink(const size_type range_begin, const size_type range_end)` | 在SmartVector容器中，删掉从range_begin到range_end之间的内存。 |
| `swap(op::internal::SmallVector<T, N, Alloc> &sv1, op::internal::SmallVector<T, N, Alloc> &sv2)` | 交换两个SmallVector容器sv1和sv2的元素。 |
```
