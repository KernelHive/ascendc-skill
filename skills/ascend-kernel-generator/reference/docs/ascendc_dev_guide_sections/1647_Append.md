##### Append

## 函数功能

向后添加 tiling data，若添加超过可容纳的最大长度，则添加失败。

## 函数原型

```cpp
template<typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
ge::graphStatus Append(const T &data)

template<typename T, typename std::enable_if<std::is_standard_layout<T>::value, int>::type = 0>
ge::graphStatus Append(const T *data, size_t append_num)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| T | 输入 | 添加的 tiling data 的类型 |
| data | 输入 | ● 引用类型：添加的 tiling data 实例<br>● 指针类型：添加的 tiling data 起始地址 |
| append_num | 输入 | 添加的 tiling data 的个数，共添加 append_num 个 T 类型的 tiling data |

## 返回值说明

- 成功返回 `ge::GRAPH_SUCCESS`
- 失败返回 `ge::GRAPH_FAILED`

## 约束说明

添加的 tiling data 必须为符合 standard_layout，即内存平铺。

## 调用示例

```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());

// 示例 1
struct AppendData {
    int a = 10;
    int b = 100;
};
AppendData ad;
auto ret = td->Append<AppendData>(ad); // ge::GRAPH_SUCCESS

// 示例 2
size_t append_num = 10;
int32_t *td = new int32_t[append_num];
auto ret = td->Append<int32_t>(td, append_num); // ge::GRAPH_SUCCESS

// 示例 3
size_t append_num = 50;
int32_t *td = new int32_t[append_num];
auto ret = td->Append<int32_t>(td, append_num); // ge::GRAPH_FAILED
```
