##### operator<<

## 函数功能

使用 `<<` 运算符重载的方式，实现向后添加 tiling data 的功能。若添加超过可容纳的最大长度，则忽略本次操作。

使用 `<<` 添加 tiling data，可以实现和 15.2.2.36.7 Append 相同的功能，使用该运算符更加直观方便。

## 函数原型

```cpp
template<typename T>
TilingData &operator<<(TilingData &out, const T &data)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| T    | 输入      | 添加的 tiling data 的类型 |
| out  | 输出      | TilingData 类实例 |
| data | 输入      | 添加的 tiling data 的实例 |

## 返回值说明

追加完 data 的 TilingData 对象。

## 约束说明

无。

## 调用示例

```cpp
auto td_buf = TilingData::CreateCap(100U);
auto td = reinterpret_cast<TilingData *>(td_buf.get());

struct AppendData {
    int a = 10;
    int b = 100;
};

AppendData ad;
(*td) << ad;
auto data_size = td->GetDataSize(); // 2 * sizeof(int)
```
