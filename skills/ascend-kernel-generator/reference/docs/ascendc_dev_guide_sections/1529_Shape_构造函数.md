##### Shape 构造函数

## 函数功能
Shape 构造函数。

## 函数原型
下文中的 `dim_num_` 为维度个数，即有几维；`dims_` 为具体的维度值信息。

- **默认构造函数**  
  默认构造一个 shape，默认构造的 shape 实例中，`dim_num_` 长度为 0。  
  ```cpp
  Shape() : dim_num_(0), dims_{0}
  ```

- **通过 `dims_` 值构造 shape**  
  例如：`Shape({8,3,224,224})` 表示创建一个 Shape 实例，有 4 个维度，每个维度的值分别是 8, 3, 224, 224。  
  ```cpp
  Shape(const std::initializer_list<int64_t> &args) : Shape()
  ```

- **拷贝构造函数**  
  为了提升性能，`dims_` 超过源 Shape `dim_num_` 的空间没有拷贝，可能有脏数据。  
  ```cpp
  Shape(const Shape &other)
  ```

- **拷贝赋值运算符**  
  为了提升性能，`dims_` 超过源 Shape `dim_num_` 的空间没有拷贝，可能有脏数据。  
  ```cpp
  Shape &operator=(const Shape &other)
  ```

## 参数说明

| 参数   | 输入/输出 | 说明                   |
|--------|------------|------------------------|
| args   | 输入       | shape 的所有 dim 值。  |
| other  | 输入       | 源 Shape 对象。        |

## 返回值说明
生成一个初始化的 Shape 对象。

## 约束说明
无。

## 调用示例
```cpp
Shape shape({3, 256, 256}); // dim_num_=3 dims_的前三维的维度为3,256,256
```
