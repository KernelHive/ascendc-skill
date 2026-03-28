##### Range 构造函数

## 函数功能

Range 构造函数，对应如下 3 种构造方法：

- 可以默认构造一个上下界为 `nullptr` 的 range 实例；
- 也可以构造一个通过指定上下界的 range 实例；
- 还可以只传入一个任意类型的指针构造一个上下界相同的 range 实例。

## 函数原型

```cpp
// 默认构造函数，上下界均为空指针
Range()

// 用户指定上界 max，下界 min
Range(T *min, T* max) : min_(min), max_(max)

// 上下界均为 same_ele
explicit Range(T *same_ele) : min_(same_ele), max_(same_ele)
```

## 参数说明

| 参数      | 输入/输出 | 说明                                                                 |
|-----------|-----------|----------------------------------------------------------------------|
| `min`     | 输入      | 下界的指针，类型为 `T*`                                              |
| `max`     | 输入      | 上界的指针，类型为 `T*`                                              |
| `same_ele`| 输入      | 构造相同上下界 range 实例时使用，上下界均使用 `same_ele` 赋值，类型为 `T*` |

## 返回值说明

返回用户指定构造的 range 对象。

## 约束说明

无。

## 调用示例

```cpp
// 1. 默认构造
Range<int> range1; // 上下界均为 nullptr

// 2. 用户指定上下界
int min = 0;
int max = 1024;
Range<int> range2(&min, &max); // 上界为 1024，下界为 0

// 3. 构造上下界相同的 range
Range<int> range3(&min); // 上下界均为 0
```
