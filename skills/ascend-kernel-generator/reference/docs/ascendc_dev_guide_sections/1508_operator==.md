##### operator==

## 函数功能

判断与另外一个 `Range` 对象是否相等。如果两个 `Range` 的上下界的地址相同，或者上下界的值相同，则这两个对象相等。

## 函数原型

```cpp
bool operator==(const Range<T>& rht) const
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| rht  | 输入      | 另一个 `Range` 对象 |

## 返回值说明

- `true`：相等
- `false`：不相等

## 约束说明

无。

## 调用示例

```cpp
int min = 0;
int max = 1024;
int max2 = 1024;
Range<int> range1(&min, &max);  // 上界为1024，下界为0
Range<int> range2(&min, &max);  // 上界为1024，下界为0
Range<int> range3(&min, &max2); // 上界为1024，下界为0

auto ret1 = range1 == range2; // true
auto ret2 = range1 == range3; // true
```
