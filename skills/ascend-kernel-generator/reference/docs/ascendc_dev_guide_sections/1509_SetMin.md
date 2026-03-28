##### SetMin

## 函数功能
设置最小的T对象指针。

## 函数原型
```cpp
void SetMin(T *min)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| min  | 输入      | 最小的T对象指针 |

## 返回值说明
无。

## 约束说明
无。

## 调用示例
```cpp
Range<int> range;
int min = -1;
range.SetMin(&min);
```
