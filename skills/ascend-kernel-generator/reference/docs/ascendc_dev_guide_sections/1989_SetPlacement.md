##### SetPlacement

## 函数功能
设置 Tensor 的数据存放位置。

## 函数原型
```cpp
graphStatus SetPlacement(const ge::Placement &placement)
```

## 参数说明

| 参数名    | 输入/输出 | 描述                     |
|-----------|-----------|--------------------------|
| placement | 输入      | 需设置的数据地址的值。 |

**枚举值定义如下：**
```cpp
enum Placement {
    kPlacementHost = 0,   // host data addr
    kPlacementDevice = 1  // device data addr
};
```

## 返回值
graphStatus 类型：设置成功返回 `GRAPH_SUCCESS`，否则返回 `GRAPH_FAILED`。

## 异常处理
无。

## 约束说明
无。
