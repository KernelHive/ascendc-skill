###### GetOffsetC

## 功能说明

预留接口，为后续功能做预留。

获取本次计算时当前分片在整个C矩阵中的位置。

## 函数原型

```cpp
__aicore__ inline MatrixOffset GetOffsetC()
```

## 参数说明

无

## 返回值说明

MatrixOffset结构体如下：

```cpp
struct MatrixOffset {
    int32_t offset;
    int32_t row, col;
    int32_t height, width;
};
```

## 约束说明

无
