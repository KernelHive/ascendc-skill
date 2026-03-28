##### ResetData

## 函数功能

释放 Tensor 中数据内存。

## 函数原型

```cpp
std::unique_ptr<uint8_t[], Tensor::DeleteFunc> ResetData()
```

## 参数说明

无

## 返回值

返回释放后的内存地址和删除器。

## 异常处理

无

## 约束说明

无
