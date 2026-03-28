#### TensorPlacementUtils

## 功能概述

提供一组函数，用于判断 TensorPlacement 的位置。

## 函数原型

```cpp
class TensorPlacementUtils {
public:
    // 判断 Tensor 是否位于 Device 上的内存
    static bool IsOnDevice(TensorPlacement placement) {
        ...
    }
    
    // 判断 Tensor 是否位于 Host 上
    static bool IsOnHost(TensorPlacement placement) {
        ...
    }
    
    // 判断 Tensor 是否位于 Host 上，且数据紧跟在结构体后面
    static bool IsOnHostFollowing(TensorPlacement placement) {
        ...
    }
    
    // 判断 Tensor 是否位于 Host 上，且数据不紧跟在结构体后面
    static bool IsOnHostNotFollowing(TensorPlacement placement) {
        ...
    }
    
    // 判断 Tensor 是否位于 Device 上的内存
    static bool IsOnDeviceHbm(TensorPlacement placement) {
        ...
    }
    
    // 判断 Tensor 是否位于 Device 上的 P2p 内存
    static bool IsOnDeviceP2p(TensorPlacement placement) {
        ...
    }
};
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| placement | 输入 | 需要进行判断的 TensorPlacement 枚举 |

## 返回值说明

- `true`：表示是
- `false`：表示不是

## 约束说明

无

## 调用示例

```cpp
TensorData tensor_data;
tensor_data.SetPlacement(TensorPlacement::kOnHost);
auto on_host = TensorPlacementUtils::IsOnHost(tensor_data.GetPlacement()); // on_host is true
```
