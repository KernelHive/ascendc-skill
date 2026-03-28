##### RegOpLibInit

## 函数功能
注册自定义算子动态库的初始化函数。

## 函数原型
```cpp
OpLibRegister &RegOpLibInit(OpLibInitFunc func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| func | 输入 | 要注册的自定义初始化函数，类型为 `OpLibInitFunc`。 |

```cpp
using OpLibInitFunc = uint32_t (*)();
```

## 返回值说明
返回一个 `OpLibRegister` 对象，该对象新增注册了 `OpLibInitFunc` 函数 `func`。

## 约束说明
无。

## 调用示例
```cpp
uint32_t Init() {
    // init func
    return 0;
}

REGISTER_OP_LIB(vendor_1).RegOpLibInit(Init); // 注册厂商名为 vendor_1 的初始化函数 Init
```
