##### OpAICoreConfig 注册接口（REGISTER_OP_AICORE_CONFIG）

## 功能说明

不同的硬件形态算子原型定义不同的情况，可以通过新增 `OpAICoreConfig` 的方式，针对不同的 AI 处理器型号注册差异化的算子原型。`REGISTER_OP_AICORE_CONFIG` 宏在不改变原有注册的基础上，允许单独新增文件来注册算子在不同硬件形态上的差异化信息。

使用该注册宏需要包含以下头文件：

```cpp
#include "register/op_config_registry.h"
```

## 函数原型

```cpp
REGISTER_OP_AICORE_CONFIG(opType, socVersion, opFunc)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|------|-----------|------|
| opType | 输入 | 算子类型 |
| socVersion | 输入 | 支持的 AI 处理器型号 |
| opFunc | 输入 | 返回 `OpAICoreConfig` 的回调函数指针，回调函数原型定义为：<br>`OpAICoreConfig (*)()` |

## 返回值说明

无

## 约束说明

若算子同时使用 `AddConfig` 注册算子支持的 AI 处理器型号以及 `OpAICoreConfig` 信息，且 AI 处理器型号相同时，通过 `AddConfig` 方式注册的配置优先级更高，会覆盖 `REGISTER_OP_AICORE_CONFIG` 宏注册的 `OpAICoreConfig` 信息。

## 调用示例

假设，已有原型注册文件 `op_host/add_custom.cpp` 实现如下，配置了算子支持的 AI 处理器型号 `ascendxxx1` 及算子输入输出原型信息：

```cpp
...
namespace ops {
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        // 请替换为实际的AI处理器型号
        this->AICore().AddConfig("ascendxxx1");
    }
};
OP_ADD(AddCustom);
} // namespace ops
```

可新增文件 `op_host/add_custom_xxx.cpp`，使用 `REGISTER_OP_AICORE_CONFIG` 单独注册算子支持的 AI 处理器型号 `ascendxxx2`，示例如下：

```cpp
#include "register/op_config_registry.h"
namespace ops {
REGISTER_OP_AICORE_CONFIG(AddCustom, ascendxxx2, []() {
    ops::OpAICoreConfig config("ascendxxx2");
    return config;
});
}
```
