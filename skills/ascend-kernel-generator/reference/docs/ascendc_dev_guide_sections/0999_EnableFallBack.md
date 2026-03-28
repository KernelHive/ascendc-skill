###### EnableFallBack

## 功能说明

通过本接口启用 fallback 配置，启用后将自动生成一个 fallback 函数并注册给 GE。

fallback 函数的核心逻辑是将 GE 的输入、输出及属性转换为 aclnn 单算子 API 所需的参数格式，随后调用 aclnn 接口。动态图场景下，GE 可直接调用 fallback 函数（函数中调用了 aclnn 接口），从而简化调度流程。关于 fallback 下发算子的详细介绍请参考《图模式开发指南》中的“自定义算子入图开发 > 基于 fallback 形式下发算子”章节。

## 函数原型

```cpp
OpDef &EnableFallBack(void)
```

## 参数说明

无

## 返回值说明

OpDef 算子定义，OpDef 请参考 15.1.6.1.3 OpDef。

## 约束说明

- 算子需要注册并实现 InferShape 函数。
- 算子需要注册并实现 InferDataType 函数。

## 调用示例

```cpp
class AddCustom : public OpDef {
public:
    AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->AICore().AddConfig("ascendxxx");
        this->SetInferShape(ge::InferShapeFunc);
        this->SetInferDataType(ge::InferDataTypeFunc);
        this->EnableFallBack();
    }
};
OP_ADD(AddCustom);
```
