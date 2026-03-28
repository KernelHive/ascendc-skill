###### FormatMatchMode

## 功能说明

设置输入输出 tensor 的 format 匹配模式。

## 函数原型

```cpp
OpDef &FormatMatchMode(FormatCheckOption option)
```

## 参数说明

| 参数    | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| option  | 输入      | 匹配模式配置参数，类型为 `FormatCheckOption` 枚举。支持以下几种取值： |

- **DEFAULT**：对 NCHW/NHWC/DHWCN/NCDHW/NCL 格式的输入输出转成 ND 格式进行处理；
- **STRICT**：对数据格式需要严格区分，针对 NCHW/NHWC/DHWCN/NCDHW/NCL 格式，aclnn 框架侧不做转换处理。

## 返回值说明

`OpDef` 算子定义，`OpDef` 请参考 15.1.6.1.3 OpDef。

## 约束说明

不调用该接口的情况下，默认将 NCHW/NHWC/DHWCN/NCDHW/NCL 格式的输入输出转成 ND 格式进行处理。

## 调用示例

下面示例中，算子 `AddCustom` 输入 `x` 只支持 format 为 `NCHW`，输入 `y` 只支持 format 为 `NHWC`，需要配置 `FormatMatchMode(FormatCheckOption::STRICT)`，如果不配置 aclnn 框架会转成 ND 格式传给算子 tiling。

```cpp
AddCustom(const char* name) : OpDef(name)
{
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_NCHW});
    this->Input("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_NHWC});
    this->Output("z")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .FormatList({ge::FORMAT_ND});
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascendxxx");
    this->FormatMatchMode(FormatCheckOption::STRICT);
}
```
