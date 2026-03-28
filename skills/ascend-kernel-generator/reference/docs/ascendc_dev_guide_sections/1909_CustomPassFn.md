##### CustomPassFn

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

注册自定义Pass的执行函数。

关于接口的详细使用方法请参见自定义Pass开发 > 使用自定义Pass修改Graph。

## 函数原型

```cpp
PassRegistrationData &CustomPassFn(const CustomPassFunc &custom_pass_fn)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| custom_pass_fn | 输入 | 自定义Pass的执行函数。 |

## 返回值说明

返回自身对象的引用。

## 约束说明

无

## 回调函数 CustomPassFunc

用户自定义并实现CustomPassFunc类函数，即自定义的改图函数。

```cpp
Status CustomPassFunc(GraphPtr &graph, CustomPassContext &custom_context)
```

### 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| graph | 输入 | 要修改的图。 |
| custom_context | 输入 | 维测类对象，通过此对象向框架注册维测信息。 |

### 返回值说明

- **SUCCESS**：成功。
- 其他值：失败。
