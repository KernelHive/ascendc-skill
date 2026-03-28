##### aclrtcCompileProg

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品AI Core | × |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

编译接口，编译指定的程序。

## 函数原型

```c
aclError aclrtcCompileProg(aclrtcProg prog, int numOptions, const char **options)
```

## 参数说明

**表 15-1035 接口参数说明**

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| prog | 输入 | 运行时编译程序的句柄 |
| numOptions | 输入 | 编译选项数量 |
| options | 输入 | 编译选项数组，保存具体的编译选项（默认添加 `-std=c++17`）<br>支持的编译选项可以参考《毕昇编译器用户指南》 |

## 返回值说明

`aclError` 为 int 类型变量，详细说明请参考 15.1.8.1.9 RTC错误码。

## 约束说明

无

## 调用示例

```c
aclrtcProg prog;
const char *options[] = {"--npu-arch=dav-2201"};
int numOptions = sizeof(options) / sizeof(options[0]);
aclError result = aclrtcCompileProg(prog, numOptions, options);
```
