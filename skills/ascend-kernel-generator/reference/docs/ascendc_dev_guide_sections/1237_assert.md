#### assert

## 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | × |
| Atlas 训练系列产品 | × |

## 功能说明

基于算子工程开发的算子，可以使用该接口实现CPU/NPU域assert断言功能。算子执行中，如果assert内部条件判断不为真，则输出assert条件并将输入的信息格式化打印在屏幕上。

在算子kernel侧实现代码中需要增加断言的地方使用assert检查代码，并格式化输出一些调测信息。示例如下：

```cpp
int assertFlag = 10;

assert(assertFlag == 10);
assert(assertFlag == 10, "The assertFlag value is 10.\n");
assert(assertFlag == 10, "The assertFlag value is %d.\n", assertFlag);
```

### 注意

assert接口打印功能会对算子实际运行的性能带来一定影响（每一条assert，系统会额外增加一条逻辑判断，具体性能影响取决于代码中assert的使用数量），通常在调测阶段使用。开发者可以按需通过如下方式关闭打印功能。

#### 自定义算子工程

修改算子工程op_kernel目录下的CMakeLists.txt文件，首行增加编译选项`-DASCENDC_DUMP=0`，关闭ASCENDC_DUMP开关，示例如下：

```cmake
// 关闭所有算子的打印功能
add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0)
```

#### Kernel直调工程

- 修改cmake目录下的npu_lib.cmake文件，在ascendc_compile_definitions命令中增加`ASCENDC_DUMP=0`宏定义来关闭NPU侧ASCENDC_DUMP开关。

示例如下：

```cmake
// 关闭所有算子的打印功能
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
ASCENDC_DUMP=0
)
```

- 修改cmake目录下的cpu_lib.cmake文件，在target_compile_definitions命令中增加`ASCENDC_DUMP=0`宏定义来关闭CPU侧ASCENDC_DUMP开关。

示例如下：

```cmake
target_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
ASCENDC_DUMP=0
)
```

在assert条件被触发时，在断言信息前面会自动打印`CANN_VERSION_STR`值与`CANN_TIMESTAMP`值。其中，`CANN_VERSION_STR`与`CANN_TIMESTAMP`为宏定义，`CANN_VERSION_STR`代表CANN软件包的版本号信息，形式为字符串，`CANN_TIMESTAMP`为CANN软件包发布时的时间戳，形式为数值(uint64_t)。开发者也可在代码中直接使用这两个宏。

assert打印信息示例如下：

```
[ASSERT][CANN_VERSION : XXX.XX][TimeStamp : 20240807140556417] /home/.../add_custom.cpp:44:
Assertion `assertFlag != 10' The assertFlag value is 10.
```

## 函数原型

```cpp
assert(expr)
assert(expr, __gm__ const char *fmt, Args&&... args)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| expr | 输入 | assert断言是否终止程序的条件。为true则程序继续执行，为false则终止程序。 |
| fmt | 输入 | 格式控制字符串，包含两种类型的对象：普通字符和转换说明。<br>● 普通字符将原样不动地打印输出。<br>● 转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型。<br>支持的转换类型包括：<br>– %d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t<br>– %f：输出实数，支持打印的数据类型：float/half<br>– %x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t<br>– %s：输出字符串<br>– %u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t<br>– %p：输出指针地址<br><br>注意：<br>● 上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。<br>● 在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。 |
| args | 输入 | 附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。 |

## 返回值说明

无

## 约束说明

- 该功能仅在如下场景支持：
  - 通过Kernel直调方式调用算子
  - 通过单算子API调用方式调用算子
  - 间接调用单算子API(aclnnxxx)接口：Pytorch框架单算子直调的场景

- kernel开发不要包含系统的assert.h，会导致宏定义冲突
- assert接口调用形式与C语言一致，不需要使用AscendC命名空间
- 本接口不支持打印除换行符之外的其他转义字符
- 该接口使用Dump功能，所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1M。请开发者自行控制待打印的内容数据量，超出则不会打印
- 程序中只调用assert接口，assert接口使用空间每个核上不可超过1k。请开发者自行控制待打印的内容数据量，超出则不会打印

## 调用示例

```cpp
int assertFlag = 10;

// 断言条件
assert(assertFlag == 10);

// 打印消息
assert(assertFlag == 10, "The assertFlag value is 10.\n");

// 格式化打印
assert(assertFlag == 10, "The assertFlag value is %d.\n", assertFlag);
assert(assertFlag != 10, "The assertFlag value is %d.\n", assertFlag);
```

程序运行时会触发assert，打印效果如下：

```
[ASSERT][CANN_VERSION : XXX.XX][TimeStamp : 20240807140556417] /home/.../add_custom.cpp:44:
Assertion `assertFlag != 10' The assertFlag value is 10.
```
