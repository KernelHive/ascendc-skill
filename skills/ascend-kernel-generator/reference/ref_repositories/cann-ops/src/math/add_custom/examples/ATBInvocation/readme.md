## 概述

本样例基于AscendC自定义[AddCustom](https://gitee.com/ascend/cann-ops/tree/master/src/math/add_custom)算子,开发了ATB插件并进行了插件调用测试.

## 项目结构介绍

```
├── AddOperationATBPlugin               //AddOperation ATB插件代码

├── AddOperationTest                   //AddOperation 测试代码
```

## 样例运行

### Add AscendC自定义算子部署

参照[add_custom算子](https://gitee.com/ascend/cann-ops/tree/master/src/math/add_custom)" **算子包编译部署** "章节

### AddOperation ATB插件部署

- 运行编译脚本完成部署(脚本会生成静态库.a文件,同时将头文件拷贝到/usr/include,.a文件拷贝到/usr/local/lib下)

  ```
  cd AddOperationATBPlugin
  bash build.sh
  ```

### AddOperation测试

- 运行脚本完成算子测试

  ```shell
  cd AddOperationTest  
  bash script/run.sh
  ```

## AddOperation算子介绍

### 功能

实现两个输入张量相加


### 参数列表

该算子参数为空

### 输入

| **参数** | **维度**                   | **数据类型**          | **格式** | 描述       |
| -------- | -------------------------- | --------------------- | -------- | ---------- |
| x        | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输入tensor |
| y        | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输入tensor |

### 输出

| **参数** | **维度**                   | **数据类型**          | **格式** | 描述                                     |
| -------- | -------------------------- | --------------------- | -------- | ---------------------------------------- |
| output   | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输出tensor。数据类型和shape与x保持一致。 |

### 规格约束

暂无
