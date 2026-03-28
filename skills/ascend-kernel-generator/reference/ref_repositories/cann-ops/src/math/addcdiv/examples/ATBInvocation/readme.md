## 概述

本样例基于AscendC自定义[AddCdiv](https://gitee.com/ascend/cann-ops/tree/master/src/math/addcdiv)算子,开发了ATB插件并进行了插件调用测试.

## 项目结构介绍

```
├── AddCdivOperationATBPlugin               //AddCdivOperation ATB插件代码

├── AddCdivOperationTest                   //AddCdivperation 测试代码
```

## 样例运行

### ATB环境变量部署

- 运行脚本完成部署
  ```
  source /usr/local/nnal/atb/set_env.sh
  ```

### AddCdivOperation ATB插件部署

- 运行编译脚本完成部署(脚本会生成静态库.a文件,同时将头文件拷贝到/usr/include,.a文件拷贝到/usr/local/lib下)

  ```
  cd AddCdivOperationATBPlugin
  bash build.sh
  ```

### AddCdivOperation测试

- 运行脚本完成算子测试

  ```shell
  cd AddCdivOperationTest  
  bash script/run.sh
  ```

## AddCdivOperation算子介绍

### 功能

实现了向量x1除以向量x2，乘标量value后的结果再加上向量input_data，返回计算结果的功能。

对应的数学表达式为: y = (input_data + (x1 / x2) * value)


### 参数列表

该算子参数为空

### 输入

|   **参数**  | **维度**                   | **数据类型**          | **格式** | 描述       |
|  ---------  | -------------------------- | --------------------- | -------- | ---------- |
|  inputData  | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输入tensor |
|      X      | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输入tensor |
|      y      | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输入tensor |
|    value    |             0             | float/half/int8/int32 | ND       | 输入tensor |

### 输出

| **参数** | **维度**                   | **数据类型**          | **格式** | 描述                                     |
| -------- | -------------------------- | --------------------- | -------- | ---------------------------------------- |
| output   | [dim_0，dim_1，...，dim_n] | float/half/int8/int32 | ND       | 输出tensor。数据类型和shape与x保持一致。 |

### 规格约束

暂无
