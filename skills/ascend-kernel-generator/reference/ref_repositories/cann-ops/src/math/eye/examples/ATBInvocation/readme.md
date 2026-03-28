## 概述

本样例基于AscendC自定义[Eye](https://gitee.com/ascend/cann-ops/tree/master/src/math/eye)算子,开发了ATB插件并进行了插件调用测试.

## 项目结构介绍

```
├── EyeOperationATBPlugin               //EyeOperation ATB插件代码

├── EyeOperationTest                   //EyeOperation 测试代码
```

## 样例运行

### Eye AscendC自定义算子部署

参照[eye算子](https://gitee.com/ascend/cann-ops/tree/master/src/math/eye)" **算子包编译部署** "章节

### EyeOperation ATB插件部署

- 运行编译脚本完成部署(脚本会生成静态库.a文件,同时将头文件拷贝到/usr/include,.a文件拷贝到/usr/local/lib下)

  ```
  cd EyeOperationATBPlugin
  bash build.sh
  ```

### EyeOperation测试

- 运行脚本完成算子测试

  ```shell
  cd EyeOperationTest  
  bash script/run.sh
  ```

## EyeOperation算子介绍

### 功能

创建一个二维矩阵 m×n，对角元素全为1，其它元素都为0


### 参数列表

| **成员名称** | 类型         | 默认值 | 取值范围 | **描述**                  | 是否必选 |
| ------------ | ------------ | ------ | -------- | ------------------------- | -------- |
| num_rows     | uint64_t     | /      | /        | 生成的矩阵的行数          | 是       |
| num_columns  | uint64_t     | 0      | /        | 生成的矩阵的列数          | 是       |
| batch_shape  | aclIntArray* | {1}    | -        |                           | 是       |
| dtype        | uint64_t     | 0      | 0,1      | 0表示float32,1表示float16 | 是       |



### 输入

| **参数** | **维度**                   | **数据类型**    | **格式** | 描述                                     |
| -------- | -------------------------- | --------------- | -------- | ---------------------------------------- |
| y        | [dim_0，dim_1，...，dim_n] | float16/float32 | ND       | 输入tensor且为输出tensor。 

### 输出

| **参数** | **维度**                   | **数据类型**    | **格式** | 描述                                     |
| -------- | -------------------------- | --------------- | -------- | ---------------------------------------- |
| y        | [dim_0，dim_1，...，dim_n] | float16/float32 | ND       | 输入tensor且为输出tensor。 |

### 规格约束

暂无
