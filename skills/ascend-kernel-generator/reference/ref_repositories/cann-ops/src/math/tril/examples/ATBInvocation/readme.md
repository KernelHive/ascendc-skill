## 概述

本样例基于AscendC自定义[Tril](https://gitee.com/ascend/cann-ops/tree/master/src/math/tril)算子,开发了ATB插件并进行了插件调用测试.

## 项目结构介绍

```
├── TrilOperationATBPlugin               //TrilOperation ATB插件代码

├── TrilOperationTest                   //TrilOperation 测试代码
```

## 样例运行

### Tril AscendC自定义算子部署

参照[Tril算子](https://gitee.com/ascend/cann-ops/tree/master/src/math/tril)" **算子包编译部署** "章节

### TrilOperation ATB插件部署

- 运行编译脚本完成部署(脚本会生成静态库.a文件,同时将头文件拷贝到/usr/include,.a文件拷贝到/usr/local/lib下)

  ```
  cd TrilOperationATBPlugin
  bash build.sh
  ```

### TrilOperation测试

- 运行脚本完成算子测试

  ```shell
  cd TrilOperationTest  
  bash script/run.sh
  ```

## TrilOperation算子介绍

### 功能
`Tril`算子用于提取张量的下三角部分。返回一个张量`out`，包含输入矩阵(2D张量)的下三角部分，`out`其余部分被设为0。这里所说的下三角部分为矩阵指定对角线`diagonal`之上的元素。参数`diagonal`控制对角线：默认值是`0`，表示主对角线。如果 `diagonal > 0`，表示主对角线之上的对角线；如果 `diagonal < 0`，表示主对角线之下的对角线。

计算公式为：
  $$
  y = tril(x, diagonal)
  $$


### 参数列表

| **成员名称** | 类型         | 默认值 | 取值范围 | **描述**                  | 是否必选 |
| ------------ | ------------ | ------ | -------- | ------------------------- | -------- |
| diagonal     | int          | 0      | /        | 表示对角线的位置          | 是       |




### 输入

| **参数** | **维度**                   | **数据类型**    | **格式** | 描述                                     |
| -------- | -------------------------- | --------------- | -------- | ---------------------------------------- |
| x        | [dim_0，dim_1] | float16/float32 | ND       | 输入tensor。                             |

### 输出

| **参数** | **维度**                   | **数据类型**    | **格式** | 描述                                     |
| -------- | -------------------------- | --------------- | -------- | ---------------------------------------- |
| y        | [dim_0，dim_1] | float16/float32 | ND       | 输出tensor。数据类型和shape与x保持一致。 |

### 规格约束

暂无
