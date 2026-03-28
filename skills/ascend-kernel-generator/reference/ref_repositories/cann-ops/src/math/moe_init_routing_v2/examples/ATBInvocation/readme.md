## 概述

本样例基于AscendC自定义[MoeInitRoutingV2](https://gitee.com/ascend/cann-ops/tree/master/src/math/moe_init_routing_v2)算子,开发了ATB插件并进行了插件调用测试.

## 项目结构介绍

```
├── MoeInitRoutingV2OperationATBPlugin               //MoeInitRoutingV2Operation ATB插件代码

├── MoeInitRoutingV2OperationTest                   //MoeInitRoutingV2Operation 测试代码
```

## 样例运行

### MoeInitRoutingV2 AscendC自定义算子部署

参照[MoeInitRoutingV2算子](https://gitee.com/ascend/cann-ops/tree/master/src/math/moe_init_routing_v2)" **算子包编译部署** "章节

### MoeInitRoutingV2Operation ATB插件部署

- 运行编译脚本完成部署(脚本会生成静态库.a文件,同时将头文件拷贝到/usr/include,.a文件拷贝到/usr/local/lib下)

  ```
  cd MoeInitRoutingV2OperationATBPlugin
  bash build.sh
  ```

### MoeInitRoutingV2Operation测试

- 运行脚本完成算子测试

  ```shell
  cd MoeInitRoutingV2OperationTest  
  bash script/run.sh
  ```

## MoeInitRoutingV2Operation算子介绍

### 功能

该算子对应MoE（Mixture of Experts，混合专家模型）中的**Routing计算**，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。


### 参数列表

| **成员名称**                       | 类型    | 默认值 | 取值范围   | **描述**                                                     | 是否必选 |
| ---------------------------------- | ------- | ------ | ---------- | ------------------------------------------------------------ | -------- |
| active_num                         | int64_t | 0      | >=0        | 表示是否为Active场景                                         | 否       |
| expert_capacity                    | int64_t | 0      | >=0        | 表示每个专家能够处理的tokens数                               | 否       |
| expert_num                         | int64_t | 0      | >=0        | 表示专家数                                                   | 否       |
| drop_pad_mode                      | int64_t | 0      | 0,1        | 表示是否为Drop/Pad场景                                       | 否       |
| expert_tokens_count_or_cumsum_flag | int64_t | 0      | 0,1,2      | 0：表示不输出expertTokensCountOrCumsumOut。 <br />1：表示输出的值为各个专家处理的token数量的累计值。<br /> 2：表示输出的值为各个专家处理的token数量。 | 否       |
| expertTokensBeforeCapacityFlag     | bool    | false  | false,true | false：表示不输出expertTokensBeforeCapacityOut。<br /> true：表示输出的值为在drop之前各个专家处理的token数量。 | 否       |
| start_expertId                     | int64_t | 0      | /          |                                                              | 否       |
| end_expertId                       | int64_t | 0      | /          |                                                              | 否       |
| device_id                          | int64_t | 0      | /          |                                                              | 否       |

### 输入

| **参数**  | **维度**       | **数据类型**        | **格式** | 描述                                                         | 是否必选 |
| --------- | -------------- | ------------------- | -------- | ------------------------------------------------------------ | -------- |
| x         | [dim_0，dim_1] | float/half/bfloat16 | ND       | 为MOE的输入，即token特征输入                                 | 是       |
| expertIdx | [dim_0，dim_1] | float/half/bfloat16 | ND       | 为每个Token对应的k个处理专家的序号，一般为aclnnMoeGatingTopKSoftmaxV2接口的输出 | 是       |

### 输出

| **参数**                      | **维度**                             | **数据类型**        | **格式** | 描述                                          | 是否必选 |
| ----------------------------- | ------------------------------------ | ------------------- | -------- | --------------------------------------------- | -------- |
| expandedXOut                  | [dim_0,dim_1] or [dim_0,dim_1,dim_2] | float/half/bfloat16 | ND       | 根据expertIdx进行扩展过的特征                 | 是       |
| expandedRowIdxOut             | [dim_0]                              | int32               | ND       | expandedXOut和x的索引映射关系                 | 是       |
| expertTokensCountOrCumsumOut  | [dim_0]                              | int32               | ND       | 输出每个专家处理的token数量的统计结果及累加值 | 否       |
| expertTokensBeforeCapacityOut | [dim_0]                              | int32               | ND       | 输出drop之前每个专家处理的token数量的统计结果 | 否       |

### 规格约束

暂无

