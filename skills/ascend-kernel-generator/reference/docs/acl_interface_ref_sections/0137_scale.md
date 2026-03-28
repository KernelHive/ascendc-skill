### scale

## 输入

### x
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 输入Tensor
- **规格限制**: 无

### scale
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: scale的Tensor
- **规格限制**: 无

### bias
- **是否必填**: 非必填
- **数据类型**: float16、float32
- **参数解释**: bias tensor
- **规格限制**: 无

## 属性

### axis
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认为1，输入的bottom[0]做scale的起始维度，axis不同则bottom[1]的shape不同，例如，如果bottom[0]是4维，shape为100x3x40x60，输出top[0] shape相同，且给定axis，bottom[1]可以有如下的Shape：
  - (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
  - (axis == 1 == -3) 3; 3x40; 3x40x60
  - (axis == 2 == -2) 40; 40x60
  - (axis == 3 == -1) 60
  并且，bottom[1] 的shape可以为空，表示scale是一个标量（这时axis就没有意义）
- **规格限制**: [-rank(x), rank(x))

### num_axes
- **是否必填**: 非必填
- **数据类型**: int
- **参数解释**: 默认是1，当只有一个bottom的时候有效，表示做scale的维度，-1表示axis开始全部，0表示是参数只有一个scalar
- **规格限制**: [-1, rank(x))，设axis_=axis>0:axis,axis+rank(x)，则要求num_axes+axis_<=rank(x)

## 输出

### y
- **是否必填**: 必填
- **数据类型**: float16、float32
- **参数解释**: 输出Tensor
- **规格限制**: 无
