# Version-0.2.5.1

## Update

- 选择是否保存`perm_matrix`
- 细化recall

# Version-0.2.5

## Update

- 用`similarity`替换`differ`, 并保存`ref`, `src`, `ground truth`的`similarity`

# Version-0.2.4.1

## Update

- 只保留最后一次迭代的`perm_matrix`
- 选择是否保存稀疏还是完整的`perm_matrix`

# Version-0.2.4

## Updates
- 修改了`eval.py`, 使其保存的`perm_matrix`不再是稀疏矩阵, 而是完整的所有点的对应关系
- 新增`test/load_perm_matrix.py`, 能够保存并绘制指定sample和iter的`perm matrix`
# Version-0.2.3

## Updates

- 新增对每个类别的metrics统计并保存json

# Version-0.2.2

## Updates

- 新增recall的plot绘图和保存

# Version-0.2.1

## Updates

- 在`metric`中增加了`recall`的计算, 输出和保存.
- `Parameter_net`中, `pre_pool`可选`InstanceNorm`还是`GroupNorm`.

# Version-0.2.0

## Updates

- 使partial-partial能够一对一匹配.
