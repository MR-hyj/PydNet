# Version-0.2.4
## Updates
- 修改了`eval.py`, 使其保存的`perm_metrix`不再是稀疏矩阵, 而是完整的所有点的对应关系
- 新增`test/load_perm_metrix.py`, 能够保存并绘制指定sample和iter的`perm metrix`
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