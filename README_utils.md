# tutils

`tutils` 是一个为机器学习和数据科学流程设计的常用工具模块，包含矩阵处理、随机工具、内存统计、绘图、jsonl 操作等实用功能，助力模型开发/实验高效进行。

## 安装方式

```bash
pip install .
```

## 用法示例

```python
from tutils import set_seed, Timer, load_jsonl, plot_box, calculate_model_memory
```

## 依赖要求
- numpy
- pandas
- matplotlib
- torch

## 目录结构
- io.py        通用 I/O 工具，包括 jsonl
- matrix.py    矩阵常用统计与筛选
- memory.py    内存和模型统计
- plot.py      快速可视化辅助
- random.py    随机相关（种子、hash 等）
- timing.py    定时器

