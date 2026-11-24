# pairwise-limiter-truepeak

## 领域

游戏、图形与多媒体

## 主题

限幅器真峰值与块边界

## 说明

真峰值检测、前视按样本与跨块继承。

初始状态下 `media/limiter_truepeak.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
