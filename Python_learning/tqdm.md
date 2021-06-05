tqdm使用方法：

> 和enumerate使用

```
for i, seq in enumerate(tqdm(x[feature])):
```

- enumerate tqdm 可迭代对象即可

> 平时使用

任何可迭代对象，都可以用tqdm封装。