参考：https://www.jianshu.com/p/a4c745b6ea9b

这里有个trick，修改某一层：

```
for name, layer in model.named_modules():
    if 'conv' in name:
        对layer进行处理
```

