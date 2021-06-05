如何使用bash进行debug呢？

使用set -x 和 set +x进行debug。

```
set -x
ls
set +x
```

bash -x file
