Linux tee命令用于读取标准输入的数据，并将其内容输出成文件。

tee指令会从标准输入设备读取数据，将其内容输出到标准输出设备，同时保存成文件。

**也就是tee命令从某个地方读取输入，然后将这个输入输出到终端中和文件中。**

**语法**：

```
tee -ai file
# a 表示append到后面，而不是覆盖
# i 表示忽略中断
```

**参考**：https://www.runoob.com/linux/linux-comm-tee.html

**比如要把gpu显存同时输入在终端，并且保存到文件中：**

```
watch -n 1 'gpustat | tee -a gpu.log'
```

- 每隔1s就将gpu的状态输入到log中
- -a表示添加在log后面，不加的话就会重新覆盖
- 管道命令将gpustat输出作为tee的输入

