what is Macro?

Macro是计算机中宏的英文单词。宏指的是批量处理的一种称呼。

宏就是将一系列指令组织在一起，作为一个单独的命令完成任务。

vim 中的宏：

宏很适合针对一系列相似的行，段落甚至文件，进行重复性的修改。

宏允许我们把一段修改序列录制下来，用于之后的回放。

录制宏命令：

q 开始录制，再按q停止录制。

`qa` use register a, and start recording

then  , input your command

`q` stop recording.

如何使用宏？

使用`:reg a` 查看寄存器a中录制的宏命令。

使用`@a` 调用该寄存器中的宏命令。

使用`10@a` 调用10次这个命令。

使用`@@` 调用上次录制的宏命令。
