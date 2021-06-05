#!/usr/bin/env expect

# 参考：https://github.com/dunwu/linux-tutorial/blob/master/docs/linux/expect.md
# 这里shebang不能设置为/bin/bash 只能设置为expect，因为下面命令都是在expect环境下执行的
# spawn 命令用来启动新的进程，spawn后的send和expect命令都是和使用spawn打开的进程进行交互。
# expect 环境中使用set命令进行赋值
set user "zwl"
set ip "******"
set password "****"
spawn ssh -L 8877:localhost:8888 ${user}@${ip}
# 获取匹配信息，匹配成功则执行 expect 后面的程序动作。
expect "*password"
# 命令接收一个字符串参数，并将该参数发送到进程。
# 这里有个知识点是字符串拼接, 直接使用引号即可将两个字符串拼接
send "${password}\n"
# expect 执行结束，退出。
# expect eof
#interact的作用是，登录完远程机器后，不要退出来，一直保持登录状态，如果不加这句，登录完后马上会退出。
# 参考：https://github.com/aminglinux/shell/blob/master/expect.txt
interact
