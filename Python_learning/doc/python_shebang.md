Linux中有shebang，设置后，就可以不用每次都自己手动选择环境了。

如果不设置shebang，那么在命令行中执行python。每次都要
`/usr/bin/python3 test.py` 

如果设置shebang，可以`#!/Users/mikizhu/miniconda3/envs/py38_env/bin/python3
`

然后`chmod +x test.py` 

`./test.py` 


