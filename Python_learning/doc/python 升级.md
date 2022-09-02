报错：

```
W: GPG 错误：http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu xenial InRelease: 由于没有公钥，无法验证下列签名： NO_PUBKEY FCAE110B1118213C
E: 仓库 “http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu xenial InRelease” 没有数字签名。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
```

解决办法：

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FCAE110B1118213C
```

参考网址：https://blog.csdn.net/LUONIANXIN08/article/details/115109557

python手动安装：https://www.codenong.com/cs107056617/

```
wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz
tar -zxvf Python-3.7.7.tgz
cd Python-3.7.7
./configure
make
sudo make install
```

切换默认的python源

```
mv /usr/bin/python /usr/bin/python.bak
ln -s /usr/local/bin/python3.7  /usr/bin/python
mv /usr/bin/pip /usr/bin/pip.bak # 这个本人调试失败了，但是没有这个代码也行
ln -s /usr/local/bin/pip3.7 /usr/bin/pip
```

