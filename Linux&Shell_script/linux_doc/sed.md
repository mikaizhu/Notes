# linux sed

mac os 下的sed命令和linux中的有点不一样, 使用下面命令，让之与linux相同

```
brew install gnu-sed
alias sed=gsed
```

sed 命令是对文本进行操作

插入文本

```
sed -i '1 i hello' toc.sh
```
