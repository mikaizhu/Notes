删除的话，有删除中间某一层和前面某几层或者最后某几层。

- **删除最后某几层**：

参考：https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2

参考：https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/4

情景1:如果要删除最后某几层

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpnsw70spuj30t60rwgsu.jpg" alt="image.png" style="zoom:50%;" />

**解决方法**：

将前几层的结果返回输出即可

```
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

model = models.resnet18(pretrained=False)
model.fc = Identity() # 将fc层直接替换成输出层，这样就删除了后面几层
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)
```

- **删除并修改某几层**

这里举botnet的例子，因为botnet只是在resnet的基础上修改了网络的最后几层。以下是模板

```
resnet = resnet.resnet50()

# 将resent的maxpool改成adaptive
for name, layer in resnet.named_modules():
    if isinstance(layer, nn.MaxPool2d):
        resnet.maxpool = nn.AdaptiveAvgPool2d((7, 7))   

# 修改最后几层
layer = BottleStack(
    dim = 108, # 图片的大小
    fmap_size = 18,        # set specifically for imagenet's 224 x 224
    dim_out = 512,
    proj_factor = 4,
    downsample = True,
    heads = 4,
    dim_head = 128,
    rel_pos_emb = True,
    activation = nn.ReLU()
)

# model surgery
n_class = 10

backbone = list(resnet.children())

model = nn.Sequential(
    *backbone[:5], # 这里取前面几层
    layer,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(1),
    nn.Linear(2048, n_class)
)
```

- 使用中间某几层

```
# 这里是测试
resnet = resnet.resnet50()
backbone = list(resnet.children())
print(len(backbone[:])) # 10
model = nn.Sequential(*backbone[:5]) # 这样就只使用了resnet的前5层
```

- **修改某一层**：

使用named_modules()

```
for name, layer in model.named_modules():
    if isinstance(layer, nn.MaxPool2d):
        model.maxpool = nn.AdaptiveAvgPool2d((7, 7))   
```

