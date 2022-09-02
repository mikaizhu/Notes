# collections模块学习

- Counter 模块

> counter 模块主要是用来进行计数的，就是统计可迭代对象元素出现的次数，并以字典的形式返回

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gjp39iykr2j30vs0ukq5s.jpg" alt="image.png" style="zoom:50%;" />

- nametuple 模块

> nametuple 模块相当于创建一个新类，并且可以传入多个变量

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gjp42tprrqj30vy0ei3zy.jpg" alt="image.png" style="zoom:50%;" />

- OrderDict

> 可以发现有序字典，他是会根据值来进行排序的，然后默认是从小到大排序，而普通的字典不会出现这样

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gjp47kkc1gj30xk0oqtas.jpg" alt="image.png" style="zoom:50%;" />

- defaultdict

> 在默认的情况下，如果访问字典中不存在的元素，会报错

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gjp4oeh2vyj319w0yen0e.jpg" alt="image.png" style="zoom: 50%;" />

- deque

> 双端队列，可以从左边pop，也可以从右边pop
>
> - 新功能，可以从左边extend，也可以从右边extend
> - rotate就是将每个元素向左或者向右边移动设置的长度，设置为-1表示元素向左移动1个单位

<img src="http://ww1.sinaimg.cn/large/005KJzqrly1gjp4umncrpj30yu0vkq5m.jpg" alt="image.png" style="zoom:50%;" />