---
layout: post
title: CS20SI-tensorflow for research笔记:Lecture2
tags: 深度学习 Tensorflow
key: 20180327_tf2
picture_frame: shadow
---
本文整理自知乎专栏[深度炼丹](https://zhuanlan.zhihu.com/c_94953554)，转载请征求原作者同意。

本文的全部代码都在原作者GitHub仓库[github](http://link.zhihu.com/?target=https%3A//github.com/SherlockLiao/tensorflow-beginner/tree/master/lab)

CS20SI是Stanford大学开设的基于Tensorflow的深度学习研究课程。
## TensorBoard可视化
安装TensorFlow的时候TensorBoard自动安装，使用`writer=tf.summary.FileWriter('./graph',sess.graph)`创建一个文件写入器,`./graph`是文件路径，`sess.graph`表示读入的图结构  
简单的例子：
```python
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(x))
writer.close()  # close the writer when you're done using it
```
打开终端运行程序，输入`tensorboard --logdir="./graphs"`，在网页输入`http://localhost:6006/`，进入tensorboard
![l21.jpg](https://i.loli.net/2018/03/26/5ab8e89eccd5b.jpg)
## 常数类型
创建一个常数：
```python
tf.constant(value, dtype=None, shape=None, name='const', verify_shape=False)
```
比如建立一维向量和矩阵，然后相乘：
```python
a = tf.constant([2,2], name='a')
b = tf.constant([[0,1],[2,3]], name='b')
x = tf.multiply(a, b, name='dot_production')
with tf.Session() as sess:
    print(sess.run(x))
>> [[0,2]
    [4,6]]
```
### 类似numpy的创建
**特殊值常量创建：**
```python
tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
tf.ones(shape, dtype=tf.float32,name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
tf.fill([2,3],8)
>> [[8,8,8],[8,8,8]]
```
**序列创建：**
```python
tf.linspace(start, stop, num, name=None)
tf.linspace(10.0, 13.0, 4)
>> [10.0, 11.0, 12.0, 13.0]
tf.range(start, limit=None, delta=1, dtyde=None, name='range')
tf.range(3, limit=18, delta=3)
>> [3,6,9,12,15]
```
>与numpy不同，tf不能迭代，即
```python
for _ in tf.range(4): #TypeError
```

### 产生随机数
```python
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

```
>tf目前与numpy数据类型通用:
```python
tf.ones([2,2],np.float32)
>> [[1.0,1.0],[1.0,1.0]]
```

## 变量
常量定义保存在计算图中，常量过多使得计算图加载缓慢。
```python
a = tf.Variable(2, name='scalar')
b = tf.Variable([2,3], name='vector')
c = tf.Variable([[0,1][2,3]],name='matrix')
w = tf.Variable(tf.zeros([784,10]),name='weight')
```
### 变量的几个操作
```python
x = tf.Variable()
x.initializer  # 初始化
x.eval()    # 读取里面的值
x.assign()    #分配值给该变量
```
>使用变量前必须初始化，初始化可以看作是一种变量的分配值操作

### 变量的初始化
**一次性全部初始化：**
```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
**一部分初始化：**
```python
init_ab = tf.variables_initializer([a,b],name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)
```
**某个变量的初始化：**
```python
w = tf.Variables(tf.zeros([10,10]))
with tf.Session() as sess:
    sess.run(w.initializer)
```
**注意下面这个例程：**
```python
w = tf.Variable(10)
w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(w.eval())
>> 10
```
得到的答案是`10`而不是`100`的原因是：虽然定义了`assign`操作，但是tensorflow实在`session`中执行操作，所以我们需要执行`assign`操作：
```python
w = tf.Variable(10)
assign_op = w.assign(100)
with tf.Session() as sess:
    sess.run(w.initializer)
    sess.run(assign_op)   # 赋值作为运算
    print(w.eval())
>> 100
```
**用变量定义变量**
```python
w = tf.Variable(tf.truncated_normal([700, 10]))
u = tf.Variable(w * 2)
```
## Session独立
tensorflow的`session`相互独立。
```python
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8
print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42
sess1.close()
sess2.close()
```
## 占位符(Placeholders)
tensorflow有两步，第一步定义图，第二步进行计算。对于图中暂时不知道值的量，可以定义为占位符，之后再用`feed_dict`赋值
### 定义占位符
```python
tf.placeholder(dtype, shape=None, name=None)
```
>最好指定shape，容易Debug

例程：
```python
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = a + b
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))
```
也可以给tensorflow的运算进行feed操作
```python
a = tf.add(2, 3)
b = tf.multiply(a, 3)
with tf.Session() as sess:
    print(sess.run(b, feed_dict={a: 2}))
>> 6
```

## lazy loading
azy loading是指你推迟变量的创建直到你必须要使用他的时候。下面我们看看一般的loading和lazy loading的区别。
```python
# normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(z)

# lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        sess.run(tf.add(x, y))
```
normal loading会在图中创建`x`和`y`变量，同时创建`x+y`运算，而lazy loading只会创建这两个变量：
- normal loading 在`session`中不管做多少次`x+y`，只需要执行`z`定义的加法操作就可以了
- lazy loading在`session`中每进行一次`x+y`，就会在图中创建一个加法操作，计算图就会多一个节点。严重影响图的读入速度。
