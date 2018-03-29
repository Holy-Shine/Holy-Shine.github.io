---
layout: post
title: TensorFlow.org教程笔记(二)DataSets快速入门
tags: 深度学习 Tensorflow
key: 20180329_tf_dataset
picture_frame: shadow
---
本文翻译自[www.tensorflow.org](http://www.cnblogs.com/HolyShine/p/www.tensorflow.org)的英文教程。  

`tf.data` 模块包含一组类，可以让你轻松加载数据，操作数据并将其输入到模型中。本文通过两个简单的例子来介绍这个API<!--more-->

- 从内存中的numpy数组读取数据。
- 从csv文件中读取行



## 基本输入

对于刚开始使用`tf.data`，从数组中提取切片(slices)是最简单的方法。  

[笔记(1)TensorFlow初上手](http://logwhen.cn/2018/03/20/TensorFlow.org%E6%95%99%E7%A8%8B%E7%AC%94%E8%AE%B0(%E4%B8%80)Tensorflow%E5%88%9D%E4%B8%8A%E6%89%8B.html)里提到了训练输入函数`train_input_fn`，该函数将数据传输到`Estimator`中：

```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

让我们进一步来看看这个过程。

### 参数

这个函数需要三个参数。期望“array”的参数几乎可以接受任何可以使用`numpy.array`转换为数组的东西。其中有一个例外是对`Datasets`有特殊意义的元组(tuple)。

- **features** ：一个包含原始特征输入的`{'feature_name':array}`的字典(或者`pandas.DataFrame`)
- **labels** ：一个包含每个样本标签的数组
- **batch_size**：指示所需批量大小的整数。

在前面的笔记中，我们使用`iris_data.load_data()`函数加载了鸢尾花的数据。你可以运行下面的代码来获取结果：

```python
import iris_data

# Fetch the data.
train, test = iris_data.load_data()
features, labels = train
```

然后你可以将数据输入到输入函数中，类似这样：

```python
batch_size = 100
iris_data.train_input_fn(features, labels, batch_size)
```

我们来看看这个`train_input_fn`

### 切片(Slices)

在最简单的情况下，`tf.data.Dataset.from_tensor_slices`函数接收一个`array`并返回一个表示`array`切片的`tf.data.Dataset`。例如，mnist训练集的shape是`(60000, 28, 28)`。将这个`array`传递给`from_tensor_slices`将返回一个包含60000个切片的数据集对象，每个切片大小为`28X28`的图像。（其实这个API就是把array的第一维切开）。  

这个例子的代码如下：  

```python
train, test = tf.keras.datasets.mnist.load_data()
mnist_x, mnist_y = train

mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
print(mnist_ds)
```

将产生下面的结果：显示数据集中项目的type和shape。注意，数据集不知道它含有多少个sample。

```python
<TensorSliceDataset shapes: (28,28), types: tf.uint8>
```

上面的数据集代表了简单数组的集合，但`Dataset`的功能还不止如此。`Dataset`能够透明地处理字典或元组的任何嵌套组合。例如，确保`features`是一个标准的字典，你可以将数组字典转换为字典数据集。  

先来回顾下`features`，它是一个`pandas.DataFrame`类型的数据：

| SepalLength | SepalWidth | PetalLength | PetalWidth |
| :---------: | :--------: | :---------: | :--------: |
|     0.6     |    0.8     |     0.9     |     1      |
|     ...     |    ...     |     ...     |    ...     |

而`dict(features)`是一个字典，它的形式如下：

```python
{key:value,key:value...}  # key是string类型的列名，即SepalLength等
			# value是pandas.core.series.Series类型的变量，即数据的一个列，是一个标量
```

对它进行切片

```python
dataset = tf.data.Dataset.from_tensor_slices(dict(features))
print(dataset)
```

结果如下：

```python
<TensorSliceDataset

  shapes: {
    SepalLength: (), PetalWidth: (),
    PetalLength: (), SepalWidth: ()},

  types: {
      SepalLength: tf.float64, PetalWidth: tf.float64,
      PetalLength: tf.float64, SepalWidth: tf.float64}
>
```

这里我们看到，当数据集包含结构化元素时，数据集的形状和类型采用相同的结构。该数据集包含标量字典，所有类型为tf.float64。  

`train_input_fn`的第一行使用了相同的函数，但它增加了一层结构-----创建了一个包含`(feature, labels)`对的数据集  

我们继续回顾`labels`的结构,它其实是一个`pandas.core.series.Series`类型的变量，即它与`dict(features)`的value是同一类型。且维度一致，都是标量，长度也一致。  

以下代码展示了这个`dataset`的形状:  

```python
# Convert the inputs to a Dataset.
dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
print(dataset)
```

```python
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (), PetalWidth: (),
          PetalLength: (), SepalWidth: ()},
        ()),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```

### 操纵

对于目前的数据集，将以固定的顺序遍历数据一次，并且每次只生成一个元素。在它可以被用来训练之前，还需做进一步处理。幸运的是，`tf.data.Dataset`类提供了接口以便于更好地在训练之前准备数据。输入函数的下一行利用了以下几种方法：

```python
# Shuffle, repeat, and batch the examples.
dataset = dataset.shuffle(1000).repeat().batch(batch_size)
```

**shuffle**方法使用一个固定大小的缓冲区来随机对数据进行shuffle。设置大于数据集中sample数目的buffer_size可以确保数据完全混洗。鸢尾花数据集只包含150个数据。  

**repeat**方法在读取到组后的数据时重启数据集。要限制epochs的数量，可以设置`count`参数。  

**batch**方法累计样本并堆叠它们，从而创建批次。这个操作的结果为这批数据的形状增加了一个维度。新维度被添加为第一维度。以下代码是早期使用mnist数据集上的批处理方法。这使得`28x28`的图像堆叠为三维的数据批次。

```python
print(mnist_ds.batch(100))
```

```python
<BatchDataset
  shapes: (?, 28, 28),
  types: tf.uint8>
```

请注意，数据集具有未知的批量大小，因为最后一批的元素数量较少。

在`train_input_fn`中，批处理后，数据集包含一维向量元素，其中每个标量先前都是：

```python
print(dataset)
```

```python
<TensorSliceDataset
    shapes: (
        {
          SepalLength: (?,), PetalWidth: (?,),
          PetalLength: (?,), SepalWidth: (?,)},
        (?,)),

    types: (
        {
          SepalLength: tf.float64, PetalWidth: tf.float64,
          PetalLength: tf.float64, SepalWidth: tf.float64},
        tf.int64)>
```

### 返回值

每个`Estimator`的`train`、`predict`、`evaluate`方法都需要输入函数返回一个包含Tensorflow张量的`(features, label)`对。`train_input_fn`使用以下代码将数据集转换为预期的格式：

```python
# Build the Iterator, and return the read end of the pipeline.
features_result, labels_result = dataset.make_one_shot_iterator().get_next()
```

结果是TensorFlow张量的结构，匹配数据集中的项目层。

```python
print((features_result, labels_result))
```

```python
({
    'SepalLength': <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=float64>,
    'PetalWidth': <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=float64>,
    'PetalLength': <tf.Tensor 'IteratorGetNext:0' shape=(?,) dtype=float64>,
    'SepalWidth': <tf.Tensor 'IteratorGetNext:3' shape=(?,) dtype=float64>},
Tensor("IteratorGetNext_1:4", shape=(?,), dtype=int64))
```



## 读取CSV文件

`Dataset`最常见的实际用例是按流的方式从磁盘上读取文件。`tf.data`模块包含各种文件读取器。让我们来看看如何使用`Dataset`从csv文件中分析鸢尾花数据集。  

以下对`iris_data.maybe_download`函数的调用在需要时会下载数据，并返回下载结果文件的路径名称：  

```python
import iris_data
train_path, test_path = iris_data.maybe_download()
```

`iris_data.csv_input_fn`函数包含使用`Dataset`解析csv文件的替代实现。

### 构建数据集

我们首先构建一个`TextLineDataset`对象，一次读取一行文件。然后，我们调用`skip`方法跳过文件第一行，它包含一个头部，而不是样本：

```python
ds = tf.data.TextLineDataset(train_path).skip(1)
```

### 构建csv行解析器

最终，我们需要解析数据集中的每一行，以产生必要的`(features, label)`对。  

我们将开始构建一个函数来解析单个行。  

下面的`iris_data.parse_line`函数使用`tf.decode_csv`函数和一些简单的代码完成这个任务：

我们必须解析数据集中的每一行以生成必要的`(features, label)`对。以下的`_parse_line`函数调用`tf.decode_csv`将单行解析为其`features`和`label`。由于`Estimator`要求将特征表示为字典，因此我们依靠python的内置字典和`zip`函数来构建该字典。特征名是该字典的key。然后我们调用字典的`pop`方法从特征字典中删除标签字段。

```python
# Metadata describing the text columns
COLUMNS = ['SepalLength', 'SepalWidth',
           'PetalLength', 'PetalWidth',
           'label']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)

    # Pack the result into a dictionary
    features = dict(zip(COLUMNS, fields))

    # Separate the label from the features
    label = features.pop('label')

    return features, label
```

### 解析行

`Datasets`有很多方法用于在数据传输到模型时处理数据。最常用的方法是map，它将转换应用于`Dataset`的每个元素。  

`map`方法使用一个`map_func`参数来描述`Dataset`中每个项目应该如何转换。   ![map.png](https://i.loli.net/2018/03/29/5abd008016e8d.png)

因此为了解析流出csv文件的行，我们将`_parse_line`函数传递给`map`方法：

```python
ds = ds.map(_parse_line)
print(ds)
```

```python
<MapDataset
shapes: (
    {SepalLength: (), PetalWidth: (), ...},
    ()),
types: (
    {SepalLength: tf.float32, PetalWidth: tf.float32, ...},
    tf.int32)>
```

现在的数据集不是简单的标量字符串，而是包含了`(features, label)`对。  

`iris_data.csv_input_fn`函数其余部分与基本输入部分中涵盖的`iris_data.train_input_fn`相同。  

### 试试看

该函数可以用来替代`iris_data.train_input_fn`。它可以用来提供一个如下的`Estimator`：

```python
train_path, test_path = iris_data.maybe_download()

# All the inputs are numeric
feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns,
                                    n_classes = 3)
# Train the estimator
batch_size = 100
est.train(
	steps=1000
	input_fn=lambda:iris_data.csv_input_fn(train_path, batch_size))
```

`Estimator`期望`input_fn`不带任何参数。为了解除这个限制，我们使用`lambda`来捕获参数并提供预期的接口。



## 总结

`tf.data`模块提供了一组用于轻松读取各种来源数据的类和函数。此外，`tf.data`具有简单强大的方法来应用各种标准和自定义转换。  

现在你已经了解如何有效地将数据加载到`Estimator`中的基本想法。接下来考虑以下文档：

- [创建自定义估算器](https://www.tensorflow.org/get_started/custom_estimators)，演示如何构建自己的自定义估算器模型。
- [低层次简介](https://www.tensorflow.org/programmers_guide/low_level_intro#datasets)，演示如何使用TensorFlow的低层API直接实验`tf.data.Datasets`。
- [导入](https://www.tensorflow.org/programmers_guide/datasets)详细了解数据集附加功能的数据。
