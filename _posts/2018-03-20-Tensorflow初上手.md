# Tensorflow初上手

本文翻译自[www.tensorflow.org](www.tensorflow.org)的英文教程。

本文档介绍了TensorFlow编程环境，并向您展示了如何使用Tensorflow解决鸢尾花分类问题。



## 先决条件

在本文档中使用示例代码之前，您需要执行以下操作：

- 确认安装了Tensorflow

- 如果在Anaconda的虚拟环境下安装了TF，激活你的TF环境

- 通过以下命令安装或者升级`pandas`

  ```shell
  pip install pandas
  ```

  ​

## 获取示例代码

按照以下步骤获取我们将要全程使用的示例代码

1. 通过输入以下命令从GitHub克隆TensorFlow Models仓库：

   ```shell
   git clone https://github.com/tensorflow/models
   ```

2. 进入本文档所使用的示例所在的目录

   ```shell
   cd models/samples/core/get_started/
   ```

本文档中介绍的程序是`premade_estimator.py`。该程序使用`iris_data.py`获取训练数据。

### 运行程序

像运行Python代码一样运行TF程序。例如：

```shell
python premade_estimator.py
```

程序应该在对测试集进行预测之后输出一些训练日志。例如，以下输出中的第一行显示该模型认为测试集中的第一个示例是`Setosa`的可能性为`99.6％`。由于测试集中该示例正是`Setosa`，所以这似乎是一个很不错的预测。

```
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

如果程序生成错误而没有输出答案，请自答以下问题：

- 安装Tensorflow是否正确
- 你的Tensorflow版本是否正确
- 你是否激活了Tensorflow环境



## 编程堆栈

在深入了解程序本身的细节之前，让我们来看看编程环境。如下图所示，TensorFlow提供了一个由多个API层组成的编程堆栈：![tensorflow_programming_environment.png](https://i.loli.net/2018/03/20/5ab0f64e78d47.png)

我们强烈建议使用以下API编写TensorFlow程序：

- **评估器Estimators**，代表一个完整的模型。 Estimator API提供方法来训练模型，判断模型的准确性并生成预测。

- **训练集Datasets，**它构建了一个数据输入管道。Datasets API具有加载和操作数据的方法，并将其输入到你的模型中。Datasets API与Estimators API良好地协作。、

  ​

## 分类鸢尾花：概述

本文档中的示例程序构建并测试了一个模型，该模型根据萼片和花瓣的大小将鸢尾花分为三种不同的物种。

![iris_three_species.jpg](https://i.loli.net/2018/03/20/5ab0f7e24e948.jpg)

从左到右，分别是***Iris setosa***（由Radomil，CC BY-SA 3.0），***Iris versicolor***（Dlanglois，CC BY-SA 3.0）和***Iris virginica***（by Frank Mayfield，CC BY-SA 2.0）。

### 数据集

鸢尾花数据集包含四个特征和一个标签。这四个特征确定了单个鸢尾花的以下植物学特征：

- 萼片长度
- 萼片宽度
- 花瓣长度
- 花瓣宽度

我们的模型将这些特征表示为`float32`数值数据。 

标签用来标识鸢尾所属种类，它必须是下列3者之一：

- Iris setosa
- Iris versicolor
- Iris virginica

我们的模型将标签作为`int32`分类数据。 

下表显示了数据集中的三个示例：

| 萼片长度 | 萼片宽度 | 花瓣长度 | 花瓣宽度 | 种类（标签）    |
| :------: | :------: | :------: | :------: | :-------------- |
|   5.1    |   3.3    |   1.7    |   0.5    | 0（Setosa）     |
|   5.0    |   2.3    |   3.3    |   1.0    | 1（Versicolor） |
|   6.4    |   2.8    |   5.6    |   2.2    | 2（virginica）  |

### 算法

该程序训练具有以下拓扑的深度神经网络分类器模型：

- 两个隐藏层
- 每个隐藏层10个节点

下图说明了特征，隐藏层和预测（并未显示隐藏层中的所有节点）：![full_network.png](https://i.loli.net/2018/03/20/5ab0fc68aeca3.png)

### 推论

在未标记的例子上运行训练模型会得出三个预测结果，即这朵花是给定鸢尾花种类的可能性。这些输出预测的总和将是1.0。例如，对未标记示例的预测可能如下所示：

- Iris Setosa：0.03
- Iris Versicolor：0.95
- Iris Virginica：0.02

前面的预测表明给定的未标记示例是**Iras Versicolor**的概率为95％。



## 评估器(Estimator)编程概述

`Estimator`是TensorFlow对完整模型的高级表示。它处理初始化，记录，保存和恢复以及许多其他功能的细节，以便你可以专注于您的模型。欲了解更多详情，请参[Estimator](https://www.tensorflow.org/programmers_guide/estimators)。

`Estimator`是从`tf.estimator.Estimator`派生的通用类。 TensorFlow提供了一系列定制的评估器（例如`LinearRegressor`）来实现常用的ML算法。除此之外，你也可以编写自己的定制评估器。我们建议在刚开始使用TensorFlow时使用内置的`Estimator`。在获得内置的`Estimator`的专业知识后，我们建议通过创建你自己的定制`Estimator`来优化你的模型。

要根据内置的`Estimator`编写TensorFlow程序，你必须执行以下任务：

- 创建一个或多个输入函数。
- 定义模型的特征列。
- 实例化`Estimator`，指定特征列和各种超参数。
- 在`Estimator`对象上调用一个或多个方法，传递适当的输入函数作为数据源。

让我们看看这些任务是如何实施鸢尾花分类的。



## 创建输入函数(input function)

你必须创建输入函数来为训练，评估和预测提供数据。

输入函数(input function)是一个返回一个`tf.data.Dataset`对象的函数，该对象输出一个两元素元组(tuple)：

- **features** - 一个Python字典
  - 字典的key是特征名
  - 字典的值是特征数组
- **label**-包含所有样例的标签数组

为了演示输入函数的格式，下面是一个简单的实现：

```python
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels
```

你的`input function`可能会以你喜欢的任何方式生成功能字典和标签列表。不过，我们建议使用TensorFlow的数据集API，它可以解析各种数据。在较高层次上，数据集API由以下类组成：![dataset_classes.png](https://i.loli.net/2018/03/20/5ab1018e4c6d8.png)

这里每个独立成员如下：

- **Dataset**：包含创建和转换数据集的方法的基类。还允许你从内存中的数据或Python生成器中初始化数据集。
- **TextLineDataset**：从文本文件读取行
- **TFRecordDataset**：从TFRecord文件中读取记录。
- **FixedLengthRecordDataset**：从二进制文件中读取固定大小的记录。
- **Iterator**：提供一次访问一个数据集元素的方法。

Dataset API可以为你处理很多常见情况。例如，使用Dataset API，你可以轻松地从大量文件集中并行读入记录，并将它们合并到一个流中。

在这个例子中，简单起见，我们将使用`pandas`加载数据，并从这些内存数据构建输入管道。

以下是用于在该程序中进行训练的`input function`，该函数在`iris_data.py`中可用：

```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)
```



## 定义特征列

特征列(feature column)是描述模型如何使用特征字典中的原始输入数据的对象。在构建`Estimator`模型时，你会传递一个特征列列表来描述你希望模型使用的每个特征。 `tf.feature_column`模块提供了很多用于向模型表示数据的选项。

对于鸢尾花，4个原始特征是数值类型，因此我们将构建一个特征列列表，以告诉`Estimator`模型将四个特征中的每一个都表示为32位浮点值。因此，创建特征列的代码是：

```python
# Feature columns describe how to use the input.
my_feature_columns=[]
for key in train_x,keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

特征列可能比我们在这里展示的要复杂得多。我们将在“入门指南”中稍后详细介绍特征列。

现在我们已经描述了我们希望模型如何表示原始特征，我们可以构建`Estimator`。

