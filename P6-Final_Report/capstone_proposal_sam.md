# 机器学习算法工程师纳米学位

## 毕业项目

陈正和 Sam Chen
May 23, 2019

## 猫狗大战项目

### 背景知识 Domain Background 
本项目属于计算机视觉领域的范畴。早在1960年代，计算机视觉正式成为一门学科领域，当时的目标是自动化图像分析的过程：用计算机模拟人类的视觉系统，让计算机告诉我们它看到了什么。[1]

到了2010年，得益于深度学习领域的发展与突破，我们图像识别被推到了全新的高度。其中不得不提到的是大型开源项目： [ImageNet](http://www.image-net.org/about-overview) 数据库。ImageNet数据库，包含超过1千万张手动标记的图片，这些图片来自于1000个不同的图片类别。(Udacity,算法工程师纳米学位课程)2010年开始，ImageNet每年都会举办大规模图像识别大赛(ImageNet Large Scale Visual Recognition Competition)。在这个比赛中，不同的团队会协力创造准确率最高的图像识别CNN模型。(Udacity,算法工程师纳米学位课程)

其中的重大突破，包含2012年多伦多大学提出的AlexNet架构；2014年牛津大学的VGG Net 架构；2015年微软研究部门的ResNet[3]架构等。(Udacity,算法工程师纳米学位课程)从每年胜出者的架构当中，我们可以学习到很多设计CNN架构的技巧，例如：使用了ReLU激活函数、Dropout等技巧，避免过拟合的问题；增加横跨多层的线路，解决过深的架构梯度消失的问题。(Udacity,算法工程师纳米学位课程)

### 问题描述 Problem Statement

本项目源自于Kaggle数据竞赛平台上，[猫狗大战图像识别项目](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview)，属于**监督学习二分类问题**。

通过25,000张包含猫或者狗的训练集图片，训练图像识别模型，识别图片中的主体是猫，还是狗。(若模型识别出图片中的主体是猫，则概率为0；若模型识别出图片中的主体是狗，则概率为1。) 一个潜在的解决方案，是以ImageNet的获胜者模型，例：ResNet[3]等，通过**迁移学习**的方式，预测图片中的主题为猫或者狗。

### 数据集 Datasets and Inputs

本项目的数据集由Kaggle提供，其中包含25,000张训练集数据；12,500张测试集数据。数据集可以在[这个地址](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)获取。

#### 基础信息探索
通过这段代码，我统计了训练集数据的信息
```
from glob import glob
import seaborn as sns

train_list = [i for i in glob("./input/train/*")]
count = []
for i in train_list:
    count.append(i.split('/')[-1].split('.')[0])
sns.countplot(count);
```
![train_stat.png](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/train_stat-1.png)

从上图可以发现，此数据包含12,500张小猫的照片、12,500张狗狗的照片。

#### 可视化探索
通过一个简单的`getRandomNum`函数，我探索了训练集中的部分图片：
```
import random
from IPython.display import Image
import glob

train_path = glob.glob("./input/train/*")
test_path = glob.glob("./input/test/*")

def getRandomNum(maxNum):
    rand_num = random.randint(0, maxNum)
    print(rand_num)
    return rand_num

for i in range(20000):
    if i >= 12000:
        rand_num = getRandomNum(24999)
        print(train_path[rand_num])
        display(Image(filename=train_path[rand_num]))
```

经过对图片的探索，发现以下几个特征或问题：

1. 图片大小格式差异较大
    - 部分图片过小，可能影响模型的训练(dog.9705.jpg, cat.8504.jpg, dog.7011.jpg, cat.4821.jpg, cat.8585.jpg, dog.11686.jpg)

    - ![dog.7011.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.7011.jpg)
    - ![cat.4821.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.4821.jpg)
    - ![dog.9705.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.9705.jpg)

2. 部分图片为动物侧边照，或者十分模糊(dog.7561.jpg, dog.9687.jpg, dog.10939.jpg, cat.1772.jpg, dog.7164.jpg,cat.241.jpg, cat.8475.jpg, dog.5906.jpg, cat.5324.jpg, dog.8607.jpg, cat.7535.jpg, cat.4042.jpg, dog.10622.jpg, dog.12259.jpg, dog.3430.jpg, dog.516.jpg, dog.3843.jpg, cat.241.jpg, cat.9626.jpg, cat.11337.jpg)

    - ![dog.7561.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.7561.jpg)
    - ![cat.241.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.241.jpg)
    - ![dog.10622.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.10622.jpg)

3. 部分图片存在文字(dog.12199.jpg, cat.6190.jpg, cat.8296.jpg, cat.2834.jpg, cat.11564.jpg, dog.8209.jpg)

    - ![dog.12199.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.12199.jpg)
    - ![cat.2834.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.2834.jpg)
    - ![cat.11564.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.11564.jpg)

4. 部分图片包含人类(dog.3690.jpg, , dog.4313.jpg, cat.9382.jpg, dog.10307.jpg, dog.6113.jpg,dog.2262.jpg, dog.11935.jpg, dog.3488.jpg, dog.1674.jpg, cat.4426.jpg, dog.11923.jpg,dog.8128.jpg, cat.11125.jpg, dog.7794.jpg, cat.6.jpg, dog.11935.jpg)

    - ![dog.3690.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.3690.jpg)
    - ![cat.11125.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.11125.jpg)
    - ![dog.11935.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.11935.jpg)

5. 部分图片包含物品(cat.11148.jpg, dog.7644.jpg, dog.9047.jpg, cat.7073.jpg)

    - ![cat.7073.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.7073.jpg)
    - ![dog.7644.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.7644.jpg)
    - ![cat.11148.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.11148.jpg)

6. 部分图片中的动物，在笼子里面(cat.7738.jpg, cat.5834.jpg, dog.3100.jpg, dog.5085.jpg, dog.7760.jpg, dog.10353.jpg, dog.6910.jpg, dog.3972.jpg, dog.8627.jpg, dog.1311.jpg, dog.638.jpg)

    - ![cat.7738.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.7738.jpg)
    - ![dog.6910.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.6910.jpg)
    - ![dog.8627.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.8627.jpg)

7. 有些有多只动物(cat.10127.jpg, dog.7914.jpg, cat.2042.jpg, cat.6908.jpg,dog.6487.jpg,cat.678.jpg, dog.8610.jpg, cat.4394.jpg)

    - ![cat.4394.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.4394.jpg)
    - ![dog.8610.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/dog.8610.jpg)
    - ![cat.678.jpg](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/cat.678.jpg)

在实际执行项目的过程中，将会跟据上述包含的问题进行处理。

### 解决方案 Solution Statement

这个问题，可以通过迁移学习解决。以ResNet[3]为基底，通过迁移学习的方式，去掉最后一层，并在其后加上池化层、Dense层等。迁移学习搭建的模型，可以识别图片中的主题为某个狗/猫品种的概率。在迁移学习的模型之后，加上一个函数，若识别出的前 n 个结果有 x 个为狗的品种之一，则这个图片为狗的概率为 x/n 。

### 模型标杆 Benchmark Model

这个项目中，在原始有效的比赛时间里，共有1314个提交。以Leaderboard前10%做为标杆，Log Loss值需要达到 0.06127以下。

### 评估指标 Evaluation Metrics

Kaggle的原始比赛中，提供了这个项目的评估指标 -- log loss值:

LogLoss=−1n∑i=1n[yilog(ŷi)+(1−yi)log(1−ŷ i)],

![logloss.png](https://raw.githubusercontent.com/zhsam/Udacity-MachineLearningEngineer-nd/master/P6-Final_Report/img/logloss.png)

其中：
- n 表示测试集中的图片数量
- ŷi 表示图片被预测为狗狗的概率
- yi 表示实际的情况，此值为1代表这是一张狗的图片；此值为0代表这是一张猫的图片
- log() 表示以e为底的自然对数

### 项目设计 Project Design

1. **数据预处理**
    1. 数据探索：
        - 基础信息探索：探索数据集中图片的数量等，数据的基本信息。
        - 可视化探索：随机检索某些图片，了解图片数据的分布情况。
    2. 数据异常处理：发现图片潜在的问题，针对异常的图片进行处理。
    3. 数据增强：
        - 对图片进行不同的操作(例：旋转、翻转、平移)等，增加模型的泛化能力。
    4. 数据归一化：将数据resize到224x224。[4]
    ```
    import cv2
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    ```

2. **模型搭建**
    1. 模型架构：使用ResNet[3] 进行迁移学习。
    ```
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
    model.add(Dense(2, activation='relu'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    ```   
    2. 将切分20%的训练集，做完验证集
    ```
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2018)
    ```
    3. 训练模型，并使用`checkpointer`记录最好的超参数组合
    ```
    checkpointer = ModelCheckpoint(filepath='DogCatsResnet50.weights.best.hdf5', verbose=1,
                              save_best_only=True)
    model.fit(X_train, y_train,
            batch_size=64,
            epochs=20,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[checkpointer],
            shuffle=True)
    ```
3. **测试模型、准备提交**
    1. 输出 submission.csv。
    2. 提交到Kaggle，持续优化模型(更改dropout值、进行更多数据增强、尝试不同的模型进行迁移学习等方法)，直到log loss值达到leader borad前10%。

-----------

### 文章引用
- [1] [Amir Jirbandey, A brief history of Computer Vision and AI Image Recognition, 2018](https://www.pulsarplatform.com/blog/2018/brief-history-computer-vision-vertical-ai-image-recognition/)
- [2] [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)
- [3] [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015) Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)
- [4] [Is there any particular reason why people pick 224x224 image size for imagenet experiments?](https://stackoverflow.com/questions/43434418/is-there-any-particular-reason-why-people-pick-224x224-image-size-for-imagenet-e)