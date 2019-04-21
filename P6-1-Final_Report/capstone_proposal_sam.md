# 机器学习算法工程师纳米学位
## 毕业项目
陈正和 Sam Chen
April 20st, 2019

## 猫狗大战项目

### 背景知识 Domain Background 

本项目属于计算机视觉领域的范畴。早在1960年代，计算机视觉正式成为一门学科领域，当时的目标是自动化图像分析的过程：用计算机模拟人类的视觉系统，让计算机告诉我们它看到了什么。(Amir Jirbandey, A brief history of Computer Vision and AI Image Recognition, 2018) 

到了2010年，得益于深度学习领域的发展与突破，我们图像识别被推到了全新的高度。其中不得不提到的是大型开源项目： [ImageNet](http://www.image-net.org/about-overview) 数据库。ImageNet数据库，包含超过1千万张手动标记的图片，这些图片来自于1000个不同的图片类别。(Udacity,算法工程师纳米学位课程)2010年开始，ImageNet每年都会举办大规模图像识别大赛(ImageNet Large Scale Visual Recognition Competition)。在这个比赛中，不同的团队会协力创造准确率最高的图像识别CNN模型。(Udacity,算法工程师纳米学位课程)

其中的重大突破，包含2012年多伦多大学提出的AlexNet架构；2014年牛津大学的VGG Net 架构；2015年微软研究部门的ResNet架构等。(Udacity,算法工程师纳米学位课程)从每年胜出者的架构当中，我们可以学习到很多设计CNN架构的技巧，例如：使用了ReLU激活函数、Dropout等技巧，避免过拟合的问题；增加横跨多层的线路，解决过深的架构梯度消失的问题。(Udacity,算法工程师纳米学位课程)


### 问题描述 Problem Statement

本项目源自于Kaggle数据竞赛平台上，[猫狗大战图像识别项目](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview)。项目包含了37,500张猫或者狗的图片，欲通过训练一个深度学习图像识别的模型，识别图片中的主体是猫，还是狗。若模型识别出图片中的主体是猫，则标记为0；若模型识别出图片中的主体是狗，则标记为1。一个潜在的解决方案，是以ImageNet的获胜者模型(VGG, ResNet等)为基底，通过迁移学习的方式，最后加上池化层、Dense层。迁移学习搭建的模型，可以识别狗的品种、猫的品种，最后搭建一个函数，若识别的结果为狗的品种之一，则输出为1；若为猫的品种之一，则输出为0，如此即可建立猫狗图像识别的模型。


### 数据集 Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.


### 解决方案 Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).


### 模型标杆 Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### 评估指标 Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).


### 项目设计 Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

### 文章引用
- [Amir Jirbandey, A brief history of Computer Vision and AI Image Recognition, 2018](https://www.pulsarplatform.com/blog/2018/brief-history-computer-vision-vertical-ai-image-recognition/)


**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
