![](/extra/img/1_HNjcfB3lccaqB95fSWXJpQ.png)

**Artificial Intelligence** is purely math and scientific exercise but when it becomes computational, it starts to solve human problems.

**Machine Learning** is a subset of Artificial Intelligence. ML is the study of computer algorithms that improve automatically through experience. ML explores the study and construction of algorithms that can learn from data and make predictions on data. Based on more data, machine learning can change actions and responses which will make it more efficient, adaptable, and scalable. [source](https://www.analyticsvidhya.com/blog/2021/03/everything-you-need-to-know-about-machine-learning/)

**Deep Learning** is a technique for implementing machine learning algorithms. It uses **Artificial Neural Networks** for training data to achieve highly promising decision making. The neural network performs micro calculations with computational on many layers and can handle tasks like humans.

## 1. What is Machine Learning?

### 1.1 Machine Learning definition

Machine learning is a discipline of artificial intelligence. The main objective is to create systems that are able to learn automatically, ie they are able to find complex patterns in large sets of data on their own.

Two definitions of Machine Learning are offered.

Arthur Samuel described it as: "**the field of study that gives computers the ability to learn without being explicitly programmed.**" This is an older, informal definition.

Tom Mitchell provides a more modern definition: "**A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E**."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

![](/extra/img/week01-lecture1-01.png)

**Machine learning is the technology that is concerned with teaching computers different algorithms to perform different tasks, and making machines capable of taking care of themselves**. Different ideas are framed and fed to machines. There are mainly three recognized categories of framing ideas, which we reckon as the three types of machine learning. In general, any machine learning problem can be assigned to one of three broad classifications: **Supervised learning and Unsupervised learning and Reinforcement Learning.** [source](https://www.analyticssteps.com/blogs/types-machine-learning) [source](https://www.analyticssteps.com/blogs/what-confusion-matrix)

![](/extra/img/types_ml.jpg)

#### 1.1.1 Introduction Supervised Learning

Supervised learning is a set of techniques that allows future predictions based on behaviors or characteristics analyzed in historical data. In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. Supervised learning problems are categorized into "regression" and "classification" trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are trying to map input variables into discrete categories. [source](https://blogs.nvidia.com/blog/2018/08/02/supervised-unsupervised-learning/)

![](/extra/img/Supervised_machine_learning_in_a_nutshell.svg_.png)

A labeled dataset of animal images would tell the model whether an image is of a dog, a cat, etc.. Using which, a model gets training, and so, whenever a new image comes up to the model, it can compare that image with the labeled dataset for predicting the correct label. [source](https://intellipaat.com/blog/supervised-learning-vs-unsupervised-learning-vs-reinforcement-learning/)

![](/extra/img/2021-12-18_151840.png)

There are two main areas where supervised learning is useful: **classification problems** and **regression problems**.

![](/extra/img/week1_04.png) 

![](/extra/img/2021-12-18_130447.png) 

[source](https://www.congrelate.com/18-machine-learning-algorithm-for-classification-gif/)

**classification problems**

Classification problems ask the algorithm to predict a discrete value, identifying the input data as the member of a particular class, or group. In a training dataset of animal images, that would mean each photo was pre-labeled as cat, koala or turtle. The algorithm is then evaluated by how accurately it can correctly classify new images of other koalas and turtles.

![](/extra/img/week1_02.png)

[source](https://developers.google.com/machine-learning/guides/text-classification/?hl=it-CH)

**regression problems**

A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”. Many different models can be used, the simplest is the linear regression. It tries to fit data with the best hyper-plane which goes through the points.

![](/extra/img/week01-lecture2-01.png)

**Linear Regression vs Logistic Regression** [source](https://www.javatpoint.com/linear-regression-vs-logistic-regression-in-machine-learning)

Linear Regression and Logistic Regression are the two famous Machine Learning Algorithms which come under supervised learning technique. Since both the algorithms are of supervised in nature hence these algorithms use labeled dataset to make the predictions. But the main difference between them is how they are being used. The Linear Regression is used for solving Regression problems whereas Logistic Regression is used for solving the Classification problems.

![](/extra/img/linear-regression-vs-logistic-regression.png)

**Linear Regression:**

- Linear Regression is one of the most simple Machine learning algorithm that comes under Supervised Learning technique and used for solving regression problems.
- It is used for predicting the continuous dependent variable with the help of independent variables.
- The goal of the Linear regression is to find the best fit line that can accurately predict the output for the continuous dependent variable.
- If single independent variable is used for prediction then it is called Simple Linear Regression and if there are more than two independent variables then such regression is called as Multiple Linear Regression.
- By finding the best fit line, algorithm establish the relationship between dependent variable and independent variable. And the relationship should be of linear nature.
- The output for Linear regression should only be the continuous values such as price, age, salary, etc.

**Logistic Regression:** 

- Logistic regression is one of the most popular Machine learning algorithm that comes under Supervised Learning techniques.
- It can be used for Classification as well as for Regression problems, but mainly used for Classification problems.
- Logistic regression is used to predict the categorical dependent variable with the help of independent variables.
- The output of Logistic Regression problem can be only between the 0 and 1.
- Logistic regression can be used where the probabilities between two classes is required. Such as whether it will rain today or not, either 0 or 1, true or false etc.
- Logistic regression is based on the concept of Maximum Likelihood estimation. According to this estimation, the observed data should be most probable.
- In logistic regression, we pass the weighted sum of inputs through an activation function that can map values in between 0 and 1. Such activation function is known as **sigmoid function** and the curve obtained is called as sigmoid curve or S-curve. 

| Linear Regression                                            | Logistic Regression                                          |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| Linear regression is used to predict the continuous dependent variable using a given set of independent variables. | Logistic Regression is used to predict the categorical dependent variable using a given set of independent variables. |
| Linear Regression is used for solving Regression problem.    | Logistic regression is used for solving Classification problems. |
| In Linear regression, we predict the value of continuous variables. | In logistic Regression, we predict the values of categorical variables. |
| In linear regression, we find the best fit line, by which we can easily predict the output. | In Logistic Regression, we find the S-curve by which we can classify the samples. |
| Least square estimation method is used for estimation of accuracy. | Maximum likelihood estimation method is used for estimation of accuracy. |
| The output for Linear Regression must be a continuous value, such as price, age, etc. | The output of Logistic Regression must be a Categorical value such as 0 or 1, Yes or No, etc. |
| In Linear regression, it is required that relationship between dependent variable and independent variable must be linear. | In Logistic regression, it is not required to have the linear relationship between the dependent and independent variable. |
| In linear regression, there may be collinearity between the independent variables. | In logistic regression, there should not be collinearity between the independent variable. |

##### 1.1.1.1 **Linear Regression with One Variable**

Recall that in regression problems, we are taking input variables and trying to result function. Linear regression with one variable is also known as "univariate linear regression." Univariate linear regression is used when you want to predict a single output value y from a single input value x. We're doing supervised learning here, so that means we already have an idea about what the input/output cause and effect should be.

![](/extra/img/week01-lecture2-01.png)

![](/extra/img/week01-lecture2-02.png)

**The Hypothesis Function**

![](/extra/img/week01-lecture2-03.png)

Our hypothesis function has the general form: 

<img src="https://latex.codecogs.com/gif.latex?\hat{y}=h_\theta%20\left(%20x%20\right)=\theta_{0}%20+%20\theta_{1}x">

Note that this is like the equation of a straight line. We give to  <img src="https://latex.codecogs.com/gif.latex?h_\theta%20\left(%20x%20\right)">  values for  <img src="https://latex.codecogs.com/gif.latex?\theta_{0}%20"> and  <img src="https://latex.codecogs.com/gif.latex?\theta_{1}%20"> to get our estimated output <img src="https://latex.codecogs.com/gif.latex?\hat{y}"> .In other words, we are trying to create a function called that is trying to map our input data to our output data.

**Cost Function**

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average (actually a fancier version of an average) of all the results of the hypothesis with inputs from x compared to the actual output y.

<img src="https://latex.codecogs.com/gif.latex?J%20\left(%20\theta_0,%20\theta_1%20\right)%20=%20\frac{1}{2m}\sum\limits_{i=1}^m%20\left(%20\hat{y}^{(i)}%20-y^{(i)}%20\right)^{2}=%20\frac{1}{2m}\sum\limits_{i=1}^m%20\left(%20h_{\theta}(x^{(i)})-y^{(i)}%20\right)^{2}">

This function is otherwise called the "Squared error function", or "Mean squared error". Now we are able to concretely measure the accuracy of our predictor function against the correct results we have so that we can predict new results we don't have. If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make straight line  which passes through this scattered set of data. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. In the best case, the line should pass through all the points of our training data set. In such a case the value of  <img src="https://latex.codecogs.com/gif.latex?J%20\left(%20\theta_0,%20\theta_1%20\right)%20">  will be 0.

1.1.1.2 **Logistic Regression**

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, **logistic regression** is estimating the parameters of a logistic model (a form of binary regression).

![](/extra/img/0__z9yOXB3ukbTvyV1.png)

**The Hypothesis Function**

**Cost Function**

The cost function represents optimization objective i.e. we create a cost function and minimize it so that we can develop an accurate model with minimum error.

#### 1.1.2 Introduction Unsupervised Learning

Clean, perfectly labeled datasets aren’t easy to come by. And sometimes, researchers are asking the algorithm questions they don’t know the answer to. That’s where unsupervised learning comes in. The training dataset is a collection of examples without a specific desired outcome or correct answer. 

This type of machine learning algorithm learns from a dataset without any labels. The algorithm can automatically classify or categorize the input data. The application of unsupervised learning mainly includes cluster analysis, association rule or dimensionality reduce. [source](https://www.ecloudvalley.com/mlintroduction/) [source](https://intellipaat.com/blog/supervised-learning-vs-unsupervised-learning-vs-reinforcement-learning/)

![](/extra/img/2021-12-18_145009.png)

#### 1.1.3 Introduction Reinforcement Learning

After discussing on supervised and unsupervised learning models, now, let me explain to you reinforcement learning. As it is based on neither supervised learning nor unsupervised learning, what is it? To be straight forward, in reinforcement learning, algorithms learn to react to an environment on their own.

To be a little more specific, reinforcement learning is a type of learning that is based on interaction with the environment. It is rapidly growing, along with producing a huge variety of learning algorithms that can be used for various applications.

To begin with, there is always a start and an end state for an agent (the AI-driven system); however, there might be different paths for reaching the end state, like a maze. This is the scenario wherein reinforcement learning is able to find a solution for a problem. Examples of reinforcement learning include self-navigating vacuum cleaners, driverless cars, scheduling of elevators, etc. [source](https://intellipaat.com/blog/supervised-learning-vs-unsupervised-learning-vs-reinforcement-learning/)

![](/extra/img/2021-12-18_152435.png)



### Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it estimate the parameters in hypothesis function. That's where gradient descent comes in.







https://blogs.nvidia.com/blog/2018/08/02/supervised-unsupervised-learning/

https://developers.google.com/machine-learning/guides/text-classification/?hl=it-CH

https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/

https://www.congrelate.com/18-machine-learning-algorithm-for-classification-gif/

https://vinodsblog.com/

https://www.diegocalvo.es/en/machine-learning-supervised-unsupervised/

https://www.mathworks.com/help/stats/machine-learning-in-matlab.html

https://towardsdatascience.com/a-beginners-guide-to-regression-analysis-in-machine-learning-8a828b491bbf

https://www.oatext.com/a-supervised-machine-learning-approach-to-generate-the-auto-rule-for-clinical-decision-support-system.php

https://www.v7labs.com/blog/supervised-vs-unsupervised-learning

https://www.analyticsvidhya.com/blog/2021/03/everything-you-need-to-know-about-machine-learning/

https://intellipaat.com/blog/supervised-learning-vs-unsupervised-learning-vs-reinforcement-learning/

https://search.slidegeeks.com/search?af=cat1%3apowerpointtemplates&lbc=slidegeeks&method=and&p=Q&ts=custom&w=roadmap&cnt=200

https://intellipaat.com/blog/tutorial/machine-learning-tutorial/

https://www.analyticssteps.com/blogs/introduction-machine-learning-supervised-and-unsupervised-learning

https://vinodsblog.com/2018/11/28/machine-learningml-introduction-to-basics/

https://vinodsblog.com/2018/10/15/everything-you-need-to-know-about-convolutional-neural-networks/

https://vinodsblog.com/2021/01/01/2021-the-year-of-transformer-deep-learning/

https://vinodsblog.com/2020/07/28/deep-learning-introduction-to-boltzmann-machines/

https://vinodsblog.com/2018/11/28/machine-learningml-introduction-to-basics/

https://vinodsblog.com/2018/11/28/machine-learningml-introduction-to-basics/

https://vinodsblog.com/2019/03/31/supervised-vs-unsupervised-machine-learning/

https://vinodsblog.com/2018/03/26/machine-learning-introduction-to-its-algorithms-mlalgos/

https://vinodsblog.com/2018/03/11/the-exciting-evolution-of-machine-learning/

https://vinodsblog.com/2018/03/11/the-exciting-evolution-of-machine-learning/
