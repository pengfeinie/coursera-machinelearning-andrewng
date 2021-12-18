## Lecture 1

### What is Machine Learning?

#### Machine Learning definition

Two definitions of Machine Learning are offered.

Arthur Samuel described it as: "**the field of study that gives computers the ability to learn without being explicitly programmed.**" This is an older, informal definition.

Tom Mitchell provides a more modern definition: "**A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E**."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

![](/extra/img/week01-lecture1-01.png)

In general, any machine learning problem can be assigned to one of two broad classifications: 

**Supervised learning and Unsupervised learning.**



#### Introduction Supervised Learning

![](/extra/img/Supervised_machine_learning_in_a_nutshell.svg_.png)

There are two main areas where supervised learning is useful: classification problems and regression problems.

Classification problems ask the algorithm to predict a discrete value, identifying the input data as the member of a particular class, or group. In a training dataset of animal images, that would mean each photo was pre-labeled as cat, koala or turtle. The algorithm is then evaluated by how accurately it can correctly classify new images of other koalas and turtles.

![](/extra/img/week1_02.png)

A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”. Many different models can be used, the simplest is the linear regression. It tries to fit data with the best hyper-plane which goes through the points.

![](/extra/img/week01-lecture2-01.png)

### Linear Regression with One Variable - Lecture 2

Recall that in regression problems, we are taking input variables and trying to result function. Linear regression with one variable is also known as "univariate linear regression." Univariate linear regression is used when you want to predict a single output value y from a single input value x. We're doing supervised learning here, so that means we already have an idea about what the input/output cause and effect should be.

![](/extra/img/week01-lecture2-01.png)

![](/extra/img/week01-lecture2-02.png)

### The Hypothesis Function

![](/extra/img/week01-lecture2-03.png)

Our hypothesis function has the general form: 

<img src="https://latex.codecogs.com/gif.latex?\hat{y}=h_\theta%20\left(%20x%20\right)=\theta_{0}%20+%20\theta_{1}x">

Note that this is like the equation of a straight line. We give to  <img src="https://latex.codecogs.com/gif.latex?h_\theta%20\left(%20x%20\right)">  values for  <img src="https://latex.codecogs.com/gif.latex?\theta_{0}%20"> and  <img src="https://latex.codecogs.com/gif.latex?\theta_{1}%20"> to get our estimated output <img src="https://latex.codecogs.com/gif.latex?\hat{y}"> .In other words, we are trying to create a function called that is trying to map our input data to our output data.

### Cost Function

We can measure the accuracy of our hypothesis function by using a cost function. This takes an average (actually a fancier version of an average) of all the results of the hypothesis with inputs from x compared to the actual output y.

<img src="https://latex.codecogs.com/gif.latex?J%20\left(%20\theta_0,%20\theta_1%20\right)%20=%20\frac{1}{2m}\sum\limits_{i=1}^m%20\left(%20\hat{y}^{(i)}%20-y^{(i)}%20\right)^{2}=%20\frac{1}{2m}\sum\limits_{i=1}^m%20\left(%20h_{\theta}(x^{(i)})-y^{(i)}%20\right)^{2}">

This function is otherwise called the "Squared error function", or "Mean squared error". Now we are able to concretely measure the accuracy of our predictor function against the correct results we have so that we can predict new results we don't have. If we try to think of it in visual terms, our training data set is scattered on the x-y plane. We are trying to make straight line  which passes through this scattered set of data. Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. In the best case, the line should pass through all the points of our training data set. In such a case the value of  <img src="https://latex.codecogs.com/gif.latex?J%20\left(%20\theta_0,%20\theta_1%20\right)%20">  will be 0.

### Gradient Descent

So we have our hypothesis function and we have a way of measuring how well it estimate the parameters in hypothesis function. That's where gradient descent comes in.







https://blogs.nvidia.com/blog/2018/08/02/supervised-unsupervised-learning/

https://developers.google.com/machine-learning/guides/text-classification/?hl=it-CH

https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/
