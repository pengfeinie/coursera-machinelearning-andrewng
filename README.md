## What is Machine Learning?

Two definitions of Machine Learning are offered.

Arthur Samuel described it as: "**the field of study that gives computers the ability to learn without being explicitly programmed.**" This is an older, informal definition.

Tom Mitchell provides a more modern definition: "**A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E**."

Example: playing checkers.

E = the experience of playing many games of checkers

T = the task of playing checkers.

P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:

Supervised learning and Unsupervised learning.

## Week 2

### 2.1 Multiple Features

We will start to talk about a new version of linear regression that's more powerful. One that works with multiple variables or with multiple features. Here's what I mean. In the original version of linear regression that we developed, we have a single feature x, the size of the house, and we wanted to use that to predict why the price of the house and this was our form of our hypothesis.

![](https://pengfeinie.github.io/images/44c68412e65e62686a96ad16f278571f.png)

But now imagine, what if we had not only the size of the house as a feature or as a variable of which to try to predict the price, but that we also knew the number of bedrooms, the number of house and the age of the home and years. 
$$
(x_1,x_2,x_3,......x_n)
$$
It seems like this would give us a lot more information with which to predict the price.

![](https://pengfeinie.github.io/images/591785837c95bca369021efa14a8bb1c.png)

To introduce a little bit of notation, we sort of started to talk about this earlier, in this case, four features and I'm going to continue to use Y to denote the variable, the output variable price that we're trying to predict. 

Let's introduce a little bit more notation. Now that we have four features, I'm going to use lowercase "n" to denote the number of features. So in this example , you know, one, two, three, four features. And "n" is different from our earlier notation where we were using "n" to denote the number of examples. So if you have 47 rows "M" is the number of rows on this table or the number of training examples. So I'm also going to use X superscript "i" to denote the input features of the "i" training example.

![image-20211206181321485](https://pengfeinie.github.io/images/image-20211206181321485.png)


$$
{x}^{(2)}\text{=}\begin{bmatrix} 1416\\\ 3\\\ 2\\\ 40 \end{bmatrix}
$$

$$
x_{2}^{\left( 2 \right)}=3,x_{3}^{\left( 2 \right)}=2
$$

$$
h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}
$$

$$
h_{\theta} \left( x \right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}
$$





http://www.ai-start.com/ml2014/html/week2.html