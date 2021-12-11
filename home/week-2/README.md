## Week 2

### 2.1 Multiple Features

We will start to talk about a new version of linear regression that's more powerful. One that works with multiple variables or with multiple features. Here's what I mean. In the original version of linear regression that we developed, we have a single feature x, the size of the house, and we wanted to use that to predict why the price of the house and this was our form of our hypothesis.

![](/extra/img/week2_01.png)

But now imagine, what if we had not only the size of the house as a feature or as a variable of which to try to predict the price, but that we also knew the number of bedrooms, the number of house and the age of the home and years.  <img src="https://latex.codecogs.com/gif.latex?(x_1,x_2,x_3,......x_n)">

It seems like this would give us a lot more information with which to predict the price.

![](/extra/img/week2_02.png)

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

![](/extra/img/week2_03.png)

To introduce a little bit of notation, we sort of started to talk about this earlier, in this case, four features and I'm going to continue to use Y to denote the variable, the output variable price that we're trying to predict.

 <img src="https://latex.codecogs.com/gif.latex?{x}^{(2)}\text{=}\begin{bmatrix}%201416\\\%203\\\%202\\\%2040%20\end{bmatrix}"> <img src="https://latex.codecogs.com/gif.latex?x_{2}^{\left(%202%20\right)}=3,x_{3}^{\left(%202%20\right)}=2">





Now define the multivariable form of the hypothesis function as follows, accommodating these multiple features:  <img src="https://latex.codecogs.com/gif.latex?h_{\theta}\left(%20x%20\right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}">

In order to develop intuition about this function, we can think about  <img src="https://latex.codecogs.com/gif.latex?{\theta_{0}}"> as the basic price of a house, <img src="https://latex.codecogs.com/gif.latex?{\theta_{1}}"> as the price per square meter, <img src="https://latex.codecogs.com/gif.latex?{\theta_{2}}"> as the price per floor, etc.  <img src="https://latex.codecogs.com/gif.latex?{x_{1}}"> will be the number of square meters in the house, <img src="https://latex.codecogs.com/gif.latex?{x_{2}}"> the number of  floors, etc.

There is a parameter and a variable in this formula, and to simplify the formula a little bit, the formula is converted to (e.g <img src="https://latex.codecogs.com/gif.latex?x_{0}=1">):
$$

$$
<img src="https://latex.codecogs.com/gif.latex?h_{\theta}%20\left(%20x%20\right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}">

At this point, the parameters in the model are a vector of (n+1) dimensions, and any training instance is also a vector of (n+1) dimensions, and the dimension of the matrix is m x (n+1).  So the formula can be Simply to : 

<img src="https://latex.codecogs.com/gif.latex?h_{\theta}%20\left(%20x%20\right)={\theta^{T}}X">

Using the definition of matrix multiplication , our multivariable hypothesis function can be concisely represented as:

![](/extra/img/week2_04.png)


http://www.ai-start.com/ml2014/html/week2.html

