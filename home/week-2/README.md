## Week 2

### 2.1 Multiple Features

We will start to talk about a new version of linear regression that's more powerful. One that works with multiple variables or with multiple features. Here's what I mean. In the original version of linear regression that we developed, we have a single feature x, the size of the house, and we wanted to use that to predict why the price of the house and this was our form of our hypothesis.

![](/extra/img/week2_01.png)

But now imagine, what if we had not only the size of the house as a feature or as a variable of which to try to predict the price, but that we also knew the number of bedrooms, the number of house and the age of the home and years. 
$$
(x_1,x_2,x_3,......x_n)
$$
It seems like this would give us a lot more information with which to predict the price.

![](/extra/img/week2_02.png)





<iframe>/home/week-2/lectures/notes.pdf</iframe>







To introduce a little bit of notation, we sort of started to talk about this earlier, in this case, four features and I'm going to continue to use Y to denote the variable, the output variable price that we're trying to predict. Let's introduce a little bit more notation. Now that we have four features, I'm going to use lowercase "n" to denote the number of features. So in this example , you know, one, two, three, four features. And "n" is different from our earlier notation where we were using "n" to denote the number of examples. So if you have 47 rows "M" is the number of rows on this table or the number of training examples. So I'm also going to use X superscript "i" to denote the input features of the "i" training example.

<img src="https://latex.codecogs.com/gif.latex?{x}^{(2)}\text{=}\begin{bmatrix}%201416\\\%203\\\%202\\\%2040%20\end{bmatrix}">

<img src="https://latex.codecogs.com/gif.latex?x_{2}^{\left(%202%20\right)}=3,x_{3}^{\left(%202%20\right)}=2">

The hypothesis supporting multi variable is expressed as:

<img src="https://latex.codecogs.com/gif.latex?h_{\theta}\left(%20x%20\right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}">

There is a parameter and a variable in this formula, and to simplify the formula a little bit, the formula is converted to:

<img src="https://latex.codecogs.com/gif.latex?h_{\theta}%20\left(%20x%20\right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}">

At this point, the parameters in the model are a vector of (n+1) dimensions, and any training instance is also a vector of (n+1) dimensions, and the dimension of the matrix is m x (n+1).  So the formula can be Simply to : 

<img src="https://latex.codecogs.com/gif.latex?h_{\theta}%20\left(%20x%20\right)={\theta^{T}}X">






http://www.ai-start.com/ml2014/html/week2.html
