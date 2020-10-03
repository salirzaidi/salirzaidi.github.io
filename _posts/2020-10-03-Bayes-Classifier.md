<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
In this blog post, I summarie how Bayes Classifiers operation. Let $$(X,Y)$$ be a pair which takes values in $$\mathbb{R}^d\times\mathcal{Y}=\{1,2,...,K\}$$ where $$Y$$ is the class label associated with the observation $$X$$. The classification problem involves selecting class label $$Y|X=x$$.Therefore the problem can be concisely stated as:\
**Input:** $$X=x\in \mathbb{R}^d$$ \
**Output:** $$Y\in\mathcal{Y}$$ \
**Goal:** To learn $$C(x):\mathbb{R}^d\rightarrow\mathcal{Y}$$ \
**Joint Distribution:**  The joint distribution of $$(X,Y)$$ is given by $$P(X,Y)$$ \
**Loss Function:**  As $$\mathcal{Y}$$ is a discrete valued-set the squared-error function makes less sense, in this case the loss function can be defined as

$$L= \begin{bmatrix}  
0 & 1 & 1&...&1\\  
1 & 0 & 1&...&1\\
1 & 1&0&....&1\\
\vdots & \vdots&\ddots & \vdots&\vdots\\
1 & 1&1&....&0 
\end{bmatrix}_{K\times K}$$

where the $L(k,l)$ is the cost of misclassifying the samples belonging to class $$k$$ as $$l$$. This is known as zero-one loss function. The expicted prediction error in this case is given by:\
\$$\mathbb{E}(C(x))=\mathbb{E}(L(\mathcal{Y},C(x)))$$\
\$$\mathbb{E}(C(x))=\mathbb{E}_X\mathbb{E}_{Y|X}(L(\mathcal{Y},C(x)|X))$$\
\$$\mathbb{E}(C(x))=\mathbb{E}_X\sum_{i=1}^KL(i,C(x))\Pr\{Y=i|X=x\}$$\
\$$\mathbb{E}(C(x))=\sum_{i=1}^KL(i,C(x))\Pr\{Y=i|X=x\}\Pr\{X=x\}$$\
Therefore the $\mathbb{E}(C(x))$ can be minimised by selecting 
\$$C(x)=\underset{y}\arg\min\sum_{i=1}^KL(i,y)\Pr{Y=i|X=x}$$\
Consequently, the Bayes estimator can be given as
$$C(x)=\underset{i}\arg\max\Pr\{Y=i|X=x\}$$

**Further References & Sources**
M´ario A. T. Figueiredo, Lecture Notes on Bayesian Estimation and Classification
T. M. Cover, Nearest Neighbor Pattern Classification
