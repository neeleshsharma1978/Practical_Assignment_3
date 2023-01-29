# Practical_Application_III-UC-Berkeley

# Overview:
In this practical application,The goal is to compare the performance of the classifiers, namely K Nearest Neighbor, Logistic Regression, Decision Trees,and Support Vector Machines. I utilized a dataset related to marketing bank products over the telephone. The dataset comes from the UCI Machine Learning repository https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing. It is one of the common and powerful dataset repository for machine learing. 

Data Set Information:
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Attribute Information:
Input variables:

##bank client data:

1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')

##related with the last contact of the current campaign:

8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

##other attributes:

12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

##social and economic context attributes

16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

# The Task:
The work starts with the data understanding and data source. The data is collected from the powerful data repository called a UCI machine learning repository.
The dataset is related with direct markting compaigns (phone calls) of a Portuguess banking institution. 

There are four datasets:
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).
The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

##Exploratory data anaylysis and data visualization:

An exploratory data analysis and data visualization have been done to clearly understand the relationship among the dataset and their trends. The matplotlib graph of the data set are not symmetrical. The graph is skewed to the right, most of the data falls to the right of the graph's peak, that mean the mean > median > mode. However, the distribution is become almost symmetrical on the logarithmic transformed data set. A seaborn Kernel Density Estimate (KDE) Plot also utilized to estimate the probability density function of the continuous age data set. The plot has been used on the transformed dataset. A plotly histogram, seaborn pairplot, seaborn joint plot, seaborn bar plot, seaborn box plot, and seaborn heatmap plot are also utilized for a better visualization to understand the data relationship. 
A seaborn from the heatmap plot the 'euribor3m' and 'emp.var.rate' have the highest strong positive relationship. On the other hand the 'previous' and 'emp.var.rate' have a strong negative relationship. A boxplot helps to easily grasp the average numbers of the jobs those are subscribed or not. The retired have got an average age highest age who subscribed the term deposit.


##Engineering features:

Using the bank information features, an encoding data transformation has been applied. The transformation tools are imported from the machine learning package called a scikit learn. A OneHotEncoder has been used to the 'job','marital','default','housing','loan','contact','month','day_of_week’, and 'poutcome' attributes. The 'education' data set has been encoded by Oridinal Encoder. Then, a pipeline has been prepared for each classifier. The K Nearest Neighbor, Decision tree, and Support Vector Machine pipeline includes transformer and model, but the Logistic Regression pipeline have an extractor on top of the transformer and model. Finally, the data prepared into X and y data set. X has all features and y has the desired target. The data then splits into train and test data set by using a tool from the scikit learning preprocessing tools.

##Baseline Model:

The baseline model is simple model that acts as a comparison. Moreover, the baseline model should be based on the dataset to create the actual model. It helps to compare the complex model result. With the baseline model, we could assess whether we need a complex model or the simple one already working for the business. There are two categories to calculate the baseline model, either use a Simple Baseline Model or Machine Learning Baseline Model. A dummy classifier from the scikit learning packages is used to generate a baseline results. A different result of metrics based on the different dummy classifier strategy has been generated. Since the class of the data set are nor balance, a stratified strategy is suitable for imbalanced data as it reflects the actual distribution. The accuracy of the dummy classifier based on the stratified strategy is 0.805778101480942. Therefore, this accuracy result is used as a benchmark for our complex models.

##Simple Model:

The work starts by train the Logistic Regression model based on the default parameters to build a basic model. The model took 0.585625 seconds to train and scored an accuracy of 0.8929351784413693. Then, K nearest Neighbor, Decision tree, and support Vector Machine classifiers are trained based on their default parameters. The models took 0.080644 seconds, 0.240950 seconds, and 53.349359 seconds and scored 0.890426, 0.834992, and 0.899409 respectively. All models are scored a similar result. But the Support Vector Machine took a long time compared to the others. K Nearest Neighbor process faster any other models. The Logestic Regression, KNN, and SVM model are good on both train and test data set. Decision model score 0.995595 on the train data set and 0.834992 on the test data set. The model is probaly overfitting. This 
model is required to tune the hyperparameters. 

##Improving the Model:

Eventually, a gridsearchCV has been used to tune the hyperparameters of the models. The Logistic Regression model grid search include only l2 regularization and 0.01,0.5,1,1.5,2 inverse of regularization strength. The smaller values specify stronger regularization. A 21 number of K Nearest Neighbor model has been used and the grid search also includes weights and power. The possible weights are uniform and distance and the power parameter for the Minkowski metric. p = 1 is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. Four hypermeters are tuned for the Decision tree model. The criterion: used to measure the quality of a split a Gini and entropy splitter are used. The max_depth: the classifier will check up to max depth 2,3,4. The Min_sample_split: it is the minimum number of samples required to split an internal node, only 0.05, 0.1,0.2 are used and finally, the min_samples_leaf:it is the minimum number of samples required to be at a leaf node. The minimum number of leaf checked at 1,2,3. Last, the support Vector Machine hyperparameters are prepared. The model has been trained on the different kernels, gammas and polynomial degrees.

##Improved Results:

The time taken to train Logistic Regression, KNN, Decision Tree, and SVM model is 13.304939,226.731269, 28.096495, and 202.620631 and their accuracy on the test data set is 0.894796, 0.896820,0.899166, and 0.900542 respectively. The KNN and SVM model took longer to learn the data. But the Logistic regression is better. May, be the reason could be the model did not used many hyperparameters. Decision trees learn the data fast. The model scored 0.900212 on the train data set	and 0.899166 on the test data set. After tunning the hyperparameters the decision gets better results. The other models do not have a big change on the accuracy results from their defualt results. 

##Findings:

In this problem, the Decision tree model is overfitting based on the default parameters. But the model quick back when tuned the hyperparameters. The logistic Regression model is well performed in terms of the accuracy and training time. KNN and SVM model are also well performed on the default parameters. But they took longer time to learn on when a grid search applied.

##Recommendation:

The Logistic Regression model is the best model on both accuracy and time.






