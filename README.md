# Fetal-Health
Classification of fetal health in order to prevent child and maternal mortality

The dataset contains 2126 records and 22 columns of features extracted from Cardiotocogram exams, it was then categorised by medical professionals who specialised in obstetrics into 3 categories: 
•	1 : Normal 
•	2 : Suspect 
•	3 : Pathological

	Data Analysis and Visualisation
Dataset contain no null values in both independent features and target features. Target features were furthered explored and we find that 78% of target features are labelled as “Normal” while 14% and 8% are labelled as “Suspect” and “Pathological” respectively.
 
 

"prolongued_decelerations", "abnormal_short_term_variability", "mean_value_of_long_term_variability" and "percentage_of_time_with_abnormal_long_term_variability" have high correlation to “fetal_health” are the most important features. These features should be further explored for a more clearer picture.

   
   
The rates of change for all the features shows consistent trend except for "accelerations" feature. There are also outliers in the features that can be further explored.
 
There are outliers in the dataset. But the outlier won't be removed yet as it may lead to overfitting. 
It is a CTG report so it is very unlikely it is human error.
To solve the outlier problem, we will scaled the features columns.


	Scaling the Data
Feature scaling in machine learning is one of the most critical steps during the pre-processing of data before creating a machine learning model. Scaling can make a difference between a weak machine learning model and a better one. Normalization is used when we want to bound our values between two numbers, typically, between [0,1] which is what we are going to do for our next step.
 
	After we scaled the data, the features are clearly in the same range which we can see from the plot.
	Outliers can be spotted in certain features, which we have to make a call whether to take it along or drop it off.
	To prevent from overfitting, we will keep the outliers as it is not a result of measurement error or human error.
	Model Building
The data set are split into train and test set with the test set size sets at 0.2. Then the processed datasets is fit into five classification models, namely, Logistics Regression, Decision Tree, Random Forest, Support Vector Classification and Gradient Boost to find out which models would work for our dataset.
 
                              
From above table we can see that, after performing data preparation, preprocessing, and feature engineering to get the results, Gradient Boost and Random Forest are the top two performing model giving accuracy of 0.9459 and 0.9412 respectively. While Decision Tree gives 0.9123 and SVC scores 0.9123. On the other side, Logistics Regression showed least accuracy score with 0.8959. 
Gradient Boost and Random Forest are the two models that we would further enhanced to get the best model for our dataset.
	Random Forest Classifier
Processed dataset are passed through GridSearch CV with predefined parameters to search for the best parameters to have the best model for our dataset.
The result of best parameters is {'criterion': 'gini',  'max_depth': 16, 'max_features': 'log2', 'n_estimators': 100, 'n_jobs': None}
The test dataset are then pass through the model using the best parameters and the final result that we achieve is 95.31% with the Random Forest Model.

The model perform quite consistently across labels even and that proves that the models works well.

	Gradient Boosting Classifier
The same procedures are used with Gradient Boosting Classifier with the fitting of dataset into the GridSearch CV model with predefined parameters and the best parameters for Gradient Boosting Classification is {'learning_rate': 0.5, 'loss': 'deviance', 'max_depth': 3, 'n_estimators': 200}.
 
The model works consistenly across different label and the recall and f1-score are consistently high.
The best parameter on Gradient Boosting model scores 95.77% using our dataset. It is higher than the Random Forest and we would choose this model as our final model.

2.2.2 Fetal Health Classification
	Summarizing dataset
The dataset contains 5001 records with 7 features and a label column. The 7 features are forehead width measured in cm and forehead height measured in cm with an additional 5 columns, long hair, nose wide, nose long, lips thin and distance between nose to lip is long which are labelled “0” and “1”. The label column, “gender” are either labelled “Male” or “Female”.
	Exploratory Data Analysis
Dataset contain no null values in both independent features and target features. Target features were furthered explored and we find that 50.01% of target features are “Male” and 49.99% are “Female”. This would prevent the model from becoming biased towards one class.

All features shows correlation with gender classification except for long hair feature. It scores only -0.011. Forehead_width_cm and forehead_height_cm scores lower compares to other features at 0.334 and 0.277.

The scatterplot shows that people with forehead height longer than 6.5cm and 14.4cm are males.
it is evenly distributed for both males and females with long hair and the ones that is not.
Persons with wide nose are more likely to be male and persons with narrower nose are more likely to be female.
Persons with long nose are more likely to be male and persons with short nose are more likely to be female.
Persons with thin lips are more likely to be male and persons with thick lips are more likely to be female.

 



Persons with long distance between lips and nose are more likely to be male and persons with short distance between lips and nose are more likely to be female.
	Data Preparation
Data is split into train and test datasets with test dataset size at 25% of the whole dataset. Train dataset contain 3750 entries and 7 features and test datasets contain 1251 columns.
	Model Building and Hyperparameters Tuning
Five Models were chosen for initial modelling to find out which model is suitable for the datasets. Logistics Regression, SGD Classifier, Support Vector Classifier, K Nearest Neighbors Classifier, Decision Tree Classifier and Random Forest Classifier are paired with their respective hyperparameters to find the best model for the datasets.

From the modelling, , we will choose the random forest model as the classification model as scores the highest in the Cross Validation. Decision trees is going to be further explored to make sure we get the best model. 
	Random Forest Classifier
The result of best parameters is (max_depth=5, max_features='log2', n_estimators=400, random_state=0)
The test dataset are then pass through the model using the best parameters and the final result that we achieve is 97.28% with the Random Forest Model.
 
The model perform quite consistently across labels even and that proves that the models works well.

	Decision Tree Classifier
The same procedures are used with Decision Tree Classifier with the best parameters for Decision Tree Classification is (criterion='entropy', max_depth=6).
The test dataset are then pass through the model using the best parameters and the final result that we achieve is 97.12% with the Decision Tree Model.
 
The model works consistenly across different label and the recall and f1-score are consistently high.
The best parameter on Gradient Boosting model scores 95.77% using our dataset. It is higher than the Random Forest and we would choose this model as our final model.
Conclusion:
Maximum accuracy was achieved with Random Forest Classifier. Six classifiers were modelled and hyperparameter tuning for each classifier were done. The results showed us that Random Forest Classifier works best for this dataset and achieved 0.9784 accuracy on the trainset and 0.9728 on the test set.
2.2.2 Fetal Health Classification
	Summarizing dataset
The dataset contains 500 records with 2 columns with numerical data types. One column is temperature and the other column is revenue.
	Exploratory Data Analysis
The data has no NA values after initial analysis.

The dataset is then plotted in a scatterplot and it shows a clear trend.
The scatterplot shows that the dataset is scattered linearly.
The graph shows a strong correlation between the feature temperature and the revenue generated as the datapoints are lying very closely with the intercepts. The revenue generated is directly proportional to the temperature.

The jointplot on the datasets highlighted that the data points are distributed more heavily in the centre part of the plot.
The boxplot shows that there are outliers in the revenue column but after further investigation, the outlier is determined as correct value and not a measurement or human error.The outliers will be kept to prevent overfitting of dataset. 	

	Data Preparation
Data is split into train and test datasets with test dataset size at 25% of the whole dataset. Train dataset contain 3750 entries and 7 features and test datasets contain 1251 columns.
	Modelling
According to the results, the mean absolute error is only 19.997 and the R-Squared is scored at 98.01% which tells is that the variance is relatively low.

In conclusion, the datapoints in the datasets is distributed linearly and the linear regression models is the best model to handle the ice cream dataset that is provided.
