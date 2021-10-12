---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Introduction
   
&emsp; While there have been many studies on building models to predict student performance, the predictive accuracy of these models are often complicated by the myriad of factors inside and outside the classroom that affect student performance. For example, factors ranging from a student's previous academic achievements to the socio-economic background of a student's family are all considered when building predictive models on student performance. The features of such models and the predictive model of this project are variables that describe a student's academic background, demographics, and family background, among others, that may influence how a student performs. The target variable then is the student's academic performance in the classroom, which is commonly measured with the final grade a student achieves in a class. As with most prediction models, there is no guarantee of perfect predictive accuracy for a model predicting student performance. However, there is considerable value in building even a moderately accurate predictive tool in that it could immensely benefit education professionals in their efforts to identify struggling students, which consequently would help education professionals better allocate school resources and improve the education quality. This is hugely important in current times, as education professionals work to fight against growing inequalities and falling standards in education in certain countries.
    
&emsp; This project attempts to build a highly accurate predictive tool that utilizes a binary classification model in machine learning to classify a student's performance as pass or fail. The dataset used for this project came from the UCI Machine Learning Repository and was published by the University of Minho in Portugal. The data itself is derived from school reports and questionaires and consists of 32 total attributes across 395 students that measures student grades specifically in mathematics, and describes the demographics and the social and school features of students. While the dataset utilizes features 'G1', 'G2', and target variable 'G3' to measure student performance in the first, second, and final period respectively on a 0 (low) - 20 (high) integer scale, a binary scale for 'G1', 'G2', and 'G3' will be used to build the binary classification model in this project, where 0 represents a failing score ranging from 0-9 on the original integer scale and 1 represents a passing score ranging from 10-20 on the original integer scale.
   
&emsp; There were several studies conducted that used this dataset to similarly predict student performance based on student attributes. In Cortez and Silva's original publication of the database, they used a combination of three data mining goals, which included Binary Classification, 5-Level Classification, and Regression, and four data mining methods, which included Decision Trees, Random Forests, Neural Networks, and Support Vector Machines to study whether it would be possible to achieve a high predictive accuracy of student performance. The models all performed well for accuracy, where one Decision Tree model recorded a classification accuracy of 93.0%, proving that it was indeed possible to achieve a high predictive accuracy [1]. In another study by Satyanarayana and Nuckowski, the authors have used the dataset to use multiple classifiers (Decision Trees, Naive Bayes, and Random Forest) to answer the question of whether or not predictive accuracy can be further improved if an ensemble filtering technique is utilized to imporve the quality of student data. In fact, they found that predictive accuracy did increase with ensemble filtering, with a peak accuracy score of 95% [2]. However, what the predictive accuracies of these two studies show is that there is still room to improve and build on the models in these two studies to achieve the highest predictive accuracy possible, which is the goal of this project.
    
    
# Exploratory Data Analysis

The following figures in this section were created during exploratory data analysis. 

![image](figures/SEX-GRADE.jpg)


**Figure 1** This figure depicts a stacked bar plot showing the proportion of male and female students who receive a final grade of pass or fail in mathematics. The proportion of male students passing the class is just slightly higher than the proportion of female students passing the class. The slight difference in the proportion of male students and female students passing the class suggests that the 'sex' feature may not play a significant role in the machine learning model and may not be as influential to student performance as other features.

![image](figures/ABSENCE-GRADE.jpg)

**Figure 2** This figure displays a category-specific histogram showing the proportion of students passing or failing in mathematics according to the number of days a student missed class. From the histogram, it is evident that there is a slightly larger proportion of students who achieve a passing final grade than those who achieve a failing final grade when the number absences are low. As the number of absences increases, there is a slight increase in the proportion of students who receive a failing final grade, which suggests absences may have a small negative influence on student performance.

![image](figures/SCHOOL-GRADE.jpg)

**Figure 3** This figure depicts another stacked bar plot showing the proportion of students from two different schools in Portugal who receive a final grade of pass or fail in mathematics. The proportion of students who achieve a final passing score in school 'GP' is nearly identical to that of students in school 'MS', indicating that the 'school' feature may not be as strong indicator of student performance as other features in the dataset.


# Methods

## Data Splitting and Preprocessing

&emsp; Each row, or observation, of the data accounted for one individual student, which meant that the data was assumed to be independent and identically distributed with no time-series or group structure. To account for the small number of observations in the data and to address the need for more cases to test and validate the models given the small dataset, 20% of observations were initially split into testing using train_test_split and the other 80% of observations were allocated to 5-fold cross-validation. This splitting approach was trained on the machine learning model as it also accounted for variability in random splits that could occur with a small dataset. The preprocessor then fit and transformed training folds before going on to fit and transform the testing and validation sets in each iteration of cross-validation. Some categorical features such as 'Medu' and 'Fedu' were already ordered and converted into integer values in the data, so these features did not need to be encoded using the ordinal encoder. However, the preprocessor applied the StandardScaler on these features for the purpose of converting them to have mean = 0 and standard deviation = 1. The preprocessor applied one-hot encoder on the remaining categorical features such as 'school' and 'sex', as they were all unordered and could not be clearly ranked. Finally the preprocessor applied the MinMaxEncoder on the remaining continuous features such as 'age' and 'absences', as they were all bounded by reasonable ranges. The target variable 'G3' was not encoded in the preprocessor. In all, there were 32 total features in the final preprocessed data.
   
# References

[1] P. Cortez and A. Silva. Using Data Mining to Predict Secondary 
    School Student Performance. In A. Brito and J. Teixeira Eds., 
    Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 
    2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-'
    9077381-39-7. 

[2] A. Satyanarayana and M. Nuckowski, “Data Mining using Ensemble 
    Classifiers for Improved Prediction of Student Academic 
    Performance,” Middle Atl. Sect. Spring 2016 Conf. (ASEE 2016), no. 
    April.2016.

**Github Repository:** https://github.com/jamesjooyoung/Data-1030-Project.git
<!-- #endregion -->
