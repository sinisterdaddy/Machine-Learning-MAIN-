# Multi-Class SVMs for Classifying Thyroid Diseases
Implemented a Support Vector Machine model for diagnosing thyroid disease. 
- Specifically, the SVM is used to distinguish between 4 different thyroid states: 
    - Hyperthyroid
    - Hypothyroid
    - Euthyroid-sick
    - Euthyroid (negative)




## Motivation
- In the U.S. alone, an estimated 27 million Americans have Thyroid disease
- More than half of this population are undiagnosed or misdiagnosed 
- Women are seven times more likely to contract thyroid problems than men and nearly half of all women and a quarter of all men in the US will die with evidence of an inflamed thyroid.
- The symptoms of this disease often vary from person to person and are non-specific, so a correct diagnosis can easily be missed or misdiagnosed for irrelevant issues.

## Objective:
- Finding an accurate solution to this problem is a must
- Provide an efficient solution for healthcare practitioners via Support Vector Machines for diagnosing/classifying a particular thyroid disease that a person may have.
- This tool will cause an immense decrease in misdiagnoses as it is capable of distinguishing between problems of the thyroid gland and other illnesses in the body.
- As well as providing the ability to detect the disease before it forms into a more destructive anomaly. 

## Thyroid: What is it?
The thyroid is a gland that produces thyroid hormone.
This hormone regulates vital body functions:
- Breathing
- Body weight 
- Heart rate
- Muscle Strength 

## Types of Thyroid Diseases

- Euthyroid: A normal functioning thyroid gland

- Euthyroid Sick: The serum levels of thyroid hormones are low in clinically euthyroid patients with nonthyroidal systemic illness

- Hypothyroidism: The thyroid fails to make enough thyroid hormones and slows down many of your body's functions

- Hyperthyroidism: The thyroid makes more thyroid hormone than the body needs, which speeds up many of your body's functions.

## Thyroid Data Preprocessing: Exploratory Analysis
- Goal: gain a solid understanding of the data and make corrections before training model
- Imputation of missing values
  - For Continuous values: substituted in with the mode of the column 
    - Mode is the most frequently occurring number
Robust to outliers
  - For Categorical values (Sex): wasn’t many missing, so deleted those rows
    - Why? Because the sex of the person matters in respect to their diagnosis
- Balanced data: leaned heavily towards the negative samples
- Once found, able to train model

Data can be found here: https://archive.ics.uci.edu/ml/datasets/thyroid+disease

## Support Vector Machines
- A geometric approach to classification and regression tasks
- Main Objective: to find a hyperplane that distinctly classifies the data points.
- Hyperplanes are decision boundaries that help classify the data points. 
- If a data point falls on either side of the hyperplane, it is attributed to different classes. 
- There exist many possible hyperplanes, but the one that has the maximum margin, or distance, between data points of the classes is ideal.
- This maximum marginal distance provides confidence that future data points are assigned to the correct class

## One-vs-Rest Scheme
- SVMs are typically used for binary classification; classifying between two classes
- Strategy involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. 
- Essentially, it decomposes a multiclass classification problem into a multiple binary classification problems

## Results:
- Obtained an accuracy of about 69%
- For further details on the entire process, visit the [notebook here](https://github.com/deontaepharr/Multiclass-SVM-Thyroid-Classification/blob/master/Notebooks/Classification%20of%20Thyroid%20Disorders%20Using%20Support%20Vector%20Machines.ipynb)

## Required Packages
Python Version: 3.6.5

[Pandas](https://pandas.pydata.org/): Powerful data structures for data analysis, time series, and statistics 
```bash
pip install pandas
```

[Numpy](https://www.numpy.org/): a general-purpose array-processing package designed to efficiently manipulate large multi-dimensional arrays  
```bash
pip install numpy
```

[Scikit-Learn](https://scikit-learn.org/stable/index.html): Simple and efficient tools for data mining and data analysis  
```bash
pip install scikit-learn
```

[Seaborn](https://scikit-learn.org/stable/index.html): A library for making statistical graphics in Python  
```bash
pip install seaborn
```

[Matplotlib](https://matplotlib.org/): Python plotting package
```bash
pip install matplotlib
```
