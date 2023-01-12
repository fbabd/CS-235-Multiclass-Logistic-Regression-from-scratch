
Multiclass Logistic Regression implementation
Implemented by: Faisal Bin Ashaf

* To run the model implementated from scratch by me "faisal_sklearn_LR_final.ipynb" should be used. 

== Files and Folders ==

Occupancy_Estimation.csv: 
        Original dataset file from the source. 

data:   This folder contains all the dataset divided into X_train, X_test, y_train, and y_test.
        These secondary files are created in the midterm after applying different data representation techniques.
        secondary datasets are - Original, z normaimized, min-max normalized, PCA (10 components), LDA (2 components), 
                                t-SNE (2 components), FA (10 components)

faisal_sklearn_LR_mid.ipynb: 
        This file contains all the data representation techniques codes and Logistic Regression
        codes using sklearn python library. 

faisal_sklearn_LR_final.ipynb: 
        This file contains codes that uses my scratched implementation of Logistic regression. 

MyLogisticRegression.py: 
        This is the file where I have written all the scratched code for Logistic Regression.
        The classifier is written as a class that has fit(X, y), predict(X), and plot_history() function.
        It is imported in the notebook for using the functionalities. 

cv_scratch.py: 
        This file contains helper function to run my scratch implemented model with cross-validation.
        It also has the function to run the model without cross-validation. 
        It has functions for plotting the cross-validation results. 
        It is imported in the notebook for using the functionalities. 

