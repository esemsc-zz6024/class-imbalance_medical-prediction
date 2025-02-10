# class-imbalance_medical-prediction

## Project introduction
Medical prediction tasks involve the application of predictive analyses to complex medical cases, utilizing extensive medical datasets. However, these tasks commonly encounter the challenge of class imbalance, which can exert varying degrees of influence on their performance.This paper outlines an experimental approach for a medical prediction task aimed at forecasting Acute Kidney Injury (AKI) risk.

## Dataset
A medical data set was collected and constructed from the MIMIC-III database.

## Conclusion
When the IR is 5, this work used seven data resampling methods under three prediction models to address the class imbalance. 
The performance of each resampling method under three prediction models were evaluated and compared based on the precision, recall and AUC. 
The experimental results indicate that the RUS method performs well in all three prediction models. Additionally, the SMOTE method achieved the best result in the LR model, with the highest AUC value of 0.7203.

This project evaluated and analyzed various data resampling methods and their priorities in different scenarios. It summarized the following ideas: 
When the IR is between 1 and 5, if the assessment index for model training is the PR curve , so data resampling methods can be excluded. If the assessment index is not considered, the RUS method is recommended. 
When the IR is between 5 and 10, either over-sampling or combined-sampling methods are preferred. 
When the IR is between 10 and 100, it is recommended to use the combined-sampling method.