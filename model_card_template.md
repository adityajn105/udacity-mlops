# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Aditya Jain created this model. It is RandomForest Classifier with 100 estimators and everything else is default.

## Intended Use
This model can be used to predict if salary of a person is greater than 50k or not, based on the persons demographics. 

## Training Data
Data used to train model is provided by UCI. https://archive.ics.uci.edu/ml/datasets/census+income ; training is done using 80% of this data.

## Evaluation Data
Evaluation data is subset of original data by UCI and 20% of this data is used for evaluation.

## Metrics
The model was evaluated on Accuracy score. The value is around 0.812.


## Ethical Considerations
Dataset contains data related race, gender and origin country. This will drive to a model that may potentially discriminate people; further investigation before using it should be done.

## Caveats and Recommendations
Currently dataset provides limited number of occupations, country, gender, race etc. Further work is needed to be done to consider people with different demographics.