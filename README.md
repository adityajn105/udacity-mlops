# Udacity MLOPs Nanodegree Project

## ML pipeline to expose API on Render

Udacity MLOPS nanodegree project to train a model on census data and publish it via API on Render

## Developer Environment
1. Have python3.8 installed.
2. Firstly install all requirements: `pip install -r requirements.txt`
3. Build models using `python mlpipeline.py`
4. You can test using `pytest`

## EDA
EDA notebook is provided: [link](https://github.com/adityajn105/udacity-mlops/blob/main/eda/eda.ipynb). This is referred while developing cleaning procedures.

## Running individual Steps:

### Cleaning Steps

To clean the data, it will create cleaned_census.csv in data folder.
`python mlpipeline.py --step=clean`


### Train/save model

To train model, it will create necessary model files in model folder.
`python mlpipeline.py --step=train_save_model`

### Slicing evaluation

Evaluate model on different slices, output can be found in data folder.
`python mlpipeline.py --step=evaluate`

### Run all steps one after other

`python mlpipeline.py`

### Check if Render server working

`python check_render_api.py`

## CI/CD

This project is supported by CI/CD, every new commit to `main` or pull request to `main` will trigger a github action workflow. 

Link to [CI pipeline](https://github.com/adityajn105/udacity-mlops/actions/runs/6086935936/job/16514414503)

## For Rubriks

1. [live_get.png](https://github.com/adityajn105/udacity-mlops/blob/main/screenshots/live_get.png)
2. [live_post.png](https://github.com/adityajn105/udacity-mlops/blob/main/screenshots/live_post.png)
3. [example.png](https://github.com/adityajn105/udacity-mlops/blob/main/screenshots/example.png)
4. [Model card](https://github.com/adityajn105/udacity-mlops/blob/main/model_card_template.md)
5. [continuous_integration.png](https://github.com/adityajn105/udacity-mlops/blob/main/screenshots/continuous_integration.png)
6. [continuous_deloyment.png](https://github.com/adityajn105/udacity-mlops/blob/main/screenshots/continuous_deloyment.png)