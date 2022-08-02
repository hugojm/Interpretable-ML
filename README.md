
# Installation

To install the framework we will need to create a new environment and install the dependences. To create the environment and to activate it:

```
conda create -n iML-project python=3.9
conda activate iML-project
```
Then we will need to install the requirements:

```
pip install -r requirements.txt
```

This project has been tested on Google Colab as well as a Kali Linux VM with 2 cores and 4GB RAM.

# About the project


## Framework

We have used Jupyter Notebook to implement the framework since it makes it very easy to create visually appealing plots and general output.

This framework can be applied on (nearly) every dataset, given the correct information, on which basis it creates an adversarial model that successfully fools perturbation-based methods like LIME and SHAP. The framework is also able to plot PDPs of the adversarial models, but unable to fool PDP itself.

The basic usage is the following: The user provides at least a pandas dataframe, a sensitive feature name, a list of values in the sensitive feature column which should be used to discrimate against, a mode that describes if an existing feature or new features should be used to shift the influence away from the sensitive feature, as well as a list of categorical feature names. The unprotected class name and its list of values also needs be provided and would be best described as the y-values of this dataset. Example calls to this framework could look like:

```
fools = FooLS(pd_dataset=dataset, protected_class_name='race', protected_class_name_value=['black', 'hispanic'], unprotected_class_name='score', unprotected_class_name_value=['HIGH'], column_flag=ONE_EXISTING_COLUMN, correlated_column='income', categorical_feature_names=['has_job', 'postal_code'], unprotected_class_name='score')

fools = FooLS(pd_dataset=dataset, protected_class_name='race', protected_class_name_value=['black', 'hispanic'], unprotected_class_name='score', protected_class_name_value=['HIGH'], column_flag=ONE_UNCORRELATED_COLUMN, categorical_feature_names=['has_job', 'postal_code'])
```

Invalid entries (e.g. NAs) should be removed before applying this framework on a dataset. If a dataset contains numerical values and the user wants to discriminate against a value above/below a threshold then he must convert it to a boolean entry himself.

The framework can return the adversarial models as well as their explanations by calling:

```
fools.LIME_execute()
fools.SHAP_execute()
```

A PDP can be generated for a model by calling:

```
fools.PDP_execute()
```

## Datasets description

### COMPAS dataset

This dataset captures detailed information about the criminal history, jail and prison time, demographic attributes, and COMPAS risk scores for 6172 defendants from Broward Couty, Florida. The sensitive attribute in this dataset is race – 51.4% of the defendants are African-American. Each defendant in the data is labeled either as high-risk or low-risk for recidivism.


### German credit dataset

This dataset captures financial and demo- graphic information (including account information, credit history, employment, gender) of about 1000 loan applicants. The sensitive attribute in this dataset is gender – 69% of the individuals are male. Each individual in the dataset is labeled either as a good customer or a bad one depending on their credit risk.


### Boston housing dataset

This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. The objective of this dataset is to predict the price of a house given some variables. We have found a variable with a potential bias, the black variable, that measures the proportion of black people by town.

We preprocessed the dataset in a way that the black population in a neighborhood in no longer interpreted as a numeric value, but as a boolean with the threshold being the 50th percentile.

### Telecust Dataset

Telecust Dataset collects information about 1.000 clients of a company and categorizes them into 4 classes (according to whether they are good or bad clients). Any model could be biased by the gender variable and base the prediction on this alone. If we manage to deceive the explanation method, we will be able to create a model that discriminates against clients only because of their gender and therefore prevent them from being good customer no matter what they do. In our case women would never be able to get a good grading because of the algorithm's bias.

Link: https://www.kaggle.com/prathamtripathi/customersegmentation


### Absenteeism Dataset

This dataset has information about 8336 employees. The main objective is to obtain the number of hours in the absence of the workers, the potential bias is given by the gender variable since it could give greater weight to one gender than another.

According to our implementation, how we have managed to fool the interpretability method, the predictions would always tell us that women are more likely to be absent from work since neither LIME nor SHAP are able to detect the bias.

Link: https://www.kaggle.com/HRAnalyticRepository/absenteeism-dataset

## PDP implementation

We have implemented a PDP function to see if we can fool this explanation method. The results of the experiments are attached in the notebook with their corresponding graphics.

## Hyperparameter Sensivity

In this section, we wanted to see how it will change the ability to fool the techniques depending on the hyperparameters. To do that, we have changed the hyperparameters of the local model (random forest estimators) and the perturbation algorithms (k-means, standard derivation, perturbation multiplier).

## Perturbation methods

In addition to the perturbation methods explained in the paper, we have used Soft Brownian Offset and Gaussian Hyperspheric Offset for LIME and the BIRCH clustering algorithm for SHAP. We have used the python sbo library (https://pypi.org/project/sbo/) as an implementation.
