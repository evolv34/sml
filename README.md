
# SMLT - Simple Machine learning tool.

This is a machine learning pipeline tool that focuses mainly on using `Tensorflow Extended` library to train machine learning model using data from various data storage.




## Architecture


![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)


## Description

Notebook files are stored in `/notebooks` folder

| File | Description |
| :----| :-----------|
|`IEEE-CIS-Fraud-Detection-preprocessor.ipynb`| pyspark preprocessor notebook |
|`IEEE-CIS-Fraud-Detection-Train-TF.ipynb`| Tensorflow extended model training and publishing code. |
|`IEEE-CIS-Fraud-Detection-Score-Spark.ipynb` | pyspark score notebook. |


## Installation

The stack is deployed using docker and docker-compose. `docker` and `docker-compose` are prerequisite.


```bash
  docker-compose -f sml.yml up -d
```
    