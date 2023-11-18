#   Inventory Monitoring at Distribution Centers

In this project, we'll work on how to count the objects in bins. Our goal is to create a pipeline with AWS tools.

**Note**: This repository relates to AWS Machine Learning Engineer nanodegree provided by Udacity.

## Project Summary
Our tasks are structured into five distinct categories:

1. Gathering data from the primary source and arranging it within an S3 bucket.
2. Conducting exploratory data analysis (EDA) on the dataset within SageMaker Studio.
3. Creating a model and optimizing its hyperparameters through SageMaker.
4. Training and assessing the model's performance using SageMaker.
5. Overseeing the management of model resources with the assistance of SageMaker Debugger.


## Environment

We utilized an AWS SageMaker instance of type `ml.t3.medium` because `ml.t3.medium` is often a good choice for a notebook instance because it offers a balance between cost and performance. The `ml.t3.medium` instance type provides a good amount of CPU and memory resources. It offers 2 vCPUs, 4 GB of memory, and moderate network performance. 

Additionally, the essential software prerequisites for the project include:
- Python version 3.10
- Transformer version 4.28.1
- Pytorch version 2.0

## Initial setup

1. Clone the repository.
2. Run [create_dataset.ipynb](./create_dataset.ipynb) to create dataset to train and evaluate
3. Run [sagemaker.ipynb](./sagemaker.ipynb) cells in order and follow its instructions!

## Data

In this work, we aim to have 300 clusters of documents extracted from news. To this end, we made use of the Vietnamese language version of Google News. Due to the copyright issue, we did not collect articles from every source listed on Google News, but limited to some sources that are open for research purposes. The collected articles belong to five genres: world news, domestic news, business, entertainment, and sports. Every cluster contains from four to ten news articles. Each article is represented by the following information: the title, the plain text content, the news source, the date of publication, the author(s), the tag(s) and the headline summary.

After that, two summaries are created for each cluster (produced in the first subtask above) by two distinguished annotators using the MDSWriter system (Meyer, Christian M., et al. "MDSWriter: Annotation tool for creating high-quality multi-document summarization corpora." Proceedings of ACL-2016 System Demonstrations). These annotators are Vietnamese native speakers and they are undergraduate students or graduate students. Most of them know about natural language processing. The full annotation process consists of seven steps that must be done sequentially from the first to the seventh one.

Github dataset used [https://github.com/CLC-HCMUS/ViMs-Dataset]

## Pipeline

After splitting our dataset into train, validation, test. We can store them into S3 bucket as shown below:

You can use [train.py](./train.py) for hyperparameter tuning for benchmark and refined model, respectively. This point is similar for [train.py](./train.py).

And finally, you can use [sagemaker.ipynb](./sagemaker.ipynb) as an orchestrator for all the mentioned above scripts to create the pipeline in SageMaker.

## Profiler Reports
The reports of the SageMaker profiler is organized in [profiler reports](./ProfilerReports/benchmark)

## Technical Reports
You can read about the introduction and development phase of the project in [proposal.pdf](./propsoal.pdf) and [report.pdf](./report.pdf).