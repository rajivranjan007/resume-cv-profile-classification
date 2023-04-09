# Project Title
Resume Matching and Analysis using NLP

## Overview
The project involves using Natural Language Processing techniques to train a model that can analyze resumes and match them to specific job profiles. The project involves preprocessing raw data in doc and docx formats, training various classification algorithms, and deploying the final model on Streamlit.

## Installation
To run this project, you will need to have Python 3 installed along with the following libraries:

* Pandas
* Numpy
* Scikit-learn
* NLTK
* Docx2txt
* Streamlit
## Data Preparation
The data for this project consisted of resumes in doc and docx formats. The first step was to read the resumes using the docx2txt library and convert them into a dataframe. The data was then preprocessed using NLTK to remove stop words, punctuation, and perform stemming.

The preprocessed data was then visualized using various plots to gain insights into the characteristics of the data.
## Model Training
Various classification algorithms were trained on the preprocessed data, including K-Nearest Neighbors (KNN), Naive Bayes, Random Forest, and Support Vector Machines (SVM). The performance of each algorithm was evaluated using various metrics such as accuracy, precision, recall, and F1 score.

SVM was chosen as the main model due to its superior performance in the project.
## Deployment
The final model was deployed on Streamlit, which is a platform for building and sharing data applications. The outcome of the project was a model that could take a resume as input and output which job profile the resume matched, along with the candidate's mobile number and email address, and the skills listed on the resume.
## Conclusion
The project demonstrates the effectiveness of NLP techniques for analyzing and processing textual data. The model can automate and improve the efficiency of tasks such as resume screening and job matching. The project has the potential to be extended to other languages and to incorporate more advanced NLP techniques.

## Contributors
- [Rajiv Ranjan Kumar](rajivranjan819@gmail.com)

