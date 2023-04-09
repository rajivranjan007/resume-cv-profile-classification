import pickle
import streamlit as st
import pandas as pd
import numpy as np
import re
import docx2txt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter

svm_model_final = pickle.load(open('svm_model_final.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
skills_list = pickle.load(open('skills_list.pkl', 'rb'))

st.set_page_config(page_title = "Resume classification", layout = "wide") 

st.title('Resume classification')

uploaded = st.file_uploader('Upload your Resume', type = ["pdf", "docx", "doc", "txt"])


if uploaded is not None:
    data = []
    text = docx2txt.process(uploaded)
    data.append(text)
    df = pd.DataFrame({'Resume': data})
    
    

    def extract_skills(resume_text):
        # Create a dictionary to hold skill matches and their frequency
        skills_freq = {skill: 0 for skill in skills_list}

        # Loop through each word in the resume text and check if it matches any skill
        for word in re.findall(r'\w+', resume_text):
            for skill in skills_list:
                if word.lower() == skill.lower():
                    skills_freq[skill] += 1

        # Filter out skills with frequency 0 and return a list of skills sorted by frequency
        skills = [skill for skill, freq in skills_freq.items() if freq > 0]
        skills.sort(key=lambda x: skills_freq[x], reverse=True)

        return skills
   
    # regular expressions for phone number and email
    phone_regex = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    
    # search for phone number and email in the resume text
    phone_number = re.search(phone_regex, text)
    email = re.search(email_regex, text)

    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    
    def clean_resume_text(resume_text):
        """
        This function takes in a string of text (resume) as input and returns a cleaned version of the text.
        """
        # Convert to lowercase
        resume_text = resume_text.lower()
    
        # Remove numbers and special characters
        resume_text = re.sub('[^a-zA-Z]', ' ', resume_text)
    
        # Remove punctuation
        resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))
    
        # Remove extra whitespaces
        resume_text = ' '.join(resume_text.split())
    
        # Remove words with two or one letter
        resume_text = ' '.join(word for word in resume_text.split() if len(word) > 2)

        # Remove stop words
        resume_text = ' '.join(word for word in resume_text.split() if word not in stopwords)

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        resume_text = ' '.join(lemmatizer.lemmatize(word) for word in resume_text.split())

        return resume_text

    df["clean_text"] = df["Resume"].apply(clean_resume_text)
    all_resume_text = ' '.join(df["clean_text"])

    all_words = all_resume_text.split()
    word_counts = Counter(all_words) 
    
    
    # Clean the text by removing short words and noise words
    noise_words = ['xff','xffcj', 'xbabp','xddn','xaek','xcdf','xedv','xfe', 'xfeoj', 'xbe', 'xed', 'xbf', 'xef',
                   "xcf","xfe",'xfd', 'xea', 'xdd', 'xde', 'xba', 'xdc', 'xae', 'xdf', 'xec', 'xeb', 'xbb', 'xca',
                   'xaf', 'xac', 'xaa', 'xcf', 'xda', 'xcd', 'xab', 'xfb', 'xce', 'xbd', 'xdb', 'xcc', 
                   'xbc', 'xfc', 'xfa', 'xee', 'xad', 'xcb','hxai','xban']

    df['clean_text'] = df['clean_text'].apply(lambda x: re.sub(r'\b\w{{1,2}}\b|\b(?:{})\b'.format('|'.join(noise_words)), '', x))
    
    resume = df.loc[:, 'clean_text']
    
    X_train_tfidf = tfidf_vectorizer.transform(resume)
    
    y = svm_model_final.predict(X_train_tfidf)
    
    if y == 0:
        st.subheader("Person's Resume Match's to SQL Developer")
    elif y == 1:
        st.subheader("Person's Resume Match's to People Soft")
    elif y == 2:
        st.subheader("Person's Resume Match's to React Developer")
    else:
        st.subheader("Person's Resume Match's to Workday")
        
    if phone_number is not None:
        st.write("Phone number:", phone_number.group(0))
    else:
        st.write("No phone number found")

    if email is not None:
        st.write("Email:", email.group(0))
    else:
        st.write("No email found")
       
    skills = extract_skills(text)
    st.table({"Skills" : skills})
        
        
       
   
    