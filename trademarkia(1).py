#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data_from_json(file_path):
    with open('idmanual', 'r') as f:
        data = json.load(f)
    return data

def recommend_classes(input_description, train_data, vectorizer):
    train_features = vectorizer.transform([item['description'] for item in train_data])
    input_features = vectorizer.transform([input_description])
    similarity_scores = cosine_similarity(input_features, train_features)
    recommended_classes = train_data[similarity_scores.argmax()]['class_id']
    return recommended_classes

data = load_data_from_json('idmanual.json')
train_data, test_data = train_test_split(data, test_size=0.2)

vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform([item['description'] for item in train_data])


input_description = "Ear plugs for divers"
recommended_classes = recommend_classes(input_description, train_data, vectorizer)
print("Recommended classes:", recommended_classes)



# In[ ]:




