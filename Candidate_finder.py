import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

job_description = "We are looking for a skilled UI Developer to join our dynamic team. The ideal candidate will have a strong background in front-end development, with proficiency in HTML, CSS, JavaScript, and modern frameworks like React or Angular. Your primary responsibility will be to create visually appealing and user-friendly web interfaces that enhance user experience and align with our brand guidelines."

df_candidates = pd.read_csv('candidates.csv')


def concatenate_columns(row):
    return " ".join([str(row[col]) for col in ['Job Skills', 'Projects', 'Comments'] 
                     if pd.notnull(row[col])])

df_candidates['combined_keywords'] = df_candidates.apply(concatenate_columns, axis=1)


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_candidates['combined_keywords'])
query_tfidf = vectorizer.transform([job_description])


cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()



top_indices = cosine_similarities.argsort()[-10:][::-1]


matching = df_candidates.iloc[top_indices]


table = matching[matching.columns[:3]]


print("Top 10 candidates which will be suitable for the job description are :\n ", table)




