import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("Cleaned_Audible_Catalog.csv")

# TF-IDF for content-based similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Book Name']).drop_duplicates()

# Recommend books by content similarity
def recommend_books(title, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame({'Message': [f"'{title}' not found."]})
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return df.iloc[book_indices][['Book Name', 'Author', 'Genre', 'Rating']]

# Streamlit UI
st.title("📚 Audible Book Recommendation App")

# Sidebar navigation
option = st.sidebar.radio("Go to", ['📖 Recommend by Book', '🎯 Recommend by Genre', '📊 EDA'])

# Page 1: Recommend by Book
if option == '📖 Recommend by Book':
    st.header("🔍 Find Similar Books")
    book_title = st.selectbox("Choose a book you like", sorted(df['Book Name'].unique()))
    if st.button("Recommend"):
        results = recommend_books(book_title)
        st.subheader("📚 Recommended Books:")
        st.dataframe(results)

# Page 2: Recommend by Genre
elif option == '🎯 Recommend by Genre':
    st.header("🎧 Genre-Based Book Finder")
    genre = st.selectbox("Choose a genre", sorted(df['Genre'].dropna().unique()))
    top_n = st.slider("Number of books to show", 1, 10, 5)
    filtered = df[df['Genre'] == genre].sort_values(by="Rating", ascending=False).head(top_n)
    st.dataframe(filtered[['Book Name', 'Author', 'Rating', 'Listening Time (min)']])

# Page 3: EDA
elif option == '📊 EDA':
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Distribution of Ratings")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Rating'], bins=10, kde=True, ax=ax1, color="skyblue")
    st.pyplot(fig1)

    st.subheader("Top Genres")
    top_genres = df['Genre'].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax2, palette="viridis")
    st.pyplot(fig2)
