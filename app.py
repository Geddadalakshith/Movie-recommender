import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import joblib
import os

DATA_FILE = 'imdb_top_1000.csv'
CACHE_FILE = 'tfidf_cache.joblib'
RECOMMEND_PER_PAGE = 5

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.head(300).reset_index(drop=True)
    df.fillna({
        'Genre': '', 'Star1': '', 'Star2': '', 'Star3': '', 'Star4': '',
        'IMDBRating': 0, 'ReleasedYear': 'Unknown', 'PosterLink': '',
        'SeriesTitle': 'Unknown', 'Director': 'Unknown', 'Overview': ''
    }, inplace=True)
    df.rename(columns={
        'Series_Title': 'SeriesTitle',
        'IMDB_Rating': 'IMDBRating',
        'Released_Year': 'ReleasedYear',
        'Poster_Link': 'PosterLink'
    }, inplace=True)
    return df

def combine_features(row):
    return f"{row['Genre']} {row['Star1']} {row['Star2']} {row['Star3']} {row['Star4']}"

@st.cache_resource
def compute_similarity(df):
    combined = df.apply(combine_features, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined)
    sim = cosine_similarity(tfidf_matrix)
    joblib.dump((tfidf_matrix, sim), CACHE_FILE)
    return sim

def load_similarity():
    if os.path.exists(CACHE_FILE):
        _, sim = joblib.load(CACHE_FILE)
        return sim
    return None

def fuzzy_search(query, choices, limit=10, score_cutoff=60):
    return [match[0] for match in process.extract(query, choices, limit=limit, score_cutoff=score_cutoff)]

def get_recommendations(title, df, sim, genre_filter=None):
    if title not in df['SeriesTitle'].values:
        return pd.DataFrame()
    idx = df[df['SeriesTitle'] == title].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:51]
    indices = [i[0] for i in scores]
    recs = df.iloc[indices].reset_index(drop=True)
    if genre_filter and genre_filter != "All":
        recs = recs[recs['Genre'].str.contains(genre_filter)]
    return recs.reset_index(drop=True)

for key in ['page', 'search_term', 'selected_movie', 'genre_filter', 'recommendations']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'selected_movie' else '' if key == 'search_term' else 1 if key == 'page' else 'All' if key == 'genre_filter' else pd.DataFrame()

def reset_state():
    st.session_state.page = 1
    st.session_state.search_term = ''
    st.session_state.selected_movie = None
    st.session_state.genre_filter = "All"
    st.session_state.recommendations = pd.DataFrame()

def next_page():
    if st.session_state.recommendations is not None:
        total = len(st.session_state.recommendations)
        total_pages = (total + RECOMMEND_PER_PAGE - 1) // RECOMMEND_PER_PAGE
        if st.session_state.page < total_pages:
            st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

st.set_page_config(page_title="Movie Recommender with Genre and Pagination", layout="wide")
st.markdown('<h1 style="color:#4B0082;">🎬 Movie Recommender System 🎥</h1>', unsafe_allow_html=True)
st.write("Genre filtering, fuzzy search, and pagination supported.")

df = load_data()
sim = load_similarity()
if sim is None:
    with st.spinner("Computing similarity matrix..."):
        sim = compute_similarity(df)

genre_list = sorted(set(g.strip() for s in df['Genre'] for g in s.split(',')))
genre_list = ["All"] + genre_list

st.sidebar.header("🔍 Search & Filters")
genre_filter = st.sidebar.selectbox("Filter by Genre", genre_list, index=genre_list.index(st.session_state.genre_filter) if st.session_state.genre_filter in genre_list else 0)
st.session_state.genre_filter = genre_filter

search_input = st.sidebar.text_input("Search a movie", value=st.session_state.search_term)
if search_input != st.session_state.search_term:
    st.session_state.search_term = search_input
    st.session_state.page = 1

df_filtered = df
if genre_filter != "All":
    df_filtered = df[df['Genre'].str.contains(genre_filter)]

choices = df_filtered['SeriesTitle'].tolist()
search_results = fuzzy_search(search_input, choices) if search_input else choices

selected_movie = st.sidebar.selectbox("Select a movie", search_results,
                                      index=search_results.index(st.session_state.selected_movie) if st.session_state.selected_movie in search_results else 0)
if selected_movie != st.session_state.selected_movie:
    st.session_state.selected_movie = selected_movie
    st.session_state.page = 1

sort_option = st.sidebar.selectbox("Sort recommendations by", ["Default", "IMDB Rating", "Release Year"])

if st.sidebar.button("Reset Filters", key="reset_filters"):
    reset_state()

if st.sidebar.button("Show Recommendations", key="show_recommendations") and selected_movie:
    recs = get_recommendations(selected_movie, df, sim, genre_filter=genre_filter if genre_filter != "All" else None)
    st.session_state.recommendations = recs.reset_index(drop=True)
    st.session_state.page = 1

recs = st.session_state.recommendations

if recs is not None and not recs.empty:
    if sort_option == "IMDB Rating":
        recs = recs.sort_values(by='IMDBRating', ascending=False)
    elif sort_option == "Release Year":
        recs = recs.sort_values(by='ReleasedYear', ascending=False)

    total = len(recs)
    total_pages = (total + RECOMMEND_PER_PAGE - 1) // RECOMMEND_PER_PAGE
    page = st.session_state.page
    start_idx = (page - 1) * RECOMMEND_PER_PAGE
    end_idx = start_idx + RECOMMEND_PER_PAGE
    recs_page = recs.iloc[start_idx:end_idx]

    st.markdown(f"### Recommendations for **{selected_movie}** (Page {page}/{total_pages})")
    for _, row in recs_page.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                if row['PosterLink']:
                    st.image(row['PosterLink'], width=160)
                else:
                    st.write("No Image")
            with col2:
                st.markdown(f"<h3>{row['SeriesTitle']}</h3>", unsafe_allow_html=True)
                st.write(f"Year: {row['ReleasedYear']} | Rating: {row['IMDBRating']}")
                st.write(f"Genre: {row['Genre']}")
                st.write(f"Director: {row['Director']}")
                st.write(f"Main Cast: {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}")
                with st.expander("Plot Summary"):
                    st.write(row['Overview'])
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", disabled=page == 1, key="prev_page"):
            prev_page()
    with col2:
        if st.button("Next", disabled=page == total_pages, key="next_page"):
            next_page()
else:
    if st.sidebar.button("Show Recommendations", key="show_recommendations_empty"):
        st.warning("No recommendations found.")

st.markdown('<div style="text-align:center;">© 2025 Movie Recommender with Pagination & Keys</div>', unsafe_allow_html=True)
