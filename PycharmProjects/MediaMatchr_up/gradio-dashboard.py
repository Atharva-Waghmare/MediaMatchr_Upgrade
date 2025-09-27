import pandas as pd
import numpy as np
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

movies=pd.read_csv('movies_with_emotion_new.csv')
movies['large thumbnail']=movies['Poster_Url']+'&fife=800'
movies['large thumbnail']=np.where(
    movies['Poster_Url'].isna(),
    'cover-not-available.png',
    movies['large thumbnail']
)

raw_document=TextLoader('tagged_des_new.txt', encoding='utf-8').load()
text_splitter=CharacterTextSplitter(chunk_size=1,chunk_overlap=0,separator='\n')
documents=text_splitter.split_documents(raw_document)
db_movies=Chroma.from_documents(documents,
                               embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

def retrieve_semantic_recommendations(
        query:str,
        tone:str=None,
        initial_top_k=50,
        final_top_k=25,
)->pd.DataFrame:
    recs=db_movies.similarity_search(query, k=initial_top_k)
    movies_list=[int(rec.page_content.strip('"').split()[0]) for rec in recs]
    movies_recs=movies[movies['movie_id'].isin(movies_list)].head(final_top_k)

    if tone and tone != 'All':
        if tone == 'Happy':
            movies_recs = movies_recs.sort_values(by='joy', ascending=False)
        elif tone == 'Sad':
            movies_recs = movies_recs.sort_values(by='sadness', ascending=False)
        elif tone == 'Angry':
            movies_recs = movies_recs.sort_values(by='anger', ascending=False)
        elif tone == 'Suspenseful':
            movies_recs = movies_recs.sort_values(by='fear', ascending=False)
        elif tone == 'Surprising':
            movies_recs = movies_recs.sort_values(by='surprise', ascending=False)

    return movies_recs

def recommend_movies(
        query:str,
        tone:str=None,
        category: str = None
):

    recommendations=retrieve_semantic_recommendations(query, tone)
    if category and category != 'All':
        recommendations = recommendations[
            recommendations['listed_in'].str.contains(category, na=False)
        ].sort_values(by='Popularity', ascending=False)
    recommendations = recommendations.head(16)
    results=[]

    for _, row in recommendations.iterrows():
        description=row['description']
        truncated_desc_split=description.split()
        truncated_description=' '.join(truncated_desc_split[:30])+'...'


        caption=f"{row['title']} {truncated_description}"
        results.append((row['large thumbnail'], caption))
    return results

tones=['All']+['Happy', 'Sad', 'Angry', 'Suspenseful', 'Surprising']
genre_choices = ['All']+["Action", "Romance", "Horror", "Comedy", "Drama", "Sci-Fi", "Fantasy"]
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# MediaMatchr')

    with gr.Row():
        user_query=gr.Textbox(label='Please enter the description of a movie you like:',
                              placeholder='e.g., A superhero movie...')

    with gr.Row():
        category_dropdown = gr.Dropdown(choices=genre_choices, label='Select a Category', value='All')
        tone_description = gr.Dropdown(choices=tones, label='Select the emotional tone ', value='All')


    with gr.Row():
        submit_button = gr.Button('Find Movies',elem_classes="center-btn")


    gr.Markdown('## Recommended Movies')
    output=gr.Gallery(label='Recommended Movies',columns=8,rows=2)

    submit_button.click(fn=recommend_movies,
                           inputs=[user_query, tone_description,category_dropdown],
                           outputs=output)

if __name__=='__main__':
    dashboard.launch()