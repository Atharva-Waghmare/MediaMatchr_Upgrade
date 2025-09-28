
# MediaMatchr:
This project is an intelligent movie recommendation system where users can search for movies by description or filter by category and tone. It uses transformer models to turn descriptions into embeddings, stores them in ChromaDB for fast search, and leverages LangChain to handle retrieval. By combining semantic similarity with emotional tone and genres, it delivers accurate and personalized movie suggestions.
## Features 
- Personalized Suggestions: Users can search movies by description and refine results using category and emotional tone.

- NLP-Powered Search: Movie descriptions and queries are embedded with transformer models for semantic matching.

- Vector Database: ChromaDB enables fast similarity searches and efficient retrieval.

- Hybrid Filtering: Combines semantic similarity with mood-to-genre filtering for more accurate recommendations.

- Interactive UI: Built with Gradio for a simple, intuitive, and visually engaging experience.

- Robust Backend: Powered by Python, pandas, numpy, and LangChain for seamless integration and data handling.
## Techstack
- Programming Language: Python

- Libraries & Frameworks:

    - LangChain – Orchestrating embeddings and retrieval pipeline

    - ChromaDB – Vector database for similarity search

    - Hugging Face Transformers – Embedding model (all-MiniLM-L6-v2)

    - Gradio – Interactive user interface

    - Pandas & Numpy – Data preprocessing and manipulation

Dataset: https://www.kaggle.com/datasets/disham993/9000-movies-dataset

Environment: Pycharm
## Installation
- Clone the repository
```
git clone https://github.com/Atharva-Waghmare/MediaMatchr_Upgrade
cd PycharmProjects/MediaMatchr_up
```

- Create a virtual environment (recommended)
```
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

- Install dependencies
```
pip install -r requirements.txt
```

- Add datasets

  Place movies_with_emotion_new.csv and tagged_des_new.txt inside the project folder.
  
- Run the app
```
python app.py
```

or if the entry file is different, replace app.py with your script name (e.g., main.py).

- Open in browser

  Gradio will provide a local URL (e.g., http://127.0.0.1:7860/)

  Open it in your browser to use the app.
    
## Usage
- Enter a movie description in the search box (e.g., "A thrilling space adventure with emotional depth").
- Select a genre and emotional tone to refine recommendations.
- Click "Find Movies" to view recommended movies with posters and brief descriptions.
## Future Improvements
- Add support for book recommendations using similar description-based search.
- Integrate TV shows and web series into the recommendation engine.
- Enhance search with advanced filters (release year, language, ratings).
- Personalize recommendations based on user profiles and history.
Improve UI/UX with more interactive elements and better visuals.
