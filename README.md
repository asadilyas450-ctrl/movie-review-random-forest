# Movie Review Sentiment Analyzer — Random Forest

A Streamlit app that classifies movie/product/restaurant review text as Positive or Negative.

## Model
- Dataset: UCI Sentiment Labelled Sentences (IMDb, Yelp, Amazon review sentences; 3,000 labelled rows).
- Text features: TF-IDF, unigrams and bigrams.
- Classifier: RandomForestClassifier (300 trees).
- Evaluation: stratified 80/20 train/test split, random_state=42.
- The app calculates and displays the actual held-out test accuracy, classification report, and confusion matrix at startup.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and this README to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, set the main file path to `app.py`, then deploy.
4. Wait for the build to finish. Streamlit will show your live app URL.

The dataset is downloaded from the UCI Machine Learning Repository when the app starts; no separate dataset upload is needed.
