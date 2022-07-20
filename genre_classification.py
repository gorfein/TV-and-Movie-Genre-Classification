import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score


class PredictPrimaryGenre:

    def __init__(self, threshold = 0, file_name="PeerLogix Titles (IMDb Metadata).csv"):

        self.file_to_predict = file_name

        self.meta_file = "MovieSummaries/movie.metadata.tsv"
        self.plot_file = "MovieSummaries/plot_summaries.txt.gz"
        self.peerlogix_genres_file = "peerlogix_genres.csv"

        self.only_peerlogix_valid_genres = True

        self.valid_genres = [
            'Action', 'Action & Adventure', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
            'Drama', 'Family', 'Fantasy', 'Horror', 'Musical', 'Mystery', 'Romance', 'Science Fiction',
            'Thriller', 'War', ]

        self.valid_genres = pd.read_csv(self.peerlogix_genres_file).genre.unique().tolist() ## This one has around 8-10 more unique labels

        self.genre_corrections = {
                            'Action/Adventure' : 'Action & Adventure',
                            'Crime Fiction' : 'Crime',
                            'Family Film' : 'Family',
                            'Romance Film' : 'Romance',
                            'War Film' : 'War',
                            'Comedy Film' : 'Comedy'
                            }

        self.threshold = threshold
        self.max_df = 0.8
        self.stop_words = "english"
        self.max_features = 50000

        print('Predictor Initialized.')


    def loadMetaData(self):
        self.meta = pd.read_csv(self.meta_file, sep = '\t', header = None)
        self.meta.columns = ["movie_id",1,"movie_name",3,4,5,6,7,"genre"]
        self.meta['movie_id'] = self.meta['movie_id'].astype(str)

        print("Loading of Meta Data Complete.")

    def loadMoviePlotsData(self):
        plots = []

        with gzip.open(self.plot_file, 'rt', encoding = 'UTF-8') as f:
            reader = csv.reader(f, dialect='excel-tab')
            for row in tqdm(reader):
                plots.append(row)

        movie_id = []
        plot = []

        # extract movie Ids and plot summaries
        for i in tqdm(plots):
            movie_id.append(i[0])
            plot.append(i[1])

        # create dataframe
        self.movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})

        print("Loading of Movies Plot Data Complete.")

    def loadExistingPeerLogixGenres(self):
        existing_peerlogix_genres = pd.read_csv(self.peerlogix_genres_file)
        existing_peerlogix_genres = existing_peerlogix_genres.groupby("imdb_id").aggregate(lambda x: list(x)).reset_index()
        self.existing_peerlogix_genres = dict(zip(existing_peerlogix_genres["imdb_id"].values.tolist(), existing_peerlogix_genres["genre"].values.tolist()))

        print("Loading of Existing PeerLogix Genres Complete.")

    def mergeMovieAndPlotsData(self):
        self.movies = pd.merge(self.movies, self.meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

        print("Merging of Genre and Plots Data Complete.")

    def cleanGenreData(self):
        genres = []

        for i in self.movies['genre']:
            movie_genres = list(json.loads(i).values())

            movie_genres = [self.genre_corrections.get(genre, genre) for genre in movie_genres] ## Genre Correction in Data

            if self.only_peerlogix_valid_genres: ## Keep only peerlgix valid genres
                movie_genres = [genre for genre in movie_genres if genre in self.valid_genres]

            genres.append(movie_genres)

        self.movies['genre_new'] = genres

        print("Cleaning of Genre Data Complete.")

    def cleanPlotData(self):
        self.movies['clean_plot'] = self.movies['plot'].apply(lambda x: " ".join(re.findall("[a-zA-Z]+", x.lower())))

        print("Cleaning of Plots Data Complete.")

    def LoadPeerLogixFileToPredictPrimaryGenre(self):
        self.peerlogix = pd.read_csv(self.file_to_predict)

        print("Loading of Prediction File Complete.")

    def LoadAndCleanData(self):
        self.loadMetaData()
        self.loadMoviePlotsData()
        self.loadExistingPeerLogixGenres()
        self.mergeMovieAndPlotsData()
        self.cleanGenreData()
        self.cleanPlotData()
        self.LoadPeerLogixFileToPredictPrimaryGenre()

    def PrepareMultiLabelTargetValues(self):
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.multilabel_binarizer.fit(self.movies['genre_new'])

        # transform target variable
        self.Y = self.multilabel_binarizer.transform(self.movies['genre_new'])

        print("Preparation of Multi-Label Target Complete.")

    def PrepareTrainTestSplit(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.movies['clean_plot'], self.Y, test_size=0.1, random_state=123)

        print("Train Test Split Complete.")

    def TransformPlotsFromTextToFloats(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=self.max_df, stop_words=self.stop_words, max_features=self.max_features)

        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(self.X_test)

        print("Text Data Vectorization Complete.")

    def PrepreDataForMLModel(self):
        self.PrepareMultiLabelTargetValues()
        self.PrepareTrainTestSplit()
        self.TransformPlotsFromTextToFloats()


    def TrainClassifier(self):
        lr = LogisticRegression(max_iter=500)
        self.clf = OneVsRestClassifier(lr)

        self.clf.fit(self.X_train_tfidf, self.Y_train)

    def EvaluateClassifier(self):
        Y_test_pred = self.clf.predict(self.X_test_tfidf)
        print("F1-Score : {:.3f}".format(f1_score(self.Y_test, Y_test_pred, average="micro")))

    def PredictGenre(self, text):
        if isinstance(text, str):
            cleaned_text = " ".join(re.findall("[a-zA-Z]+", text.lower()))
            X = self.tfidf_vectorizer.transform([cleaned_text])
            probs = self.clf.predict_proba(X) ## Probabilities/ liklihood

            # Isolate highest probable genre
            primary_genre_idx = probs.argsort()[0][-1] ## Taking highest probability/liklihood
            idx2, idx3, idx4, idx5 = probs.argsort()[0][-2], probs.argsort()[0][-3], probs.argsort()[0][-4], probs.argsort()[0][-5]

            # Discard results if none were above threshold
            if probs[0][primary_genre_idx] < self.threshold:
                return ["NA", ] * 5

            # Else, return top genre
            primary_genre = self.multilabel_binarizer.classes_[primary_genre_idx]
            genre2, genre3, genre4, genre5 = self.multilabel_binarizer.classes_[idx2], self.multilabel_binarizer.classes_[idx3], self.multilabel_binarizer.classes_[idx4], self.multilabel_binarizer.classes_[idx5]

            primary_genre = self.genre_corrections.get(primary_genre, primary_genre)
            genre2, genre3, genre4, genre5 = self.genre_corrections.get(genre2, genre2), self.genre_corrections.get(genre3, genre3), self.genre_corrections.get(genre4, genre4), self.genre_corrections.get(genre5, genre5)
            return primary_genre, genre2, genre3, genre4, genre5
        else:
            return ["NA", ] * 5

    def PredictGenreForFile(self):
        primary_genre = []

        for i, (imdb_id, plot) in enumerate(self.peerlogix[["imdb_id","description"]].values):
            existing_genres = self.existing_peerlogix_genres.get(imdb_id, []) ## Retrieve Existing Genres for id
            predicted_genres = self.PredictGenre(plot) ## Make Prediction on Plot

            if existing_genres: ## If Genres present for IMDB ID then choose from it else append predicted one.
                if len(existing_genres) == 1: ## If single Genre then it'll be primary Genre
                    primary_genre.append(existing_genres[0])
                else:
                    selected_genre = None
                    for genre in predicted_genres: ### Check for predicted Genre in existing Genres
                        if genre in existing_genres:
                            selected_genre = genre
                            break
                    primary_genre.append(selected_genre if selected_genre else "NA")

            else:
                primary_genre.append(predicted_genres[0]) ## Append first one which is primary

            if (i+1)%5000 == 0:
                print("{} iteration completed".format(i+1))

        self.peerlogix["Primary_Genre1"] = primary_genre

    def run(self):

        # Download the training data
        self.LoadAndCleanData()
        self.PrepreDataForMLModel()

        print("\n=========== Data Loading and Cleaning Complete =============\n")

        # train the classifier
        self.TrainClassifier()
        self.EvaluateClassifier()

        print("\n=========== Model Training and Evaluation Complete =============\n")

        # Loop through each and choose primary genre
        self.PredictGenreForFile()

        print("\n=========== Prediction of Genre Complete =============\n")

        # Save CSV locally
        self.peerlogix.to_csv("file_with_genre.csv")

        return self.peerlogix
