# Predicting the Primary Genre Of Movies/TV-Shows using Plot Text

## Introduction

Everyone loves binge-watching their favorite Movies and TV Shows. Nowadays, movies can pull elements from multiple genres (e.g., action, adventure, comedy, etc.) through complex themes intertwined within a single plot. For example, a movie can be primarily an action movie while also containing undercurrents of romance and comedy (for example, Thor: Ragnarok, Thor: Love and Thunder). The majority of online platforms (e.g., IMDB, rotten tomatoes, etc.) that maintain movie/tv-shows details include all genres BUT do not specifically mention a "primary genre." We seek to tackle this problem today by using machine learning to classify any movie/tv-show with a single, primary genre that best represents the title's plot.

## Commercial Applications

Highlighting a movie or tv show's primary genre can have many commercial applications, including improved content recommendation and increased precision in understanding various actors' performance and affinities per particular genres (and consequently, across niche fan bases).

## Methodology

As stated, we will use **machine learning** to **predict the primary genre** of movies/tv-shows. The majority of movies have a plot mentioned in a few lines of text that can be utilized for predicting target genres. This will be a **multi-label classification task** as we'll be predicting multiple genres per movie based on plot text. A simple **Logistic Regression** model was trained for this task. The dataset used for training the model is publicly available **CMU Movie Summary Corpus** dataset. The text of the plot is encoded using the **TF-IDF** text encoding method. The dataset was divided into train (90%) and test (10%) subsets. The evaluation metric used for performance evaluation was **F1-score**, which was **0.436** for test datasets which is quite a good score as we are able to recover the majority of genres. The model outputs probabilities for each genre, and we choose the genre with the highest probability as the "primary genre" of the particular movie/tv-show. The process's total code is present as a single class which can be run by simply initiating and calling the run() method.

## Individual Results (single example)

##### Final Kill (2020)

![Crime](https://github.com/gorfein/TV-and-Movie-Genre-Classification/blob/main/Images/Crime.png)

##### LIME Explanation

![Crime Explanation](https://github.com/gorfein/TV-and-Movie-Genre-Classification/blob/main/Images/Crime%20-%20explanation.png)

## High-level Results

![High-level Summary of Results](https://github.com/gorfein/TV-and-Movie-Genre-Classification/blob/main/Images/High%20level.png)

## Individual Results (single example)

##### Space Pirate Captain Harlock (1978-1979)

![Sci-Fi](https://github.com/gorfein/TV-and-Movie-Genre-Classification/blob/main/Images/Science%20Fiction.png)

##### LIME Explanation

![Sci-Fi Explanation](https://github.com/gorfein/TV-and-Movie-Genre-Classification/blob/main/Images/Science%20Fiction%20-%20explanation.png)
