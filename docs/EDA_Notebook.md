# EDA for Amazon Books Reviews

## Dataset Overview

This dataset contains **2 files**.

### 1. Books Reviews File

The first file contains feedback from approximately **3 million users** on **212,404 unique books**.

This dataset is part of the **Amazon Review Dataset**, which contains product reviews and metadata from Amazon, including **142.8 million reviews** spanning from **May 1996 to July 2014**.

### Features

| Feature | Description |
|---|---|
| `Id` | The ID of the book |
| `Title` | Book title |
| `Price` | Price of the book |
| `User_id` | ID of the user who rated the book |
| `profileName` | Name of the user who rated the book |
| `review/helpfulness` | Helpfulness rating of the review, e.g. `2/3` |
| `review/score` | Rating from 0 to 5 for the book |
| `review/time` | Time when the review was given |
| `review/summary` | Summary of the text review |
| `review/text` | Full text of the review |

---

### 2. Books Details File

The second file, **Books Details**, contains detailed information about **212,404 unique books**.

The file was built using the **Google Books API** to retrieve additional information about books appearing in the reviews dataset.

### Features

| Feature | Description |
|---|---|
| `Title` | Book title |
| `description` | Description of the book |
| `authors` | Names of the book authors |
| `image` | URL of the book cover |
| `previewLink` | Link to access the book on Google Books |
| `publisher` | Name of the publisher |
| `publishedDate` | Publication date |
| `infoLink` | Link to more information about the book on Google Books |
| `categories` | Genres/categories of the book |
| `ratingsCount` | Number of ratings for the book |

---

# Import the Data

## Install Required Packages

```python
!pip install missingno
!pip install wordcloud
```

## Import Libraries

```python
import numpy as np
import pandas as pd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud
from IPython.core.display import display, HTML

from skimage import io
from PIL import Image

import requests
from io import BytesIO
```

---

# Load Data

```python
df_books = pd.read_csv(
    '../input/amazon-books-reviews/books_data.csv'
)

df_books
```

```python
df_reviews = pd.read_csv(
    '../input/amazon-books-reviews/Books_rating.csv'
)

df_reviews
```

---

# Exploratory Data Analysis

## Display a Book Sample

```python
def book_sample(row):
    print("##########################################")
    print('\t\t\t{}\t\t\n\n'.format(row['Title']))

    print("description : " + str(row['description']))
    print("\n\n authors are " + str(row['authors']))

    try:
        io.imshow(io.imread(row['image']))
    except:
        print(
            "can't read image maybe it's Null "
            "or don't have internet connection"
        )

    try:
        print("\n\n previewLink is ")
        display(row['previewLink'])
    except:
        print(
            "can't read previewLink maybe it's Null "
            "or don't have internet connection"
        )

    try:
        print("\n\n infoLink is ")
        display(row['infoLink'])
    except:
        print(
            "can't read infoLink maybe it's Null "
            "or don't have internet connection"
        )

    print("\n\n publisher is " + str(row['publisher']))
    print("\n\n published Date is " + str(row['publishedDate']))
    print("\n\n categories are " + str(row['categories']))
    print("\n\n rating is " + str(row['ratingsCount']))

    print("##########################################")
```

Example:

```python
book_sample(df_books.iloc[1, :])
```

---

## Display a Review Sample

```python
def review_sample(row):
    print("##########################################")
    print('\t\t\t{}\t\t\n\n'.format(row['Title']))

    print("\n\n User_id is " + str(row['User_id']))
    print("\n\n profileName is " + str(row['profileName']))
    print(
        "\n\n review/helpfulness is "
        + str(row['review/helpfulness'])
    )
    print("\n\n review/score is " + str(row['review/score']))
    print("\n\n review/time is " + str(row['review/time']))
    print(
        "\n\n review/summary is "
        + str(row['review/summary'])
    )
    print("\n\n review/text are " + str(row['review/text']))

    print("##########################################")
```

Example:

```python
review_sample(df_reviews.iloc[1, :])
```

---

# Dataset Shape

```python
df_books.shape
```

```python
df_reviews.shape
```

---

# Dataset Information

```python
df_reviews.info()
```

```python
df_books.info()
```

---

# Dataset Columns

```python
df_reviews.columns
```

```python
df_books.columns
```

---

# Check Missing Values

## Missing Values in Reviews Dataset

```python
df_reviews.isnull().sum()
```

Visualize missing values:

```python
msno.matrix(
    df_reviews,
    color=(0.99, 0.75, 0.023)
)
```

## Missing Values in Books Dataset

```python
df_books.isnull().sum()
```

Visualize missing values:

```python
msno.matrix(
    df_books,
    color=(0.99, 0.75, 0.023)
)
```

---

# Compare Rating Class Sizes

```python
colors = [
    'gold',
    'mediumturquoise',
    'brown'
]

labels = (
    df_reviews['review/score']
    .value_counts()
    .keys()
    .map(str)
)

values = (
    df_reviews['review/score'].value_counts()
    / df_reviews['review/score'].value_counts().shape[0]
)

fig = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=.3
        )
    ]
)

fig.update_traces(
    hoverinfo='label+percent',
    textinfo='percent',
    textfont_size=20,
    marker=dict(
        colors=colors,
        line=dict(
            color='white',
            width=0.1
        )
    )
)

fig.show()
```

---

# Browse Reviews for Individual Books

Group reviews by book title:

```python
groups = df_reviews.groupby('Title')
```

Display a book:

```python
book_sample(df_books.iloc[4])
```

Display all reviews associated with a selected book:

```python
groups.get_group(
    df_books.loc[15, 'Title']
)
```

---

# Number of Reviews per Book

```python
user_per_book = (
    df_reviews
    .groupby('Title')['User_id']
    .count()
)

user_per_book = user_per_book.sort_values(
    ascending=False
)
```

Visualize the **50 books with the most reviews**:

```python
fig = px.bar(
    user_per_book.head(50)
)

fig.show()
```

---

# Explore the Most Reviewed Book

Display information about the most reviewed book:

```python
book_sample(
    df_books[
        df_books['Title'] == user_per_book.keys()[0]
    ]
)
```

Get its reviews:

```python
df_habbit = groups.get_group(
    user_per_book.keys()[0]
)
```

Check the shape:

```python
df_habbit.shape
```

---

# Rating Distribution for the Most Reviewed Book

```python
colors = [
    'gold',
    'mediumturquoise',
    'brown'
]

labels = (
    df_habbit['review/score']
    .value_counts()
    .keys()
    .map(str)
)

values = (
    df_habbit['review/score'].value_counts()
    / df_habbit['review/score'].value_counts().shape[0]
)

fig = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=.3
        )
    ]
)

fig.update_traces(
    hoverinfo='label+percent',
    textinfo='percent',
    textfont_size=20,
    marker=dict(
        colors=colors,
        line=dict(
            color='white',
            width=0.1
        )
    )
)

fig.show()
```

---

# Word Cloud from Book Reviews

Download an image to use as a mask for the word cloud:

```python
response = requests.get(
    "https://thumbs.dreamstime.com/b/"
    "open-book-silhouette-black-white-open-book-silhouette-black"
)

mask = np.array(
    Image.open(
        BytesIO(response.content)
    )
)
```

Define the word cloud function:

```python
def generate_better_wordcloud(data, title, mask=None):

    cloud = WordCloud(
        scale=3,
        max_words=200,
        background_color="white",
        mask=mask,
        collocations=False,
        contour_color='black',
        contour_width=3
    ).generate(data)

    plt.figure(figsize=(10, 8))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
```

Generate a word cloud:

```python
generate_better_wordcloud(
    df_habbit.iloc[0]['review/text'],
    "Sample from Text Reviews",
    mask
)
```