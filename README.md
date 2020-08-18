
### DS/ML review

```
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder

file_name = 'input/wine-reviews/winemag-data-130k-v2.csv'

stock_data = pd.read_csv('something.csv')
             pd.read_csv(file_name)
```
### import with first column indexed
```stock_data = pd.read_csv('something.csv', index_col=0)```

### basic dataframe info
```
stock_data.shape
stock_data.describe()
stock_data.head()
stock_data.dtypes #OR
stock_data['ticker'].dtype 
```

### an array of the unique names/values
```stock_data.Ticker.unique()```

### a list of unique values and how often they occur
```stock_data.sell_date.value_counts() # how many stocks were sold each day```

### a different way of doing the previous
```
stock_data.groupby('sell_date').sell_date.count()
reviews.groupby('region').region.count()
```

### or get the cheapest wine in each region
```reviews.groupby('region').price.min()```

### access individual columns
```
stock_data.price
stock_data['price']
```

### indexed by numbers
```stock_data['price'][25]```

### indexed by ticker
```stock_data['price']['TSLA']```

### pandas indexing with iloc and loc
#### iloc - grabs the entire row based on index - can grab a column
```stock_data.iloc[25] # row of Tesla stock```
#### loc - grabs a column
```stock_data.loc[:, ['price', 'sell_date']]```

### change the index
```stock_data.set_index('Ticker')```

### selecting data based on a condition
```
stock_data.q3_growth > 0 # filters for stocks that have appreciated in the 3rd quarter
stock_data.loc[stock_data.q3_growth > 0] # filters a dataframe of these stocks
stock_data.loc[(stock_data.q3_growth > 0) & (stock_data.q2_growth > 0)] # filters for stocks that have appreciated for 2 quartes
```

### filter based on a condition or conditions
```
reviews.loc[(reviews.country == 'Italy') | (reviews.country == 'France') ]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
```

### filter out nulls
```reviews.loc[reviews.price.notnull()]```

### filter for just the nulls
```wine_reviews.loc[wine_reviews.price.isnull()]```

### assigning or building out new data - common practice for feature engineering
```wine_reviews['top_rated_regions'] = wine_reviews['country'] + ' - ' + wine_reviews['region_1']```

### recast a column as a different data type
```wine_reviews.points.astype('float64')  # convert from int to float``` 

### rename a column
```wine_reviews.rename(columns={'points': 'score'})```

### if we collected data based on country and the data was identically formatted
```pd.concat([us_wine_reviews, france_wine_reviews])```

### various pandas join methods for combining dataframes

### figure out what *percentage* of the values are missing
#### how many total missing values do we have?
```total_cells = np.product(wine_reviews.shape)```

#### get the number of missing data points per column
```
missing_values_count = wine_reviews.isnull().sum()
total_missing = missing_values_count.sum()
```

#### percent of data that is missing
```
percent_missing = (total_missing/total_cells) * 100
print(f'Percent missing: {percent_missing}')
```

### if it's an extremely small percentage of data with NaNs, drop those rows
```wine_reviews.dropna(subset=['variety'], inplace=True)```

### or remove a singular problematic column that's rife with NaNs or errors
```wine_reviews.drop(['region_2'], axis=1)```

### if there's a date present, it's a good idea to check if that column is being recognized as a date dtype
```stock_data['sale_date'].dtype```

### if not - parse the likely string type into a datetime object
### if it's a standard format like 2/8/18 or 23-10-1998
```stock_data['sale_date_2'] = pd.to_datetime(stock_data['sale_date'], format='%m/%d/%y')```

### then double check the reformat
```stock_data['sale_date_2'].head()```

### if you run into multiple date formats, try 'infering' - infer_datetime_format=True

### select and plot the day of the month that stocks were sold
```
stock_data_sell_dates = stock_data['sale_date_2'].dt.day
sns.distplot(stock_data_sell_dates, kde=False, bins=31)
```

### if strings - particularly names of something (country, city, etc...) - are similar
### use something like the fuzzywuzzy library to correct this


### ----------------------------------------------------------------------------------

### if you have datetime data and you're trying to model, try separating them into hour-day-month-year
### if you have categorical data - you'll prob need to one-hot-encode (multiple columns) or label-encode (single column)
#### may need to drop NaN's before transforming
```wine_reviews.dropna(subset=['country'], inplace=True)```

### Apply the label encoder to each column
```encoded = wine_reviews[cat_features].apply(encoder.fit_transform)

cat_features = ['country']
encoder = LabelEncoder() #from scikit-learn
```

### Apply the label encoder to each column
```
encoded = wine_reviews[cat_features].apply(encoder.fit_transform)
wine_reviews['country'] = wine_reviews[cat_features].apply(encoder.fit_transform)
```

### interactions or combining categorical columns/variables is a great way to feature engineer
```wine_reviews['country-region'] = wine_reviews['country'] + ' - ' + wine_reviews['region_1']```

When selecting and narrowing features for a model, there's 2 general approaches to take
Univariate methods which consider only one feature at a time or selecting all the best features at once with L1 (Lasso regression) or L2 (Ridge regression) regularization
- L1 - linear model
- L2 - penalizes the square of the coefficients


### visualize the data with the right type of graph (seaborn or matplotlib)

### -------------------------------------------------------------------------------------

### SQL query examples
```
query_1 = """
        SELECT COUNT(consecutive_number) AS num_accidents, 
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """

query_2 = """ 
            WITH time AS 
            (
                SELECT DATE(block_timestamp) AS trans_date
                FROM `bigquery-public-data.crypto_bitcoin.transactions`
            )
            SELECT COUNT(1) AS transactions,
                trans_date
            FROM time
            GROUP BY trans_date
            ORDER BY trans_date
            """
```

### -------------------------------------------------------------------------------------

### modeling steps: Define - Fit - Predict - Evaluate
#### Define
```
from sklearn.tree import DecisionTreeRegressor
melbourne_model = DecisionTreeRegressor(random_state=1)
```

#### Fit (after train - test split)
```
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
melbourne_model.fit(train_X, train_y)
```

### Predict
```melbourne_model.predict(X_val()```

### Evaluate (loads of metrics for this)
```mean_absolute_error(val_y, val_predictions)```

### Modeling
Decision Trees - parameters to play with - size of a node - depth of tree
Random forest - makes many trees and averages their predictions (distribution of sorts)
