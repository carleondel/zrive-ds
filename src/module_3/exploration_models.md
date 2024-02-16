# Exploration Phase - Linear Models Only - Fast Proof of Concept (POC)

**GOAL: Create a Classification Model that can predict whether or not a person would buy an item we offered them (via push notification) based on behavioral and personal features of that user (user id, ordered before, etc), features of that specific order (date, etc) and features of the items themselves (popularity, price, avg days to buy, etc)**

We must notice that sending too many notifications would have a negative impact on user experience.


## Imports


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_curve, auc
```


```python
path = r'/home/carleondel/data-zrive-ds/box_builder_dataset/feature_frame.csv'
df = pd.read_csv(path)
```

## Data

This dataset contains 26 attributes based on the orders users make. Whenever an order is made, we take a picture of the whole inventory. The 'outcome' feature refers to the presence of each inventory's item in the order (0 for no presence, 1 for presence)

Content


- item id (variant_id)
- product type
- order id
- user id
- created at 
- order date
- sequence of the item within the order
- outcome: 0 for item not ordered, 1 for item ordered
- ordered before (0,1)
- abandoned before (0,1)
- active snoozed (0,1)
- set as regular (0,1)
- normalised price
- discount percentage
- vendor
- global popularity
- count adults
- count children
- count babies
- count pets
- people except babies
- days since last purchase of that item
- avg days to buy that item
- std days to buy that item
- days since last purchase of that product type
- avg days to buy that product type
- std days to buy that product type

### EDA & Visualizations


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
df.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>variant_id</th>
      <td>2880549.0</td>
      <td>3.401250e+13</td>
      <td>2.786246e+11</td>
      <td>3.361529e+13</td>
      <td>3.380354e+13</td>
      <td>3.397325e+13</td>
      <td>3.428495e+13</td>
      <td>3.454300e+13</td>
    </tr>
    <tr>
      <th>order_id</th>
      <td>2880549.0</td>
      <td>2.978388e+12</td>
      <td>2.446292e+11</td>
      <td>2.807986e+12</td>
      <td>2.875152e+12</td>
      <td>2.902856e+12</td>
      <td>2.922034e+12</td>
      <td>3.643302e+12</td>
    </tr>
    <tr>
      <th>user_id</th>
      <td>2880549.0</td>
      <td>3.750025e+12</td>
      <td>1.775710e+11</td>
      <td>3.046041e+12</td>
      <td>3.745901e+12</td>
      <td>3.812775e+12</td>
      <td>3.874925e+12</td>
      <td>5.029635e+12</td>
    </tr>
    <tr>
      <th>user_order_seq</th>
      <td>2880549.0</td>
      <td>3.289342e+00</td>
      <td>2.140176e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>4.000000e+00</td>
      <td>2.100000e+01</td>
    </tr>
    <tr>
      <th>outcome</th>
      <td>2880549.0</td>
      <td>1.153669e-02</td>
      <td>1.067876e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>ordered_before</th>
      <td>2880549.0</td>
      <td>2.113868e-02</td>
      <td>1.438466e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>abandoned_before</th>
      <td>2880549.0</td>
      <td>6.092589e-04</td>
      <td>2.467565e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>active_snoozed</th>
      <td>2880549.0</td>
      <td>2.290188e-03</td>
      <td>4.780109e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>set_as_regular</th>
      <td>2880549.0</td>
      <td>3.629864e-03</td>
      <td>6.013891e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>normalised_price</th>
      <td>2880549.0</td>
      <td>1.272808e-01</td>
      <td>1.268378e-01</td>
      <td>1.599349e-02</td>
      <td>5.394416e-02</td>
      <td>8.105178e-02</td>
      <td>1.352670e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>discount_pct</th>
      <td>2880549.0</td>
      <td>1.862744e-01</td>
      <td>1.934480e-01</td>
      <td>-4.016064e-02</td>
      <td>8.462238e-02</td>
      <td>1.169176e-01</td>
      <td>2.234637e-01</td>
      <td>1.325301e+00</td>
    </tr>
    <tr>
      <th>global_popularity</th>
      <td>2880549.0</td>
      <td>1.070302e-02</td>
      <td>1.663389e-02</td>
      <td>0.000000e+00</td>
      <td>1.628664e-03</td>
      <td>6.284368e-03</td>
      <td>1.418440e-02</td>
      <td>4.254386e-01</td>
    </tr>
    <tr>
      <th>count_adults</th>
      <td>2880549.0</td>
      <td>2.017627e+00</td>
      <td>2.098915e-01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>count_children</th>
      <td>2880549.0</td>
      <td>5.492182e-02</td>
      <td>3.276586e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
    </tr>
    <tr>
      <th>count_babies</th>
      <td>2880549.0</td>
      <td>3.538562e-03</td>
      <td>5.938048e-02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>count_pets</th>
      <td>2880549.0</td>
      <td>5.134091e-02</td>
      <td>3.013646e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>6.000000e+00</td>
    </tr>
    <tr>
      <th>people_ex_baby</th>
      <td>2880549.0</td>
      <td>2.072549e+00</td>
      <td>3.943659e-01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>days_since_purchase_variant_id</th>
      <td>2880549.0</td>
      <td>3.312961e+01</td>
      <td>3.707162e+00</td>
      <td>0.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.300000e+01</td>
      <td>3.300000e+01</td>
      <td>1.480000e+02</td>
    </tr>
    <tr>
      <th>avg_days_to_buy_variant_id</th>
      <td>2880549.0</td>
      <td>3.523734e+01</td>
      <td>1.057766e+01</td>
      <td>0.000000e+00</td>
      <td>3.000000e+01</td>
      <td>3.400000e+01</td>
      <td>4.000000e+01</td>
      <td>8.400000e+01</td>
    </tr>
    <tr>
      <th>std_days_to_buy_variant_id</th>
      <td>2880549.0</td>
      <td>2.645304e+01</td>
      <td>7.168323e+00</td>
      <td>1.414214e+00</td>
      <td>2.319372e+01</td>
      <td>2.769305e+01</td>
      <td>3.059484e+01</td>
      <td>5.868986e+01</td>
    </tr>
    <tr>
      <th>days_since_purchase_product_type</th>
      <td>2880549.0</td>
      <td>3.143513e+01</td>
      <td>1.227511e+01</td>
      <td>0.000000e+00</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>1.480000e+02</td>
    </tr>
    <tr>
      <th>avg_days_to_buy_product_type</th>
      <td>2880549.0</td>
      <td>3.088810e+01</td>
      <td>4.330262e+00</td>
      <td>7.000000e+00</td>
      <td>2.800000e+01</td>
      <td>3.100000e+01</td>
      <td>3.400000e+01</td>
      <td>3.950000e+01</td>
    </tr>
    <tr>
      <th>std_days_to_buy_product_type</th>
      <td>2880549.0</td>
      <td>2.594969e+01</td>
      <td>3.278860e+00</td>
      <td>2.828427e+00</td>
      <td>2.427618e+01</td>
      <td>2.608188e+01</td>
      <td>2.796118e+01</td>
      <td>3.564191e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(x='outcome', data=df)
```




    <Axes: xlabel='outcome', ylabel='count'>




    
![png](exploration_models_files/exploration_models_9_1.png)
    


We are considering all the items on inventory for each order. In the column 'outcome' we have 1 if the item was bought and 0 if it was not. Therefore, it is normal that most of our order_ids have an 'outcome' of 0.


```python
df['outcome'].value_counts()
```




    outcome
    0.0    2847317
    1.0      33232
    Name: count, dtype: int64



### Filtering 
We are asked to consider only those orders with 5 items or more to build the dataset to work with.


```python
# We group by 'order_id' and sum the outcomes
df_grouped = orders_filtered = df.groupby('order_id')['outcome'].sum().reset_index()
df_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2807985930372</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2808027644036</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2808099078276</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2808393957508</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2808429314180</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3441</th>
      <td>3643254800516</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3442</th>
      <td>3643274788996</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3443</th>
      <td>3643283734660</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3444</th>
      <td>3643294515332</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3445</th>
      <td>3643301986436</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>3446 rows × 2 columns</p>
</div>




```python
# We filter the orders by the outcome sum >= 5
filtered_orders = df_grouped[df_grouped['outcome'] >=5 ]['order_id']
filtered_orders
```




    0       2807985930372
    1       2808027644036
    2       2808099078276
    3       2808393957508
    5       2808434524292
                ...      
    3438    3643241300100
    3441    3643254800516
    3442    3643274788996
    3443    3643283734660
    3444    3643294515332
    Name: order_id, Length: 2603, dtype: int64




```python
# We filter the original df using the filtered order ids 
df_filtered = df[df['order_id'].isin(filtered_orders)]
```


```python
print(f"We have kept {100*len(df_filtered) / len(df):.2f} % of the original dataset")
```

    We have kept 75.12 % of the original dataset



```python
# Quick check to make sure we filtered properly
sum(df_filtered.groupby('order_id')['outcome'].sum() <5)
```




    0



### Dividing our columns


```python
df_filtered.columns
```




    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
           'order_date', 'user_order_seq', 'outcome', 'ordered_before',
           'abandoned_before', 'active_snoozed', 'set_as_regular',
           'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
           'count_adults', 'count_children', 'count_babies', 'count_pets',
           'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')




```python
categorical_cols = ['variant_id', 'product_type', 'order_id', 'user_id', 'vendor']
binary_cols = ['outcome', 'ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular',]
numerical_cols = [cols for cols in df_filtered.columns if cols not in (categorical_cols+binary_cols)]
```

----
----

# Machine Learning

## Feature engineering

- Decide how to encode the categorical variables. By now we are going to remove all the categorical variables in order to simplify the problem. 

- Extract features from the dates columns. We are only going to keep the 'created_at' column and extract the year, month, day and hour from it (the column 'order_date' contains redundant information)



```python
df_filtered['created_at']
```




    0          2020-10-05 16:46:19
    1          2020-10-05 17:59:51
    2          2020-10-05 20:08:53
    3          2020-10-06 08:57:59
    5          2020-10-06 10:50:23
                      ...         
    2880541    2021-03-03 12:56:04
    2880544    2021-03-03 13:19:28
    2880545    2021-03-03 13:57:35
    2880546    2021-03-03 14:14:24
    2880547    2021-03-03 14:30:30
    Name: created_at, Length: 2163953, dtype: object




```python
df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])
```

    /tmp/ipykernel_2827/3071389778.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['created_at'] = pd.to_datetime(df_filtered['created_at'])



```python
# We extract features from our date info
df_filtered['year'] = df_filtered['created_at'].dt.year
df_filtered['month'] = df_filtered['created_at'].dt.month
df_filtered['day'] = df_filtered['created_at'].dt.day
df_filtered['hour'] = df_filtered['created_at'].dt.hour
```

    /tmp/ipykernel_2827/1622433905.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['year'] = df_filtered['created_at'].dt.year
    /tmp/ipykernel_2827/1622433905.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['month'] = df_filtered['created_at'].dt.month
    /tmp/ipykernel_2827/1622433905.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['day'] = df_filtered['created_at'].dt.day
    /tmp/ipykernel_2827/1622433905.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_filtered['hour'] = df_filtered['created_at'].dt.hour



```python
categorical_cols
```




    ['variant_id', 'product_type', 'order_id', 'user_id', 'vendor']




```python
# cols i'm not going to use this time
unwanted = ['order_date'] + categorical_cols
```


```python
# We drop those columns 
df_filtered = df_filtered.drop(columns=unwanted)
```


```python
df_filtered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>global_popularity</th>
      <th>...</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-05 16:46:19</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.000000</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-05 17:59:51</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.000000</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-05 20:08:53</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.000000</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-06 08:57:59</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.038462</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-10-06 10:50:23</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.038462</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## Train | Validation | Test Split and Scaling

In order to avoid data leakage, since we are working with temporal data, we need to sort our dataset by date and divide it sequentially. We are going to take a 70-15-15 split for the dataset.



```python
# We order our dataset by date
df_filtered = df_filtered.sort_values(by= 'created_at', ascending= True)

# We drop the 'created_at' column 
df_filtered = df_filtered.drop(columns= 'created_at')
```


```python
df_filtered
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>...</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>481583</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.089184</td>
      <td>0.151976</td>
      <td>0.111111</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>30.0</td>
      <td>30.234265</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>23.827826</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2398555</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.043101</td>
      <td>1.006289</td>
      <td>0.037037</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>41.5</td>
      <td>28.238356</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.826713</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>478137</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.026837</td>
      <td>1.020202</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>54.0</td>
      <td>35.319072</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>30.506129</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2402001</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.129845</td>
      <td>0.041754</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>55.0</td>
      <td>34.085746</td>
      <td>30.0</td>
      <td>37.0</td>
      <td>27.032264</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1313555</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.032258</td>
      <td>0.260504</td>
      <td>0.008571</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>28.556854</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.826713</td>
      <td>2021</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>746573</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.121713</td>
      <td>0.066815</td>
      <td>0.013571</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>21.694590</td>
      <td>36.0</td>
      <td>27.0</td>
      <td>23.827826</td>
      <td>2021</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1563628</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.121713</td>
      <td>0.069042</td>
      <td>0.011429</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>21.0</td>
      <td>10.662606</td>
      <td>30.0</td>
      <td>27.0</td>
      <td>23.634873</td>
      <td>2021</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1926032</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.024126</td>
      <td>0.112360</td>
      <td>0.010000</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>38.0</td>
      <td>24.908885</td>
      <td>30.0</td>
      <td>31.0</td>
      <td>25.535369</td>
      <td>2021</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2880547</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
      <td>2021</td>
      <td>3</td>
      <td>3</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>2163953 rows × 24 columns</p>
</div>




```python
# We define the size of each dataset
train_size = int(0.7 * len(df_filtered))
val_size = int(0.15 * len(df_filtered))

# We divide the df after being sorted
df_train = df_filtered.iloc[:train_size]
df_val = df_filtered.iloc[train_size:train_size + val_size]
df_test = df_filtered.iloc[train_size + val_size:]

# We verify there is no data in common (Assertion error if it fails)
assert df_train.index.intersection(df_val.index).empty
assert df_val.index.intersection(df_test.index).empty
assert df_train.index.intersection(df_test.index).empty
```


```python
X_train = df_train.drop('outcome', axis=1)
y_train = df_train['outcome']

X_val = df_val.drop('outcome', axis=1)
y_val = df_val['outcome']

X_test = df_test.drop('outcome', axis=1)
y_test = df_test['outcome']
```


```python
# Initialize the scaler
scaler = StandardScaler()
```

Now we normalize the X train, test and validation data. We only fit to the training data in order to avoid data leakage (statistics coming from the val/test data).


```python
scaled_X_train = scaler.fit_transform(X_train)

scaled_X_val = scaler.transform(X_val)

scaled_X_test = scaler.transform(X_test)
```

## Logistic Regression Model


```python
# Initialize our model
model = LogisticRegression()

# Train our model
model.fit(scaled_X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression()</pre></div> </div></div></div></div>




```python
model.get_params()
```




    {'C': 1.0,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'l1_ratio': None,
     'max_iter': 100,
     'multi_class': 'auto',
     'n_jobs': None,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'lbfgs',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}



----


## Model Performance Evaluation
- Confusion Matrix
- Accuracy, Precision, etc

Now we use our validation dataset to improve the selection of hyperparameters
(Not yet)


```python

```


```python
y_pred = model.predict(scaled_X_val)
```


```python
confusion_matrix(y_val, y_pred)
```




    array([[319850,    348],
           [  4161,    233]])




```python
ConfusionMatrixDisplay.from_estimator(model,scaled_X_val,y_val)
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fb77c443510>




    
![png](exploration_models_files/exploration_models_45_1.png)
    



```python
print(classification_report(y_val, y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      0.99    320198
             1.0       0.40      0.05      0.09      4394
    
        accuracy                           0.99    324592
       macro avg       0.69      0.53      0.54    324592
    weighted avg       0.98      0.99      0.98    324592
    


We have a good performance on the negative instances, but a really poor performance for our positive cases. It can be explained by the existing inbalance of our classes in the dataset and also because we need to improve this model.

Out of all the predicted positives (orders predicted), only 40% are correct (precision) and out of all the true positives (true orders), we are only predicting 5% of them correctly (and predicting 95% as negatives) (recall).

### Performance Curves
- Precision-recall curve
- ROC curve

(Next steps) What are we comparing our model to? We can consider: choosing at random and choosing based on item popularity


```python
# We calculate the probabilities of being positive for the validation test

y_val_proba = model.predict_proba(scaled_X_val)[:, 1]
```


```python
# We calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
```


```python
# We plot the precision-recall curve

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```


    
![png](exploration_models_files/exploration_models_52_0.png)
    





```python
# ROC curve
fpr, tpr, thresholds =  roc_curve(y_val, y_val_proba)

# Calculate Area Under Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
```


    
![png](exploration_models_files/exploration_models_54_0.png)
    


We have an AUC of 0.81, much better than guessing at random. But this curve is not really informative in our case since we have very unbalanced classes
