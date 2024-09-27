```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# load the data saved in local downloaded from Amazon S3
inventory = pd.read_parquet('../../../data-zrive-ds/sampled-datasets/inventory.parquet')
abandoned_carts = pd.read_parquet('../../../data-zrive-ds/sampled-datasets/abandoned_carts.parquet')
orders = pd.read_parquet('../../../data-zrive-ds/sampled-datasets/orders.parquet')
regulars = pd.read_parquet('../../../data-zrive-ds/sampled-datasets/regulars.parquet')
users = pd.read_parquet('../../../data-zrive-ds/sampled-datasets/users.parquet')
```

## Part 1. Understanding the dataset

The data is partitioned over multiple datasets and comes from a groceries e-
commerce platform selling products directly to consumers (think of it as an online
supermarket):

- **orders**: An order history of customers. Each row is an order and the
item_ids for the order are stored as a list in the item_ids column.

- **regulars**: Users are allowed to specify items that they wish to buy
regularly. This data gives the items each user has asked to get regularly, along
with when they input that information.

- **abandoned_cart**: If a user has added items to their basket but not
bought them, we capture that information. Items that were abandoned are stored
as a list in item_ids.

- **inventory**: Some information about each item_id.

- **users**: Information about users.


#### STRUCTURE:
1. First we will check each of the tables we will be working with in order to understand deeply the information. Then we can start thinking about creating hypothesis.

2. We now look for null values, weird outliers and make sure our data is OK. Then we make decisions and justify why we impute values or delete certain features/rows.

3. Once cleaned, we join tables and create new features.

4. We test the hypothesis we created.


### Inventory
Info about each item id


```python
inventory.head()
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
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>




```python
inventory.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB


No null data in inventory

### Abandoned Carts
If a user has added items to their basket but not
bought them, we capture that information. Items that were abandoned are stored
as a list in item_ids.


```python
abandoned_carts.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>




```python
abandoned_carts.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB


No null data in abandoned_carts. Weird indexes, it's probably a sample of the whole dataset.

### Orders
An order history of customers. Each row is an order and the
items ids of each order are stored in a list in the ordered_items column.


```python
orders.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>




```python
orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB


The orders dataset is probably also a sample. No null data.

### Regulars
Users are allowed to specify items that they wish to buy
regularly. This data contains the items each user has asked to get regularly, along with the date when they input that information.


```python
regulars.head()
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
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>




```python
regulars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB


regulars is also a sample. No null data.

### Users
Info about users


```python
users.head()
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
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB


Users contains interesting data. Nuts are a way of dividing the UK in small and similar sized regions. Many null values in the 'count' categories. If this data comes from optional answers we may be taking some risk of skewing heavily our sample. Again, we can guess this is a sample based on the indexes.

There are also less nuts than user ids, so some people don't have any nut assigned. 

Let's get some info about our missing values


```python
users.isnull().sum()
```




    user_id                     0
    user_segment                0
    user_nuts1                 51
    first_ordered_at            0
    customer_cohort_month       0
    count_people             4658
    count_adults             4658
    count_children           4658
    count_babies             4658
    count_pets               4658
    dtype: int64




```python
users[['user_nuts1', 'count_people','count_adults', 'count_children', 'count_babies', 'count_pets']].describe()
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
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
      <td>325.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.787692</td>
      <td>2.003077</td>
      <td>0.707692</td>
      <td>0.076923</td>
      <td>0.636923</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.365753</td>
      <td>0.869577</td>
      <td>1.026246</td>
      <td>0.289086</td>
      <td>0.995603</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in users.columns[5:]:
    print(users[col].value_counts())
```

    count_people
    2.0    97
    3.0    68
    4.0    67
    1.0    57
    5.0    24
    6.0     8
    0.0     2
    7.0     1
    8.0     1
    Name: count, dtype: int64
    count_adults
    2.0    201
    1.0     71
    3.0     32
    4.0     11
    5.0      5
    0.0      3
    7.0      1
    6.0      1
    Name: count, dtype: int64
    count_children
    0.0    195
    2.0     58
    1.0     55
    3.0     12
    4.0      3
    5.0      1
    6.0      1
    Name: count, dtype: int64
    count_babies
    0.0    302
    1.0     21
    2.0      2
    Name: count, dtype: int64
    count_pets
    0.0    193
    1.0     87
    2.0     28
    3.0      8
    4.0      6
    5.0      2
    6.0      1
    Name: count, dtype: int64


I'm guessing these columns were created as extra info added voluntarily by clients and that's the reason why we have so many missing values. We can keep these features as they can provide some insights in the future.


```python
users[users['user_nuts1'].isnull()].head()
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
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1145</th>
      <td>8afa3289c56fd4225f42bd3427615204eca9fea9e46261...</td>
      <td>Top Up</td>
      <td>None</td>
      <td>2021-09-27 16:42:50</td>
      <td>2021-09-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1858</th>
      <td>361235e2decd78440db688cc8ca0abf6ec6725dd5a2e6c...</td>
      <td>Proposition</td>
      <td>None</td>
      <td>2021-11-26 13:19:47</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3884</th>
      <td>e99c9a29e3d1fc49649568cc5625a23b2aa8e3a4577527...</td>
      <td>Top Up</td>
      <td>None</td>
      <td>2022-02-21 12:50:19</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4573</th>
      <td>e5ff40c78b3cdcf2b10c9dc69e5e893d95246741300f26...</td>
      <td>Proposition</td>
      <td>None</td>
      <td>2020-05-30 13:28:27</td>
      <td>2020-05-01 00:00:00</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12992</th>
      <td>b34e8a5c1464868734143e3e3b49d951e8dc7efb346852...</td>
      <td>Proposition</td>
      <td>None</td>
      <td>2021-05-23 17:56:47</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We also have 51 users with no nut info. We will also keep this data by now.

### Joining Tables
We are going to join **orders** and **inventory** tables on variant_id indexes ('variant_ids' are contained in 'ordered_items' lists in the orders table). By doing so, we will be able to analyze the categories and tags of the products that are being ordered. We can also check if we have products in the inventory that aren't being sold.


```python
'''we break down each list 'ordered_items' into different rows.
Then we reset the indexes'''

orders_split = orders.explode('ordered_items').reset_index(drop=True)
orders_split.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618849693828</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618860179588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618874040452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618907005060</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618907005060</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f" Number of orders: {len(orders)}\n Total products bought: {len(orders_split)}\n Average number of items per order: {(len(orders_split)/len(orders)):.2f}")
```

     Number of orders: 8773
     Total products bought: 107958
     Average number of items per order: 12.31


### How should we join the tables?
We have to think first about the questions each join could answer.

1. *Inner Join*: We would have only the items that are available in inventory *AND* have been ordered. It is useful because by checking the length of this table we can know how many of the items in inventory are being sold.

2. *Left Join* (on inventory): We now preserve all the rows in inventory and complete them with the info of the orders of these products that exist. Do we have products in our inventory that have never been ordered? We can answer this question looking for empty values in the orders-related columns.

3. *Right Join* (on orders): We now keep the rows in orders and fill them with info from inventory. Do we have products that have been ordered but don't exist in our inventory? In that case we need to look for empty values in the inventory-related columns. If that was the case this can mean two things: we don't have the item in our inventory yet or it could be an error (mistaken register).

4. *Outer Join*: We keep all the data. We have a whole vision of our data.


```python
inner = inventory.merge(orders_split, how='inner', left_on='variant_id', right_on='ordered_items')
inventory_complete = inventory.merge(orders_split, how='left', left_on='variant_id', right_on='ordered_items')
orders_complete = inventory.merge(orders_split, how='right', left_on='variant_id', right_on='ordered_items')
outer = inventory.merge(orders_split, how='outer', left_on='variant_id', right_on='ordered_items')
```


```python
print(f"Total orders: {len(orders)}")
print(f"Volume of existing inventory items ordered: {len(inner)}")
print(f"Items on inventory that have NEVER been ordered: {len(inventory_complete) - len(inner)}")
print(f"Volume of items ordered: {len(orders_complete)}")
print(f"Unique items on inventory: {len(inventory)}")
print(f"Unique items ordered: {orders_complete['ordered_items'].nunique()}")
print(f"Unique items ordered not in stock: {orders_complete[orders_complete['variant_id'].isnull()]['ordered_items'].nunique()}")
print(f"Volume of existing items on inventory or ordered: {len(outer)}")
```

    Total orders: 8773
    Volume of existing inventory items ordered: 92361
    Items on inventory that have NEVER been ordered: 256
    Volume of items ordered: 107958
    Unique items on inventory: 1733
    Unique items ordered: 2117
    Unique items ordered not in stock: 640
    Volume of existing items on inventory or ordered: 108214



```python
print(f"Products sold not on inventory: {100*orders_complete.price.isna().sum() / len(orders_complete) :.2f}%")
print(f"Products on inventory never sold: {100*inventory_complete.ordered_items.isna().sum() / len(inventory_complete) :.2f}%")
```

    Products sold not on inventory: 14.45%
    Products on inventory never sold: 0.28%


Let's explore our sales. We will now focus on the orders_complete table.


```python
# Unique Ordered Items out of stock
orders_complete[orders_complete['variant_id'].isnull()]['ordered_items'].nunique()
```




    640




```python
# Amount of Items ordered out of inventory
len(orders_complete[orders_complete['variant_id'].isnull()]['ordered_items'])
```




    15597




```python
inventory_complete.isnull().sum()
```




    variant_id            0
    price                 0
    compare_at_price      0
    vendor                0
    product_type          0
    tags                  0
    id                  256
    user_id             256
    created_at          256
    order_date          256
    user_order_seq      256
    ordered_items       256
    dtype: int64




```python
inventory_complete.head()
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
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
      <td>2.297562e+12</td>
      <td>70741713dc3d0ca898200632ca4fb44f5105416d0d66a7...</td>
      <td>2020-06-12 00:00:29</td>
      <td>2020-06-11</td>
      <td>1.0</td>
      <td>39587297165444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
      <td>2.302065e+12</td>
      <td>b2cc7053eea2abd9033fb47469ac0e29c5deb2752b43a2...</td>
      <td>2020-06-16 14:55:56</td>
      <td>2020-06-16</td>
      <td>1.0</td>
      <td>39587297165444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
      <td>2.304104e+12</td>
      <td>5e662d203c16c72443d1f97e6a29604a7fb0c8ae836cd7...</td>
      <td>2020-06-18 16:38:36</td>
      <td>2020-06-18</td>
      <td>1.0</td>
      <td>39587297165444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
      <td>2.322977e+12</td>
      <td>1e29453f19121f338edd09ac8d0bfc6431883377d39241...</td>
      <td>2020-07-02 14:57:26</td>
      <td>2020-07-02</td>
      <td>1.0</td>
      <td>39587297165444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
      <td>2.326898e+12</td>
      <td>9155c7d1ea2729fd64fd9243afd4c347e7a1b65c2b4302...</td>
      <td>2020-07-07 13:59:15</td>
      <td>2020-07-07</td>
      <td>2.0</td>
      <td>39587297165444</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6),dpi=200)
sns.kdeplot(data = orders_complete, x='price',fill=True, label='Orders')
sns.kdeplot(data = inventory, x='price',fill=True, label='Inventory')
plt.xlabel("Price (GBP)")
plt.ylabel("Density")
plt.title('Price Distribution by orders vs inventory',fontsize =10,fontweight = 'bold')

plt.legend()

plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_42_0.png)
    


Watching the distribution of prices by the amount of orders received, we can see that most of our sold products are relatively cheap (0-5 GBP) but we are also getting a spike in sales in the 10-13 GBP range.

What are our best selling products?


```python
# We get the top 10 sales
top_10_sales = orders_complete['product_type'].value_counts()[:10]

plt.figure(figsize=(12, 6), dpi=200)
sns.countplot(data = orders_complete, x='product_type', order=top_10_sales.index, color='red')
plt.xlabel('Product Type')
plt.ylabel('Number of Orders')
plt.title('Top 10 Most Ordered Products by Categories', fontsize =10, fontweight = 'bold')
plt.xticks(rotation=45, ha='right')  # Rotate x axis tags
plt.show()

```


    
![png](module_2_EDA_files/module_2_EDA_45_0.png)
    


I's surprising that some of our most sold products are milk substitutes.


```python
# We get the top 10 vendors 
top_10_vendors = orders_complete['vendor'].value_counts()[:10]

plt.figure(figsize=(10, 6), dpi=200)
sns.countplot(data = orders_complete, x='vendor', order=top_10_vendors.index, color='red')
plt.xlabel('Vendor')
plt.ylabel('Number of Orders')
plt.title('Top 10 Vendors by Volume', fontsize =10, fontweight = 'bold')
plt.xticks(rotation=45, ha='right')  # Rotate xf axis tags
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_47_0.png)
    


We found again more evidence in favor of our hypothesis that we are a e-commerce specialized in eco and vegan products as our top vendors are oatly (oat milk) and ecover (eco laundry).


```python
top_vendors_revenue = orders_complete.groupby('vendor')['price'].sum().reset_index().sort_values(by='price',ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='vendor', y='price', data=top_vendors_revenue, color='red')
plt.xlabel('Vendor')
plt.ylabel('Revenue')
plt.title('Top 10 Vendors by Revenue', fontsize =10, fontweight = 'bold')
plt.xticks(rotation=45) # rotate x axis
plt.show()

```


    
![png](module_2_EDA_files/module_2_EDA_49_0.png)
    


We have slight differences compared to Volume. Persil (detergent) comes third in facturation and CocaCola appears at top 8 (meaning these two sell more expensive products).

How much does our top 10 products sells account for?


```python
orders_complete.tail()
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
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107953</th>
      <td>34537169453188</td>
      <td>0.99</td>
      <td>1.38</td>
      <td>whiskas</td>
      <td>cat-food</td>
      <td>[cat-treats]</td>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>34537169453188</td>
    </tr>
    <tr>
      <th>107954</th>
      <td>39368665432196</td>
      <td>4.99</td>
      <td>5.09</td>
      <td>whiskas</td>
      <td>cat-food</td>
      <td>[wet-cat-food]</td>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39368665432196</td>
    </tr>
    <tr>
      <th>107955</th>
      <td>39403033428100</td>
      <td>3.29</td>
      <td>3.40</td>
      <td>napolina</td>
      <td>tins-packaged-foods</td>
      <td>[beans-pulses-lentils, discontinue, gluten-fre...</td>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39403033428100</td>
    </tr>
    <tr>
      <th>107956</th>
      <td>39459281272964</td>
      <td>4.49</td>
      <td>9.00</td>
      <td>cif</td>
      <td>cleaning-products</td>
      <td>[bathroom-limescale-cleaner, refills]</td>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39459281272964</td>
    </tr>
    <tr>
      <th>107957</th>
      <td>39830155034756</td>
      <td>4.49</td>
      <td>9.00</td>
      <td>cif</td>
      <td>cleaning-products</td>
      <td>[kitchen-oven-cleaner, refills]</td>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39830155034756</td>
    </tr>
  </tbody>
</table>
</div>




```python
# most luxurious vendors
orders_complete.groupby('vendor')['price'].mean().sort_values(ascending=False).head(10)
```




    vendor
    moëtchandon        40.49
    vitalproteins      39.99
    jackdaniels        30.99
    jameson            29.99
    malfy              27.99
    grenade            27.99
    creativenature     26.99
    waterwipes         26.49
    seedlip            25.99
    funktionalfoods    23.99
    Name: price, dtype: float64




```python
# top 10 products by revenue
top_10_prod = orders_complete.groupby(['ordered_items'])['price'].sum().reset_index().sort_values(by='price',ascending=False).head(10)
# others revenue
not_top_10_prod = orders_complete.groupby(['ordered_items'])['price'].sum().reset_index().sort_values(by='price',ascending=False)[10:]
# total revenue
total_rev = orders_complete['price'].sum()
```


```python
plt.bar(['Top 10', 'Others'],[top_10_prod['price'].sum(), not_top_10_prod['price'].sum()])
plt.ylabel('Revenue')
plt.title('Top 10 vs Others Revenue')
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_55_0.png)
    



```python
print(f"Top 10 Products account for {100*top_10_prod['price'].sum()/total_rev:.2f}% of the Total revenue")
print(f"The top Product accounts for {100*top_10_prod['price'].iloc[0] / total_rev:.2f}% of the Total revenue")
```

    Top 10 Products account for 20.68% of the Total revenue
    The top Product accounts for 9.88% of the Total revenue



```python
# how much is 10 relative to the total of products (pct)
round(100*10/orders_complete['ordered_items'].nunique(), 2)
```




    0.47




```python
# Let's check the top tags on inventory by merging all tags into one dataframe
all_tags = []
for list in inventory['tags']:
    for tag in list:
        all_tags.append(tag)

top_10_tags = pd.DataFrame(all_tags).value_counts().head(10)
'''top_10_tags'''
```




    'top_10_tags'




```python
top_10_tags.plot.bar()
plt.title('Top 10 most common tags on inventory', fontsize = 10, fontweight = 'bold')
plt.xlabel("Tags")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right') 
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_59_0.png)
    


## Sales & Inventory Insights

- Most of our sales (and products on inventory) are vegan and eco-friendly products.
- Most of the ordered items fall into the next categories: cleaning products, packaged foods and milk substitutes 
- We have 0.28% of products in inventory that have never been ordered
- We have sold 14.45% of products that are not in inventory 
- 10 products (0.47%) are responsible for 20.68% (!!) of the total revenue, while the top product accounts for almost 10% of the total revenue  

## Regulars overview


```python
regulars_complete = regulars.merge(inventory, on= 'variant_id', how='left')
# We are only going to work with the regulars existing on inventory
regulars_clean = regulars_complete.dropna()
```


```python
regulars_clean.head()
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
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3.49</td>
      <td>3.50</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan, window-glass-cleaner]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667182493828</td>
      <td>2020-05-06 14:42:11</td>
      <td>4.29</td>
      <td>5.40</td>
      <td>bulldog</td>
      <td>skincare</td>
      <td>[cruelty-free, eco, facial-skincare, vegan]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d883991facbc3b07b62da342d00c97d1e6cea8d2176695...</td>
      <td>33667198910596</td>
      <td>2020-07-06 10:12:08</td>
      <td>14.99</td>
      <td>16.55</td>
      <td>ecover</td>
      <td>dishwashing</td>
      <td>[cruelty-free, dishwasher-tablets, eco, vegan]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>66a195720d6988ff4d32155cc03631b84f68b34d3b0a1e...</td>
      <td>33826459320452</td>
      <td>2020-07-06 17:17:52</td>
      <td>5.09</td>
      <td>5.65</td>
      <td>treeoflife</td>
      <td>snacks-confectionery</td>
      <td>[christmas, nuts-dried-fruit-seeds, organic]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0b7e02fee4b9e215da3bdae70050f20c5ffd18264454a5...</td>
      <td>33667247276164</td>
      <td>2020-07-18 16:56:55</td>
      <td>2.49</td>
      <td>3.00</td>
      <td>method</td>
      <td>hand-soap-sanitisers</td>
      <td>[cruelty-free, eco, hand-soap, vegan]</td>
    </tr>
  </tbody>
</table>
</div>




```python
regulars_clean.groupby('variant_id')['user_id'].nunique().sort_values(ascending=False).head(10)
```




    variant_id
    34081589887108    207
    34284951863428     78
    33667282436228     78
    34370915041412     77
    34370361229444     74
    34284949766276     72
    34037939372164     61
    33667302555780     56
    33826430648452     55
    33667247407236     53
    Name: user_id, dtype: int64




```python
# We group by 'variant_id', count the users (nunique in case some users marked the same product as regular twice)
# and we rename the users_id column as 'count'
top10_var_and_users = regulars_clean.groupby('variant_id')['user_id'].nunique().sort_values(ascending=False).head(10).reset_index().rename(columns={'user_id': 'count'})
top10_var_and_users
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34081589887108</td>
      <td>207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34284951863428</td>
      <td>78</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33667282436228</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34370915041412</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34370361229444</td>
      <td>74</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34284949766276</td>
      <td>72</td>
    </tr>
    <tr>
      <th>6</th>
      <td>34037939372164</td>
      <td>61</td>
    </tr>
    <tr>
      <th>7</th>
      <td>33667302555780</td>
      <td>56</td>
    </tr>
    <tr>
      <th>8</th>
      <td>33826430648452</td>
      <td>55</td>
    </tr>
    <tr>
      <th>9</th>
      <td>33667247407236</td>
      <td>53</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We join the df that counts the regulars with the rest of columns in order to use it as labels for our graph
top10_products = pd.merge(top10_var_and_users, regulars_clean[['variant_id', 'product_type','vendor','price']], on='variant_id', how='left')

```


```python
# we need this to use it as labels
top10_products = top10_products.drop_duplicates()
top10_products
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
      <th>count</th>
      <th>product_type</th>
      <th>vendor</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34081589887108</td>
      <td>207</td>
      <td>long-life-milk-substitutes</td>
      <td>oatly</td>
      <td>10.79</td>
    </tr>
    <tr>
      <th>253</th>
      <td>34284951863428</td>
      <td>78</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>plenty</td>
      <td>3.69</td>
    </tr>
    <tr>
      <th>358</th>
      <td>33667282436228</td>
      <td>78</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>thecheekypanda</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>459</th>
      <td>34370915041412</td>
      <td>77</td>
      <td>tins-packaged-foods</td>
      <td>mutti</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>571</th>
      <td>34370361229444</td>
      <td>74</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>whogivesacrap</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>698</th>
      <td>34284949766276</td>
      <td>72</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>andrex</td>
      <td>8.49</td>
    </tr>
    <tr>
      <th>794</th>
      <td>34037939372164</td>
      <td>61</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>andrex</td>
      <td>4.99</td>
    </tr>
    <tr>
      <th>883</th>
      <td>33667302555780</td>
      <td>56</td>
      <td>bin-bags</td>
      <td>symphony</td>
      <td>1.99</td>
    </tr>
    <tr>
      <th>961</th>
      <td>33826430648452</td>
      <td>55</td>
      <td>cooking-ingredients</td>
      <td>maldon</td>
      <td>2.29</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>33667247407236</td>
      <td>53</td>
      <td>cleaning-products</td>
      <td>method</td>
      <td>3.49</td>
    </tr>
  </tbody>
</table>
</div>




```python
regulars_clean.groupby('variant_id')['user_id'].nunique().sort_values(ascending=False).head(10).plot(kind= 'bar')
plt.title('Top 10 Regulars Items')
plt.xlabel('Category')
plt.xticks(ticks= range(0,10),labels= top10_products['product_type'].head(10),rotation=45, ha='right')
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_68_0.png)
    


The regulars and the most ordered products are very similar. It's interesting that the product most added as regular is the oat milk sold by Oatly.


```python
sns.kdeplot(regulars_clean['price'], label= 'regulars price')
sns.kdeplot(inventory['price'], label= 'inventory price')
sns.kdeplot(orders_complete['price'], label= 'orders price')
plt.xlabel("Price")
plt.ylabel("Density")

plt.legend()

plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_70_0.png)
    


Regulars prices are slightly cheaper than the ordered items prices and way cheaper than the inventory prices. It's probably because people mark the basics as regulars (which are usually cheaper).

## Part 2. EDA

We will now work with the csv 


```python
feature_frame = pd.read_csv('../../../data-zrive-ds/box_builder_dataset/feature_frame.csv')
```


```python
feature_frame.head()
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
feature_frame.info()
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
feature_frame.columns
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
info_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
label_col = ['outcome']

numerical_cols = ['user_order_seq', 'normalised_price', 'discount_pct', 'global_popularity',
            'count_adults', 'count_children', 'count_babies', 'count_pets', 
            'people_ex_baby', 'days_since_purchase_variant_id', 
            'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
            'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
                'std_days_to_buy_product_type']

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
```


```python
feature_frame['outcome'].value_counts()
```




    outcome
    0.0    2847317
    1.0      33232
    Name: count, dtype: int64




```python
corr = feature_frame[numerical_cols + label_col].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmax=0.5)
plt.title('Correlation Matrix')
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_80_0.png)
    


We have very weak correlations with outcome. And we have some variables very correlated between themselves, so we must be careful with possible mulitcolinearity problems in the future.


```python
for col in feature_frame[binary_cols]:
    plt.figure(figsize = (6, 4))
    sns.countplot(data = feature_frame, x = col, palette='Set1', hue=col, legend=False)
    plt.title(f'Distribution of {col}')
    plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_82_0.png)
    



    
![png](module_2_EDA_files/module_2_EDA_82_1.png)
    



    
![png](module_2_EDA_files/module_2_EDA_82_2.png)
    



    
![png](module_2_EDA_files/module_2_EDA_82_3.png)
    


Our binary features are really unbalanced. So we must consider this in order to address future potential problems when modelling.


```python
feature_frame[categorical_cols].describe()
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
      <th>product_type</th>
      <th>vendor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2880549</td>
      <td>2880549</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>62</td>
      <td>264</td>
    </tr>
    <tr>
      <th>top</th>
      <td>tinspackagedfoods</td>
      <td>biona</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>226474</td>
      <td>146828</td>
    </tr>
  </tbody>
</table>
</div>



We won't be able to use one-hot encoding for the categorical columns since we have too many unique variables, so we must consider different alternatives.
