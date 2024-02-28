## Exploration Phase - Non Linear Models & Model Selection

**GOAL**: Create a Classification Model that can predict whether or not a person would buy an item we offered them (via push notification) based on behavioral and personal features of that user (user id, ordered before, etc), features of that specific order (date, etc) and features of the items themselves (popularity, price, avg days to buy, etc)

We must notice that sending too many notifications would have a negative impact on user experience.

**MODULE 4 GOAL: We will now continue the work done in Module 3 and keep exploring models, now focusing on non linear ones.**

At the end, we will select the best performing model (among baseline, linear and non linear models)

### **STRUCTURE OF THIS NOTEBOOK**

- Split the data, create Baseline model
- Non Linear models exploration with different hyperparameters: Random Forests, Gradient Boosting Trees, CatBoost, XGBoost
- Final comparison of all models & saving the best performing one


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from typing import Tuple
import joblib
import os

from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    average_precision_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import Pool
from catboost import CatBoostClassifier


```


```python
path = '/home/carleondel/data-zrive-ds/box_builder_dataset/'
df = pd.read_csv(path + 'feature_frame.csv')
```

We already explored our dataset so we will jump directly into splitting it

### Data splitting
- Before splitting we need to filter our df to keep only orders with more than 5 items
- Then we make a temporal split of 70-20-10 for train | validation | test
- Finally we make the X | y splits


```python
def push_relevant_dataframe(df: pd.DataFrame, min_products: int=5) -> pd.DataFrame:
    """We are only interested in orders with at least 5 products"""
    order_size = df.groupby('order_id').outcome.sum()
    orders_of_min_size = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size)]

df_selected = (
    df
    .pipe(push_relevant_dataframe)
    .assign(created_at=lambda x: pd.to_datetime(x.created_at))
    .assign(order_date=lambda x: pd.to_datetime(x.order_date).dt.date)
)
```


```python
df_selected.head()
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
      <td>2020-10-05</td>
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
      <td>2020-10-05</td>
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
      <td>2020-10-05</td>
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
      <td>2020-10-06</td>
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
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06</td>
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
daily_orders = df_selected.groupby('order_date')['order_id'].nunique()
```


```python
daily_orders
```




    order_date
    2020-10-05     3
    2020-10-06     7
    2020-10-07     6
    2020-10-08    12
    2020-10-09     4
                  ..
    2021-02-27    32
    2021-02-28    32
    2021-03-01    42
    2021-03-02    25
    2021-03-03    14
    Name: order_id, Length: 149, dtype: int64




```python
fig, ax = plt.subplots()
ax.plot(daily_orders, label="daily orders")
ax.set_title("Daily orders")
```




    Text(0.5, 1.0, 'Daily orders')




    
![png](push_notifications_project_files/push_notifications_project_9_1.png)
    


We already discussed in Module 3 why it makes sense to make a temporal split


```python
cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
""" We are getting the pct of acummulated orders out of the all orders after each day"""

train_val_cutoff = cumsum_daily_orders[cumsum_daily_orders <= 0.7].idxmax()
""" Now we are taking the date of the order just before surpassing 0.7 of the total"""
val_test_cutoff = cumsum_daily_orders[cumsum_daily_orders <= 0.9].idxmax()

print("Train since:", cumsum_daily_orders.index.min())
print("Train until:", train_val_cutoff)
print("Val until:", val_test_cutoff)
print("Test until:", cumsum_daily_orders.index.max())
```

    Train since: 2020-10-05
    Train until: 2021-02-04
    Val until: 2021-02-22
    Test until: 2021-03-03


We must notice our time splits are very narrow and we could be skewing our samples in case of rare events (Halloween, holidays, etc)


```python
train_df = df_selected[df_selected['order_date'] <= train_val_cutoff]
val_df = df_selected[(df_selected['order_date'] > train_val_cutoff) & (df_selected['order_date'] <= val_test_cutoff)]
test_df = df_selected[df_selected['order_date'] > val_test_cutoff]
```

We make the temporal split, taking the first 70% for training, the next 20% for validation and the last 10% for test

### Explicit declaration of columns
We make our columns selection explicit in order to avoid problems in case columns names/order changed in the future


```python
df_selected.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2163953 entries, 0 to 2880547
    Data columns (total 27 columns):
     #   Column                            Dtype         
    ---  ------                            -----         
     0   variant_id                        int64         
     1   product_type                      object        
     2   order_id                          int64         
     3   user_id                           int64         
     4   created_at                        datetime64[ns]
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
    dtypes: datetime64[ns](1), float64(19), int64(4), object(3)
    memory usage: 462.3+ MB



```python
info_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
label_col = 'outcome'
features_cols = [col for col in df.columns if col not in (info_cols + [label_col])]

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols = [col for col in features_cols if col not in categorical_cols + binary_cols]
```

By now we will work without the categorical features and without info_cols.
In the future we can explore different types of encoding and also creating new features extracted from the datetime columns such as month, day or hour the order was made (not year since we have less than 1 full year on our dataset).


```python
train_cols = numerical_cols + binary_cols
```


```python
train_cols
```




    ['user_order_seq',
     'normalised_price',
     'discount_pct',
     'global_popularity',
     'count_adults',
     'count_children',
     'count_babies',
     'count_pets',
     'people_ex_baby',
     'days_since_purchase_variant_id',
     'avg_days_to_buy_variant_id',
     'std_days_to_buy_variant_id',
     'days_since_purchase_product_type',
     'avg_days_to_buy_product_type',
     'std_days_to_buy_product_type',
     'ordered_before',
     'abandoned_before',
     'active_snoozed',
     'set_as_regular']



Before creating our baseline model we are going to define some functions that will help us plot the different metrics we will use


```python
def plot_metrics(
        model_name : str, y_pred : pd.Series, y_test : pd.Series,
        figure : Tuple[matplotlib.figure.Figure, np.array] = None
    ):
    precision_, recall_, _ = precision_recall_curve(
        y_test, y_pred
    )
    pr_auc = auc(recall_, precision_)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)


    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    else:
        fig, ax = figure

    ax[0].plot(recall_, precision_, label=f"{model_name}; AUC: {pr_auc:.2f}")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].set_title("Precision-recall Curve")
    ax[0].legend()



    ax[1].plot(fpr, tpr, label=f"AUC: {roc_auc:.2f}")
    ax[1].set_xlabel("FPR")
    ax[1].set_ylabel("TPR")
    ax[1].set_title("ROC Curve")
    ax[1].legend()
```

### Baseline model
Since we are now exploring non linear models, our baseline model will be the best linear model we got in module 3, which is Ridge Regression with C=1e-6


```python
def feature_label_split(df : pd.DataFrame, label_col : str ) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(label_col, axis=1), df[label_col]

X_train, y_train = feature_label_split(train_df, label_col)
X_val, y_val = feature_label_split(val_df, label_col)
X_test, y_test = feature_label_split(test_df, label_col)
```


```python
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")
    
ridge = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1e-6)
)
ridge.fit(X_train[train_cols], y_train)

train_proba = ridge.predict_proba(X_train[train_cols])[:,1]
plot_metrics(f"LR; C=1e-6", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = ridge.predict_proba(X_val[train_cols])[:, 1]
plot_metrics(f"LR; C=1e-6", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

```


    
![png](push_notifications_project_files/push_notifications_project_25_0.png)
    



    
![png](push_notifications_project_files/push_notifications_project_25_1.png)
    



```python
X_train[(ridge.predict_proba(X_train[train_cols])[:,1] > 0.1)][numerical_cols]
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
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>global_popularity</th>
      <th>count_adults</th>
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
      <th>87573</th>
      <td>4</td>
      <td>0.284359</td>
      <td>0.138227</td>
      <td>0.387146</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>19.68243</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>21.04899</td>
    </tr>
    <tr>
      <th>87612</th>
      <td>3</td>
      <td>0.284359</td>
      <td>0.138227</td>
      <td>0.387146</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>19.68243</td>
      <td>6.0</td>
      <td>25.0</td>
      <td>21.04899</td>
    </tr>
  </tbody>
</table>
</div>



We only have 2 instances classified as 1 with probabilities >10%. So we should consider lowering the threshold


```python
# Obtaining class probabilities
probabilities_test = ridge.predict_proba(X_val[train_cols])

# personalized threshold
new_threshold = 0.03

# Adjust predictions by the new threshold
new_predictions = (probabilities_test[:, 1] > new_threshold).astype(int)

# Evaluate performance of the model with the new threshold
accuracy = accuracy_score(val_df[label_col], new_predictions)
precision = precision_score(val_df[label_col], new_predictions)
recall = recall_score(val_df[label_col], new_predictions)

print(f'Performance with new threshold ({new_threshold}):')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
```

    Performance with new threshold (0.03):
    Accuracy: 0.99
    Precision: 0.70
    Recall: 0.04



```python
# Obtaining class probabilities
probabilities_test = ridge.predict_proba(X_val[train_cols])

# personalized threshold
new_threshold = 0.02

# Adjust predictions by the new threshold
new_predictions = (probabilities_test[:, 1] > new_threshold).astype(int)

# Evaluate performance of the model with the new threshold
accuracy = accuracy_score(val_df[label_col], new_predictions)
precision = precision_score(val_df[label_col], new_predictions)
recall = recall_score(val_df[label_col], new_predictions)

print(f'Performance with new threshold ({new_threshold}):')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
```

    Performance with new threshold (0.02):
    Accuracy: 0.98
    Precision: 0.25
    Recall: 0.22


In our context it makes more sense to accept more false positive cases, so we could choose a threshold between 0.03 and 0.02.

## Non-linear models
We will explore different models, including Random Forests and Gradient Boosting Trees

### Random Forest


```python
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")

# Number of trees
# In random forests we don't need to scale our training data
ns =  [5, 25, 50, 100]   
for n in ns:  
    rf = make_pipeline(
        RandomForestClassifier(n_estimators = n)
    )  
    rf.fit(X_train[train_cols], y_train)
    train_proba = rf.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"RF; Number of Trees = {n}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


    val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"RF; Number of Trees = {n}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

# We plot our baseline model ridge lr with C = 1e-6
val_proba = ridge.predict_proba(X_val[train_cols])[:, 1]    
plot_metrics(f"Baseline; Ridge LR", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))
    
```


    
![png](push_notifications_project_files/push_notifications_project_32_0.png)
    



    
![png](push_notifications_project_files/push_notifications_project_32_1.png)
    


### Insights 
- At first glance, we can clearly see how this model works. The performance metrics are much better on train than on validation (overfit signal). This is because each tree is built in its entirety, making the model learn some noise. But because we are averaging the predictions from multiple trees, this overfitting is reduced.
- After creating different models using different number of trees, there is not much improvement when going from 50 to 100, so it doesn't make sense to keep training with more trees.
- Overall the performance of our Random Forest models is slightly worse than the performance of our Baseline Model (Ridge C=1e-6)

### Feature Importance & Retraining
Now we are going to reduce our features and retrain our model to see if we make any improvement. Since there wasn't much improvement when moving from 50 to 100 trees, we will base our decisions on our model with 50 trees.


```python
rf50 = make_pipeline(
        RandomForestClassifier(n_estimators = 50)
    )  
rf50.fit(X_train[train_cols], y_train)


# Access the model within the pipeline
rf_model = rf50.named_steps['randomforestclassifier']

# Extract importance
feature_importance = rf_model.feature_importances_

# Create df with importance
rf50_coefs_df = pd.DataFrame({'Feature': train_cols, 'Importance': feature_importance})

# Order the df
rf50_coefs_df = rf50_coefs_df.sort_values(by='Importance', ascending=False)

```


```python
rf50_coefs_df
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
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>global_popularity</td>
      <td>0.368934</td>
    </tr>
    <tr>
      <th>12</th>
      <td>days_since_purchase_product_type</td>
      <td>0.145682</td>
    </tr>
    <tr>
      <th>0</th>
      <td>user_order_seq</td>
      <td>0.118486</td>
    </tr>
    <tr>
      <th>9</th>
      <td>days_since_purchase_variant_id</td>
      <td>0.045590</td>
    </tr>
    <tr>
      <th>16</th>
      <td>abandoned_before</td>
      <td>0.044895</td>
    </tr>
    <tr>
      <th>11</th>
      <td>std_days_to_buy_variant_id</td>
      <td>0.043964</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_variant_id</td>
      <td>0.037541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>discount_pct</td>
      <td>0.036915</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ordered_before</td>
      <td>0.036334</td>
    </tr>
    <tr>
      <th>1</th>
      <td>normalised_price</td>
      <td>0.033050</td>
    </tr>
    <tr>
      <th>14</th>
      <td>std_days_to_buy_product_type</td>
      <td>0.026591</td>
    </tr>
    <tr>
      <th>13</th>
      <td>avg_days_to_buy_product_type</td>
      <td>0.018657</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count_pets</td>
      <td>0.011481</td>
    </tr>
    <tr>
      <th>8</th>
      <td>people_ex_baby</td>
      <td>0.007611</td>
    </tr>
    <tr>
      <th>18</th>
      <td>set_as_regular</td>
      <td>0.006999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>count_adults</td>
      <td>0.005638</td>
    </tr>
    <tr>
      <th>5</th>
      <td>count_children</td>
      <td>0.005598</td>
    </tr>
    <tr>
      <th>17</th>
      <td>active_snoozed</td>
      <td>0.005004</td>
    </tr>
    <tr>
      <th>6</th>
      <td>count_babies</td>
      <td>0.001029</td>
    </tr>
  </tbody>
</table>
</div>



We are going to keep the top 3 features by importance, train again and compare


```python
reduced_cols = ['global_popularity', 'days_since_purchase_product_type', 'user_order_seq']
```


```python
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")

# Number of trees
# In random forests we don't need to scale our training data
ns =  [5, 25, 50, 100]   
for n in ns:  
    rf = make_pipeline(
        RandomForestClassifier(n_estimators = n)
    )  
    rf.fit(X_train[reduced_cols], y_train)
    train_proba = rf.predict_proba(X_train[reduced_cols])[:, 1]
    plot_metrics(f"RF; Number of Trees = {n}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


    val_proba = rf.predict_proba(X_val[reduced_cols])[:, 1]
    plot_metrics(f"RF; Number of Trees = {n}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

# We plot our baseline model ridge lr with C = 1e-6
val_proba = ridge.predict_proba(X_val[train_cols])[:, 1]    
plot_metrics(f"Baseline; Ridge LR", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))
    
```


    
![png](push_notifications_project_files/push_notifications_project_39_0.png)
    



    
![png](push_notifications_project_files/push_notifications_project_39_1.png)
    


### Insights
- The performance metrics are significantly worse than the metrics when training with all the features. (PR-AUC = 0.04 vs 0.13)

## Gradient Boosting Trees



```python
'''The number of weak learners (i.e. regression trees) is controlled by
 the parameter n_estimators; 
 The size of each tree can be controlled either by setting the tree depth 
 via max_depth or by setting the number of leaf nodes via max_leaf_nodes.
 The learning_rate is a hyper-parameter in the range (0.0, 1.0] that controls
   overfitting via shrinkage .'''

```




    'The number of weak learners (i.e. regression trees) is controlled by\n the parameter n_estimators; \n The size of each tree can be controlled either by setting the tree depth \n via max_depth or by setting the number of leaf nodes via max_leaf_nodes.\n The learning_rate is a hyper-parameter in the range (0.0, 1.0] that controls\n   overfitting via shrinkage .'




```python
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")

# We define our parameters
ns_trees = [5,100,250]
ns_max_depth = [1,3,5]
lr = 1.0

trained_gbt_models = []
y_preds_gbt = []
gbt_model_names = []

for n_trees in ns_trees:
    for depth in ns_max_depth:
        gbt = GradientBoostingClassifier(
            n_estimators = n_trees,
            max_depth = depth,
            learning_rate = lr
            )
        gbt.fit(X_train[train_cols], y_train)

        train_proba = gbt.predict_proba(X_train[train_cols])[:, 1]
        plot_metrics(f"GBT; Number of estimators = {n_trees}, Depth = {depth}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


        val_proba = gbt.predict_proba(X_val[train_cols])[:, 1]
        plot_metrics(f"GBT; Number of estimators = {n_trees}, Depth = {depth}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))


        # We save the models and the predictions
        trained_gbt_models.append(gbt)
        y_preds_gbt.append(gbt.predict_proba(X_val[train_cols])[:, 1])
        gbt_model_names.append(f"GBT (n_trees={n_trees}, depth={depth})")
```


    
![png](push_notifications_project_files/push_notifications_project_43_0.png)
    



    
![png](push_notifications_project_files/push_notifications_project_43_1.png)
    


## Insights
This took >20 mins training...

Our Baseline model had PR-AUC = 0.16, ROC-AUC = 0.83. So we have finally made some improvement.
- When we used max_depth of 5, all models had a similar performance, independently of the other hyperparameter (number of estimators). PR-AUC = 0.22 and ROC-AUC = 0.79.
- So our model selected from Gradient Boosting Trees will be the one with hyperparameters: Number of estimators = 100, max_depth = 5


## Catboost

With catboost we can use our categorical features without preprocessing them.


```python
catboost_cols = train_cols + categorical_cols
```


```python
categorical_cols
```




    ['product_type', 'vendor']




```python
# Let's explore CatBoost with a simple model
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")


cbc = CatBoostClassifier(iterations=2,
                           learning_rate=1,
                           depth=2)
cbc.fit(X_train[catboost_cols], y_train, categorical_cols)

train_proba = cbc.predict_proba(X_train[catboost_cols])[:, 1]
plot_metrics(f"CBC; Iterations = {2}, Depth = {2}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = cbc.predict_proba(X_val[catboost_cols])[:, 1]
plot_metrics(f"CBC; Iterations = {2}, Depth = {2}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

```

    0:	learn: 0.0764781	total: 343ms	remaining: 343ms
    1:	learn: 0.0711741	total: 564ms	remaining: 0us



    
![png](push_notifications_project_files/push_notifications_project_49_1.png)
    



    
![png](push_notifications_project_files/push_notifications_project_49_2.png)
    



```python
# We create this function in order to keep track of the metrics of our grid search

def auc_score(y_test: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    """
    Returns the ROC and PR curves AUC score for a given set of true labels and predicted probabilities.

    Args:
        y_test (array): True labels.
        y_preds (array): Predicted probabilities or scores.

    Returns:
        Tuple (float, float): ROC AUC score, PR curve AUC score.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall_, precision_)
    return roc_auc, pr_auc
```


```python
# We are going to create a grid search based on my criteria
roc_scores_train = []
roc_scores_val = []

pr_scores_train = []
pr_scores_val = []

hyper_iterations = [] 
hyper_depths = []
hyper_lr = []

n = 0

for lr in [0.01, 0.05, 0.1]:
    for depth in [6, 8, 10]:
        for iterations in [100, 200, 300]:
            cbc = CatBoostClassifier(
                learning_rate = lr, depth = depth, iterations = iterations
            )
            # We need to pass the categorical cols this time
            
            cbc.fit(X_train[catboost_cols], y_train, categorical_cols)

            train_proba = cbc.predict_proba(X_train[catboost_cols])[:, 1]   
            auc_scores_train = auc_score(y_train, train_proba)
            roc_scores_train.append(auc_scores_train[0])
            pr_scores_train.append(auc_scores_train[1])

            val_proba = cbc.predict_proba(X_val[catboost_cols])[:, 1]
            auc_scores_val = auc_score(y_val, val_proba)
            roc_scores_val.append(auc_scores_val[0])
            pr_scores_val.append(auc_scores_val[1])
            
            hyper_iterations.append(iterations)
            hyper_depths.append(depth)
            hyper_lr.append(lr)

            n = n+1
            # Actualizations of the training
            print(
                f"Trained model nº {n}/27. Iterations: {iterations}"
                f"Depth: {depth}"
                f"Learning rate: {lr}"
            )


```

    0:	learn: 0.6723575	total: 406ms	remaining: 40.2s
    1:	learn: 0.6501879	total: 583ms	remaining: 28.6s
    2:	learn: 0.6298137	total: 753ms	remaining: 24.4s
    3:	learn: 0.6101644	total: 912ms	remaining: 21.9s
    4:	learn: 0.5919772	total: 1.07s	remaining: 20.4s
    5:	learn: 0.5738644	total: 1.23s	remaining: 19.3s
    6:	learn: 0.5549505	total: 1.4s	remaining: 18.6s
    7:	learn: 0.5388479	total: 1.55s	remaining: 17.9s
    8:	learn: 0.5200021	total: 1.71s	remaining: 17.3s
    9:	learn: 0.5028545	total: 1.87s	remaining: 16.8s
    10:	learn: 0.4878148	total: 2.03s	remaining: 16.4s
    11:	learn: 0.4732861	total: 2.18s	remaining: 16s
    12:	learn: 0.4599667	total: 2.34s	remaining: 15.7s
    13:	learn: 0.4463596	total: 2.49s	remaining: 15.3s
    14:	learn: 0.4330468	total: 2.65s	remaining: 15s
    15:	learn: 0.4204244	total: 2.81s	remaining: 14.8s
    16:	learn: 0.4090390	total: 2.98s	remaining: 14.5s
    17:	learn: 0.3974503	total: 3.13s	remaining: 14.3s
    18:	learn: 0.3852765	total: 3.29s	remaining: 14s
    19:	learn: 0.3740613	total: 3.45s	remaining: 13.8s
    20:	learn: 0.3642472	total: 3.6s	remaining: 13.6s
    21:	learn: 0.3536468	total: 3.76s	remaining: 13.3s
    22:	learn: 0.3445329	total: 3.92s	remaining: 13.1s
    23:	learn: 0.3350627	total: 4.07s	remaining: 12.9s
    24:	learn: 0.3260265	total: 4.23s	remaining: 12.7s
    25:	learn: 0.3177889	total: 4.38s	remaining: 12.5s
    26:	learn: 0.3098418	total: 4.54s	remaining: 12.3s
    27:	learn: 0.3016708	total: 4.7s	remaining: 12.1s
    28:	learn: 0.2936624	total: 4.85s	remaining: 11.9s
    29:	learn: 0.2859252	total: 5.01s	remaining: 11.7s
    30:	learn: 0.2773540	total: 5.17s	remaining: 11.5s
    31:	learn: 0.2708016	total: 5.33s	remaining: 11.3s
    32:	learn: 0.2632934	total: 5.5s	remaining: 11.2s
    33:	learn: 0.2553408	total: 5.67s	remaining: 11s
    34:	learn: 0.2490541	total: 5.83s	remaining: 10.8s
    35:	learn: 0.2434853	total: 5.99s	remaining: 10.7s
    36:	learn: 0.2368589	total: 6.15s	remaining: 10.5s
    37:	learn: 0.2304743	total: 6.31s	remaining: 10.3s
    38:	learn: 0.2243507	total: 6.48s	remaining: 10.1s
    39:	learn: 0.2187521	total: 6.63s	remaining: 9.95s
    40:	learn: 0.2135813	total: 6.77s	remaining: 9.74s
    41:	learn: 0.2079965	total: 6.92s	remaining: 9.56s
    42:	learn: 0.2028560	total: 7.09s	remaining: 9.39s
    43:	learn: 0.1976444	total: 7.25s	remaining: 9.22s
    44:	learn: 0.1927907	total: 7.43s	remaining: 9.08s
    45:	learn: 0.1884086	total: 7.58s	remaining: 8.9s
    46:	learn: 0.1840093	total: 7.74s	remaining: 8.72s
    47:	learn: 0.1797904	total: 7.89s	remaining: 8.55s
    48:	learn: 0.1760554	total: 8.05s	remaining: 8.38s
    49:	learn: 0.1720131	total: 8.21s	remaining: 8.21s
    50:	learn: 0.1682758	total: 8.36s	remaining: 8.03s
    51:	learn: 0.1647124	total: 8.52s	remaining: 7.87s
    52:	learn: 0.1614551	total: 8.68s	remaining: 7.7s
    53:	learn: 0.1581725	total: 8.83s	remaining: 7.52s
    54:	learn: 0.1549823	total: 8.99s	remaining: 7.36s
    55:	learn: 0.1521923	total: 9.15s	remaining: 7.19s
    56:	learn: 0.1495247	total: 9.31s	remaining: 7.02s
    57:	learn: 0.1466614	total: 9.48s	remaining: 6.86s
    58:	learn: 0.1438751	total: 9.63s	remaining: 6.7s
    59:	learn: 0.1412870	total: 9.79s	remaining: 6.53s
    60:	learn: 0.1388011	total: 9.95s	remaining: 6.36s
    61:	learn: 0.1366200	total: 10.1s	remaining: 6.19s
    62:	learn: 0.1343182	total: 10.3s	remaining: 6.02s
    63:	learn: 0.1322079	total: 10.4s	remaining: 5.86s
    64:	learn: 0.1302441	total: 10.6s	remaining: 5.69s
    65:	learn: 0.1280744	total: 10.7s	remaining: 5.52s
    66:	learn: 0.1260693	total: 10.9s	remaining: 5.36s
    67:	learn: 0.1240396	total: 11s	remaining: 5.2s
    68:	learn: 0.1221543	total: 11.2s	remaining: 5.04s
    69:	learn: 0.1205356	total: 11.4s	remaining: 4.88s
    70:	learn: 0.1189803	total: 11.6s	remaining: 4.72s
    71:	learn: 0.1174242	total: 11.7s	remaining: 4.55s
    72:	learn: 0.1157429	total: 11.9s	remaining: 4.39s
    73:	learn: 0.1143307	total: 12s	remaining: 4.23s
    74:	learn: 0.1128187	total: 12.2s	remaining: 4.07s
    75:	learn: 0.1114100	total: 12.4s	remaining: 3.9s
    76:	learn: 0.1100077	total: 12.5s	remaining: 3.74s
    77:	learn: 0.1086397	total: 12.7s	remaining: 3.58s
    78:	learn: 0.1073425	total: 12.9s	remaining: 3.42s
    79:	learn: 0.1060327	total: 13s	remaining: 3.25s
    80:	learn: 0.1047586	total: 13.2s	remaining: 3.09s
    81:	learn: 0.1036338	total: 13.3s	remaining: 2.92s
    82:	learn: 0.1024827	total: 13.5s	remaining: 2.76s
    83:	learn: 0.1013388	total: 13.6s	remaining: 2.6s
    84:	learn: 0.1002781	total: 13.8s	remaining: 2.44s
    85:	learn: 0.0992700	total: 14s	remaining: 2.27s
    86:	learn: 0.0982523	total: 14.1s	remaining: 2.11s
    87:	learn: 0.0973169	total: 14.3s	remaining: 1.95s
    88:	learn: 0.0963894	total: 14.5s	remaining: 1.79s
    89:	learn: 0.0954948	total: 14.6s	remaining: 1.63s
    90:	learn: 0.0946000	total: 14.8s	remaining: 1.46s
    91:	learn: 0.0937364	total: 15s	remaining: 1.3s
    92:	learn: 0.0928752	total: 15.1s	remaining: 1.14s
    93:	learn: 0.0920618	total: 15.3s	remaining: 976ms
    94:	learn: 0.0912588	total: 15.5s	remaining: 814ms
    95:	learn: 0.0904425	total: 15.6s	remaining: 651ms
    96:	learn: 0.0897925	total: 15.8s	remaining: 488ms
    97:	learn: 0.0890728	total: 15.9s	remaining: 325ms
    98:	learn: 0.0883865	total: 16.1s	remaining: 163ms
    99:	learn: 0.0877798	total: 16.3s	remaining: 0us
    Trained model nº 1/27. Iterations: 100Depth: 6Learning rate: 0.01
    0:	learn: 0.6721173	total: 240ms	remaining: 47.8s
    1:	learn: 0.6511277	total: 470ms	remaining: 46.6s
    2:	learn: 0.6320133	total: 719ms	remaining: 47.2s
    3:	learn: 0.6122770	total: 961ms	remaining: 47.1s
    4:	learn: 0.5932728	total: 1.21s	remaining: 47.2s
    5:	learn: 0.5750998	total: 1.44s	remaining: 46.7s
    6:	learn: 0.5553126	total: 1.68s	remaining: 46.5s
    7:	learn: 0.5380592	total: 1.94s	remaining: 46.4s
    8:	learn: 0.5191167	total: 2.21s	remaining: 46.9s
    9:	learn: 0.5033274	total: 2.47s	remaining: 46.9s
    10:	learn: 0.4881556	total: 2.7s	remaining: 46.4s
    11:	learn: 0.4736366	total: 2.94s	remaining: 46.1s
    12:	learn: 0.4604138	total: 3.17s	remaining: 45.6s
    13:	learn: 0.4466343	total: 3.41s	remaining: 45.3s
    14:	learn: 0.4335158	total: 3.62s	remaining: 44.6s
    15:	learn: 0.4195680	total: 3.85s	remaining: 44.3s
    16:	learn: 0.4075345	total: 4.09s	remaining: 44s
    17:	learn: 0.3954725	total: 4.32s	remaining: 43.7s
    18:	learn: 0.3841871	total: 4.54s	remaining: 43.3s
    19:	learn: 0.3727671	total: 4.78s	remaining: 43s
    20:	learn: 0.3627640	total: 5.01s	remaining: 42.7s
    21:	learn: 0.3531812	total: 5.26s	remaining: 42.6s
    22:	learn: 0.3422263	total: 5.49s	remaining: 42.3s
    23:	learn: 0.3329761	total: 5.73s	remaining: 42s
    24:	learn: 0.3220980	total: 5.98s	remaining: 41.9s
    25:	learn: 0.3126139	total: 6.25s	remaining: 41.8s
    26:	learn: 0.3037511	total: 6.49s	remaining: 41.6s
    27:	learn: 0.2950041	total: 6.71s	remaining: 41.2s
    28:	learn: 0.2857028	total: 6.93s	remaining: 40.9s
    29:	learn: 0.2780346	total: 7.17s	remaining: 40.6s
    30:	learn: 0.2706072	total: 7.4s	remaining: 40.4s
    31:	learn: 0.2635030	total: 7.63s	remaining: 40s
    32:	learn: 0.2566218	total: 7.88s	remaining: 39.9s
    33:	learn: 0.2496485	total: 8.12s	remaining: 39.7s
    34:	learn: 0.2430494	total: 8.34s	remaining: 39.3s
    35:	learn: 0.2367357	total: 8.57s	remaining: 39.1s
    36:	learn: 0.2309237	total: 8.84s	remaining: 39s
    37:	learn: 0.2250167	total: 9.06s	remaining: 38.6s
    38:	learn: 0.2192896	total: 9.31s	remaining: 38.4s
    39:	learn: 0.2138682	total: 9.63s	remaining: 38.5s
    40:	learn: 0.2086918	total: 10s	remaining: 38.9s
    41:	learn: 0.2036367	total: 10.3s	remaining: 38.8s
    42:	learn: 0.1983810	total: 10.5s	remaining: 38.5s
    43:	learn: 0.1937409	total: 10.8s	remaining: 38.2s
    44:	learn: 0.1890846	total: 11s	remaining: 37.9s
    45:	learn: 0.1848253	total: 11.2s	remaining: 37.6s
    46:	learn: 0.1810035	total: 11.5s	remaining: 37.3s
    47:	learn: 0.1773768	total: 11.7s	remaining: 37s
    48:	learn: 0.1734652	total: 11.9s	remaining: 36.7s
    49:	learn: 0.1697887	total: 12.1s	remaining: 36.4s
    50:	learn: 0.1662473	total: 12.3s	remaining: 36s
    51:	learn: 0.1627369	total: 12.6s	remaining: 35.7s
    52:	learn: 0.1597490	total: 12.8s	remaining: 35.4s
    53:	learn: 0.1568650	total: 13s	remaining: 35.1s
    54:	learn: 0.1538726	total: 13.2s	remaining: 34.8s
    55:	learn: 0.1508596	total: 13.4s	remaining: 34.5s
    56:	learn: 0.1481001	total: 13.6s	remaining: 34.2s
    57:	learn: 0.1453093	total: 13.9s	remaining: 34s
    58:	learn: 0.1429340	total: 14.1s	remaining: 33.7s
    59:	learn: 0.1404031	total: 14.3s	remaining: 33.4s
    60:	learn: 0.1380116	total: 14.5s	remaining: 33.1s
    61:	learn: 0.1356813	total: 14.8s	remaining: 32.9s
    62:	learn: 0.1336399	total: 15s	remaining: 32.7s
    63:	learn: 0.1316033	total: 15.2s	remaining: 32.3s
    64:	learn: 0.1295008	total: 15.4s	remaining: 32s
    65:	learn: 0.1275148	total: 15.6s	remaining: 31.7s
    66:	learn: 0.1255376	total: 15.8s	remaining: 31.4s
    67:	learn: 0.1236912	total: 16s	remaining: 31.1s
    68:	learn: 0.1218867	total: 16.2s	remaining: 30.8s
    69:	learn: 0.1201495	total: 16.5s	remaining: 30.6s
    70:	learn: 0.1183830	total: 16.7s	remaining: 30.3s
    71:	learn: 0.1167508	total: 16.9s	remaining: 30s
    72:	learn: 0.1151789	total: 17.1s	remaining: 29.7s
    73:	learn: 0.1137948	total: 17.3s	remaining: 29.5s
    74:	learn: 0.1124710	total: 17.5s	remaining: 29.2s
    75:	learn: 0.1110674	total: 18s	remaining: 29.3s
    76:	learn: 0.1097009	total: 18.2s	remaining: 29.1s
    77:	learn: 0.1083161	total: 18.4s	remaining: 28.8s
    78:	learn: 0.1071248	total: 18.7s	remaining: 28.6s
    79:	learn: 0.1059220	total: 18.9s	remaining: 28.3s
    80:	learn: 0.1047703	total: 19.1s	remaining: 28s
    81:	learn: 0.1035720	total: 19.3s	remaining: 27.8s
    82:	learn: 0.1024514	total: 19.5s	remaining: 27.5s
    83:	learn: 0.1013786	total: 19.7s	remaining: 27.3s
    84:	learn: 0.1003245	total: 20s	remaining: 27s
    85:	learn: 0.0992763	total: 20.2s	remaining: 26.8s
    86:	learn: 0.0982705	total: 20.4s	remaining: 26.5s
    87:	learn: 0.0972547	total: 20.6s	remaining: 26.3s
    88:	learn: 0.0962828	total: 20.9s	remaining: 26s
    89:	learn: 0.0954401	total: 21.1s	remaining: 25.8s
    90:	learn: 0.0946769	total: 21.2s	remaining: 25.4s
    91:	learn: 0.0938398	total: 21.4s	remaining: 25.2s
    92:	learn: 0.0930304	total: 21.7s	remaining: 24.9s
    93:	learn: 0.0922417	total: 21.9s	remaining: 24.7s
    94:	learn: 0.0914560	total: 22.1s	remaining: 24.4s
    95:	learn: 0.0906504	total: 22.3s	remaining: 24.2s
    96:	learn: 0.0899103	total: 22.5s	remaining: 23.9s
    97:	learn: 0.0892892	total: 22.8s	remaining: 23.7s
    98:	learn: 0.0886195	total: 23s	remaining: 23.4s
    99:	learn: 0.0879507	total: 23.2s	remaining: 23.2s
    100:	learn: 0.0873033	total: 23.4s	remaining: 23s
    101:	learn: 0.0866354	total: 23.6s	remaining: 22.7s
    102:	learn: 0.0860123	total: 23.9s	remaining: 22.5s
    103:	learn: 0.0853802	total: 24.1s	remaining: 22.2s
    104:	learn: 0.0848050	total: 24.3s	remaining: 22s
    105:	learn: 0.0842556	total: 24.5s	remaining: 21.7s
    106:	learn: 0.0837288	total: 24.7s	remaining: 21.5s
    107:	learn: 0.0832147	total: 25s	remaining: 21.3s
    108:	learn: 0.0826971	total: 25.2s	remaining: 21s
    109:	learn: 0.0821804	total: 25.4s	remaining: 20.8s
    110:	learn: 0.0816756	total: 25.6s	remaining: 20.5s
    111:	learn: 0.0812060	total: 25.8s	remaining: 20.3s
    112:	learn: 0.0807615	total: 26s	remaining: 20s
    113:	learn: 0.0802775	total: 26.3s	remaining: 19.8s
    114:	learn: 0.0798120	total: 26.5s	remaining: 19.6s
    115:	learn: 0.0794000	total: 26.7s	remaining: 19.3s
    116:	learn: 0.0789685	total: 26.9s	remaining: 19.1s
    117:	learn: 0.0785724	total: 27.1s	remaining: 18.8s
    118:	learn: 0.0781943	total: 27.3s	remaining: 18.6s
    119:	learn: 0.0778646	total: 27.5s	remaining: 18.4s
    120:	learn: 0.0775095	total: 27.7s	remaining: 18.1s
    121:	learn: 0.0771810	total: 28s	remaining: 17.9s
    122:	learn: 0.0768102	total: 28.2s	remaining: 17.6s
    123:	learn: 0.0764574	total: 28.4s	remaining: 17.4s
    124:	learn: 0.0761381	total: 28.6s	remaining: 17.1s
    125:	learn: 0.0758326	total: 28.8s	remaining: 16.9s
    126:	learn: 0.0755357	total: 29s	remaining: 16.7s
    127:	learn: 0.0752280	total: 29.2s	remaining: 16.4s
    128:	learn: 0.0749452	total: 29.4s	remaining: 16.2s
    129:	learn: 0.0746580	total: 29.7s	remaining: 16s
    130:	learn: 0.0743638	total: 29.9s	remaining: 15.7s
    131:	learn: 0.0740666	total: 30.1s	remaining: 15.5s
    132:	learn: 0.0737921	total: 30.3s	remaining: 15.3s
    133:	learn: 0.0735132	total: 30.5s	remaining: 15s
    134:	learn: 0.0732679	total: 30.7s	remaining: 14.8s
    135:	learn: 0.0730187	total: 30.9s	remaining: 14.6s
    136:	learn: 0.0727910	total: 31.2s	remaining: 14.3s
    137:	learn: 0.0725480	total: 31.4s	remaining: 14.1s
    138:	learn: 0.0723216	total: 31.6s	remaining: 13.9s
    139:	learn: 0.0720954	total: 31.8s	remaining: 13.6s
    140:	learn: 0.0718645	total: 32s	remaining: 13.4s
    141:	learn: 0.0716419	total: 32.2s	remaining: 13.2s
    142:	learn: 0.0714442	total: 32.4s	remaining: 12.9s
    143:	learn: 0.0712480	total: 32.7s	remaining: 12.7s
    144:	learn: 0.0710362	total: 32.9s	remaining: 12.5s
    145:	learn: 0.0708529	total: 33.1s	remaining: 12.2s
    146:	learn: 0.0706747	total: 33.3s	remaining: 12s
    147:	learn: 0.0705213	total: 33.5s	remaining: 11.8s
    148:	learn: 0.0703559	total: 33.7s	remaining: 11.6s
    149:	learn: 0.0701748	total: 33.9s	remaining: 11.3s
    150:	learn: 0.0700150	total: 34.1s	remaining: 11.1s
    151:	learn: 0.0698341	total: 34.4s	remaining: 10.9s
    152:	learn: 0.0696820	total: 34.6s	remaining: 10.6s
    153:	learn: 0.0695164	total: 34.8s	remaining: 10.4s
    154:	learn: 0.0693738	total: 35s	remaining: 10.2s
    155:	learn: 0.0692336	total: 35.2s	remaining: 9.93s
    156:	learn: 0.0690728	total: 35.4s	remaining: 9.7s
    157:	learn: 0.0689352	total: 35.6s	remaining: 9.47s
    158:	learn: 0.0687943	total: 35.8s	remaining: 9.24s
    159:	learn: 0.0686662	total: 36.1s	remaining: 9.01s
    160:	learn: 0.0685261	total: 36.3s	remaining: 8.79s
    161:	learn: 0.0683965	total: 36.5s	remaining: 8.56s
    162:	learn: 0.0682561	total: 36.7s	remaining: 8.33s
    163:	learn: 0.0681315	total: 36.9s	remaining: 8.11s
    164:	learn: 0.0680151	total: 37.1s	remaining: 7.88s
    165:	learn: 0.0678858	total: 37.4s	remaining: 7.65s
    166:	learn: 0.0677736	total: 37.6s	remaining: 7.43s
    167:	learn: 0.0676463	total: 37.8s	remaining: 7.2s
    168:	learn: 0.0675613	total: 38s	remaining: 6.97s
    169:	learn: 0.0674611	total: 38.2s	remaining: 6.74s
    170:	learn: 0.0673480	total: 38.4s	remaining: 6.51s
    171:	learn: 0.0672336	total: 38.6s	remaining: 6.29s
    172:	learn: 0.0671432	total: 38.8s	remaining: 6.06s
    173:	learn: 0.0670383	total: 39s	remaining: 5.83s
    174:	learn: 0.0669515	total: 39.2s	remaining: 5.61s
    175:	learn: 0.0668545	total: 39.5s	remaining: 5.38s
    176:	learn: 0.0667625	total: 39.7s	remaining: 5.16s
    177:	learn: 0.0666636	total: 39.9s	remaining: 4.93s
    178:	learn: 0.0665829	total: 40.1s	remaining: 4.71s
    179:	learn: 0.0664845	total: 40.3s	remaining: 4.48s
    180:	learn: 0.0664085	total: 40.5s	remaining: 4.25s
    181:	learn: 0.0663144	total: 40.7s	remaining: 4.03s
    182:	learn: 0.0662447	total: 41s	remaining: 3.8s
    183:	learn: 0.0661620	total: 41.2s	remaining: 3.58s
    184:	learn: 0.0660850	total: 41.4s	remaining: 3.35s
    185:	learn: 0.0660059	total: 41.6s	remaining: 3.13s
    186:	learn: 0.0659341	total: 41.8s	remaining: 2.91s
    187:	learn: 0.0658576	total: 42s	remaining: 2.68s
    188:	learn: 0.0657828	total: 42.2s	remaining: 2.46s
    189:	learn: 0.0657065	total: 42.4s	remaining: 2.23s
    190:	learn: 0.0656464	total: 42.7s	remaining: 2.01s
    191:	learn: 0.0655805	total: 42.9s	remaining: 1.79s
    192:	learn: 0.0655236	total: 43.1s	remaining: 1.56s
    193:	learn: 0.0654587	total: 43.3s	remaining: 1.34s
    194:	learn: 0.0653928	total: 43.5s	remaining: 1.12s
    195:	learn: 0.0653391	total: 43.8s	remaining: 893ms
    196:	learn: 0.0652753	total: 44s	remaining: 670ms
    197:	learn: 0.0652225	total: 44.2s	remaining: 447ms
    198:	learn: 0.0651578	total: 44.4s	remaining: 223ms
    199:	learn: 0.0651027	total: 44.6s	remaining: 0us
    Trained model nº 2/27. Iterations: 200Depth: 6Learning rate: 0.01
    0:	learn: 0.6721173	total: 263ms	remaining: 1m 18s
    1:	learn: 0.6511277	total: 485ms	remaining: 1m 12s
    2:	learn: 0.6320133	total: 718ms	remaining: 1m 11s
    3:	learn: 0.6122770	total: 931ms	remaining: 1m 8s
    4:	learn: 0.5932728	total: 1.15s	remaining: 1m 7s
    5:	learn: 0.5750998	total: 1.38s	remaining: 1m 7s
    6:	learn: 0.5553126	total: 1.58s	remaining: 1m 6s
    7:	learn: 0.5380592	total: 1.79s	remaining: 1m 5s
    8:	learn: 0.5191167	total: 2.01s	remaining: 1m 5s
    9:	learn: 0.5033274	total: 2.24s	remaining: 1m 4s
    10:	learn: 0.4881556	total: 2.46s	remaining: 1m 4s
    11:	learn: 0.4736366	total: 2.69s	remaining: 1m 4s
    12:	learn: 0.4604138	total: 2.9s	remaining: 1m 3s
    13:	learn: 0.4466343	total: 3.12s	remaining: 1m 3s
    14:	learn: 0.4335158	total: 3.32s	remaining: 1m 3s
    15:	learn: 0.4195680	total: 3.54s	remaining: 1m 2s
    16:	learn: 0.4075345	total: 3.77s	remaining: 1m 2s
    17:	learn: 0.3954725	total: 3.99s	remaining: 1m 2s
    18:	learn: 0.3841871	total: 4.2s	remaining: 1m 2s
    19:	learn: 0.3727671	total: 4.41s	remaining: 1m 1s
    20:	learn: 0.3627640	total: 4.63s	remaining: 1m 1s
    21:	learn: 0.3531812	total: 4.85s	remaining: 1m 1s
    22:	learn: 0.3422263	total: 5.06s	remaining: 1m
    23:	learn: 0.3329761	total: 5.28s	remaining: 1m
    24:	learn: 0.3220980	total: 5.5s	remaining: 1m
    25:	learn: 0.3126139	total: 5.72s	remaining: 1m
    26:	learn: 0.3037511	total: 5.96s	remaining: 1m
    27:	learn: 0.2950041	total: 6.19s	remaining: 1m
    28:	learn: 0.2857028	total: 6.42s	remaining: 1m
    29:	learn: 0.2780346	total: 6.63s	remaining: 59.7s
    30:	learn: 0.2706072	total: 6.85s	remaining: 59.5s
    31:	learn: 0.2635030	total: 7.08s	remaining: 59.3s
    32:	learn: 0.2566218	total: 7.29s	remaining: 59s
    33:	learn: 0.2496485	total: 7.51s	remaining: 58.8s
    34:	learn: 0.2430494	total: 7.72s	remaining: 58.5s
    35:	learn: 0.2367357	total: 7.93s	remaining: 58.2s
    36:	learn: 0.2309237	total: 8.16s	remaining: 58s
    37:	learn: 0.2250167	total: 8.35s	remaining: 57.6s
    38:	learn: 0.2192896	total: 8.56s	remaining: 57.3s
    39:	learn: 0.2138682	total: 8.78s	remaining: 57s
    40:	learn: 0.2086918	total: 9s	remaining: 56.9s
    41:	learn: 0.2036367	total: 9.22s	remaining: 56.6s
    42:	learn: 0.1983810	total: 9.42s	remaining: 56.3s
    43:	learn: 0.1937409	total: 9.61s	remaining: 55.9s
    44:	learn: 0.1890846	total: 9.82s	remaining: 55.7s
    45:	learn: 0.1848253	total: 10.1s	remaining: 55.5s
    46:	learn: 0.1810035	total: 10.3s	remaining: 55.3s
    47:	learn: 0.1773768	total: 10.5s	remaining: 55s
    48:	learn: 0.1734652	total: 10.7s	remaining: 54.8s
    49:	learn: 0.1697887	total: 10.9s	remaining: 54.5s
    50:	learn: 0.1662473	total: 11.1s	remaining: 54.2s
    51:	learn: 0.1627369	total: 11.3s	remaining: 53.9s
    52:	learn: 0.1597490	total: 11.5s	remaining: 53.7s
    53:	learn: 0.1568650	total: 11.7s	remaining: 53.5s
    54:	learn: 0.1538726	total: 12s	remaining: 53.2s
    55:	learn: 0.1508596	total: 12.2s	remaining: 53.1s
    56:	learn: 0.1481001	total: 12.4s	remaining: 52.9s
    57:	learn: 0.1453093	total: 12.6s	remaining: 52.7s
    58:	learn: 0.1429340	total: 12.9s	remaining: 52.5s
    59:	learn: 0.1404031	total: 13s	remaining: 52.2s
    60:	learn: 0.1380116	total: 13.3s	remaining: 52s
    61:	learn: 0.1356813	total: 13.5s	remaining: 51.8s
    62:	learn: 0.1336399	total: 13.7s	remaining: 51.6s
    63:	learn: 0.1316033	total: 13.9s	remaining: 51.3s
    64:	learn: 0.1295008	total: 14.1s	remaining: 51.1s
    65:	learn: 0.1275148	total: 14.4s	remaining: 50.9s
    66:	learn: 0.1255376	total: 14.6s	remaining: 50.6s
    67:	learn: 0.1236912	total: 14.8s	remaining: 50.4s
    68:	learn: 0.1218867	total: 15s	remaining: 50.1s
    69:	learn: 0.1201495	total: 15.2s	remaining: 49.9s
    70:	learn: 0.1183830	total: 15.4s	remaining: 49.7s
    71:	learn: 0.1167508	total: 15.6s	remaining: 49.4s
    72:	learn: 0.1151789	total: 15.8s	remaining: 49.2s
    73:	learn: 0.1137948	total: 16.1s	remaining: 49s
    74:	learn: 0.1124710	total: 16.2s	remaining: 48.7s
    75:	learn: 0.1110674	total: 16.4s	remaining: 48.5s
    76:	learn: 0.1097009	total: 16.7s	remaining: 48.2s
    77:	learn: 0.1083161	total: 16.9s	remaining: 48s
    78:	learn: 0.1071248	total: 17.1s	remaining: 47.8s
    79:	learn: 0.1059220	total: 17.3s	remaining: 47.6s
    80:	learn: 0.1047703	total: 17.5s	remaining: 47.4s
    81:	learn: 0.1035720	total: 17.7s	remaining: 47.2s
    82:	learn: 0.1024514	total: 17.9s	remaining: 46.9s
    83:	learn: 0.1013786	total: 18.1s	remaining: 46.6s
    84:	learn: 0.1003245	total: 18.4s	remaining: 46.4s
    85:	learn: 0.0992763	total: 18.6s	remaining: 46.2s
    86:	learn: 0.0982705	total: 18.8s	remaining: 46s
    87:	learn: 0.0972547	total: 19s	remaining: 45.8s
    88:	learn: 0.0962828	total: 19.2s	remaining: 45.6s
    89:	learn: 0.0954401	total: 19.4s	remaining: 45.4s
    90:	learn: 0.0946769	total: 19.6s	remaining: 45s
    91:	learn: 0.0938398	total: 19.8s	remaining: 44.8s
    92:	learn: 0.0930304	total: 20s	remaining: 44.6s
    93:	learn: 0.0922417	total: 20.2s	remaining: 44.3s
    94:	learn: 0.0914560	total: 20.5s	remaining: 44.1s
    95:	learn: 0.0906504	total: 20.7s	remaining: 43.9s
    96:	learn: 0.0899103	total: 20.9s	remaining: 43.7s
    97:	learn: 0.0892892	total: 21.1s	remaining: 43.5s
    98:	learn: 0.0886195	total: 21.3s	remaining: 43.3s
    99:	learn: 0.0879507	total: 21.5s	remaining: 43.1s
    100:	learn: 0.0873033	total: 21.8s	remaining: 42.9s
    101:	learn: 0.0866354	total: 22s	remaining: 42.7s
    102:	learn: 0.0860123	total: 22.2s	remaining: 42.5s
    103:	learn: 0.0853802	total: 22.4s	remaining: 42.2s
    104:	learn: 0.0848050	total: 22.6s	remaining: 42s
    105:	learn: 0.0842556	total: 22.9s	remaining: 41.8s
    106:	learn: 0.0837288	total: 23.1s	remaining: 41.6s
    107:	learn: 0.0832147	total: 23.3s	remaining: 41.4s
    108:	learn: 0.0826971	total: 23.5s	remaining: 41.2s
    109:	learn: 0.0821804	total: 23.7s	remaining: 41s
    110:	learn: 0.0816756	total: 23.9s	remaining: 40.7s
    111:	learn: 0.0812060	total: 24.2s	remaining: 40.5s
    112:	learn: 0.0807615	total: 24.4s	remaining: 40.3s
    113:	learn: 0.0802775	total: 24.6s	remaining: 40.1s
    114:	learn: 0.0798120	total: 24.8s	remaining: 39.9s
    115:	learn: 0.0794000	total: 25s	remaining: 39.7s
    116:	learn: 0.0789685	total: 25.2s	remaining: 39.4s
    117:	learn: 0.0785724	total: 25.4s	remaining: 39.2s
    118:	learn: 0.0781943	total: 25.6s	remaining: 39s
    119:	learn: 0.0778646	total: 25.9s	remaining: 38.8s
    120:	learn: 0.0775095	total: 26.1s	remaining: 38.6s
    121:	learn: 0.0771810	total: 26.3s	remaining: 38.4s
    122:	learn: 0.0768102	total: 26.5s	remaining: 38.2s
    123:	learn: 0.0764574	total: 26.7s	remaining: 37.9s
    124:	learn: 0.0761381	total: 26.9s	remaining: 37.7s
    125:	learn: 0.0758326	total: 27.2s	remaining: 37.5s
    126:	learn: 0.0755357	total: 27.4s	remaining: 37.3s
    127:	learn: 0.0752280	total: 27.6s	remaining: 37.1s
    128:	learn: 0.0749452	total: 27.8s	remaining: 36.9s
    129:	learn: 0.0746580	total: 28s	remaining: 36.6s
    130:	learn: 0.0743638	total: 28.2s	remaining: 36.4s
    131:	learn: 0.0740666	total: 28.4s	remaining: 36.2s
    132:	learn: 0.0737921	total: 28.6s	remaining: 36s
    133:	learn: 0.0735132	total: 28.9s	remaining: 35.8s
    134:	learn: 0.0732679	total: 29.1s	remaining: 35.5s
    135:	learn: 0.0730187	total: 29.3s	remaining: 35.3s
    136:	learn: 0.0727910	total: 29.5s	remaining: 35.1s
    137:	learn: 0.0725480	total: 29.7s	remaining: 34.9s
    138:	learn: 0.0723216	total: 29.9s	remaining: 34.7s
    139:	learn: 0.0720954	total: 30.2s	remaining: 34.5s
    140:	learn: 0.0718645	total: 30.4s	remaining: 34.2s
    141:	learn: 0.0716419	total: 30.6s	remaining: 34s
    142:	learn: 0.0714442	total: 30.8s	remaining: 33.8s
    143:	learn: 0.0712480	total: 31s	remaining: 33.6s
    144:	learn: 0.0710362	total: 31.2s	remaining: 33.4s
    145:	learn: 0.0708529	total: 31.4s	remaining: 33.2s
    146:	learn: 0.0706747	total: 31.6s	remaining: 32.9s
    147:	learn: 0.0705213	total: 31.9s	remaining: 32.7s
    148:	learn: 0.0703559	total: 32.1s	remaining: 32.5s
    149:	learn: 0.0701748	total: 32.3s	remaining: 32.3s
    150:	learn: 0.0700150	total: 32.5s	remaining: 32.1s
    151:	learn: 0.0698341	total: 32.7s	remaining: 31.8s
    152:	learn: 0.0696820	total: 32.9s	remaining: 31.6s
    153:	learn: 0.0695164	total: 33.1s	remaining: 31.4s
    154:	learn: 0.0693738	total: 33.4s	remaining: 31.2s
    155:	learn: 0.0692336	total: 33.6s	remaining: 31s
    156:	learn: 0.0690728	total: 33.8s	remaining: 30.7s
    157:	learn: 0.0689352	total: 34s	remaining: 30.5s
    158:	learn: 0.0687943	total: 34.2s	remaining: 30.3s
    159:	learn: 0.0686662	total: 34.4s	remaining: 30.1s
    160:	learn: 0.0685261	total: 34.6s	remaining: 29.9s
    161:	learn: 0.0683965	total: 34.9s	remaining: 29.7s
    162:	learn: 0.0682561	total: 35.1s	remaining: 29.5s
    163:	learn: 0.0681315	total: 35.3s	remaining: 29.3s
    164:	learn: 0.0680151	total: 35.5s	remaining: 29s
    165:	learn: 0.0678858	total: 35.7s	remaining: 28.8s
    166:	learn: 0.0677736	total: 35.9s	remaining: 28.6s
    167:	learn: 0.0676463	total: 36.1s	remaining: 28.4s
    168:	learn: 0.0675613	total: 36.3s	remaining: 28.2s
    169:	learn: 0.0674611	total: 36.6s	remaining: 28s
    170:	learn: 0.0673480	total: 36.8s	remaining: 27.7s
    171:	learn: 0.0672336	total: 37s	remaining: 27.5s
    172:	learn: 0.0671432	total: 37.2s	remaining: 27.3s
    173:	learn: 0.0670383	total: 37.4s	remaining: 27.1s
    174:	learn: 0.0669515	total: 37.6s	remaining: 26.9s
    175:	learn: 0.0668545	total: 37.8s	remaining: 26.7s
    176:	learn: 0.0667625	total: 38s	remaining: 26.4s
    177:	learn: 0.0666636	total: 38.2s	remaining: 26.2s
    178:	learn: 0.0665829	total: 38.5s	remaining: 26s
    179:	learn: 0.0664845	total: 38.6s	remaining: 25.8s
    180:	learn: 0.0664085	total: 38.8s	remaining: 25.5s
    181:	learn: 0.0663144	total: 39.1s	remaining: 25.3s
    182:	learn: 0.0662447	total: 39.3s	remaining: 25.1s
    183:	learn: 0.0661620	total: 39.5s	remaining: 24.9s
    184:	learn: 0.0660850	total: 39.7s	remaining: 24.7s
    185:	learn: 0.0660059	total: 39.9s	remaining: 24.5s
    186:	learn: 0.0659341	total: 40.1s	remaining: 24.3s
    187:	learn: 0.0658576	total: 40.4s	remaining: 24s
    188:	learn: 0.0657828	total: 40.6s	remaining: 23.8s
    189:	learn: 0.0657065	total: 40.8s	remaining: 23.6s
    190:	learn: 0.0656464	total: 41s	remaining: 23.4s
    191:	learn: 0.0655805	total: 41.2s	remaining: 23.2s
    192:	learn: 0.0655236	total: 41.4s	remaining: 23s
    193:	learn: 0.0654587	total: 41.7s	remaining: 22.8s
    194:	learn: 0.0653928	total: 41.9s	remaining: 22.6s
    195:	learn: 0.0653391	total: 42.1s	remaining: 22.3s
    196:	learn: 0.0652753	total: 42.3s	remaining: 22.1s
    197:	learn: 0.0652225	total: 42.5s	remaining: 21.9s
    198:	learn: 0.0651578	total: 42.7s	remaining: 21.7s
    199:	learn: 0.0651027	total: 42.9s	remaining: 21.5s
    200:	learn: 0.0650418	total: 43.1s	remaining: 21.2s
    201:	learn: 0.0649955	total: 43.3s	remaining: 21s
    202:	learn: 0.0649363	total: 43.5s	remaining: 20.8s
    203:	learn: 0.0648864	total: 43.8s	remaining: 20.6s
    204:	learn: 0.0648304	total: 44s	remaining: 20.4s
    205:	learn: 0.0647819	total: 44.2s	remaining: 20.2s
    206:	learn: 0.0647392	total: 44.4s	remaining: 19.9s
    207:	learn: 0.0646903	total: 44.6s	remaining: 19.7s
    208:	learn: 0.0646364	total: 44.8s	remaining: 19.5s
    209:	learn: 0.0645976	total: 45s	remaining: 19.3s
    210:	learn: 0.0645510	total: 45.2s	remaining: 19.1s
    211:	learn: 0.0645125	total: 45.4s	remaining: 18.8s
    212:	learn: 0.0644634	total: 45.6s	remaining: 18.6s
    213:	learn: 0.0644168	total: 45.8s	remaining: 18.4s
    214:	learn: 0.0643815	total: 46s	remaining: 18.2s
    215:	learn: 0.0643434	total: 46.2s	remaining: 18s
    216:	learn: 0.0643055	total: 46.5s	remaining: 17.8s
    217:	learn: 0.0642606	total: 46.6s	remaining: 17.5s
    218:	learn: 0.0642222	total: 46.9s	remaining: 17.3s
    219:	learn: 0.0641909	total: 47.1s	remaining: 17.1s
    220:	learn: 0.0641496	total: 47.3s	remaining: 16.9s
    221:	learn: 0.0641121	total: 47.5s	remaining: 16.7s
    222:	learn: 0.0640711	total: 47.7s	remaining: 16.5s
    223:	learn: 0.0640374	total: 47.9s	remaining: 16.3s
    224:	learn: 0.0640094	total: 48.2s	remaining: 16.1s
    225:	learn: 0.0639729	total: 48.3s	remaining: 15.8s
    226:	learn: 0.0639463	total: 48.6s	remaining: 15.6s
    227:	learn: 0.0639119	total: 48.8s	remaining: 15.4s
    228:	learn: 0.0638749	total: 49s	remaining: 15.2s
    229:	learn: 0.0638534	total: 49.2s	remaining: 15s
    230:	learn: 0.0638188	total: 49.4s	remaining: 14.8s
    231:	learn: 0.0637940	total: 49.6s	remaining: 14.5s
    232:	learn: 0.0637679	total: 49.8s	remaining: 14.3s
    233:	learn: 0.0637444	total: 50s	remaining: 14.1s
    234:	learn: 0.0637249	total: 50.3s	remaining: 13.9s
    235:	learn: 0.0636933	total: 50.5s	remaining: 13.7s
    236:	learn: 0.0636719	total: 50.7s	remaining: 13.5s
    237:	learn: 0.0636411	total: 50.9s	remaining: 13.3s
    238:	learn: 0.0636199	total: 51.1s	remaining: 13.1s
    239:	learn: 0.0635911	total: 51.4s	remaining: 12.8s
    240:	learn: 0.0635723	total: 51.6s	remaining: 12.6s
    241:	learn: 0.0635499	total: 51.8s	remaining: 12.4s
    242:	learn: 0.0635255	total: 52s	remaining: 12.2s
    243:	learn: 0.0635054	total: 52.3s	remaining: 12s
    244:	learn: 0.0634800	total: 52.5s	remaining: 11.8s
    245:	learn: 0.0634524	total: 52.7s	remaining: 11.6s
    246:	learn: 0.0634281	total: 53s	remaining: 11.4s
    247:	learn: 0.0634017	total: 53.2s	remaining: 11.2s
    248:	learn: 0.0633838	total: 53.5s	remaining: 10.9s
    249:	learn: 0.0633581	total: 53.7s	remaining: 10.7s
    250:	learn: 0.0633408	total: 54s	remaining: 10.5s
    251:	learn: 0.0633242	total: 54.2s	remaining: 10.3s
    252:	learn: 0.0633077	total: 54.5s	remaining: 10.1s
    253:	learn: 0.0632915	total: 54.7s	remaining: 9.91s
    254:	learn: 0.0632688	total: 55s	remaining: 9.7s
    255:	learn: 0.0632517	total: 55.2s	remaining: 9.49s
    256:	learn: 0.0632354	total: 55.5s	remaining: 9.28s
    257:	learn: 0.0632206	total: 55.7s	remaining: 9.07s
    258:	learn: 0.0632062	total: 56s	remaining: 8.86s
    259:	learn: 0.0631894	total: 56.2s	remaining: 8.65s
    260:	learn: 0.0631706	total: 56.5s	remaining: 8.44s
    261:	learn: 0.0631595	total: 56.7s	remaining: 8.22s
    262:	learn: 0.0631428	total: 57s	remaining: 8.01s
    263:	learn: 0.0631270	total: 57.2s	remaining: 7.8s
    264:	learn: 0.0631137	total: 57.4s	remaining: 7.59s
    265:	learn: 0.0631012	total: 57.7s	remaining: 7.37s
    266:	learn: 0.0630878	total: 57.9s	remaining: 7.16s
    267:	learn: 0.0630745	total: 58.2s	remaining: 6.95s
    268:	learn: 0.0630567	total: 58.4s	remaining: 6.73s
    269:	learn: 0.0630424	total: 58.7s	remaining: 6.52s
    270:	learn: 0.0630324	total: 58.9s	remaining: 6.3s
    271:	learn: 0.0630163	total: 59.1s	remaining: 6.09s
    272:	learn: 0.0630041	total: 59.4s	remaining: 5.87s
    273:	learn: 0.0629899	total: 59.6s	remaining: 5.66s
    274:	learn: 0.0629777	total: 59.9s	remaining: 5.44s
    275:	learn: 0.0629658	total: 1m	remaining: 5.22s
    276:	learn: 0.0629522	total: 1m	remaining: 5.01s
    277:	learn: 0.0629405	total: 1m	remaining: 4.79s
    278:	learn: 0.0629290	total: 1m	remaining: 4.57s
    279:	learn: 0.0629155	total: 1m 1s	remaining: 4.36s
    280:	learn: 0.0629022	total: 1m 1s	remaining: 4.14s
    281:	learn: 0.0628904	total: 1m 1s	remaining: 3.92s
    282:	learn: 0.0628796	total: 1m 1s	remaining: 3.71s
    283:	learn: 0.0628640	total: 1m 1s	remaining: 3.49s
    284:	learn: 0.0628536	total: 1m 2s	remaining: 3.27s
    285:	learn: 0.0628452	total: 1m 2s	remaining: 3.05s
    286:	learn: 0.0628351	total: 1m 2s	remaining: 2.83s
    287:	learn: 0.0628224	total: 1m 2s	remaining: 2.62s
    288:	learn: 0.0628119	total: 1m 3s	remaining: 2.4s
    289:	learn: 0.0627974	total: 1m 3s	remaining: 2.18s
    290:	learn: 0.0627882	total: 1m 3s	remaining: 1.96s
    291:	learn: 0.0627763	total: 1m 3s	remaining: 1.74s
    292:	learn: 0.0627669	total: 1m 3s	remaining: 1.53s
    293:	learn: 0.0627574	total: 1m 4s	remaining: 1.31s
    294:	learn: 0.0627437	total: 1m 4s	remaining: 1.09s
    295:	learn: 0.0627362	total: 1m 4s	remaining: 873ms
    296:	learn: 0.0627264	total: 1m 4s	remaining: 654ms
    297:	learn: 0.0627148	total: 1m 5s	remaining: 436ms
    298:	learn: 0.0627065	total: 1m 5s	remaining: 218ms
    299:	learn: 0.0626979	total: 1m 5s	remaining: 0us
    Trained model nº 3/27. Iterations: 300Depth: 6Learning rate: 0.01
    0:	learn: 0.6716917	total: 199ms	remaining: 19.7s
    1:	learn: 0.6486968	total: 393ms	remaining: 19.2s
    2:	learn: 0.6277014	total: 610ms	remaining: 19.7s
    3:	learn: 0.6079818	total: 812ms	remaining: 19.5s
    4:	learn: 0.5874282	total: 1.01s	remaining: 19.2s
    5:	learn: 0.5667149	total: 1.2s	remaining: 18.8s
    6:	learn: 0.5502262	total: 1.39s	remaining: 18.5s
    7:	learn: 0.5333407	total: 1.58s	remaining: 18.2s
    8:	learn: 0.5172447	total: 1.74s	remaining: 17.6s
    9:	learn: 0.5015513	total: 1.91s	remaining: 17.2s
    10:	learn: 0.4874725	total: 2.04s	remaining: 16.5s
    11:	learn: 0.4721071	total: 2.21s	remaining: 16.2s
    12:	learn: 0.4566255	total: 2.39s	remaining: 16s
    13:	learn: 0.4425078	total: 2.57s	remaining: 15.8s
    14:	learn: 0.4280891	total: 2.73s	remaining: 15.5s
    15:	learn: 0.4162141	total: 2.91s	remaining: 15.3s
    16:	learn: 0.4041370	total: 3.09s	remaining: 15.1s
    17:	learn: 0.3913720	total: 3.26s	remaining: 14.9s
    18:	learn: 0.3802428	total: 3.44s	remaining: 14.6s
    19:	learn: 0.3687054	total: 3.6s	remaining: 14.4s
    20:	learn: 0.3570653	total: 3.78s	remaining: 14.2s
    21:	learn: 0.3479400	total: 3.91s	remaining: 13.9s
    22:	learn: 0.3372182	total: 4.09s	remaining: 13.7s
    23:	learn: 0.3276129	total: 4.26s	remaining: 13.5s
    24:	learn: 0.3178711	total: 4.43s	remaining: 13.3s
    25:	learn: 0.3084866	total: 4.61s	remaining: 13.1s
    26:	learn: 0.2999429	total: 4.78s	remaining: 12.9s
    27:	learn: 0.2916178	total: 4.97s	remaining: 12.8s
    28:	learn: 0.2833927	total: 5.15s	remaining: 12.6s
    29:	learn: 0.2758657	total: 5.3s	remaining: 12.4s
    30:	learn: 0.2681847	total: 5.47s	remaining: 12.2s
    31:	learn: 0.2607587	total: 5.65s	remaining: 12s
    32:	learn: 0.2533396	total: 5.82s	remaining: 11.8s
    33:	learn: 0.2469949	total: 5.99s	remaining: 11.6s
    34:	learn: 0.2408163	total: 6.17s	remaining: 11.5s
    35:	learn: 0.2345747	total: 6.33s	remaining: 11.3s
    36:	learn: 0.2284505	total: 6.51s	remaining: 11.1s
    37:	learn: 0.2227302	total: 6.67s	remaining: 10.9s
    38:	learn: 0.2165614	total: 6.83s	remaining: 10.7s
    39:	learn: 0.2115232	total: 7s	remaining: 10.5s
    40:	learn: 0.2064321	total: 7.17s	remaining: 10.3s
    41:	learn: 0.2014625	total: 7.33s	remaining: 10.1s
    42:	learn: 0.1966798	total: 7.5s	remaining: 9.94s
    43:	learn: 0.1920346	total: 7.67s	remaining: 9.76s
    44:	learn: 0.1876774	total: 7.83s	remaining: 9.57s
    45:	learn: 0.1833681	total: 8s	remaining: 9.39s
    46:	learn: 0.1793437	total: 8.15s	remaining: 9.2s
    47:	learn: 0.1755163	total: 8.32s	remaining: 9.02s
    48:	learn: 0.1717338	total: 8.49s	remaining: 8.84s
    49:	learn: 0.1683886	total: 8.66s	remaining: 8.66s
    50:	learn: 0.1648603	total: 8.82s	remaining: 8.48s
    51:	learn: 0.1615626	total: 9s	remaining: 8.31s
    52:	learn: 0.1582928	total: 9.18s	remaining: 8.14s
    53:	learn: 0.1553477	total: 9.36s	remaining: 7.97s
    54:	learn: 0.1524492	total: 9.53s	remaining: 7.8s
    55:	learn: 0.1497351	total: 9.7s	remaining: 7.62s
    56:	learn: 0.1469873	total: 9.87s	remaining: 7.44s
    57:	learn: 0.1445197	total: 10s	remaining: 7.27s
    58:	learn: 0.1419653	total: 10.2s	remaining: 7.09s
    59:	learn: 0.1395764	total: 10.4s	remaining: 6.91s
    60:	learn: 0.1373688	total: 10.5s	remaining: 6.73s
    61:	learn: 0.1350071	total: 10.7s	remaining: 6.56s
    62:	learn: 0.1327459	total: 10.9s	remaining: 6.4s
    63:	learn: 0.1305321	total: 11.1s	remaining: 6.22s
    64:	learn: 0.1283937	total: 11.2s	remaining: 6.05s
    65:	learn: 0.1264149	total: 11.4s	remaining: 5.88s
    66:	learn: 0.1245101	total: 11.6s	remaining: 5.7s
    67:	learn: 0.1226378	total: 11.7s	remaining: 5.52s
    68:	learn: 0.1208312	total: 11.9s	remaining: 5.35s
    69:	learn: 0.1192209	total: 12.1s	remaining: 5.17s
    70:	learn: 0.1175415	total: 12.2s	remaining: 5s
    71:	learn: 0.1158897	total: 12.4s	remaining: 4.82s
    72:	learn: 0.1143424	total: 12.6s	remaining: 4.65s
    73:	learn: 0.1129057	total: 12.7s	remaining: 4.47s
    74:	learn: 0.1115530	total: 12.9s	remaining: 4.3s
    75:	learn: 0.1102228	total: 13.1s	remaining: 4.13s
    76:	learn: 0.1088423	total: 13.2s	remaining: 3.96s
    77:	learn: 0.1074926	total: 13.4s	remaining: 3.78s
    78:	learn: 0.1062215	total: 13.6s	remaining: 3.61s
    79:	learn: 0.1049664	total: 13.8s	remaining: 3.44s
    80:	learn: 0.1037792	total: 13.9s	remaining: 3.26s
    81:	learn: 0.1025948	total: 14.1s	remaining: 3.09s
    82:	learn: 0.1014431	total: 14.3s	remaining: 2.92s
    83:	learn: 0.1003361	total: 14.4s	remaining: 2.75s
    84:	learn: 0.0994052	total: 14.6s	remaining: 2.57s
    85:	learn: 0.0983543	total: 14.7s	remaining: 2.4s
    86:	learn: 0.0973314	total: 14.9s	remaining: 2.23s
    87:	learn: 0.0963498	total: 15.1s	remaining: 2.06s
    88:	learn: 0.0954288	total: 15.3s	remaining: 1.89s
    89:	learn: 0.0945429	total: 15.4s	remaining: 1.71s
    90:	learn: 0.0936426	total: 15.6s	remaining: 1.54s
    91:	learn: 0.0928824	total: 15.8s	remaining: 1.37s
    92:	learn: 0.0920679	total: 15.9s	remaining: 1.2s
    93:	learn: 0.0912614	total: 16.1s	remaining: 1.03s
    94:	learn: 0.0904905	total: 16.3s	remaining: 857ms
    95:	learn: 0.0897267	total: 16.5s	remaining: 686ms
    96:	learn: 0.0889936	total: 16.6s	remaining: 515ms
    97:	learn: 0.0882726	total: 16.8s	remaining: 343ms
    98:	learn: 0.0875891	total: 17s	remaining: 172ms
    99:	learn: 0.0869403	total: 17.2s	remaining: 0us
    Trained model nº 4/27. Iterations: 100Depth: 8Learning rate: 0.01
    0:	learn: 0.6720913	total: 315ms	remaining: 1m 2s
    1:	learn: 0.6511983	total: 600ms	remaining: 59.4s
    2:	learn: 0.6307706	total: 887ms	remaining: 58.2s
    3:	learn: 0.6120845	total: 1.16s	remaining: 56.6s
    4:	learn: 0.5922499	total: 1.43s	remaining: 55.7s
    5:	learn: 0.5729459	total: 1.69s	remaining: 54.7s
    6:	learn: 0.5553558	total: 1.97s	remaining: 54.3s
    7:	learn: 0.5376874	total: 2.23s	remaining: 53.6s
    8:	learn: 0.5194403	total: 2.49s	remaining: 52.8s
    9:	learn: 0.5042661	total: 2.75s	remaining: 52.3s
    10:	learn: 0.4889410	total: 3.02s	remaining: 52s
    11:	learn: 0.4732775	total: 3.3s	remaining: 51.7s
    12:	learn: 0.4598494	total: 3.57s	remaining: 51.3s
    13:	learn: 0.4461213	total: 3.85s	remaining: 51.1s
    14:	learn: 0.4323786	total: 4.1s	remaining: 50.6s
    15:	learn: 0.4197129	total: 4.37s	remaining: 50.3s
    16:	learn: 0.4076167	total: 4.63s	remaining: 49.8s
    17:	learn: 0.3958119	total: 4.9s	remaining: 49.5s
    18:	learn: 0.3822534	total: 5.17s	remaining: 49.3s
    19:	learn: 0.3713165	total: 5.45s	remaining: 49s
    20:	learn: 0.3600068	total: 5.71s	remaining: 48.7s
    21:	learn: 0.3486922	total: 5.98s	remaining: 48.4s
    22:	learn: 0.3378056	total: 6.24s	remaining: 48s
    23:	learn: 0.3275013	total: 6.52s	remaining: 47.8s
    24:	learn: 0.3178892	total: 6.78s	remaining: 47.5s
    25:	learn: 0.3079708	total: 7.05s	remaining: 47.2s
    26:	learn: 0.2984632	total: 7.31s	remaining: 46.8s
    27:	learn: 0.2895187	total: 7.58s	remaining: 46.6s
    28:	learn: 0.2811951	total: 7.85s	remaining: 46.3s
    29:	learn: 0.2731970	total: 8.12s	remaining: 46s
    30:	learn: 0.2658162	total: 8.39s	remaining: 45.7s
    31:	learn: 0.2586741	total: 8.66s	remaining: 45.5s
    32:	learn: 0.2514165	total: 8.92s	remaining: 45.2s
    33:	learn: 0.2448900	total: 9.17s	remaining: 44.8s
    34:	learn: 0.2382106	total: 9.45s	remaining: 44.6s
    35:	learn: 0.2319971	total: 9.73s	remaining: 44.3s
    36:	learn: 0.2263649	total: 10s	remaining: 44s
    37:	learn: 0.2206516	total: 10.3s	remaining: 43.8s
    38:	learn: 0.2154710	total: 10.5s	remaining: 43.5s
    39:	learn: 0.2104291	total: 10.8s	remaining: 43.2s
    40:	learn: 0.2056595	total: 11.1s	remaining: 42.9s
    41:	learn: 0.2006243	total: 11.3s	remaining: 42.6s
    42:	learn: 0.1958041	total: 11.6s	remaining: 42.3s
    43:	learn: 0.1913844	total: 11.8s	remaining: 42s
    44:	learn: 0.1873334	total: 12.1s	remaining: 41.7s
    45:	learn: 0.1834195	total: 12.4s	remaining: 41.6s
    46:	learn: 0.1797204	total: 12.7s	remaining: 41.4s
    47:	learn: 0.1760928	total: 13s	remaining: 41.2s
    48:	learn: 0.1725395	total: 13.3s	remaining: 40.9s
    49:	learn: 0.1689647	total: 13.5s	remaining: 40.6s
    50:	learn: 0.1653324	total: 13.8s	remaining: 40.3s
    51:	learn: 0.1620204	total: 14.1s	remaining: 40s
    52:	learn: 0.1586776	total: 14.3s	remaining: 39.8s
    53:	learn: 0.1556471	total: 14.6s	remaining: 39.5s
    54:	learn: 0.1525812	total: 14.9s	remaining: 39.2s
    55:	learn: 0.1496661	total: 15.1s	remaining: 39s
    56:	learn: 0.1468765	total: 15.4s	remaining: 38.7s
    57:	learn: 0.1442367	total: 15.7s	remaining: 38.4s
    58:	learn: 0.1417207	total: 15.9s	remaining: 38.1s
    59:	learn: 0.1394943	total: 16.2s	remaining: 37.9s
    60:	learn: 0.1371112	total: 16.5s	remaining: 37.6s
    61:	learn: 0.1348353	total: 16.8s	remaining: 37.3s
    62:	learn: 0.1327871	total: 17s	remaining: 37s
    63:	learn: 0.1306288	total: 17.3s	remaining: 36.8s
    64:	learn: 0.1284786	total: 17.6s	remaining: 36.5s
    65:	learn: 0.1263921	total: 17.8s	remaining: 36.2s
    66:	learn: 0.1245089	total: 18.1s	remaining: 35.9s
    67:	learn: 0.1226621	total: 18.4s	remaining: 35.7s
    68:	learn: 0.1208106	total: 18.6s	remaining: 35.4s
    69:	learn: 0.1190936	total: 18.9s	remaining: 35.1s
    70:	learn: 0.1174569	total: 19.2s	remaining: 34.9s
    71:	learn: 0.1159807	total: 19.4s	remaining: 34.6s
    72:	learn: 0.1143706	total: 19.7s	remaining: 34.3s
    73:	learn: 0.1128958	total: 20s	remaining: 34s
    74:	learn: 0.1114191	total: 20.3s	remaining: 33.8s
    75:	learn: 0.1100454	total: 20.5s	remaining: 33.5s
    76:	learn: 0.1086097	total: 20.8s	remaining: 33.2s
    77:	learn: 0.1072864	total: 21.1s	remaining: 33s
    78:	learn: 0.1061411	total: 21.3s	remaining: 32.7s
    79:	learn: 0.1048442	total: 21.6s	remaining: 32.4s
    80:	learn: 0.1037857	total: 21.9s	remaining: 32.2s
    81:	learn: 0.1026599	total: 22.2s	remaining: 31.9s
    82:	learn: 0.1015656	total: 22.4s	remaining: 31.6s
    83:	learn: 0.1005055	total: 22.7s	remaining: 31.4s
    84:	learn: 0.0995809	total: 22.9s	remaining: 31s
    85:	learn: 0.0985730	total: 23.1s	remaining: 30.7s
    86:	learn: 0.0975716	total: 23.4s	remaining: 30.4s
    87:	learn: 0.0965866	total: 23.7s	remaining: 30.2s
    88:	learn: 0.0956786	total: 24s	remaining: 29.9s
    89:	learn: 0.0948761	total: 24.2s	remaining: 29.6s
    90:	learn: 0.0939884	total: 24.5s	remaining: 29.3s
    91:	learn: 0.0931054	total: 24.7s	remaining: 29s
    92:	learn: 0.0922875	total: 25s	remaining: 28.8s
    93:	learn: 0.0914858	total: 25.3s	remaining: 28.5s
    94:	learn: 0.0907087	total: 25.5s	remaining: 28.2s
    95:	learn: 0.0899684	total: 25.8s	remaining: 28s
    96:	learn: 0.0892213	total: 26.1s	remaining: 27.7s
    97:	learn: 0.0884699	total: 26.3s	remaining: 27.4s
    98:	learn: 0.0878150	total: 26.6s	remaining: 27.1s
    99:	learn: 0.0871336	total: 26.9s	remaining: 26.9s
    100:	learn: 0.0864798	total: 27.1s	remaining: 26.6s
    101:	learn: 0.0858699	total: 27.4s	remaining: 26.3s
    102:	learn: 0.0852402	total: 27.7s	remaining: 26s
    103:	learn: 0.0846712	total: 27.9s	remaining: 25.8s
    104:	learn: 0.0841205	total: 28.2s	remaining: 25.5s
    105:	learn: 0.0835904	total: 28.4s	remaining: 25.2s
    106:	learn: 0.0830729	total: 28.7s	remaining: 24.9s
    107:	learn: 0.0825613	total: 29s	remaining: 24.7s
    108:	learn: 0.0820759	total: 29.3s	remaining: 24.4s
    109:	learn: 0.0815796	total: 29.5s	remaining: 24.1s
    110:	learn: 0.0811203	total: 29.8s	remaining: 23.9s
    111:	learn: 0.0806515	total: 30s	remaining: 23.6s
    112:	learn: 0.0801746	total: 30.3s	remaining: 23.3s
    113:	learn: 0.0797852	total: 30.6s	remaining: 23.1s
    114:	learn: 0.0793666	total: 30.9s	remaining: 22.8s
    115:	learn: 0.0789081	total: 31.1s	remaining: 22.5s
    116:	learn: 0.0784785	total: 31.4s	remaining: 22.2s
    117:	learn: 0.0780618	total: 31.6s	remaining: 22s
    118:	learn: 0.0776921	total: 31.9s	remaining: 21.7s
    119:	learn: 0.0772886	total: 32.2s	remaining: 21.4s
    120:	learn: 0.0769325	total: 32.4s	remaining: 21.2s
    121:	learn: 0.0765935	total: 32.7s	remaining: 20.9s
    122:	learn: 0.0762531	total: 33s	remaining: 20.6s
    123:	learn: 0.0759389	total: 33.2s	remaining: 20.4s
    124:	learn: 0.0756339	total: 33.5s	remaining: 20.1s
    125:	learn: 0.0752907	total: 33.8s	remaining: 19.8s
    126:	learn: 0.0749597	total: 34s	remaining: 19.6s
    127:	learn: 0.0746387	total: 34.3s	remaining: 19.3s
    128:	learn: 0.0743658	total: 34.5s	remaining: 19s
    129:	learn: 0.0740942	total: 34.8s	remaining: 18.7s
    130:	learn: 0.0738262	total: 35.1s	remaining: 18.5s
    131:	learn: 0.0735503	total: 35.3s	remaining: 18.2s
    132:	learn: 0.0732717	total: 35.6s	remaining: 17.9s
    133:	learn: 0.0730281	total: 35.9s	remaining: 17.7s
    134:	learn: 0.0727646	total: 36.1s	remaining: 17.4s
    135:	learn: 0.0725347	total: 36.4s	remaining: 17.1s
    136:	learn: 0.0723131	total: 36.7s	remaining: 16.9s
    137:	learn: 0.0720616	total: 36.9s	remaining: 16.6s
    138:	learn: 0.0718252	total: 37.2s	remaining: 16.3s
    139:	learn: 0.0715842	total: 37.5s	remaining: 16.1s
    140:	learn: 0.0713673	total: 37.7s	remaining: 15.8s
    141:	learn: 0.0711417	total: 38s	remaining: 15.5s
    142:	learn: 0.0709506	total: 38.2s	remaining: 15.2s
    143:	learn: 0.0707728	total: 38.5s	remaining: 15s
    144:	learn: 0.0705657	total: 38.8s	remaining: 14.7s
    145:	learn: 0.0703617	total: 39.1s	remaining: 14.4s
    146:	learn: 0.0701785	total: 39.3s	remaining: 14.2s
    147:	learn: 0.0699824	total: 39.6s	remaining: 13.9s
    148:	learn: 0.0698101	total: 39.9s	remaining: 13.6s
    149:	learn: 0.0696268	total: 40.1s	remaining: 13.4s
    150:	learn: 0.0694579	total: 40.4s	remaining: 13.1s
    151:	learn: 0.0693236	total: 40.6s	remaining: 12.8s
    152:	learn: 0.0691757	total: 40.9s	remaining: 12.6s
    153:	learn: 0.0690268	total: 41.2s	remaining: 12.3s
    154:	learn: 0.0688655	total: 41.4s	remaining: 12s
    155:	learn: 0.0687173	total: 41.7s	remaining: 11.8s
    156:	learn: 0.0685626	total: 42s	remaining: 11.5s
    157:	learn: 0.0684319	total: 42.2s	remaining: 11.2s
    158:	learn: 0.0683049	total: 42.5s	remaining: 11s
    159:	learn: 0.0681751	total: 42.7s	remaining: 10.7s
    160:	learn: 0.0680682	total: 43s	remaining: 10.4s
    161:	learn: 0.0679444	total: 43.3s	remaining: 10.2s
    162:	learn: 0.0678052	total: 43.5s	remaining: 9.88s
    163:	learn: 0.0676935	total: 43.8s	remaining: 9.61s
    164:	learn: 0.0675628	total: 44.1s	remaining: 9.35s
    165:	learn: 0.0674354	total: 44.3s	remaining: 9.07s
    166:	learn: 0.0673315	total: 44.6s	remaining: 8.81s
    167:	learn: 0.0672118	total: 44.9s	remaining: 8.54s
    168:	learn: 0.0671005	total: 45.1s	remaining: 8.28s
    169:	learn: 0.0669933	total: 45.4s	remaining: 8.01s
    170:	learn: 0.0668912	total: 45.7s	remaining: 7.74s
    171:	learn: 0.0667918	total: 45.9s	remaining: 7.48s
    172:	learn: 0.0666931	total: 46.2s	remaining: 7.21s
    173:	learn: 0.0665988	total: 46.5s	remaining: 6.94s
    174:	learn: 0.0664978	total: 46.8s	remaining: 6.68s
    175:	learn: 0.0664040	total: 47s	remaining: 6.42s
    176:	learn: 0.0663078	total: 47.3s	remaining: 6.14s
    177:	learn: 0.0662392	total: 47.5s	remaining: 5.88s
    178:	learn: 0.0661510	total: 47.8s	remaining: 5.61s
    179:	learn: 0.0660572	total: 48.1s	remaining: 5.34s
    180:	learn: 0.0659645	total: 48.3s	remaining: 5.07s
    181:	learn: 0.0658842	total: 48.6s	remaining: 4.81s
    182:	learn: 0.0658069	total: 48.9s	remaining: 4.54s
    183:	learn: 0.0657320	total: 49.2s	remaining: 4.28s
    184:	learn: 0.0656530	total: 49.4s	remaining: 4.01s
    185:	learn: 0.0655824	total: 49.7s	remaining: 3.74s
    186:	learn: 0.0655163	total: 50s	remaining: 3.47s
    187:	learn: 0.0654371	total: 50.2s	remaining: 3.2s
    188:	learn: 0.0653847	total: 50.5s	remaining: 2.94s
    189:	learn: 0.0653242	total: 50.7s	remaining: 2.67s
    190:	learn: 0.0652604	total: 51s	remaining: 2.4s
    191:	learn: 0.0651856	total: 51.3s	remaining: 2.13s
    192:	learn: 0.0651238	total: 51.5s	remaining: 1.87s
    193:	learn: 0.0650738	total: 51.8s	remaining: 1.6s
    194:	learn: 0.0650056	total: 52.1s	remaining: 1.33s
    195:	learn: 0.0649375	total: 52.3s	remaining: 1.07s
    196:	learn: 0.0648726	total: 52.6s	remaining: 801ms
    197:	learn: 0.0648103	total: 52.8s	remaining: 534ms
    198:	learn: 0.0647624	total: 53.1s	remaining: 267ms
    199:	learn: 0.0647077	total: 53.4s	remaining: 0us
    Trained model nº 5/27. Iterations: 200Depth: 8Learning rate: 0.01
    0:	learn: 0.6720913	total: 284ms	remaining: 1m 24s
    1:	learn: 0.6511983	total: 561ms	remaining: 1m 23s
    2:	learn: 0.6307706	total: 825ms	remaining: 1m 21s
    3:	learn: 0.6120845	total: 1.09s	remaining: 1m 20s
    4:	learn: 0.5922499	total: 1.34s	remaining: 1m 19s
    5:	learn: 0.5729459	total: 1.6s	remaining: 1m 18s
    6:	learn: 0.5553558	total: 1.87s	remaining: 1m 18s
    7:	learn: 0.5376874	total: 2.12s	remaining: 1m 17s
    8:	learn: 0.5194403	total: 2.37s	remaining: 1m 16s
    9:	learn: 0.5042661	total: 2.65s	remaining: 1m 16s
    10:	learn: 0.4889410	total: 2.91s	remaining: 1m 16s
    11:	learn: 0.4732775	total: 3.19s	remaining: 1m 16s
    12:	learn: 0.4598494	total: 3.45s	remaining: 1m 16s
    13:	learn: 0.4461213	total: 3.71s	remaining: 1m 15s
    14:	learn: 0.4323786	total: 3.96s	remaining: 1m 15s
    15:	learn: 0.4197129	total: 4.22s	remaining: 1m 14s
    16:	learn: 0.4076167	total: 4.48s	remaining: 1m 14s
    17:	learn: 0.3958119	total: 4.75s	remaining: 1m 14s
    18:	learn: 0.3822534	total: 5.02s	remaining: 1m 14s
    19:	learn: 0.3713165	total: 5.28s	remaining: 1m 13s
    20:	learn: 0.3600068	total: 5.55s	remaining: 1m 13s
    21:	learn: 0.3486922	total: 5.8s	remaining: 1m 13s
    22:	learn: 0.3378056	total: 6.07s	remaining: 1m 13s
    23:	learn: 0.3275013	total: 6.34s	remaining: 1m 12s
    24:	learn: 0.3178892	total: 6.61s	remaining: 1m 12s
    25:	learn: 0.3079708	total: 6.87s	remaining: 1m 12s
    26:	learn: 0.2984632	total: 7.14s	remaining: 1m 12s
    27:	learn: 0.2895187	total: 7.41s	remaining: 1m 12s
    28:	learn: 0.2811951	total: 7.68s	remaining: 1m 11s
    29:	learn: 0.2731970	total: 7.95s	remaining: 1m 11s
    30:	learn: 0.2658162	total: 8.22s	remaining: 1m 11s
    31:	learn: 0.2586741	total: 8.49s	remaining: 1m 11s
    32:	learn: 0.2514165	total: 8.76s	remaining: 1m 10s
    33:	learn: 0.2448900	total: 9.01s	remaining: 1m 10s
    34:	learn: 0.2382106	total: 9.27s	remaining: 1m 10s
    35:	learn: 0.2319971	total: 9.54s	remaining: 1m 9s
    36:	learn: 0.2263649	total: 9.81s	remaining: 1m 9s
    37:	learn: 0.2206516	total: 10.1s	remaining: 1m 9s
    38:	learn: 0.2154710	total: 10.4s	remaining: 1m 9s
    39:	learn: 0.2104291	total: 10.6s	remaining: 1m 9s
    40:	learn: 0.2056595	total: 10.9s	remaining: 1m 8s
    41:	learn: 0.2006243	total: 11.2s	remaining: 1m 8s
    42:	learn: 0.1958041	total: 11.4s	remaining: 1m 8s
    43:	learn: 0.1913844	total: 11.7s	remaining: 1m 7s
    44:	learn: 0.1873334	total: 12s	remaining: 1m 8s
    45:	learn: 0.1834195	total: 12.3s	remaining: 1m 8s
    46:	learn: 0.1797204	total: 12.6s	remaining: 1m 7s
    47:	learn: 0.1760928	total: 12.9s	remaining: 1m 7s
    48:	learn: 0.1725395	total: 13.2s	remaining: 1m 7s
    49:	learn: 0.1689647	total: 13.5s	remaining: 1m 7s
    50:	learn: 0.1653324	total: 13.8s	remaining: 1m 7s
    51:	learn: 0.1620204	total: 14.1s	remaining: 1m 7s
    52:	learn: 0.1586776	total: 14.3s	remaining: 1m 6s
    53:	learn: 0.1556471	total: 14.6s	remaining: 1m 6s
    54:	learn: 0.1525812	total: 14.9s	remaining: 1m 6s
    55:	learn: 0.1496661	total: 15.2s	remaining: 1m 6s
    56:	learn: 0.1468765	total: 15.4s	remaining: 1m 5s
    57:	learn: 0.1442367	total: 15.7s	remaining: 1m 5s
    58:	learn: 0.1417207	total: 15.9s	remaining: 1m 5s
    59:	learn: 0.1394943	total: 16.2s	remaining: 1m 4s
    60:	learn: 0.1371112	total: 16.5s	remaining: 1m 4s
    61:	learn: 0.1348353	total: 16.8s	remaining: 1m 4s
    62:	learn: 0.1327871	total: 17s	remaining: 1m 4s
    63:	learn: 0.1306288	total: 17.3s	remaining: 1m 3s
    64:	learn: 0.1284786	total: 17.6s	remaining: 1m 3s
    65:	learn: 0.1263921	total: 17.9s	remaining: 1m 3s
    66:	learn: 0.1245089	total: 18.1s	remaining: 1m 3s
    67:	learn: 0.1226621	total: 18.4s	remaining: 1m 2s
    68:	learn: 0.1208106	total: 18.6s	remaining: 1m 2s
    69:	learn: 0.1190936	total: 18.9s	remaining: 1m 2s
    70:	learn: 0.1174569	total: 19.2s	remaining: 1m 1s
    71:	learn: 0.1159807	total: 19.5s	remaining: 1m 1s
    72:	learn: 0.1143706	total: 19.7s	remaining: 1m 1s
    73:	learn: 0.1128958	total: 20s	remaining: 1m 1s
    74:	learn: 0.1114191	total: 20.3s	remaining: 1m
    75:	learn: 0.1100454	total: 20.5s	remaining: 1m
    76:	learn: 0.1086097	total: 20.8s	remaining: 1m
    77:	learn: 0.1072864	total: 21.1s	remaining: 1m
    78:	learn: 0.1061411	total: 21.3s	remaining: 59.7s
    79:	learn: 0.1048442	total: 21.6s	remaining: 59.5s
    80:	learn: 0.1037857	total: 21.9s	remaining: 59.2s
    81:	learn: 0.1026599	total: 22.2s	remaining: 58.9s
    82:	learn: 0.1015656	total: 22.4s	remaining: 58.7s
    83:	learn: 0.1005055	total: 22.7s	remaining: 58.4s
    84:	learn: 0.0995809	total: 22.9s	remaining: 57.9s
    85:	learn: 0.0985730	total: 23.1s	remaining: 57.6s
    86:	learn: 0.0975716	total: 23.4s	remaining: 57.3s
    87:	learn: 0.0965866	total: 23.7s	remaining: 57.1s
    88:	learn: 0.0956786	total: 23.9s	remaining: 56.8s
    89:	learn: 0.0948761	total: 24.2s	remaining: 56.5s
    90:	learn: 0.0939884	total: 24.5s	remaining: 56.2s
    91:	learn: 0.0931054	total: 24.7s	remaining: 55.9s
    92:	learn: 0.0922875	total: 25s	remaining: 55.6s
    93:	learn: 0.0914858	total: 25.3s	remaining: 55.3s
    94:	learn: 0.0907087	total: 25.5s	remaining: 55.1s
    95:	learn: 0.0899684	total: 25.8s	remaining: 54.8s
    96:	learn: 0.0892213	total: 26.1s	remaining: 54.6s
    97:	learn: 0.0884699	total: 26.3s	remaining: 54.3s
    98:	learn: 0.0878150	total: 26.6s	remaining: 54s
    99:	learn: 0.0871336	total: 26.9s	remaining: 53.8s
    100:	learn: 0.0864798	total: 27.2s	remaining: 53.5s
    101:	learn: 0.0858699	total: 27.4s	remaining: 53.2s
    102:	learn: 0.0852402	total: 27.7s	remaining: 52.9s
    103:	learn: 0.0846712	total: 27.9s	remaining: 52.6s
    104:	learn: 0.0841205	total: 28.2s	remaining: 52.4s
    105:	learn: 0.0835904	total: 28.4s	remaining: 52.1s
    106:	learn: 0.0830729	total: 28.7s	remaining: 51.8s
    107:	learn: 0.0825613	total: 29s	remaining: 51.5s
    108:	learn: 0.0820759	total: 29.2s	remaining: 51.3s
    109:	learn: 0.0815796	total: 29.5s	remaining: 51s
    110:	learn: 0.0811203	total: 29.8s	remaining: 50.7s
    111:	learn: 0.0806515	total: 30s	remaining: 50.4s
    112:	learn: 0.0801746	total: 30.3s	remaining: 50.2s
    113:	learn: 0.0797852	total: 30.6s	remaining: 49.9s
    114:	learn: 0.0793666	total: 30.8s	remaining: 49.6s
    115:	learn: 0.0789081	total: 31.1s	remaining: 49.3s
    116:	learn: 0.0784785	total: 31.4s	remaining: 49s
    117:	learn: 0.0780618	total: 31.6s	remaining: 48.8s
    118:	learn: 0.0776921	total: 31.9s	remaining: 48.5s
    119:	learn: 0.0772886	total: 32.2s	remaining: 48.3s
    120:	learn: 0.0769325	total: 32.5s	remaining: 48s
    121:	learn: 0.0765935	total: 32.7s	remaining: 47.8s
    122:	learn: 0.0762531	total: 33s	remaining: 47.5s
    123:	learn: 0.0759389	total: 33.2s	remaining: 47.2s
    124:	learn: 0.0756339	total: 33.5s	remaining: 46.9s
    125:	learn: 0.0752907	total: 33.8s	remaining: 46.6s
    126:	learn: 0.0749597	total: 34s	remaining: 46.3s
    127:	learn: 0.0746387	total: 34.3s	remaining: 46.1s
    128:	learn: 0.0743658	total: 34.5s	remaining: 45.8s
    129:	learn: 0.0740942	total: 34.8s	remaining: 45.5s
    130:	learn: 0.0738262	total: 35.1s	remaining: 45.2s
    131:	learn: 0.0735503	total: 35.3s	remaining: 45s
    132:	learn: 0.0732717	total: 35.6s	remaining: 44.7s
    133:	learn: 0.0730281	total: 35.9s	remaining: 44.4s
    134:	learn: 0.0727646	total: 36.1s	remaining: 44.2s
    135:	learn: 0.0725347	total: 36.4s	remaining: 43.9s
    136:	learn: 0.0723131	total: 36.7s	remaining: 43.6s
    137:	learn: 0.0720616	total: 36.9s	remaining: 43.3s
    138:	learn: 0.0718252	total: 37.2s	remaining: 43.1s
    139:	learn: 0.0715842	total: 37.5s	remaining: 42.8s
    140:	learn: 0.0713673	total: 37.7s	remaining: 42.5s
    141:	learn: 0.0711417	total: 38s	remaining: 42.2s
    142:	learn: 0.0709506	total: 38.2s	remaining: 42s
    143:	learn: 0.0707728	total: 38.5s	remaining: 41.7s
    144:	learn: 0.0705657	total: 38.8s	remaining: 41.4s
    145:	learn: 0.0703617	total: 39s	remaining: 41.1s
    146:	learn: 0.0701785	total: 39.3s	remaining: 40.9s
    147:	learn: 0.0699824	total: 39.5s	remaining: 40.6s
    148:	learn: 0.0698101	total: 39.8s	remaining: 40.4s
    149:	learn: 0.0696268	total: 40.1s	remaining: 40.1s
    150:	learn: 0.0694579	total: 40.4s	remaining: 39.8s
    151:	learn: 0.0693236	total: 40.6s	remaining: 39.5s
    152:	learn: 0.0691757	total: 40.9s	remaining: 39.3s
    153:	learn: 0.0690268	total: 41.1s	remaining: 39s
    154:	learn: 0.0688655	total: 41.4s	remaining: 38.7s
    155:	learn: 0.0687173	total: 41.7s	remaining: 38.4s
    156:	learn: 0.0685626	total: 41.9s	remaining: 38.2s
    157:	learn: 0.0684319	total: 42.2s	remaining: 37.9s
    158:	learn: 0.0683049	total: 42.5s	remaining: 37.6s
    159:	learn: 0.0681751	total: 42.7s	remaining: 37.4s
    160:	learn: 0.0680682	total: 43s	remaining: 37.1s
    161:	learn: 0.0679444	total: 43.2s	remaining: 36.8s
    162:	learn: 0.0678052	total: 43.5s	remaining: 36.6s
    163:	learn: 0.0676935	total: 43.7s	remaining: 36.3s
    164:	learn: 0.0675628	total: 44s	remaining: 36s
    165:	learn: 0.0674354	total: 44.2s	remaining: 35.7s
    166:	learn: 0.0673315	total: 44.5s	remaining: 35.4s
    167:	learn: 0.0672118	total: 44.8s	remaining: 35.2s
    168:	learn: 0.0671005	total: 45.1s	remaining: 34.9s
    169:	learn: 0.0669933	total: 45.3s	remaining: 34.6s
    170:	learn: 0.0668912	total: 45.6s	remaining: 34.4s
    171:	learn: 0.0667918	total: 45.8s	remaining: 34.1s
    172:	learn: 0.0666931	total: 46.1s	remaining: 33.9s
    173:	learn: 0.0665988	total: 46.4s	remaining: 33.6s
    174:	learn: 0.0664978	total: 46.7s	remaining: 33.3s
    175:	learn: 0.0664040	total: 46.9s	remaining: 33.1s
    176:	learn: 0.0663078	total: 47.2s	remaining: 32.8s
    177:	learn: 0.0662392	total: 47.4s	remaining: 32.5s
    178:	learn: 0.0661510	total: 47.7s	remaining: 32.2s
    179:	learn: 0.0660572	total: 48s	remaining: 32s
    180:	learn: 0.0659645	total: 48.2s	remaining: 31.7s
    181:	learn: 0.0658842	total: 48.5s	remaining: 31.5s
    182:	learn: 0.0658069	total: 48.8s	remaining: 31.2s
    183:	learn: 0.0657320	total: 49.1s	remaining: 30.9s
    184:	learn: 0.0656530	total: 49.3s	remaining: 30.7s
    185:	learn: 0.0655824	total: 49.6s	remaining: 30.4s
    186:	learn: 0.0655163	total: 49.8s	remaining: 30.1s
    187:	learn: 0.0654371	total: 50.1s	remaining: 29.8s
    188:	learn: 0.0653847	total: 50.4s	remaining: 29.6s
    189:	learn: 0.0653242	total: 50.6s	remaining: 29.3s
    190:	learn: 0.0652604	total: 50.9s	remaining: 29s
    191:	learn: 0.0651856	total: 51.2s	remaining: 28.8s
    192:	learn: 0.0651238	total: 51.4s	remaining: 28.5s
    193:	learn: 0.0650738	total: 51.7s	remaining: 28.2s
    194:	learn: 0.0650056	total: 51.9s	remaining: 28s
    195:	learn: 0.0649375	total: 52.2s	remaining: 27.7s
    196:	learn: 0.0648726	total: 52.5s	remaining: 27.4s
    197:	learn: 0.0648103	total: 52.7s	remaining: 27.2s
    198:	learn: 0.0647624	total: 53s	remaining: 26.9s
    199:	learn: 0.0647077	total: 53.3s	remaining: 26.6s
    200:	learn: 0.0646568	total: 53.5s	remaining: 26.4s
    201:	learn: 0.0645985	total: 53.8s	remaining: 26.1s
    202:	learn: 0.0645501	total: 54.1s	remaining: 25.8s
    203:	learn: 0.0645043	total: 54.3s	remaining: 25.6s
    204:	learn: 0.0644459	total: 54.6s	remaining: 25.3s
    205:	learn: 0.0643906	total: 54.9s	remaining: 25.1s
    206:	learn: 0.0643397	total: 55.2s	remaining: 24.8s
    207:	learn: 0.0642874	total: 55.5s	remaining: 24.5s
    208:	learn: 0.0642341	total: 55.8s	remaining: 24.3s
    209:	learn: 0.0641831	total: 56s	remaining: 24s
    210:	learn: 0.0641389	total: 56.3s	remaining: 23.7s
    211:	learn: 0.0641023	total: 56.6s	remaining: 23.5s
    212:	learn: 0.0640526	total: 56.8s	remaining: 23.2s
    213:	learn: 0.0640155	total: 57.1s	remaining: 22.9s
    214:	learn: 0.0639745	total: 57.3s	remaining: 22.7s
    215:	learn: 0.0639333	total: 57.6s	remaining: 22.4s
    216:	learn: 0.0638908	total: 57.9s	remaining: 22.1s
    217:	learn: 0.0638459	total: 58.1s	remaining: 21.9s
    218:	learn: 0.0638185	total: 58.4s	remaining: 21.6s
    219:	learn: 0.0637781	total: 58.6s	remaining: 21.3s
    220:	learn: 0.0637470	total: 58.9s	remaining: 21.1s
    221:	learn: 0.0637068	total: 59.2s	remaining: 20.8s
    222:	learn: 0.0636741	total: 59.5s	remaining: 20.5s
    223:	learn: 0.0636422	total: 59.7s	remaining: 20.3s
    224:	learn: 0.0636039	total: 60s	remaining: 20s
    225:	learn: 0.0635677	total: 1m	remaining: 19.7s
    226:	learn: 0.0635299	total: 1m	remaining: 19.5s
    227:	learn: 0.0634966	total: 1m	remaining: 19.2s
    228:	learn: 0.0634605	total: 1m 1s	remaining: 18.9s
    229:	learn: 0.0634348	total: 1m 1s	remaining: 18.6s
    230:	learn: 0.0634077	total: 1m 1s	remaining: 18.4s
    231:	learn: 0.0633783	total: 1m 1s	remaining: 18.1s
    232:	learn: 0.0633537	total: 1m 2s	remaining: 17.9s
    233:	learn: 0.0633305	total: 1m 2s	remaining: 17.6s
    234:	learn: 0.0633066	total: 1m 2s	remaining: 17.3s
    235:	learn: 0.0632766	total: 1m 2s	remaining: 17s
    236:	learn: 0.0632552	total: 1m 3s	remaining: 16.8s
    237:	learn: 0.0632321	total: 1m 3s	remaining: 16.5s
    238:	learn: 0.0632008	total: 1m 3s	remaining: 16.3s
    239:	learn: 0.0631711	total: 1m 3s	remaining: 16s
    240:	learn: 0.0631434	total: 1m 4s	remaining: 15.7s
    241:	learn: 0.0631203	total: 1m 4s	remaining: 15.4s
    242:	learn: 0.0630922	total: 1m 4s	remaining: 15.2s
    243:	learn: 0.0630647	total: 1m 4s	remaining: 14.9s
    244:	learn: 0.0630420	total: 1m 5s	remaining: 14.6s
    245:	learn: 0.0630163	total: 1m 5s	remaining: 14.4s
    246:	learn: 0.0629956	total: 1m 5s	remaining: 14.1s
    247:	learn: 0.0629695	total: 1m 6s	remaining: 13.8s
    248:	learn: 0.0629472	total: 1m 6s	remaining: 13.6s
    249:	learn: 0.0629313	total: 1m 6s	remaining: 13.3s
    250:	learn: 0.0629132	total: 1m 6s	remaining: 13s
    251:	learn: 0.0628874	total: 1m 7s	remaining: 12.8s
    252:	learn: 0.0628621	total: 1m 7s	remaining: 12.5s
    253:	learn: 0.0628396	total: 1m 7s	remaining: 12.2s
    254:	learn: 0.0628162	total: 1m 7s	remaining: 12s
    255:	learn: 0.0627955	total: 1m 8s	remaining: 11.7s
    256:	learn: 0.0627788	total: 1m 8s	remaining: 11.4s
    257:	learn: 0.0627599	total: 1m 8s	remaining: 11.2s
    258:	learn: 0.0627385	total: 1m 8s	remaining: 10.9s
    259:	learn: 0.0627218	total: 1m 9s	remaining: 10.6s
    260:	learn: 0.0627014	total: 1m 9s	remaining: 10.4s
    261:	learn: 0.0626885	total: 1m 9s	remaining: 10.1s
    262:	learn: 0.0626733	total: 1m 9s	remaining: 9.84s
    263:	learn: 0.0626576	total: 1m 10s	remaining: 9.57s
    264:	learn: 0.0626383	total: 1m 10s	remaining: 9.31s
    265:	learn: 0.0626194	total: 1m 10s	remaining: 9.04s
    266:	learn: 0.0626010	total: 1m 11s	remaining: 8.78s
    267:	learn: 0.0625839	total: 1m 11s	remaining: 8.51s
    268:	learn: 0.0625646	total: 1m 11s	remaining: 8.24s
    269:	learn: 0.0625504	total: 1m 11s	remaining: 7.97s
    270:	learn: 0.0625391	total: 1m 12s	remaining: 7.71s
    271:	learn: 0.0625265	total: 1m 12s	remaining: 7.44s
    272:	learn: 0.0625145	total: 1m 12s	remaining: 7.17s
    273:	learn: 0.0625000	total: 1m 12s	remaining: 6.91s
    274:	learn: 0.0624892	total: 1m 13s	remaining: 6.64s
    275:	learn: 0.0624798	total: 1m 13s	remaining: 6.37s
    276:	learn: 0.0624696	total: 1m 13s	remaining: 6.11s
    277:	learn: 0.0624585	total: 1m 13s	remaining: 5.84s
    278:	learn: 0.0624458	total: 1m 14s	remaining: 5.57s
    279:	learn: 0.0624305	total: 1m 14s	remaining: 5.31s
    280:	learn: 0.0624212	total: 1m 14s	remaining: 5.04s
    281:	learn: 0.0624071	total: 1m 14s	remaining: 4.78s
    282:	learn: 0.0623968	total: 1m 15s	remaining: 4.51s
    283:	learn: 0.0623843	total: 1m 15s	remaining: 4.25s
    284:	learn: 0.0623746	total: 1m 15s	remaining: 3.98s
    285:	learn: 0.0623643	total: 1m 15s	remaining: 3.71s
    286:	learn: 0.0623519	total: 1m 16s	remaining: 3.45s
    287:	learn: 0.0623394	total: 1m 16s	remaining: 3.18s
    288:	learn: 0.0623242	total: 1m 16s	remaining: 2.92s
    289:	learn: 0.0623123	total: 1m 16s	remaining: 2.65s
    290:	learn: 0.0623039	total: 1m 17s	remaining: 2.39s
    291:	learn: 0.0622922	total: 1m 17s	remaining: 2.12s
    292:	learn: 0.0622823	total: 1m 17s	remaining: 1.86s
    293:	learn: 0.0622715	total: 1m 18s	remaining: 1.59s
    294:	learn: 0.0622624	total: 1m 18s	remaining: 1.33s
    295:	learn: 0.0622523	total: 1m 18s	remaining: 1.06s
    296:	learn: 0.0622414	total: 1m 18s	remaining: 796ms
    297:	learn: 0.0622287	total: 1m 19s	remaining: 531ms
    298:	learn: 0.0622194	total: 1m 19s	remaining: 265ms
    299:	learn: 0.0622087	total: 1m 19s	remaining: 0us
    Trained model nº 6/27. Iterations: 300Depth: 8Learning rate: 0.01
    0:	learn: 0.6716862	total: 251ms	remaining: 24.9s
    1:	learn: 0.6506792	total: 471ms	remaining: 23.1s
    2:	learn: 0.6301470	total: 710ms	remaining: 23s
    3:	learn: 0.6089949	total: 923ms	remaining: 22.1s
    4:	learn: 0.5913086	total: 1.14s	remaining: 21.6s
    5:	learn: 0.5734177	total: 1.3s	remaining: 20.5s
    6:	learn: 0.5542074	total: 1.52s	remaining: 20.3s
    7:	learn: 0.5348989	total: 1.73s	remaining: 19.9s
    8:	learn: 0.5184874	total: 1.95s	remaining: 19.7s
    9:	learn: 0.5012399	total: 2.15s	remaining: 19.4s
    10:	learn: 0.4857455	total: 2.36s	remaining: 19.1s
    11:	learn: 0.4702346	total: 2.57s	remaining: 18.9s
    12:	learn: 0.4569763	total: 2.74s	remaining: 18.3s
    13:	learn: 0.4435443	total: 2.95s	remaining: 18.1s
    14:	learn: 0.4303796	total: 3.17s	remaining: 18s
    15:	learn: 0.4155404	total: 3.37s	remaining: 17.7s
    16:	learn: 0.4042182	total: 3.59s	remaining: 17.5s
    17:	learn: 0.3913328	total: 3.81s	remaining: 17.4s
    18:	learn: 0.3804991	total: 4.01s	remaining: 17.1s
    19:	learn: 0.3694632	total: 4.19s	remaining: 16.8s
    20:	learn: 0.3580498	total: 4.36s	remaining: 16.4s
    21:	learn: 0.3460312	total: 4.57s	remaining: 16.2s
    22:	learn: 0.3348456	total: 4.77s	remaining: 16s
    23:	learn: 0.3251515	total: 4.98s	remaining: 15.8s
    24:	learn: 0.3157640	total: 5.13s	remaining: 15.4s
    25:	learn: 0.3060789	total: 5.34s	remaining: 15.2s
    26:	learn: 0.2967048	total: 5.55s	remaining: 15s
    27:	learn: 0.2884371	total: 5.77s	remaining: 14.8s
    28:	learn: 0.2800192	total: 5.97s	remaining: 14.6s
    29:	learn: 0.2716421	total: 6.19s	remaining: 14.4s
    30:	learn: 0.2642799	total: 6.39s	remaining: 14.2s
    31:	learn: 0.2567584	total: 6.6s	remaining: 14s
    32:	learn: 0.2501658	total: 6.82s	remaining: 13.8s
    33:	learn: 0.2432802	total: 7.03s	remaining: 13.6s
    34:	learn: 0.2362994	total: 7.26s	remaining: 13.5s
    35:	learn: 0.2299260	total: 7.51s	remaining: 13.3s
    36:	learn: 0.2240603	total: 7.69s	remaining: 13.1s
    37:	learn: 0.2187014	total: 7.91s	remaining: 12.9s
    38:	learn: 0.2130346	total: 8.13s	remaining: 12.7s
    39:	learn: 0.2079352	total: 8.35s	remaining: 12.5s
    40:	learn: 0.2031993	total: 8.52s	remaining: 12.3s
    41:	learn: 0.1984113	total: 8.72s	remaining: 12s
    42:	learn: 0.1941267	total: 8.93s	remaining: 11.8s
    43:	learn: 0.1900329	total: 9.1s	remaining: 11.6s
    44:	learn: 0.1855622	total: 9.31s	remaining: 11.4s
    45:	learn: 0.1814634	total: 9.52s	remaining: 11.2s
    46:	learn: 0.1777768	total: 9.74s	remaining: 11s
    47:	learn: 0.1742436	total: 9.94s	remaining: 10.8s
    48:	learn: 0.1706090	total: 10.2s	remaining: 10.6s
    49:	learn: 0.1671258	total: 10.4s	remaining: 10.4s
    50:	learn: 0.1637409	total: 10.6s	remaining: 10.2s
    51:	learn: 0.1603686	total: 10.8s	remaining: 9.97s
    52:	learn: 0.1570705	total: 11s	remaining: 9.77s
    53:	learn: 0.1542402	total: 11.2s	remaining: 9.56s
    54:	learn: 0.1510610	total: 11.4s	remaining: 9.36s
    55:	learn: 0.1482129	total: 11.7s	remaining: 9.16s
    56:	learn: 0.1454609	total: 11.9s	remaining: 8.95s
    57:	learn: 0.1428195	total: 12.1s	remaining: 8.75s
    58:	learn: 0.1402940	total: 12.3s	remaining: 8.55s
    59:	learn: 0.1380605	total: 12.5s	remaining: 8.35s
    60:	learn: 0.1356094	total: 12.7s	remaining: 8.14s
    61:	learn: 0.1332666	total: 12.9s	remaining: 7.93s
    62:	learn: 0.1311385	total: 13.2s	remaining: 7.73s
    63:	learn: 0.1290733	total: 13.4s	remaining: 7.52s
    64:	learn: 0.1270018	total: 13.6s	remaining: 7.32s
    65:	learn: 0.1251928	total: 13.8s	remaining: 7.11s
    66:	learn: 0.1232971	total: 14s	remaining: 6.91s
    67:	learn: 0.1214164	total: 14.2s	remaining: 6.7s
    68:	learn: 0.1195861	total: 14.5s	remaining: 6.5s
    69:	learn: 0.1178955	total: 14.7s	remaining: 6.28s
    70:	learn: 0.1161646	total: 14.9s	remaining: 6.08s
    71:	learn: 0.1147793	total: 15.1s	remaining: 5.86s
    72:	learn: 0.1134196	total: 15.3s	remaining: 5.65s
    73:	learn: 0.1119755	total: 15.5s	remaining: 5.44s
    74:	learn: 0.1104785	total: 15.7s	remaining: 5.24s
    75:	learn: 0.1091154	total: 15.9s	remaining: 5.03s
    76:	learn: 0.1077255	total: 16.1s	remaining: 4.82s
    77:	learn: 0.1064396	total: 16.4s	remaining: 4.62s
    78:	learn: 0.1052341	total: 16.6s	remaining: 4.41s
    79:	learn: 0.1039494	total: 16.8s	remaining: 4.2s
    80:	learn: 0.1027291	total: 17s	remaining: 3.99s
    81:	learn: 0.1015842	total: 17.2s	remaining: 3.78s
    82:	learn: 0.1004796	total: 17.4s	remaining: 3.57s
    83:	learn: 0.0993602	total: 17.7s	remaining: 3.36s
    84:	learn: 0.0982842	total: 17.9s	remaining: 3.15s
    85:	learn: 0.0972721	total: 18.1s	remaining: 2.94s
    86:	learn: 0.0963954	total: 18.3s	remaining: 2.74s
    87:	learn: 0.0953914	total: 18.5s	remaining: 2.53s
    88:	learn: 0.0944404	total: 18.7s	remaining: 2.32s
    89:	learn: 0.0935870	total: 19s	remaining: 2.11s
    90:	learn: 0.0927244	total: 19.2s	remaining: 1.9s
    91:	learn: 0.0919141	total: 19.4s	remaining: 1.69s
    92:	learn: 0.0911096	total: 19.7s	remaining: 1.48s
    93:	learn: 0.0903666	total: 19.9s	remaining: 1.27s
    94:	learn: 0.0896279	total: 20.1s	remaining: 1.06s
    95:	learn: 0.0889145	total: 20.3s	remaining: 846ms
    96:	learn: 0.0881836	total: 20.5s	remaining: 634ms
    97:	learn: 0.0874681	total: 20.7s	remaining: 423ms
    98:	learn: 0.0867694	total: 21s	remaining: 212ms
    99:	learn: 0.0860989	total: 21.2s	remaining: 0us
    Trained model nº 7/27. Iterations: 100Depth: 10Learning rate: 0.01
    0:	learn: 0.6709280	total: 370ms	remaining: 1m 13s
    1:	learn: 0.6487503	total: 677ms	remaining: 1m 7s
    2:	learn: 0.6280402	total: 858ms	remaining: 56.3s
    3:	learn: 0.6069655	total: 1.2s	remaining: 59s
    4:	learn: 0.5882385	total: 1.49s	remaining: 58.2s
    5:	learn: 0.5701167	total: 1.81s	remaining: 58.4s
    6:	learn: 0.5535306	total: 2.06s	remaining: 56.7s
    7:	learn: 0.5343151	total: 2.41s	remaining: 57.8s
    8:	learn: 0.5161472	total: 2.71s	remaining: 57.6s
    9:	learn: 0.5009580	total: 3.07s	remaining: 58.3s
    10:	learn: 0.4857692	total: 3.42s	remaining: 58.8s
    11:	learn: 0.4692129	total: 3.74s	remaining: 58.6s
    12:	learn: 0.4557561	total: 4.06s	remaining: 58.4s
    13:	learn: 0.4416863	total: 4.42s	remaining: 58.7s
    14:	learn: 0.4281171	total: 4.74s	remaining: 58.5s
    15:	learn: 0.4164354	total: 4.98s	remaining: 57.3s
    16:	learn: 0.4030027	total: 5.29s	remaining: 57s
    17:	learn: 0.3902485	total: 5.63s	remaining: 56.9s
    18:	learn: 0.3775050	total: 5.97s	remaining: 56.9s
    19:	learn: 0.3657411	total: 6.32s	remaining: 56.9s
    20:	learn: 0.3545054	total: 6.66s	remaining: 56.8s
    21:	learn: 0.3440675	total: 7s	remaining: 56.6s
    22:	learn: 0.3332023	total: 7.33s	remaining: 56.5s
    23:	learn: 0.3232359	total: 7.65s	remaining: 56.1s
    24:	learn: 0.3133205	total: 7.97s	remaining: 55.8s
    25:	learn: 0.3042153	total: 8.31s	remaining: 55.6s
    26:	learn: 0.2949934	total: 8.65s	remaining: 55.4s
    27:	learn: 0.2865564	total: 8.88s	remaining: 54.6s
    28:	learn: 0.2784296	total: 9.2s	remaining: 54.2s
    29:	learn: 0.2705664	total: 9.54s	remaining: 54s
    30:	learn: 0.2632143	total: 9.84s	remaining: 53.6s
    31:	learn: 0.2560832	total: 10.2s	remaining: 53.3s
    32:	learn: 0.2491029	total: 10.5s	remaining: 53.2s
    33:	learn: 0.2425508	total: 10.8s	remaining: 52.9s
    34:	learn: 0.2364102	total: 11.2s	remaining: 52.8s
    35:	learn: 0.2302936	total: 11.5s	remaining: 52.6s
    36:	learn: 0.2244040	total: 11.9s	remaining: 52.2s
    37:	learn: 0.2189867	total: 12.2s	remaining: 52s
    38:	learn: 0.2136266	total: 12.5s	remaining: 51.6s
    39:	learn: 0.2084021	total: 12.8s	remaining: 51.4s
    40:	learn: 0.2034402	total: 13.2s	remaining: 51s
    41:	learn: 0.1985118	total: 13.5s	remaining: 50.8s
    42:	learn: 0.1938799	total: 13.8s	remaining: 50.4s
    43:	learn: 0.1896552	total: 14.1s	remaining: 50.2s
    44:	learn: 0.1853294	total: 14.5s	remaining: 49.8s
    45:	learn: 0.1812028	total: 14.8s	remaining: 49.5s
    46:	learn: 0.1774908	total: 15.1s	remaining: 49.2s
    47:	learn: 0.1736911	total: 15.4s	remaining: 48.9s
    48:	learn: 0.1699636	total: 15.8s	remaining: 48.7s
    49:	learn: 0.1664491	total: 16.1s	remaining: 48.3s
    50:	learn: 0.1632937	total: 16.4s	remaining: 47.8s
    51:	learn: 0.1598989	total: 16.7s	remaining: 47.5s
    52:	learn: 0.1567814	total: 17s	remaining: 47.2s
    53:	learn: 0.1537023	total: 17.4s	remaining: 47s
    54:	learn: 0.1509399	total: 17.7s	remaining: 46.6s
    55:	learn: 0.1481542	total: 18s	remaining: 46.3s
    56:	learn: 0.1454520	total: 18.3s	remaining: 46s
    57:	learn: 0.1430012	total: 18.7s	remaining: 45.7s
    58:	learn: 0.1404791	total: 19s	remaining: 45.3s
    59:	learn: 0.1380581	total: 19.3s	remaining: 45s
    60:	learn: 0.1356393	total: 19.6s	remaining: 44.7s
    61:	learn: 0.1333484	total: 20s	remaining: 44.4s
    62:	learn: 0.1311829	total: 20.2s	remaining: 44s
    63:	learn: 0.1290624	total: 20.6s	remaining: 43.7s
    64:	learn: 0.1270112	total: 20.9s	remaining: 43.4s
    65:	learn: 0.1250588	total: 21.2s	remaining: 43.1s
    66:	learn: 0.1231406	total: 21.6s	remaining: 42.8s
    67:	learn: 0.1212340	total: 21.9s	remaining: 42.5s
    68:	learn: 0.1194287	total: 22.2s	remaining: 42.2s
    69:	learn: 0.1177351	total: 22.5s	remaining: 41.8s
    70:	learn: 0.1160932	total: 22.8s	remaining: 41.5s
    71:	learn: 0.1144977	total: 23.2s	remaining: 41.2s
    72:	learn: 0.1129631	total: 23.5s	remaining: 40.9s
    73:	learn: 0.1114941	total: 23.8s	remaining: 40.6s
    74:	learn: 0.1101338	total: 24.1s	remaining: 40.2s
    75:	learn: 0.1087533	total: 24.5s	remaining: 39.9s
    76:	learn: 0.1073412	total: 24.8s	remaining: 39.6s
    77:	learn: 0.1059998	total: 25.1s	remaining: 39.3s
    78:	learn: 0.1047555	total: 25.5s	remaining: 39s
    79:	learn: 0.1035653	total: 25.8s	remaining: 38.7s
    80:	learn: 0.1024094	total: 26.1s	remaining: 38.4s
    81:	learn: 0.1012838	total: 26.5s	remaining: 38.1s
    82:	learn: 0.1001715	total: 26.8s	remaining: 37.8s
    83:	learn: 0.0991251	total: 27.1s	remaining: 37.5s
    84:	learn: 0.0980879	total: 27.5s	remaining: 37.2s
    85:	learn: 0.0971510	total: 27.8s	remaining: 36.8s
    86:	learn: 0.0961820	total: 28.1s	remaining: 36.5s
    87:	learn: 0.0951274	total: 28.5s	remaining: 36.2s
    88:	learn: 0.0941915	total: 28.8s	remaining: 35.9s
    89:	learn: 0.0933229	total: 29.1s	remaining: 35.6s
    90:	learn: 0.0924567	total: 29.5s	remaining: 35.3s
    91:	learn: 0.0916520	total: 29.8s	remaining: 35s
    92:	learn: 0.0908694	total: 30.2s	remaining: 34.7s
    93:	learn: 0.0900869	total: 30.5s	remaining: 34.4s
    94:	learn: 0.0893477	total: 30.8s	remaining: 34.1s
    95:	learn: 0.0886123	total: 31.1s	remaining: 33.7s
    96:	learn: 0.0878920	total: 31.5s	remaining: 33.4s
    97:	learn: 0.0872134	total: 31.8s	remaining: 33.1s
    98:	learn: 0.0865423	total: 32.1s	remaining: 32.8s
    99:	learn: 0.0859105	total: 32.5s	remaining: 32.5s
    100:	learn: 0.0852920	total: 32.8s	remaining: 32.2s
    101:	learn: 0.0847021	total: 33.1s	remaining: 31.8s
    102:	learn: 0.0841028	total: 33.5s	remaining: 31.5s
    103:	learn: 0.0835659	total: 33.8s	remaining: 31.2s
    104:	learn: 0.0830216	total: 34.2s	remaining: 30.9s
    105:	learn: 0.0824817	total: 34.5s	remaining: 30.6s
    106:	learn: 0.0819508	total: 34.8s	remaining: 30.3s
    107:	learn: 0.0814435	total: 35.1s	remaining: 29.9s
    108:	learn: 0.0809512	total: 35.5s	remaining: 29.6s
    109:	learn: 0.0804703	total: 35.8s	remaining: 29.3s
    110:	learn: 0.0799904	total: 36.1s	remaining: 29s
    111:	learn: 0.0795451	total: 36.5s	remaining: 28.7s
    112:	learn: 0.0791058	total: 36.8s	remaining: 28.3s
    113:	learn: 0.0786862	total: 37.1s	remaining: 28s
    114:	learn: 0.0782558	total: 37.5s	remaining: 27.7s
    115:	learn: 0.0778432	total: 37.8s	remaining: 27.4s
    116:	learn: 0.0774485	total: 38.1s	remaining: 27s
    117:	learn: 0.0770719	total: 38.5s	remaining: 26.7s
    118:	learn: 0.0767061	total: 38.8s	remaining: 26.4s
    119:	learn: 0.0763660	total: 39.1s	remaining: 26.1s
    120:	learn: 0.0760296	total: 39.5s	remaining: 25.8s
    121:	learn: 0.0756974	total: 39.8s	remaining: 25.4s
    122:	learn: 0.0753573	total: 40.1s	remaining: 25.1s
    123:	learn: 0.0750280	total: 40.5s	remaining: 24.8s
    124:	learn: 0.0746997	total: 40.8s	remaining: 24.5s
    125:	learn: 0.0744001	total: 41.1s	remaining: 24.2s
    126:	learn: 0.0740950	total: 41.5s	remaining: 23.8s
    127:	learn: 0.0737941	total: 41.8s	remaining: 23.5s
    128:	learn: 0.0735346	total: 42.1s	remaining: 23.2s
    129:	learn: 0.0732545	total: 42.5s	remaining: 22.9s
    130:	learn: 0.0729769	total: 42.8s	remaining: 22.5s
    131:	learn: 0.0727626	total: 43s	remaining: 22.1s
    132:	learn: 0.0724996	total: 43.3s	remaining: 21.8s
    133:	learn: 0.0722436	total: 43.7s	remaining: 21.5s
    134:	learn: 0.0719925	total: 44s	remaining: 21.2s
    135:	learn: 0.0717638	total: 44.3s	remaining: 20.8s
    136:	learn: 0.0715395	total: 44.6s	remaining: 20.5s
    137:	learn: 0.0713409	total: 44.9s	remaining: 20.2s
    138:	learn: 0.0711201	total: 45.3s	remaining: 19.9s
    139:	learn: 0.0708987	total: 45.6s	remaining: 19.5s
    140:	learn: 0.0706842	total: 46s	remaining: 19.2s
    141:	learn: 0.0704847	total: 46.3s	remaining: 18.9s
    142:	learn: 0.0702836	total: 46.6s	remaining: 18.6s
    143:	learn: 0.0700962	total: 46.9s	remaining: 18.2s
    144:	learn: 0.0699176	total: 47.3s	remaining: 17.9s
    145:	learn: 0.0697255	total: 47.6s	remaining: 17.6s
    146:	learn: 0.0695489	total: 47.9s	remaining: 17.3s
    147:	learn: 0.0693782	total: 48.2s	remaining: 16.9s
    148:	learn: 0.0692055	total: 48.5s	remaining: 16.6s
    149:	learn: 0.0690343	total: 48.9s	remaining: 16.3s
    150:	learn: 0.0688687	total: 49.2s	remaining: 16s
    151:	learn: 0.0687046	total: 49.6s	remaining: 15.7s
    152:	learn: 0.0685516	total: 49.9s	remaining: 15.3s
    153:	learn: 0.0684020	total: 50.2s	remaining: 15s
    154:	learn: 0.0682651	total: 50.5s	remaining: 14.7s
    155:	learn: 0.0681327	total: 50.9s	remaining: 14.3s
    156:	learn: 0.0679913	total: 51.2s	remaining: 14s
    157:	learn: 0.0678496	total: 51.5s	remaining: 13.7s
    158:	learn: 0.0677195	total: 51.9s	remaining: 13.4s
    159:	learn: 0.0675980	total: 52.2s	remaining: 13.1s
    160:	learn: 0.0674759	total: 52.5s	remaining: 12.7s
    161:	learn: 0.0673504	total: 52.9s	remaining: 12.4s
    162:	learn: 0.0672314	total: 53.2s	remaining: 12.1s
    163:	learn: 0.0671272	total: 53.5s	remaining: 11.7s
    164:	learn: 0.0670079	total: 53.8s	remaining: 11.4s
    165:	learn: 0.0668981	total: 54.2s	remaining: 11.1s
    166:	learn: 0.0667849	total: 54.5s	remaining: 10.8s
    167:	learn: 0.0666968	total: 54.8s	remaining: 10.4s
    168:	learn: 0.0665892	total: 55.2s	remaining: 10.1s
    169:	learn: 0.0664872	total: 55.5s	remaining: 9.79s
    170:	learn: 0.0663876	total: 55.8s	remaining: 9.47s
    171:	learn: 0.0662868	total: 56.2s	remaining: 9.14s
    172:	learn: 0.0661886	total: 56.5s	remaining: 8.81s
    173:	learn: 0.0660917	total: 56.8s	remaining: 8.49s
    174:	learn: 0.0660016	total: 57.1s	remaining: 8.16s
    175:	learn: 0.0659144	total: 57.5s	remaining: 7.83s
    176:	learn: 0.0657996	total: 57.8s	remaining: 7.51s
    177:	learn: 0.0657127	total: 58.1s	remaining: 7.18s
    178:	learn: 0.0656311	total: 58.4s	remaining: 6.86s
    179:	learn: 0.0655532	total: 58.8s	remaining: 6.53s
    180:	learn: 0.0654697	total: 59.1s	remaining: 6.21s
    181:	learn: 0.0653910	total: 59.5s	remaining: 5.88s
    182:	learn: 0.0653175	total: 59.8s	remaining: 5.55s
    183:	learn: 0.0652396	total: 1m	remaining: 5.23s
    184:	learn: 0.0651645	total: 1m	remaining: 4.9s
    185:	learn: 0.0650877	total: 1m	remaining: 4.57s
    186:	learn: 0.0649897	total: 1m 1s	remaining: 4.25s
    187:	learn: 0.0649149	total: 1m 1s	remaining: 3.92s
    188:	learn: 0.0648541	total: 1m 1s	remaining: 3.59s
    189:	learn: 0.0647907	total: 1m 2s	remaining: 3.27s
    190:	learn: 0.0647240	total: 1m 2s	remaining: 2.94s
    191:	learn: 0.0646614	total: 1m 2s	remaining: 2.61s
    192:	learn: 0.0646009	total: 1m 3s	remaining: 2.29s
    193:	learn: 0.0645458	total: 1m 3s	remaining: 1.96s
    194:	learn: 0.0644857	total: 1m 3s	remaining: 1.63s
    195:	learn: 0.0644271	total: 1m 4s	remaining: 1.31s
    196:	learn: 0.0643745	total: 1m 4s	remaining: 980ms
    197:	learn: 0.0643197	total: 1m 4s	remaining: 653ms
    198:	learn: 0.0642691	total: 1m 4s	remaining: 326ms
    199:	learn: 0.0642222	total: 1m 5s	remaining: 0us
    Trained model nº 8/27. Iterations: 200Depth: 10Learning rate: 0.01
    0:	learn: 0.6709280	total: 347ms	remaining: 1m 43s
    1:	learn: 0.6487503	total: 643ms	remaining: 1m 35s
    2:	learn: 0.6280402	total: 812ms	remaining: 1m 20s
    3:	learn: 0.6069655	total: 1.16s	remaining: 1m 25s
    4:	learn: 0.5882385	total: 1.45s	remaining: 1m 25s
    5:	learn: 0.5701167	total: 1.77s	remaining: 1m 26s
    6:	learn: 0.5535306	total: 2.02s	remaining: 1m 24s
    7:	learn: 0.5343151	total: 2.36s	remaining: 1m 26s
    8:	learn: 0.5161472	total: 2.67s	remaining: 1m 26s
    9:	learn: 0.5009580	total: 3.03s	remaining: 1m 27s
    10:	learn: 0.4857692	total: 3.38s	remaining: 1m 28s
    11:	learn: 0.4692129	total: 3.71s	remaining: 1m 29s
    12:	learn: 0.4557561	total: 4.04s	remaining: 1m 29s
    13:	learn: 0.4416863	total: 4.37s	remaining: 1m 29s
    14:	learn: 0.4281171	total: 4.71s	remaining: 1m 29s
    15:	learn: 0.4164354	total: 4.93s	remaining: 1m 27s
    16:	learn: 0.4030027	total: 5.25s	remaining: 1m 27s
    17:	learn: 0.3902485	total: 5.58s	remaining: 1m 27s
    18:	learn: 0.3775050	total: 5.92s	remaining: 1m 27s
    19:	learn: 0.3657411	total: 6.28s	remaining: 1m 27s
    20:	learn: 0.3545054	total: 6.63s	remaining: 1m 28s
    21:	learn: 0.3440675	total: 6.96s	remaining: 1m 27s
    22:	learn: 0.3332023	total: 7.3s	remaining: 1m 27s
    23:	learn: 0.3232359	total: 7.61s	remaining: 1m 27s
    24:	learn: 0.3133205	total: 7.95s	remaining: 1m 27s
    25:	learn: 0.3042153	total: 8.28s	remaining: 1m 27s
    26:	learn: 0.2949934	total: 8.61s	remaining: 1m 27s
    27:	learn: 0.2865564	total: 8.84s	remaining: 1m 25s
    28:	learn: 0.2784296	total: 9.15s	remaining: 1m 25s
    29:	learn: 0.2705664	total: 9.49s	remaining: 1m 25s
    30:	learn: 0.2632143	total: 9.79s	remaining: 1m 24s
    31:	learn: 0.2560832	total: 10.1s	remaining: 1m 24s
    32:	learn: 0.2491029	total: 10.4s	remaining: 1m 24s
    33:	learn: 0.2425508	total: 10.8s	remaining: 1m 24s
    34:	learn: 0.2364102	total: 11.1s	remaining: 1m 24s
    35:	learn: 0.2302936	total: 11.5s	remaining: 1m 23s
    36:	learn: 0.2244040	total: 11.8s	remaining: 1m 23s
    37:	learn: 0.2189867	total: 12.1s	remaining: 1m 23s
    38:	learn: 0.2136266	total: 12.4s	remaining: 1m 23s
    39:	learn: 0.2084021	total: 12.7s	remaining: 1m 22s
    40:	learn: 0.2034402	total: 13.1s	remaining: 1m 22s
    41:	learn: 0.1985118	total: 13.4s	remaining: 1m 22s
    42:	learn: 0.1938799	total: 13.7s	remaining: 1m 21s
    43:	learn: 0.1896552	total: 14s	remaining: 1m 21s
    44:	learn: 0.1853294	total: 14.4s	remaining: 1m 21s
    45:	learn: 0.1812028	total: 14.8s	remaining: 1m 21s
    46:	learn: 0.1774908	total: 15.2s	remaining: 1m 21s
    47:	learn: 0.1736911	total: 15.6s	remaining: 1m 21s
    48:	learn: 0.1699636	total: 16s	remaining: 1m 21s
    49:	learn: 0.1664491	total: 16.3s	remaining: 1m 21s
    50:	learn: 0.1632937	total: 16.6s	remaining: 1m 21s
    51:	learn: 0.1598989	total: 16.9s	remaining: 1m 20s
    52:	learn: 0.1567814	total: 17.3s	remaining: 1m 20s
    53:	learn: 0.1537023	total: 17.6s	remaining: 1m 20s
    54:	learn: 0.1509399	total: 18s	remaining: 1m 20s
    55:	learn: 0.1481542	total: 18.3s	remaining: 1m 19s
    56:	learn: 0.1454520	total: 19.2s	remaining: 1m 21s
    57:	learn: 0.1430012	total: 19.8s	remaining: 1m 22s
    58:	learn: 0.1404791	total: 20.4s	remaining: 1m 23s
    59:	learn: 0.1380581	total: 21.1s	remaining: 1m 24s
    60:	learn: 0.1356393	total: 21.8s	remaining: 1m 25s
    61:	learn: 0.1333484	total: 22.5s	remaining: 1m 26s
    62:	learn: 0.1311829	total: 22.9s	remaining: 1m 26s
    63:	learn: 0.1290624	total: 23.4s	remaining: 1m 26s
    64:	learn: 0.1270112	total: 23.9s	remaining: 1m 26s
    65:	learn: 0.1250588	total: 24.3s	remaining: 1m 26s
    66:	learn: 0.1231406	total: 24.7s	remaining: 1m 25s
    67:	learn: 0.1212340	total: 25s	remaining: 1m 25s
    68:	learn: 0.1194287	total: 25.4s	remaining: 1m 24s
    69:	learn: 0.1177351	total: 25.8s	remaining: 1m 24s
    70:	learn: 0.1160932	total: 26.1s	remaining: 1m 24s
    71:	learn: 0.1144977	total: 26.4s	remaining: 1m 23s
    72:	learn: 0.1129631	total: 26.8s	remaining: 1m 23s
    73:	learn: 0.1114941	total: 27.2s	remaining: 1m 22s
    74:	learn: 0.1101338	total: 27.5s	remaining: 1m 22s
    75:	learn: 0.1087533	total: 27.8s	remaining: 1m 22s
    76:	learn: 0.1073412	total: 28.2s	remaining: 1m 21s
    77:	learn: 0.1059998	total: 28.6s	remaining: 1m 21s
    78:	learn: 0.1047555	total: 29s	remaining: 1m 21s
    79:	learn: 0.1035653	total: 29.4s	remaining: 1m 20s
    80:	learn: 0.1024094	total: 29.8s	remaining: 1m 20s
    81:	learn: 0.1012838	total: 30.1s	remaining: 1m 20s
    82:	learn: 0.1001715	total: 30.5s	remaining: 1m 19s
    83:	learn: 0.0991251	total: 30.9s	remaining: 1m 19s
    84:	learn: 0.0980879	total: 31.2s	remaining: 1m 18s
    85:	learn: 0.0971510	total: 31.6s	remaining: 1m 18s
    86:	learn: 0.0961820	total: 31.9s	remaining: 1m 18s
    87:	learn: 0.0951274	total: 32.3s	remaining: 1m 17s
    88:	learn: 0.0941915	total: 32.7s	remaining: 1m 17s
    89:	learn: 0.0933229	total: 33.1s	remaining: 1m 17s
    90:	learn: 0.0924567	total: 33.5s	remaining: 1m 16s
    91:	learn: 0.0916520	total: 33.9s	remaining: 1m 16s
    92:	learn: 0.0908694	total: 34.2s	remaining: 1m 16s
    93:	learn: 0.0900869	total: 34.6s	remaining: 1m 15s
    94:	learn: 0.0893477	total: 35s	remaining: 1m 15s
    95:	learn: 0.0886123	total: 35.3s	remaining: 1m 15s
    96:	learn: 0.0878920	total: 35.7s	remaining: 1m 14s
    97:	learn: 0.0872134	total: 36s	remaining: 1m 14s
    98:	learn: 0.0865423	total: 36.4s	remaining: 1m 13s
    99:	learn: 0.0859105	total: 36.7s	remaining: 1m 13s
    100:	learn: 0.0852920	total: 37.1s	remaining: 1m 13s
    101:	learn: 0.0847021	total: 37.4s	remaining: 1m 12s
    102:	learn: 0.0841028	total: 37.8s	remaining: 1m 12s
    103:	learn: 0.0835659	total: 38.1s	remaining: 1m 11s
    104:	learn: 0.0830216	total: 38.5s	remaining: 1m 11s
    105:	learn: 0.0824817	total: 38.8s	remaining: 1m 10s
    106:	learn: 0.0819508	total: 39.1s	remaining: 1m 10s
    107:	learn: 0.0814435	total: 39.5s	remaining: 1m 10s
    108:	learn: 0.0809512	total: 39.8s	remaining: 1m 9s
    109:	learn: 0.0804703	total: 40.1s	remaining: 1m 9s
    110:	learn: 0.0799904	total: 40.5s	remaining: 1m 8s
    111:	learn: 0.0795451	total: 40.8s	remaining: 1m 8s
    112:	learn: 0.0791058	total: 41.1s	remaining: 1m 8s
    113:	learn: 0.0786862	total: 41.5s	remaining: 1m 7s
    114:	learn: 0.0782558	total: 41.8s	remaining: 1m 7s
    115:	learn: 0.0778432	total: 42.1s	remaining: 1m 6s
    116:	learn: 0.0774485	total: 42.5s	remaining: 1m 6s
    117:	learn: 0.0770719	total: 42.8s	remaining: 1m 6s
    118:	learn: 0.0767061	total: 43.1s	remaining: 1m 5s
    119:	learn: 0.0763660	total: 43.5s	remaining: 1m 5s
    120:	learn: 0.0760296	total: 43.8s	remaining: 1m 4s
    121:	learn: 0.0756974	total: 44.2s	remaining: 1m 4s
    122:	learn: 0.0753573	total: 44.5s	remaining: 1m 4s
    123:	learn: 0.0750280	total: 44.9s	remaining: 1m 3s
    124:	learn: 0.0746997	total: 45.2s	remaining: 1m 3s
    125:	learn: 0.0744001	total: 45.6s	remaining: 1m 2s
    126:	learn: 0.0740950	total: 46s	remaining: 1m 2s
    127:	learn: 0.0737941	total: 46.3s	remaining: 1m 2s
    128:	learn: 0.0735346	total: 46.7s	remaining: 1m 1s
    129:	learn: 0.0732545	total: 47s	remaining: 1m 1s
    130:	learn: 0.0729769	total: 47.4s	remaining: 1m 1s
    131:	learn: 0.0727626	total: 47.6s	remaining: 1m
    132:	learn: 0.0724996	total: 47.9s	remaining: 1m
    133:	learn: 0.0722436	total: 48.2s	remaining: 59.7s
    134:	learn: 0.0719925	total: 48.6s	remaining: 59.3s
    135:	learn: 0.0717638	total: 48.9s	remaining: 58.9s
    136:	learn: 0.0715395	total: 49.2s	remaining: 58.6s
    137:	learn: 0.0713409	total: 49.5s	remaining: 58.2s
    138:	learn: 0.0711201	total: 49.9s	remaining: 57.8s
    139:	learn: 0.0708987	total: 50.2s	remaining: 57.4s
    140:	learn: 0.0706842	total: 50.6s	remaining: 57s
    141:	learn: 0.0704847	total: 50.9s	remaining: 56.6s
    142:	learn: 0.0702836	total: 51.2s	remaining: 56.2s
    143:	learn: 0.0700962	total: 51.5s	remaining: 55.8s
    144:	learn: 0.0699176	total: 51.9s	remaining: 55.4s
    145:	learn: 0.0697255	total: 52.2s	remaining: 55s
    146:	learn: 0.0695489	total: 52.5s	remaining: 54.7s
    147:	learn: 0.0693782	total: 52.8s	remaining: 54.2s
    148:	learn: 0.0692055	total: 53.1s	remaining: 53.8s
    149:	learn: 0.0690343	total: 53.5s	remaining: 53.5s
    150:	learn: 0.0688687	total: 53.8s	remaining: 53.1s
    151:	learn: 0.0687046	total: 54.2s	remaining: 52.7s
    152:	learn: 0.0685516	total: 54.5s	remaining: 52.4s
    153:	learn: 0.0684020	total: 54.8s	remaining: 52s
    154:	learn: 0.0682651	total: 55.2s	remaining: 51.6s
    155:	learn: 0.0681327	total: 55.5s	remaining: 51.2s
    156:	learn: 0.0679913	total: 55.8s	remaining: 50.9s
    157:	learn: 0.0678496	total: 56.2s	remaining: 50.5s
    158:	learn: 0.0677195	total: 56.5s	remaining: 50.1s
    159:	learn: 0.0675980	total: 56.9s	remaining: 49.7s
    160:	learn: 0.0674759	total: 57.2s	remaining: 49.4s
    161:	learn: 0.0673504	total: 57.5s	remaining: 49s
    162:	learn: 0.0672314	total: 57.8s	remaining: 48.6s
    163:	learn: 0.0671272	total: 58.2s	remaining: 48.2s
    164:	learn: 0.0670079	total: 58.5s	remaining: 47.9s
    165:	learn: 0.0668981	total: 58.8s	remaining: 47.5s
    166:	learn: 0.0667849	total: 59.1s	remaining: 47.1s
    167:	learn: 0.0666968	total: 59.5s	remaining: 46.7s
    168:	learn: 0.0665892	total: 59.8s	remaining: 46.4s
    169:	learn: 0.0664872	total: 1m	remaining: 46s
    170:	learn: 0.0663876	total: 1m	remaining: 45.6s
    171:	learn: 0.0662868	total: 1m	remaining: 45.3s
    172:	learn: 0.0661886	total: 1m 1s	remaining: 44.9s
    173:	learn: 0.0660917	total: 1m 1s	remaining: 44.5s
    174:	learn: 0.0660016	total: 1m 1s	remaining: 44.2s
    175:	learn: 0.0659144	total: 1m 2s	remaining: 43.8s
    176:	learn: 0.0657996	total: 1m 2s	remaining: 43.5s
    177:	learn: 0.0657127	total: 1m 2s	remaining: 43.1s
    178:	learn: 0.0656311	total: 1m 3s	remaining: 42.8s
    179:	learn: 0.0655532	total: 1m 3s	remaining: 42.4s
    180:	learn: 0.0654697	total: 1m 3s	remaining: 42s
    181:	learn: 0.0653910	total: 1m 4s	remaining: 41.7s
    182:	learn: 0.0653175	total: 1m 4s	remaining: 41.3s
    183:	learn: 0.0652396	total: 1m 4s	remaining: 41s
    184:	learn: 0.0651645	total: 1m 5s	remaining: 40.6s
    185:	learn: 0.0650877	total: 1m 5s	remaining: 40.2s
    186:	learn: 0.0649897	total: 1m 6s	remaining: 39.9s
    187:	learn: 0.0649149	total: 1m 6s	remaining: 39.5s
    188:	learn: 0.0648541	total: 1m 6s	remaining: 39.2s
    189:	learn: 0.0647907	total: 1m 7s	remaining: 38.8s
    190:	learn: 0.0647240	total: 1m 7s	remaining: 38.5s
    191:	learn: 0.0646614	total: 1m 7s	remaining: 38.1s
    192:	learn: 0.0646009	total: 1m 8s	remaining: 37.7s
    193:	learn: 0.0645458	total: 1m 8s	remaining: 37.4s
    194:	learn: 0.0644857	total: 1m 8s	remaining: 37s
    195:	learn: 0.0644271	total: 1m 9s	remaining: 36.7s
    196:	learn: 0.0643745	total: 1m 9s	remaining: 36.3s
    197:	learn: 0.0643197	total: 1m 9s	remaining: 35.9s
    198:	learn: 0.0642691	total: 1m 10s	remaining: 35.6s
    199:	learn: 0.0642222	total: 1m 10s	remaining: 35.2s
    200:	learn: 0.0641664	total: 1m 10s	remaining: 34.8s
    201:	learn: 0.0641196	total: 1m 11s	remaining: 34.5s
    202:	learn: 0.0640672	total: 1m 11s	remaining: 34.1s
    203:	learn: 0.0640181	total: 1m 11s	remaining: 33.8s
    204:	learn: 0.0639730	total: 1m 12s	remaining: 33.4s
    205:	learn: 0.0639232	total: 1m 12s	remaining: 33s
    206:	learn: 0.0638803	total: 1m 12s	remaining: 32.7s
    207:	learn: 0.0638333	total: 1m 13s	remaining: 32.3s
    208:	learn: 0.0637918	total: 1m 13s	remaining: 31.9s
    209:	learn: 0.0637492	total: 1m 13s	remaining: 31.6s
    210:	learn: 0.0637066	total: 1m 14s	remaining: 31.2s
    211:	learn: 0.0636716	total: 1m 14s	remaining: 30.9s
    212:	learn: 0.0636286	total: 1m 14s	remaining: 30.5s
    213:	learn: 0.0635910	total: 1m 15s	remaining: 30.2s
    214:	learn: 0.0635562	total: 1m 15s	remaining: 29.8s
    215:	learn: 0.0635194	total: 1m 15s	remaining: 29.5s
    216:	learn: 0.0634771	total: 1m 16s	remaining: 29.1s
    217:	learn: 0.0634415	total: 1m 16s	remaining: 28.8s
    218:	learn: 0.0634058	total: 1m 16s	remaining: 28.4s
    219:	learn: 0.0633450	total: 1m 17s	remaining: 28.1s
    220:	learn: 0.0633117	total: 1m 17s	remaining: 27.7s
    221:	learn: 0.0632753	total: 1m 17s	remaining: 27.4s
    222:	learn: 0.0632399	total: 1m 18s	remaining: 27s
    223:	learn: 0.0631925	total: 1m 18s	remaining: 26.7s
    224:	learn: 0.0631624	total: 1m 18s	remaining: 26.3s
    225:	learn: 0.0631304	total: 1m 19s	remaining: 26s
    226:	learn: 0.0631051	total: 1m 19s	remaining: 25.6s
    227:	learn: 0.0630734	total: 1m 19s	remaining: 25.3s
    228:	learn: 0.0630430	total: 1m 20s	remaining: 24.9s
    229:	learn: 0.0630101	total: 1m 20s	remaining: 24.6s
    230:	learn: 0.0629812	total: 1m 21s	remaining: 24.2s
    231:	learn: 0.0629525	total: 1m 21s	remaining: 23.9s
    232:	learn: 0.0629220	total: 1m 21s	remaining: 23.5s
    233:	learn: 0.0628915	total: 1m 22s	remaining: 23.2s
    234:	learn: 0.0628696	total: 1m 22s	remaining: 22.8s
    235:	learn: 0.0628425	total: 1m 22s	remaining: 22.5s
    236:	learn: 0.0628175	total: 1m 23s	remaining: 22.1s
    237:	learn: 0.0627933	total: 1m 23s	remaining: 21.7s
    238:	learn: 0.0627665	total: 1m 23s	remaining: 21.4s
    239:	learn: 0.0627401	total: 1m 24s	remaining: 21s
    240:	learn: 0.0627155	total: 1m 24s	remaining: 20.7s
    241:	learn: 0.0626899	total: 1m 24s	remaining: 20.3s
    242:	learn: 0.0626667	total: 1m 25s	remaining: 19.9s
    243:	learn: 0.0626468	total: 1m 25s	remaining: 19.6s
    244:	learn: 0.0626212	total: 1m 25s	remaining: 19.2s
    245:	learn: 0.0625968	total: 1m 26s	remaining: 18.9s
    246:	learn: 0.0625785	total: 1m 26s	remaining: 18.5s
    247:	learn: 0.0625474	total: 1m 26s	remaining: 18.2s
    248:	learn: 0.0625278	total: 1m 27s	remaining: 17.8s
    249:	learn: 0.0625073	total: 1m 27s	remaining: 17.5s
    250:	learn: 0.0624719	total: 1m 27s	remaining: 17.1s
    251:	learn: 0.0624502	total: 1m 28s	remaining: 16.8s
    252:	learn: 0.0624325	total: 1m 28s	remaining: 16.4s
    253:	learn: 0.0624148	total: 1m 28s	remaining: 16.1s
    254:	learn: 0.0623949	total: 1m 29s	remaining: 15.7s
    255:	learn: 0.0623775	total: 1m 29s	remaining: 15.4s
    256:	learn: 0.0623601	total: 1m 29s	remaining: 15s
    257:	learn: 0.0623435	total: 1m 30s	remaining: 14.7s
    258:	learn: 0.0623301	total: 1m 30s	remaining: 14.3s
    259:	learn: 0.0623170	total: 1m 30s	remaining: 14s
    260:	learn: 0.0622991	total: 1m 31s	remaining: 13.6s
    261:	learn: 0.0622794	total: 1m 31s	remaining: 13.3s
    262:	learn: 0.0622614	total: 1m 31s	remaining: 12.9s
    263:	learn: 0.0622482	total: 1m 32s	remaining: 12.6s
    264:	learn: 0.0622336	total: 1m 32s	remaining: 12.2s
    265:	learn: 0.0622150	total: 1m 32s	remaining: 11.9s
    266:	learn: 0.0621949	total: 1m 33s	remaining: 11.5s
    267:	learn: 0.0621770	total: 1m 33s	remaining: 11.2s
    268:	learn: 0.0621435	total: 1m 33s	remaining: 10.8s
    269:	learn: 0.0621295	total: 1m 34s	remaining: 10.5s
    270:	learn: 0.0621091	total: 1m 34s	remaining: 10.1s
    271:	learn: 0.0620954	total: 1m 34s	remaining: 9.77s
    272:	learn: 0.0620810	total: 1m 35s	remaining: 9.41s
    273:	learn: 0.0620667	total: 1m 35s	remaining: 9.06s
    274:	learn: 0.0620517	total: 1m 35s	remaining: 8.72s
    275:	learn: 0.0620263	total: 1m 36s	remaining: 8.36s
    276:	learn: 0.0620150	total: 1m 36s	remaining: 8.02s
    277:	learn: 0.0620010	total: 1m 36s	remaining: 7.67s
    278:	learn: 0.0619904	total: 1m 37s	remaining: 7.32s
    279:	learn: 0.0619772	total: 1m 37s	remaining: 6.97s
    280:	learn: 0.0619627	total: 1m 37s	remaining: 6.63s
    281:	learn: 0.0619523	total: 1m 38s	remaining: 6.28s
    282:	learn: 0.0619431	total: 1m 38s	remaining: 5.93s
    283:	learn: 0.0619323	total: 1m 39s	remaining: 5.58s
    284:	learn: 0.0619184	total: 1m 39s	remaining: 5.23s
    285:	learn: 0.0619048	total: 1m 39s	remaining: 4.88s
    286:	learn: 0.0618938	total: 1m 40s	remaining: 4.54s
    287:	learn: 0.0618807	total: 1m 40s	remaining: 4.19s
    288:	learn: 0.0618711	total: 1m 40s	remaining: 3.84s
    289:	learn: 0.0618610	total: 1m 41s	remaining: 3.49s
    290:	learn: 0.0618413	total: 1m 41s	remaining: 3.14s
    291:	learn: 0.0618272	total: 1m 41s	remaining: 2.79s
    292:	learn: 0.0618166	total: 1m 42s	remaining: 2.44s
    293:	learn: 0.0618087	total: 1m 42s	remaining: 2.09s
    294:	learn: 0.0617971	total: 1m 42s	remaining: 1.74s
    295:	learn: 0.0617852	total: 1m 43s	remaining: 1.39s
    296:	learn: 0.0617760	total: 1m 43s	remaining: 1.04s
    297:	learn: 0.0617656	total: 1m 43s	remaining: 697ms
    298:	learn: 0.0617538	total: 1m 44s	remaining: 349ms
    299:	learn: 0.0617450	total: 1m 44s	remaining: 0us
    Trained model nº 9/27. Iterations: 300Depth: 10Learning rate: 0.01
    0:	learn: 0.5939021	total: 208ms	remaining: 20.6s
    1:	learn: 0.5020045	total: 355ms	remaining: 17.4s
    2:	learn: 0.4345066	total: 531ms	remaining: 17.2s
    3:	learn: 0.3738646	total: 696ms	remaining: 16.7s
    4:	learn: 0.3267099	total: 857ms	remaining: 16.3s
    5:	learn: 0.2853132	total: 1.04s	remaining: 16.2s
    6:	learn: 0.2492302	total: 1.2s	remaining: 16s
    7:	learn: 0.2225979	total: 1.36s	remaining: 15.7s
    8:	learn: 0.1958510	total: 1.52s	remaining: 15.4s
    9:	learn: 0.1732006	total: 1.68s	remaining: 15.1s
    10:	learn: 0.1556758	total: 1.84s	remaining: 14.9s
    11:	learn: 0.1413519	total: 2s	remaining: 14.7s
    12:	learn: 0.1297884	total: 2.17s	remaining: 14.6s
    13:	learn: 0.1199937	total: 2.35s	remaining: 14.4s
    14:	learn: 0.1117620	total: 2.53s	remaining: 14.4s
    15:	learn: 0.1048972	total: 2.7s	remaining: 14.2s
    16:	learn: 0.0992098	total: 2.86s	remaining: 14s
    17:	learn: 0.0942750	total: 3.02s	remaining: 13.8s
    18:	learn: 0.0901696	total: 3.19s	remaining: 13.6s
    19:	learn: 0.0866724	total: 3.34s	remaining: 13.4s
    20:	learn: 0.0837063	total: 3.5s	remaining: 13.2s
    21:	learn: 0.0810837	total: 3.67s	remaining: 13s
    22:	learn: 0.0788245	total: 3.83s	remaining: 12.8s
    23:	learn: 0.0769275	total: 3.99s	remaining: 12.6s
    24:	learn: 0.0753221	total: 4.13s	remaining: 12.4s
    25:	learn: 0.0736928	total: 4.28s	remaining: 12.2s
    26:	learn: 0.0724847	total: 4.44s	remaining: 12s
    27:	learn: 0.0714020	total: 4.61s	remaining: 11.8s
    28:	learn: 0.0704393	total: 4.78s	remaining: 11.7s
    29:	learn: 0.0695918	total: 4.94s	remaining: 11.5s
    30:	learn: 0.0688544	total: 5.11s	remaining: 11.4s
    31:	learn: 0.0682856	total: 5.29s	remaining: 11.2s
    32:	learn: 0.0677140	total: 5.45s	remaining: 11.1s
    33:	learn: 0.0671924	total: 5.61s	remaining: 10.9s
    34:	learn: 0.0667261	total: 5.78s	remaining: 10.7s
    35:	learn: 0.0663862	total: 5.93s	remaining: 10.5s
    36:	learn: 0.0660080	total: 6.09s	remaining: 10.4s
    37:	learn: 0.0654972	total: 6.25s	remaining: 10.2s
    38:	learn: 0.0652208	total: 6.39s	remaining: 10s
    39:	learn: 0.0648566	total: 6.54s	remaining: 9.82s
    40:	learn: 0.0646395	total: 6.71s	remaining: 9.66s
    41:	learn: 0.0644451	total: 6.87s	remaining: 9.49s
    42:	learn: 0.0642861	total: 7.03s	remaining: 9.32s
    43:	learn: 0.0641388	total: 7.19s	remaining: 9.15s
    44:	learn: 0.0640009	total: 7.34s	remaining: 8.98s
    45:	learn: 0.0638623	total: 7.48s	remaining: 8.78s
    46:	learn: 0.0637508	total: 7.63s	remaining: 8.61s
    47:	learn: 0.0636512	total: 7.8s	remaining: 8.45s
    48:	learn: 0.0635562	total: 7.95s	remaining: 8.28s
    49:	learn: 0.0634605	total: 8.1s	remaining: 8.1s
    50:	learn: 0.0633797	total: 8.26s	remaining: 7.94s
    51:	learn: 0.0633097	total: 8.42s	remaining: 7.78s
    52:	learn: 0.0631316	total: 8.58s	remaining: 7.61s
    53:	learn: 0.0630752	total: 8.71s	remaining: 7.42s
    54:	learn: 0.0630253	total: 8.88s	remaining: 7.26s
    55:	learn: 0.0629417	total: 9.04s	remaining: 7.11s
    56:	learn: 0.0628992	total: 9.19s	remaining: 6.93s
    57:	learn: 0.0628594	total: 9.38s	remaining: 6.79s
    58:	learn: 0.0628233	total: 9.54s	remaining: 6.63s
    59:	learn: 0.0627875	total: 9.7s	remaining: 6.47s
    60:	learn: 0.0627566	total: 9.86s	remaining: 6.3s
    61:	learn: 0.0627254	total: 10s	remaining: 6.14s
    62:	learn: 0.0626707	total: 10.2s	remaining: 5.97s
    63:	learn: 0.0626444	total: 10.3s	remaining: 5.81s
    64:	learn: 0.0626198	total: 10.5s	remaining: 5.66s
    65:	learn: 0.0625903	total: 10.7s	remaining: 5.5s
    66:	learn: 0.0625614	total: 10.8s	remaining: 5.34s
    67:	learn: 0.0625414	total: 11s	remaining: 5.18s
    68:	learn: 0.0625185	total: 11.2s	remaining: 5.02s
    69:	learn: 0.0624932	total: 11.3s	remaining: 4.86s
    70:	learn: 0.0624723	total: 11.5s	remaining: 4.7s
    71:	learn: 0.0624601	total: 11.7s	remaining: 4.54s
    72:	learn: 0.0624185	total: 11.8s	remaining: 4.37s
    73:	learn: 0.0623958	total: 12s	remaining: 4.21s
    74:	learn: 0.0623780	total: 12.1s	remaining: 4.05s
    75:	learn: 0.0623570	total: 12.3s	remaining: 3.89s
    76:	learn: 0.0623461	total: 12.5s	remaining: 3.73s
    77:	learn: 0.0622298	total: 12.6s	remaining: 3.57s
    78:	learn: 0.0622213	total: 12.8s	remaining: 3.41s
    79:	learn: 0.0622084	total: 13s	remaining: 3.24s
    80:	learn: 0.0621955	total: 13.1s	remaining: 3.08s
    81:	learn: 0.0621787	total: 13.3s	remaining: 2.92s
    82:	learn: 0.0621659	total: 13.5s	remaining: 2.76s
    83:	learn: 0.0621554	total: 13.6s	remaining: 2.6s
    84:	learn: 0.0621147	total: 13.8s	remaining: 2.44s
    85:	learn: 0.0621048	total: 14s	remaining: 2.27s
    86:	learn: 0.0620924	total: 14.1s	remaining: 2.11s
    87:	learn: 0.0620839	total: 14.3s	remaining: 1.95s
    88:	learn: 0.0620707	total: 14.5s	remaining: 1.79s
    89:	learn: 0.0620574	total: 14.6s	remaining: 1.62s
    90:	learn: 0.0619949	total: 14.8s	remaining: 1.46s
    91:	learn: 0.0619826	total: 15s	remaining: 1.3s
    92:	learn: 0.0619729	total: 15.1s	remaining: 1.14s
    93:	learn: 0.0619642	total: 15.3s	remaining: 976ms
    94:	learn: 0.0619547	total: 15.5s	remaining: 814ms
    95:	learn: 0.0619468	total: 15.6s	remaining: 652ms
    96:	learn: 0.0618851	total: 15.8s	remaining: 489ms
    97:	learn: 0.0618755	total: 16s	remaining: 326ms
    98:	learn: 0.0618609	total: 16.1s	remaining: 163ms
    99:	learn: 0.0618540	total: 16.3s	remaining: 0us
    Trained model nº 10/27. Iterations: 100Depth: 6Learning rate: 0.05
    0:	learn: 0.5927881	total: 277ms	remaining: 55.2s
    1:	learn: 0.5057848	total: 501ms	remaining: 49.6s
    2:	learn: 0.4375461	total: 743ms	remaining: 48.8s
    3:	learn: 0.3725223	total: 956ms	remaining: 46.8s
    4:	learn: 0.3230253	total: 1.19s	remaining: 46.4s
    5:	learn: 0.2820745	total: 1.41s	remaining: 45.6s
    6:	learn: 0.2482580	total: 1.63s	remaining: 44.9s
    7:	learn: 0.2186577	total: 1.85s	remaining: 44.4s
    8:	learn: 0.1931736	total: 2.08s	remaining: 44.1s
    9:	learn: 0.1737886	total: 2.3s	remaining: 43.7s
    10:	learn: 0.1564852	total: 2.52s	remaining: 43.4s
    11:	learn: 0.1422035	total: 2.78s	remaining: 43.6s
    12:	learn: 0.1302383	total: 3.01s	remaining: 43.3s
    13:	learn: 0.1205424	total: 3.24s	remaining: 43s
    14:	learn: 0.1123842	total: 3.46s	remaining: 42.7s
    15:	learn: 0.1055140	total: 3.68s	remaining: 42.3s
    16:	learn: 0.0997503	total: 3.9s	remaining: 42s
    17:	learn: 0.0949066	total: 4.14s	remaining: 41.8s
    18:	learn: 0.0907293	total: 4.36s	remaining: 41.5s
    19:	learn: 0.0871575	total: 4.59s	remaining: 41.3s
    20:	learn: 0.0836707	total: 4.82s	remaining: 41.1s
    21:	learn: 0.0811479	total: 5.04s	remaining: 40.8s
    22:	learn: 0.0786355	total: 5.25s	remaining: 40.4s
    23:	learn: 0.0767556	total: 5.46s	remaining: 40.1s
    24:	learn: 0.0745242	total: 5.68s	remaining: 39.8s
    25:	learn: 0.0731952	total: 5.89s	remaining: 39.4s
    26:	learn: 0.0720312	total: 6.12s	remaining: 39.2s
    27:	learn: 0.0710135	total: 6.34s	remaining: 39s
    28:	learn: 0.0699668	total: 6.58s	remaining: 38.8s
    29:	learn: 0.0691981	total: 6.81s	remaining: 38.6s
    30:	learn: 0.0685048	total: 7.04s	remaining: 38.4s
    31:	learn: 0.0678997	total: 7.27s	remaining: 38.1s
    32:	learn: 0.0674083	total: 7.48s	remaining: 37.9s
    33:	learn: 0.0669600	total: 7.69s	remaining: 37.6s
    34:	learn: 0.0665514	total: 7.89s	remaining: 37.2s
    35:	learn: 0.0661725	total: 8.11s	remaining: 36.9s
    36:	learn: 0.0658496	total: 8.3s	remaining: 36.6s
    37:	learn: 0.0652886	total: 8.51s	remaining: 36.3s
    38:	learn: 0.0650541	total: 8.74s	remaining: 36.1s
    39:	learn: 0.0648216	total: 8.96s	remaining: 35.8s
    40:	learn: 0.0646312	total: 9.2s	remaining: 35.7s
    41:	learn: 0.0644627	total: 9.43s	remaining: 35.5s
    42:	learn: 0.0642846	total: 9.66s	remaining: 35.3s
    43:	learn: 0.0641258	total: 9.9s	remaining: 35.1s
    44:	learn: 0.0639258	total: 10.1s	remaining: 35s
    45:	learn: 0.0638003	total: 10.4s	remaining: 34.7s
    46:	learn: 0.0636857	total: 10.6s	remaining: 34.5s
    47:	learn: 0.0635972	total: 10.8s	remaining: 34.3s
    48:	learn: 0.0635076	total: 11s	remaining: 34s
    49:	learn: 0.0634277	total: 11.3s	remaining: 33.8s
    50:	learn: 0.0633474	total: 11.5s	remaining: 33.7s
    51:	learn: 0.0632797	total: 11.8s	remaining: 33.5s
    52:	learn: 0.0632058	total: 12s	remaining: 33.2s
    53:	learn: 0.0631457	total: 12.2s	remaining: 33s
    54:	learn: 0.0630054	total: 12.4s	remaining: 32.8s
    55:	learn: 0.0629195	total: 12.7s	remaining: 32.6s
    56:	learn: 0.0628741	total: 12.9s	remaining: 32.4s
    57:	learn: 0.0628254	total: 13.1s	remaining: 32.2s
    58:	learn: 0.0627878	total: 13.4s	remaining: 32s
    59:	learn: 0.0627535	total: 13.6s	remaining: 31.7s
    60:	learn: 0.0626886	total: 13.8s	remaining: 31.5s
    61:	learn: 0.0626641	total: 14.1s	remaining: 31.3s
    62:	learn: 0.0626329	total: 14.3s	remaining: 31.1s
    63:	learn: 0.0626127	total: 14.5s	remaining: 30.9s
    64:	learn: 0.0625884	total: 14.8s	remaining: 30.6s
    65:	learn: 0.0625529	total: 15s	remaining: 30.4s
    66:	learn: 0.0625333	total: 15.2s	remaining: 30.2s
    67:	learn: 0.0625197	total: 15.5s	remaining: 30s
    68:	learn: 0.0624929	total: 15.7s	remaining: 29.8s
    69:	learn: 0.0624722	total: 15.9s	remaining: 29.6s
    70:	learn: 0.0624506	total: 16.2s	remaining: 29.3s
    71:	learn: 0.0624389	total: 16.4s	remaining: 29.2s
    72:	learn: 0.0623952	total: 16.7s	remaining: 29s
    73:	learn: 0.0623756	total: 16.9s	remaining: 28.8s
    74:	learn: 0.0623630	total: 17.1s	remaining: 28.6s
    75:	learn: 0.0622362	total: 17.4s	remaining: 28.4s
    76:	learn: 0.0622186	total: 17.6s	remaining: 28.2s
    77:	learn: 0.0621856	total: 17.9s	remaining: 27.9s
    78:	learn: 0.0621712	total: 18.1s	remaining: 27.7s
    79:	learn: 0.0621601	total: 18.3s	remaining: 27.5s
    80:	learn: 0.0621522	total: 18.5s	remaining: 27.2s
    81:	learn: 0.0621401	total: 18.8s	remaining: 27s
    82:	learn: 0.0621033	total: 19s	remaining: 26.8s
    83:	learn: 0.0620890	total: 19.2s	remaining: 26.6s
    84:	learn: 0.0620750	total: 19.5s	remaining: 26.3s
    85:	learn: 0.0620681	total: 19.7s	remaining: 26.1s
    86:	learn: 0.0620568	total: 19.9s	remaining: 25.9s
    87:	learn: 0.0620476	total: 20.1s	remaining: 25.6s
    88:	learn: 0.0620174	total: 20.4s	remaining: 25.4s
    89:	learn: 0.0619916	total: 20.6s	remaining: 25.2s
    90:	learn: 0.0619737	total: 20.8s	remaining: 24.9s
    91:	learn: 0.0619385	total: 21s	remaining: 24.7s
    92:	learn: 0.0619289	total: 21.3s	remaining: 24.5s
    93:	learn: 0.0619170	total: 21.5s	remaining: 24.2s
    94:	learn: 0.0619121	total: 21.7s	remaining: 24s
    95:	learn: 0.0619039	total: 21.9s	remaining: 23.8s
    96:	learn: 0.0618972	total: 22.2s	remaining: 23.5s
    97:	learn: 0.0618888	total: 22.4s	remaining: 23.3s
    98:	learn: 0.0618805	total: 22.6s	remaining: 23.1s
    99:	learn: 0.0618539	total: 22.9s	remaining: 22.9s
    100:	learn: 0.0618447	total: 23.1s	remaining: 22.6s
    101:	learn: 0.0618367	total: 23.3s	remaining: 22.4s
    102:	learn: 0.0618273	total: 23.5s	remaining: 22.2s
    103:	learn: 0.0618073	total: 23.8s	remaining: 21.9s
    104:	learn: 0.0618000	total: 24s	remaining: 21.7s
    105:	learn: 0.0617935	total: 24.2s	remaining: 21.5s
    106:	learn: 0.0617787	total: 24.4s	remaining: 21.2s
    107:	learn: 0.0617627	total: 24.7s	remaining: 21s
    108:	learn: 0.0617562	total: 24.9s	remaining: 20.8s
    109:	learn: 0.0617487	total: 25.1s	remaining: 20.5s
    110:	learn: 0.0617421	total: 25.3s	remaining: 20.3s
    111:	learn: 0.0617233	total: 25.5s	remaining: 20.1s
    112:	learn: 0.0617164	total: 25.8s	remaining: 19.8s
    113:	learn: 0.0617069	total: 26s	remaining: 19.6s
    114:	learn: 0.0616652	total: 26.2s	remaining: 19.4s
    115:	learn: 0.0616595	total: 26.5s	remaining: 19.2s
    116:	learn: 0.0616262	total: 26.7s	remaining: 18.9s
    117:	learn: 0.0616112	total: 27s	remaining: 18.7s
    118:	learn: 0.0616065	total: 27.2s	remaining: 18.5s
    119:	learn: 0.0615993	total: 27.4s	remaining: 18.3s
    120:	learn: 0.0615959	total: 27.6s	remaining: 18s
    121:	learn: 0.0615898	total: 27.8s	remaining: 17.8s
    122:	learn: 0.0615846	total: 28.1s	remaining: 17.6s
    123:	learn: 0.0615761	total: 28.3s	remaining: 17.3s
    124:	learn: 0.0615710	total: 28.5s	remaining: 17.1s
    125:	learn: 0.0615632	total: 28.8s	remaining: 16.9s
    126:	learn: 0.0615574	total: 29s	remaining: 16.7s
    127:	learn: 0.0615402	total: 29.2s	remaining: 16.4s
    128:	learn: 0.0615316	total: 29.4s	remaining: 16.2s
    129:	learn: 0.0615268	total: 29.6s	remaining: 15.9s
    130:	learn: 0.0615238	total: 29.9s	remaining: 15.7s
    131:	learn: 0.0615189	total: 30.1s	remaining: 15.5s
    132:	learn: 0.0615136	total: 30.3s	remaining: 15.3s
    133:	learn: 0.0615063	total: 30.5s	remaining: 15s
    134:	learn: 0.0614964	total: 30.8s	remaining: 14.8s
    135:	learn: 0.0614913	total: 31s	remaining: 14.6s
    136:	learn: 0.0614851	total: 31.2s	remaining: 14.4s
    137:	learn: 0.0614718	total: 31.4s	remaining: 14.1s
    138:	learn: 0.0614674	total: 31.7s	remaining: 13.9s
    139:	learn: 0.0614641	total: 31.9s	remaining: 13.7s
    140:	learn: 0.0614588	total: 32.1s	remaining: 13.4s
    141:	learn: 0.0614542	total: 32.4s	remaining: 13.2s
    142:	learn: 0.0614474	total: 32.6s	remaining: 13s
    143:	learn: 0.0614438	total: 32.8s	remaining: 12.8s
    144:	learn: 0.0614416	total: 33s	remaining: 12.5s
    145:	learn: 0.0614383	total: 33.3s	remaining: 12.3s
    146:	learn: 0.0614341	total: 33.5s	remaining: 12.1s
    147:	learn: 0.0614302	total: 33.7s	remaining: 11.8s
    148:	learn: 0.0614246	total: 33.9s	remaining: 11.6s
    149:	learn: 0.0614185	total: 34.2s	remaining: 11.4s
    150:	learn: 0.0614098	total: 34.4s	remaining: 11.2s
    151:	learn: 0.0614058	total: 34.6s	remaining: 10.9s
    152:	learn: 0.0614019	total: 34.8s	remaining: 10.7s
    153:	learn: 0.0613957	total: 35s	remaining: 10.5s
    154:	learn: 0.0613879	total: 35.2s	remaining: 10.2s
    155:	learn: 0.0613837	total: 35.5s	remaining: 10s
    156:	learn: 0.0613708	total: 35.7s	remaining: 9.78s
    157:	learn: 0.0613665	total: 36s	remaining: 9.56s
    158:	learn: 0.0613613	total: 36.2s	remaining: 9.33s
    159:	learn: 0.0613581	total: 36.4s	remaining: 9.1s
    160:	learn: 0.0613531	total: 36.6s	remaining: 8.87s
    161:	learn: 0.0613481	total: 36.9s	remaining: 8.65s
    162:	learn: 0.0613342	total: 37.1s	remaining: 8.42s
    163:	learn: 0.0613281	total: 37.3s	remaining: 8.19s
    164:	learn: 0.0613226	total: 37.5s	remaining: 7.96s
    165:	learn: 0.0613178	total: 37.7s	remaining: 7.73s
    166:	learn: 0.0613147	total: 38s	remaining: 7.5s
    167:	learn: 0.0613042	total: 38.2s	remaining: 7.28s
    168:	learn: 0.0612996	total: 38.4s	remaining: 7.05s
    169:	learn: 0.0612965	total: 38.6s	remaining: 6.82s
    170:	learn: 0.0612904	total: 38.9s	remaining: 6.59s
    171:	learn: 0.0612866	total: 39.1s	remaining: 6.36s
    172:	learn: 0.0612814	total: 39.3s	remaining: 6.14s
    173:	learn: 0.0612722	total: 39.5s	remaining: 5.91s
    174:	learn: 0.0612638	total: 39.8s	remaining: 5.68s
    175:	learn: 0.0612614	total: 40s	remaining: 5.46s
    176:	learn: 0.0612601	total: 40.2s	remaining: 5.23s
    177:	learn: 0.0612563	total: 40.5s	remaining: 5s
    178:	learn: 0.0612466	total: 40.7s	remaining: 4.78s
    179:	learn: 0.0612398	total: 40.9s	remaining: 4.55s
    180:	learn: 0.0612362	total: 41.2s	remaining: 4.32s
    181:	learn: 0.0612224	total: 41.4s	remaining: 4.09s
    182:	learn: 0.0612186	total: 41.6s	remaining: 3.87s
    183:	learn: 0.0612158	total: 41.8s	remaining: 3.64s
    184:	learn: 0.0612116	total: 42.1s	remaining: 3.41s
    185:	learn: 0.0612046	total: 42.3s	remaining: 3.18s
    186:	learn: 0.0611968	total: 42.5s	remaining: 2.96s
    187:	learn: 0.0611941	total: 42.8s	remaining: 2.73s
    188:	learn: 0.0611835	total: 43s	remaining: 2.5s
    189:	learn: 0.0611790	total: 43.2s	remaining: 2.27s
    190:	learn: 0.0611737	total: 43.4s	remaining: 2.04s
    191:	learn: 0.0611711	total: 43.6s	remaining: 1.82s
    192:	learn: 0.0611694	total: 43.9s	remaining: 1.59s
    193:	learn: 0.0611658	total: 44.1s	remaining: 1.36s
    194:	learn: 0.0611616	total: 44.3s	remaining: 1.14s
    195:	learn: 0.0611591	total: 44.5s	remaining: 909ms
    196:	learn: 0.0611556	total: 44.8s	remaining: 682ms
    197:	learn: 0.0611518	total: 45s	remaining: 455ms
    198:	learn: 0.0611414	total: 45.2s	remaining: 227ms
    199:	learn: 0.0611338	total: 45.4s	remaining: 0us
    Trained model nº 11/27. Iterations: 200Depth: 6Learning rate: 0.05
    0:	learn: 0.5927881	total: 204ms	remaining: 1m
    1:	learn: 0.5057848	total: 413ms	remaining: 1m 1s
    2:	learn: 0.4375461	total: 647ms	remaining: 1m 4s
    3:	learn: 0.3725223	total: 863ms	remaining: 1m 3s
    4:	learn: 0.3230253	total: 1.09s	remaining: 1m 4s
    5:	learn: 0.2820745	total: 1.32s	remaining: 1m 4s
    6:	learn: 0.2482580	total: 1.54s	remaining: 1m 4s
    7:	learn: 0.2186577	total: 1.76s	remaining: 1m 4s
    8:	learn: 0.1931736	total: 1.98s	remaining: 1m 4s
    9:	learn: 0.1737886	total: 2.21s	remaining: 1m 4s
    10:	learn: 0.1564852	total: 2.43s	remaining: 1m 3s
    11:	learn: 0.1422035	total: 2.66s	remaining: 1m 3s
    12:	learn: 0.1302383	total: 2.87s	remaining: 1m 3s
    13:	learn: 0.1205424	total: 3.1s	remaining: 1m 3s
    14:	learn: 0.1123842	total: 3.33s	remaining: 1m 3s
    15:	learn: 0.1055140	total: 3.55s	remaining: 1m 2s
    16:	learn: 0.0997503	total: 3.77s	remaining: 1m 2s
    17:	learn: 0.0949066	total: 4s	remaining: 1m 2s
    18:	learn: 0.0907293	total: 4.21s	remaining: 1m 2s
    19:	learn: 0.0871575	total: 4.44s	remaining: 1m 2s
    20:	learn: 0.0836707	total: 4.66s	remaining: 1m 1s
    21:	learn: 0.0811479	total: 4.88s	remaining: 1m 1s
    22:	learn: 0.0786355	total: 5.08s	remaining: 1m 1s
    23:	learn: 0.0767556	total: 5.3s	remaining: 1m 1s
    24:	learn: 0.0745242	total: 5.51s	remaining: 1m
    25:	learn: 0.0731952	total: 5.72s	remaining: 1m
    26:	learn: 0.0720312	total: 5.94s	remaining: 1m
    27:	learn: 0.0710135	total: 6.16s	remaining: 59.8s
    28:	learn: 0.0699668	total: 6.37s	remaining: 59.5s
    29:	learn: 0.0691981	total: 6.59s	remaining: 59.3s
    30:	learn: 0.0685048	total: 6.8s	remaining: 59s
    31:	learn: 0.0678997	total: 7.02s	remaining: 58.8s
    32:	learn: 0.0674083	total: 7.24s	remaining: 58.6s
    33:	learn: 0.0669600	total: 7.44s	remaining: 58.2s
    34:	learn: 0.0665514	total: 7.64s	remaining: 57.8s
    35:	learn: 0.0661725	total: 7.86s	remaining: 57.6s
    36:	learn: 0.0658496	total: 8.05s	remaining: 57.2s
    37:	learn: 0.0652886	total: 8.26s	remaining: 56.9s
    38:	learn: 0.0650541	total: 8.47s	remaining: 56.7s
    39:	learn: 0.0648216	total: 8.69s	remaining: 56.5s
    40:	learn: 0.0646312	total: 8.92s	remaining: 56.4s
    41:	learn: 0.0644627	total: 9.16s	remaining: 56.2s
    42:	learn: 0.0642846	total: 9.37s	remaining: 56s
    43:	learn: 0.0641258	total: 9.59s	remaining: 55.8s
    44:	learn: 0.0639258	total: 9.8s	remaining: 55.6s
    45:	learn: 0.0638003	total: 10s	remaining: 55.3s
    46:	learn: 0.0636857	total: 10.2s	remaining: 55.1s
    47:	learn: 0.0635972	total: 10.5s	remaining: 54.9s
    48:	learn: 0.0635076	total: 10.7s	remaining: 54.7s
    49:	learn: 0.0634277	total: 10.9s	remaining: 54.5s
    50:	learn: 0.0633474	total: 11.1s	remaining: 54.4s
    51:	learn: 0.0632797	total: 11.3s	remaining: 54.1s
    52:	learn: 0.0632058	total: 11.6s	remaining: 53.9s
    53:	learn: 0.0631457	total: 11.8s	remaining: 53.7s
    54:	learn: 0.0630054	total: 12s	remaining: 53.5s
    55:	learn: 0.0629195	total: 12.2s	remaining: 53.3s
    56:	learn: 0.0628741	total: 12.5s	remaining: 53.2s
    57:	learn: 0.0628254	total: 12.7s	remaining: 52.9s
    58:	learn: 0.0627878	total: 12.9s	remaining: 52.7s
    59:	learn: 0.0627535	total: 13.1s	remaining: 52.5s
    60:	learn: 0.0626886	total: 13.4s	remaining: 52.4s
    61:	learn: 0.0626641	total: 13.6s	remaining: 52.2s
    62:	learn: 0.0626329	total: 13.8s	remaining: 51.9s
    63:	learn: 0.0626127	total: 14s	remaining: 51.7s
    64:	learn: 0.0625884	total: 14.3s	remaining: 51.5s
    65:	learn: 0.0625529	total: 14.5s	remaining: 51.3s
    66:	learn: 0.0625333	total: 14.7s	remaining: 51.1s
    67:	learn: 0.0625197	total: 14.9s	remaining: 50.9s
    68:	learn: 0.0624929	total: 15.1s	remaining: 50.6s
    69:	learn: 0.0624722	total: 15.3s	remaining: 50.4s
    70:	learn: 0.0624506	total: 15.6s	remaining: 50.2s
    71:	learn: 0.0624389	total: 15.8s	remaining: 50.1s
    72:	learn: 0.0623952	total: 16s	remaining: 49.9s
    73:	learn: 0.0623756	total: 16.3s	remaining: 49.7s
    74:	learn: 0.0623630	total: 16.5s	remaining: 49.5s
    75:	learn: 0.0622362	total: 16.7s	remaining: 49.3s
    76:	learn: 0.0622186	total: 17s	remaining: 49.1s
    77:	learn: 0.0621856	total: 17.2s	remaining: 49s
    78:	learn: 0.0621712	total: 17.4s	remaining: 48.7s
    79:	learn: 0.0621601	total: 17.7s	remaining: 48.5s
    80:	learn: 0.0621522	total: 17.9s	remaining: 48.4s
    81:	learn: 0.0621401	total: 18.1s	remaining: 48.2s
    82:	learn: 0.0621033	total: 18.4s	remaining: 48s
    83:	learn: 0.0620890	total: 18.6s	remaining: 47.9s
    84:	learn: 0.0620750	total: 18.9s	remaining: 47.8s
    85:	learn: 0.0620681	total: 19.1s	remaining: 47.6s
    86:	learn: 0.0620568	total: 19.4s	remaining: 47.4s
    87:	learn: 0.0620476	total: 19.6s	remaining: 47.2s
    88:	learn: 0.0620174	total: 19.8s	remaining: 47s
    89:	learn: 0.0619916	total: 20.1s	remaining: 46.8s
    90:	learn: 0.0619737	total: 20.3s	remaining: 46.6s
    91:	learn: 0.0619385	total: 20.5s	remaining: 46.4s
    92:	learn: 0.0619289	total: 20.7s	remaining: 46.2s
    93:	learn: 0.0619170	total: 21s	remaining: 46s
    94:	learn: 0.0619121	total: 21.2s	remaining: 45.8s
    95:	learn: 0.0619039	total: 21.4s	remaining: 45.6s
    96:	learn: 0.0618972	total: 21.7s	remaining: 45.4s
    97:	learn: 0.0618888	total: 21.9s	remaining: 45.1s
    98:	learn: 0.0618805	total: 22.1s	remaining: 44.9s
    99:	learn: 0.0618539	total: 22.4s	remaining: 44.7s
    100:	learn: 0.0618447	total: 22.6s	remaining: 44.5s
    101:	learn: 0.0618367	total: 22.8s	remaining: 44.3s
    102:	learn: 0.0618273	total: 23.1s	remaining: 44.2s
    103:	learn: 0.0618073	total: 23.3s	remaining: 44s
    104:	learn: 0.0618000	total: 23.6s	remaining: 43.8s
    105:	learn: 0.0617935	total: 23.8s	remaining: 43.5s
    106:	learn: 0.0617787	total: 24s	remaining: 43.3s
    107:	learn: 0.0617627	total: 24.3s	remaining: 43.2s
    108:	learn: 0.0617562	total: 24.5s	remaining: 43s
    109:	learn: 0.0617487	total: 24.8s	remaining: 42.8s
    110:	learn: 0.0617421	total: 25.1s	remaining: 42.7s
    111:	learn: 0.0617233	total: 25.4s	remaining: 42.6s
    112:	learn: 0.0617164	total: 25.6s	remaining: 42.4s
    113:	learn: 0.0617069	total: 26s	remaining: 42.5s
    114:	learn: 0.0616652	total: 26.3s	remaining: 42.4s
    115:	learn: 0.0616595	total: 26.6s	remaining: 42.2s
    116:	learn: 0.0616262	total: 26.8s	remaining: 42s
    117:	learn: 0.0616112	total: 27.1s	remaining: 41.8s
    118:	learn: 0.0616065	total: 27.3s	remaining: 41.6s
    119:	learn: 0.0615993	total: 27.6s	remaining: 41.4s
    120:	learn: 0.0615959	total: 27.9s	remaining: 41.2s
    121:	learn: 0.0615898	total: 28.1s	remaining: 41s
    122:	learn: 0.0615846	total: 28.4s	remaining: 40.8s
    123:	learn: 0.0615761	total: 28.7s	remaining: 40.7s
    124:	learn: 0.0615710	total: 29s	remaining: 40.6s
    125:	learn: 0.0615632	total: 29.3s	remaining: 40.5s
    126:	learn: 0.0615574	total: 29.5s	remaining: 40.2s
    127:	learn: 0.0615402	total: 29.8s	remaining: 40s
    128:	learn: 0.0615316	total: 30s	remaining: 39.8s
    129:	learn: 0.0615268	total: 30.2s	remaining: 39.6s
    130:	learn: 0.0615238	total: 30.5s	remaining: 39.4s
    131:	learn: 0.0615189	total: 30.7s	remaining: 39.1s
    132:	learn: 0.0615136	total: 30.9s	remaining: 38.9s
    133:	learn: 0.0615063	total: 31.2s	remaining: 38.6s
    134:	learn: 0.0614964	total: 31.4s	remaining: 38.4s
    135:	learn: 0.0614913	total: 31.6s	remaining: 38.1s
    136:	learn: 0.0614851	total: 31.9s	remaining: 37.9s
    137:	learn: 0.0614718	total: 32.1s	remaining: 37.7s
    138:	learn: 0.0614674	total: 32.3s	remaining: 37.4s
    139:	learn: 0.0614641	total: 32.5s	remaining: 37.2s
    140:	learn: 0.0614588	total: 32.8s	remaining: 37s
    141:	learn: 0.0614542	total: 33s	remaining: 36.7s
    142:	learn: 0.0614474	total: 33.3s	remaining: 36.5s
    143:	learn: 0.0614438	total: 33.5s	remaining: 36.3s
    144:	learn: 0.0614416	total: 33.8s	remaining: 36.1s
    145:	learn: 0.0614383	total: 34s	remaining: 35.9s
    146:	learn: 0.0614341	total: 34.2s	remaining: 35.6s
    147:	learn: 0.0614302	total: 34.5s	remaining: 35.4s
    148:	learn: 0.0614246	total: 34.8s	remaining: 35.2s
    149:	learn: 0.0614185	total: 35s	remaining: 35s
    150:	learn: 0.0614098	total: 35.2s	remaining: 34.8s
    151:	learn: 0.0614058	total: 35.5s	remaining: 34.6s
    152:	learn: 0.0614019	total: 35.8s	remaining: 34.4s
    153:	learn: 0.0613957	total: 36.1s	remaining: 34.2s
    154:	learn: 0.0613879	total: 36.3s	remaining: 34s
    155:	learn: 0.0613837	total: 36.6s	remaining: 33.8s
    156:	learn: 0.0613708	total: 36.9s	remaining: 33.6s
    157:	learn: 0.0613665	total: 37.2s	remaining: 33.4s
    158:	learn: 0.0613613	total: 37.4s	remaining: 33.2s
    159:	learn: 0.0613581	total: 37.7s	remaining: 33s
    160:	learn: 0.0613531	total: 38s	remaining: 32.8s
    161:	learn: 0.0613481	total: 38.2s	remaining: 32.5s
    162:	learn: 0.0613342	total: 38.5s	remaining: 32.3s
    163:	learn: 0.0613281	total: 38.7s	remaining: 32.1s
    164:	learn: 0.0613226	total: 39s	remaining: 31.9s
    165:	learn: 0.0613178	total: 39.2s	remaining: 31.7s
    166:	learn: 0.0613147	total: 39.5s	remaining: 31.4s
    167:	learn: 0.0613042	total: 39.7s	remaining: 31.2s
    168:	learn: 0.0612996	total: 40s	remaining: 31s
    169:	learn: 0.0612965	total: 40.2s	remaining: 30.8s
    170:	learn: 0.0612904	total: 40.5s	remaining: 30.5s
    171:	learn: 0.0612866	total: 40.7s	remaining: 30.3s
    172:	learn: 0.0612814	total: 41s	remaining: 30.1s
    173:	learn: 0.0612722	total: 41.3s	remaining: 29.9s
    174:	learn: 0.0612638	total: 41.5s	remaining: 29.6s
    175:	learn: 0.0612614	total: 41.7s	remaining: 29.4s
    176:	learn: 0.0612601	total: 42s	remaining: 29.2s
    177:	learn: 0.0612563	total: 42.3s	remaining: 29s
    178:	learn: 0.0612466	total: 42.5s	remaining: 28.7s
    179:	learn: 0.0612398	total: 42.8s	remaining: 28.5s
    180:	learn: 0.0612362	total: 43s	remaining: 28.3s
    181:	learn: 0.0612224	total: 43.3s	remaining: 28.1s
    182:	learn: 0.0612186	total: 43.5s	remaining: 27.8s
    183:	learn: 0.0612158	total: 43.8s	remaining: 27.6s
    184:	learn: 0.0612116	total: 44s	remaining: 27.4s
    185:	learn: 0.0612046	total: 44.2s	remaining: 27.1s
    186:	learn: 0.0611968	total: 44.5s	remaining: 26.9s
    187:	learn: 0.0611941	total: 44.7s	remaining: 26.6s
    188:	learn: 0.0611835	total: 45s	remaining: 26.4s
    189:	learn: 0.0611790	total: 45.2s	remaining: 26.2s
    190:	learn: 0.0611737	total: 45.4s	remaining: 25.9s
    191:	learn: 0.0611711	total: 45.7s	remaining: 25.7s
    192:	learn: 0.0611694	total: 46s	remaining: 25.5s
    193:	learn: 0.0611658	total: 46.2s	remaining: 25.3s
    194:	learn: 0.0611616	total: 46.5s	remaining: 25s
    195:	learn: 0.0611591	total: 46.8s	remaining: 24.8s
    196:	learn: 0.0611556	total: 47s	remaining: 24.6s
    197:	learn: 0.0611518	total: 47.3s	remaining: 24.4s
    198:	learn: 0.0611414	total: 47.5s	remaining: 24.1s
    199:	learn: 0.0611338	total: 47.8s	remaining: 23.9s
    200:	learn: 0.0611056	total: 48s	remaining: 23.7s
    201:	learn: 0.0611038	total: 48.3s	remaining: 23.4s
    202:	learn: 0.0610984	total: 48.5s	remaining: 23.2s
    203:	learn: 0.0610943	total: 48.8s	remaining: 23s
    204:	learn: 0.0610900	total: 49s	remaining: 22.7s
    205:	learn: 0.0610867	total: 49.3s	remaining: 22.5s
    206:	learn: 0.0610809	total: 49.5s	remaining: 22.3s
    207:	learn: 0.0610792	total: 49.8s	remaining: 22s
    208:	learn: 0.0610772	total: 50s	remaining: 21.8s
    209:	learn: 0.0610739	total: 50.2s	remaining: 21.5s
    210:	learn: 0.0610713	total: 50.5s	remaining: 21.3s
    211:	learn: 0.0610697	total: 50.7s	remaining: 21.1s
    212:	learn: 0.0610661	total: 51s	remaining: 20.8s
    213:	learn: 0.0610625	total: 51.3s	remaining: 20.6s
    214:	learn: 0.0610412	total: 51.5s	remaining: 20.4s
    215:	learn: 0.0610393	total: 51.7s	remaining: 20.1s
    216:	learn: 0.0610380	total: 52s	remaining: 19.9s
    217:	learn: 0.0610348	total: 52.2s	remaining: 19.6s
    218:	learn: 0.0610317	total: 52.4s	remaining: 19.4s
    219:	learn: 0.0610260	total: 52.7s	remaining: 19.2s
    220:	learn: 0.0610190	total: 52.9s	remaining: 18.9s
    221:	learn: 0.0610124	total: 53.2s	remaining: 18.7s
    222:	learn: 0.0610092	total: 53.4s	remaining: 18.4s
    223:	learn: 0.0610073	total: 53.7s	remaining: 18.2s
    224:	learn: 0.0610046	total: 53.9s	remaining: 18s
    225:	learn: 0.0610014	total: 54.1s	remaining: 17.7s
    226:	learn: 0.0610000	total: 54.4s	remaining: 17.5s
    227:	learn: 0.0609936	total: 54.6s	remaining: 17.2s
    228:	learn: 0.0609923	total: 54.9s	remaining: 17s
    229:	learn: 0.0609887	total: 55.1s	remaining: 16.8s
    230:	learn: 0.0609854	total: 55.4s	remaining: 16.5s
    231:	learn: 0.0609783	total: 55.6s	remaining: 16.3s
    232:	learn: 0.0609752	total: 55.8s	remaining: 16.1s
    233:	learn: 0.0609731	total: 56.1s	remaining: 15.8s
    234:	learn: 0.0609703	total: 56.3s	remaining: 15.6s
    235:	learn: 0.0609695	total: 56.5s	remaining: 15.3s
    236:	learn: 0.0609645	total: 56.8s	remaining: 15.1s
    237:	learn: 0.0609601	total: 57s	remaining: 14.8s
    238:	learn: 0.0609579	total: 57.2s	remaining: 14.6s
    239:	learn: 0.0609555	total: 57.5s	remaining: 14.4s
    240:	learn: 0.0609523	total: 57.8s	remaining: 14.1s
    241:	learn: 0.0609500	total: 58s	remaining: 13.9s
    242:	learn: 0.0609464	total: 58.2s	remaining: 13.7s
    243:	learn: 0.0609429	total: 58.5s	remaining: 13.4s
    244:	learn: 0.0609420	total: 58.7s	remaining: 13.2s
    245:	learn: 0.0609403	total: 58.9s	remaining: 12.9s
    246:	learn: 0.0609376	total: 59.1s	remaining: 12.7s
    247:	learn: 0.0609375	total: 59.3s	remaining: 12.4s
    248:	learn: 0.0609349	total: 59.6s	remaining: 12.2s
    249:	learn: 0.0609320	total: 59.8s	remaining: 12s
    250:	learn: 0.0609308	total: 1m	remaining: 11.7s
    251:	learn: 0.0609274	total: 1m	remaining: 11.5s
    252:	learn: 0.0609236	total: 1m	remaining: 11.2s
    253:	learn: 0.0609216	total: 1m	remaining: 11s
    254:	learn: 0.0609175	total: 1m	remaining: 10.8s
    255:	learn: 0.0609165	total: 1m 1s	remaining: 10.5s
    256:	learn: 0.0609150	total: 1m 1s	remaining: 10.3s
    257:	learn: 0.0609121	total: 1m 1s	remaining: 10s
    258:	learn: 0.0609103	total: 1m 1s	remaining: 9.8s
    259:	learn: 0.0609082	total: 1m 2s	remaining: 9.56s
    260:	learn: 0.0609062	total: 1m 2s	remaining: 9.32s
    261:	learn: 0.0609031	total: 1m 2s	remaining: 9.08s
    262:	learn: 0.0608998	total: 1m 2s	remaining: 8.85s
    263:	learn: 0.0608966	total: 1m 3s	remaining: 8.61s
    264:	learn: 0.0608951	total: 1m 3s	remaining: 8.37s
    265:	learn: 0.0608924	total: 1m 3s	remaining: 8.13s
    266:	learn: 0.0608900	total: 1m 3s	remaining: 7.89s
    267:	learn: 0.0608865	total: 1m 4s	remaining: 7.65s
    268:	learn: 0.0608824	total: 1m 4s	remaining: 7.41s
    269:	learn: 0.0608792	total: 1m 4s	remaining: 7.17s
    270:	learn: 0.0608779	total: 1m 4s	remaining: 6.93s
    271:	learn: 0.0608768	total: 1m 5s	remaining: 6.7s
    272:	learn: 0.0608752	total: 1m 5s	remaining: 6.46s
    273:	learn: 0.0608726	total: 1m 5s	remaining: 6.22s
    274:	learn: 0.0608684	total: 1m 5s	remaining: 5.98s
    275:	learn: 0.0608636	total: 1m 5s	remaining: 5.74s
    276:	learn: 0.0608606	total: 1m 6s	remaining: 5.5s
    277:	learn: 0.0608591	total: 1m 6s	remaining: 5.26s
    278:	learn: 0.0608571	total: 1m 6s	remaining: 5.02s
    279:	learn: 0.0608534	total: 1m 6s	remaining: 4.78s
    280:	learn: 0.0608511	total: 1m 7s	remaining: 4.54s
    281:	learn: 0.0608488	total: 1m 7s	remaining: 4.3s
    282:	learn: 0.0608451	total: 1m 7s	remaining: 4.06s
    283:	learn: 0.0608430	total: 1m 7s	remaining: 3.82s
    284:	learn: 0.0608412	total: 1m 8s	remaining: 3.58s
    285:	learn: 0.0608398	total: 1m 8s	remaining: 3.35s
    286:	learn: 0.0608380	total: 1m 8s	remaining: 3.11s
    287:	learn: 0.0608354	total: 1m 8s	remaining: 2.87s
    288:	learn: 0.0608334	total: 1m 9s	remaining: 2.63s
    289:	learn: 0.0608316	total: 1m 9s	remaining: 2.39s
    290:	learn: 0.0608304	total: 1m 9s	remaining: 2.15s
    291:	learn: 0.0608254	total: 1m 9s	remaining: 1.91s
    292:	learn: 0.0608239	total: 1m 9s	remaining: 1.67s
    293:	learn: 0.0608221	total: 1m 10s	remaining: 1.43s
    294:	learn: 0.0608204	total: 1m 10s	remaining: 1.19s
    295:	learn: 0.0608192	total: 1m 10s	remaining: 955ms
    296:	learn: 0.0608162	total: 1m 10s	remaining: 716ms
    297:	learn: 0.0608135	total: 1m 11s	remaining: 477ms
    298:	learn: 0.0608116	total: 1m 11s	remaining: 239ms
    299:	learn: 0.0608017	total: 1m 11s	remaining: 0us
    Trained model nº 12/27. Iterations: 300Depth: 6Learning rate: 0.05
    0:	learn: 0.5909020	total: 194ms	remaining: 19.2s
    1:	learn: 0.4962678	total: 373ms	remaining: 18.3s
    2:	learn: 0.4228452	total: 552ms	remaining: 17.9s
    3:	learn: 0.3636206	total: 726ms	remaining: 17.4s
    4:	learn: 0.3110298	total: 900ms	remaining: 17.1s
    5:	learn: 0.2651585	total: 1.07s	remaining: 16.7s
    6:	learn: 0.2332345	total: 1.24s	remaining: 16.5s
    7:	learn: 0.2052582	total: 1.41s	remaining: 16.2s
    8:	learn: 0.1825357	total: 1.58s	remaining: 16s
    9:	learn: 0.1637544	total: 1.74s	remaining: 15.7s
    10:	learn: 0.1483779	total: 1.91s	remaining: 15.5s
    11:	learn: 0.1349470	total: 2.09s	remaining: 15.3s
    12:	learn: 0.1245414	total: 2.25s	remaining: 15.1s
    13:	learn: 0.1152664	total: 2.42s	remaining: 14.9s
    14:	learn: 0.1075161	total: 2.62s	remaining: 14.8s
    15:	learn: 0.1012135	total: 2.82s	remaining: 14.8s
    16:	learn: 0.0961405	total: 3s	remaining: 14.7s
    17:	learn: 0.0916134	total: 3.18s	remaining: 14.5s
    18:	learn: 0.0877139	total: 3.38s	remaining: 14.4s
    19:	learn: 0.0845487	total: 3.56s	remaining: 14.2s
    20:	learn: 0.0817600	total: 3.73s	remaining: 14s
    21:	learn: 0.0793113	total: 3.9s	remaining: 13.8s
    22:	learn: 0.0773599	total: 4.08s	remaining: 13.7s
    23:	learn: 0.0754647	total: 4.25s	remaining: 13.5s
    24:	learn: 0.0739166	total: 4.45s	remaining: 13.4s
    25:	learn: 0.0726508	total: 4.63s	remaining: 13.2s
    26:	learn: 0.0714758	total: 4.82s	remaining: 13s
    27:	learn: 0.0702857	total: 5s	remaining: 12.9s
    28:	learn: 0.0693820	total: 5.18s	remaining: 12.7s
    29:	learn: 0.0685435	total: 5.35s	remaining: 12.5s
    30:	learn: 0.0678635	total: 5.53s	remaining: 12.3s
    31:	learn: 0.0672717	total: 5.7s	remaining: 12.1s
    32:	learn: 0.0666830	total: 5.87s	remaining: 11.9s
    33:	learn: 0.0662217	total: 6.04s	remaining: 11.7s
    34:	learn: 0.0658094	total: 6.22s	remaining: 11.5s
    35:	learn: 0.0653945	total: 6.39s	remaining: 11.4s
    36:	learn: 0.0650287	total: 6.58s	remaining: 11.2s
    37:	learn: 0.0647056	total: 6.75s	remaining: 11s
    38:	learn: 0.0644437	total: 6.94s	remaining: 10.9s
    39:	learn: 0.0641786	total: 7.12s	remaining: 10.7s
    40:	learn: 0.0639285	total: 7.3s	remaining: 10.5s
    41:	learn: 0.0637035	total: 7.48s	remaining: 10.3s
    42:	learn: 0.0635115	total: 7.66s	remaining: 10.1s
    43:	learn: 0.0633432	total: 7.83s	remaining: 9.97s
    44:	learn: 0.0632172	total: 8.03s	remaining: 9.81s
    45:	learn: 0.0630668	total: 8.2s	remaining: 9.63s
    46:	learn: 0.0629496	total: 8.39s	remaining: 9.46s
    47:	learn: 0.0628373	total: 8.58s	remaining: 9.3s
    48:	learn: 0.0627428	total: 8.76s	remaining: 9.11s
    49:	learn: 0.0626601	total: 8.94s	remaining: 8.94s
    50:	learn: 0.0625637	total: 9.11s	remaining: 8.75s
    51:	learn: 0.0624802	total: 9.29s	remaining: 8.58s
    52:	learn: 0.0623970	total: 9.47s	remaining: 8.39s
    53:	learn: 0.0623345	total: 9.65s	remaining: 8.22s
    54:	learn: 0.0622660	total: 9.82s	remaining: 8.04s
    55:	learn: 0.0622095	total: 10s	remaining: 7.86s
    56:	learn: 0.0621441	total: 10.2s	remaining: 7.68s
    57:	learn: 0.0621098	total: 10.4s	remaining: 7.51s
    58:	learn: 0.0620597	total: 10.5s	remaining: 7.33s
    59:	learn: 0.0620226	total: 10.7s	remaining: 7.15s
    60:	learn: 0.0619742	total: 10.9s	remaining: 6.97s
    61:	learn: 0.0619352	total: 11.1s	remaining: 6.79s
    62:	learn: 0.0619012	total: 11.3s	remaining: 6.61s
    63:	learn: 0.0618665	total: 11.4s	remaining: 6.44s
    64:	learn: 0.0618453	total: 11.6s	remaining: 6.25s
    65:	learn: 0.0618049	total: 11.8s	remaining: 6.08s
    66:	learn: 0.0617820	total: 12s	remaining: 5.9s
    67:	learn: 0.0617492	total: 12.2s	remaining: 5.72s
    68:	learn: 0.0617278	total: 12.3s	remaining: 5.54s
    69:	learn: 0.0617099	total: 12.5s	remaining: 5.37s
    70:	learn: 0.0616909	total: 12.7s	remaining: 5.18s
    71:	learn: 0.0616671	total: 12.9s	remaining: 5.01s
    72:	learn: 0.0616430	total: 13.1s	remaining: 4.83s
    73:	learn: 0.0616193	total: 13.2s	remaining: 4.65s
    74:	learn: 0.0615989	total: 13.4s	remaining: 4.48s
    75:	learn: 0.0615840	total: 13.6s	remaining: 4.3s
    76:	learn: 0.0615589	total: 13.8s	remaining: 4.12s
    77:	learn: 0.0615421	total: 14s	remaining: 3.94s
    78:	learn: 0.0615226	total: 14.2s	remaining: 3.76s
    79:	learn: 0.0615016	total: 14.3s	remaining: 3.58s
    80:	learn: 0.0614843	total: 14.5s	remaining: 3.41s
    81:	learn: 0.0614697	total: 14.7s	remaining: 3.23s
    82:	learn: 0.0614554	total: 14.9s	remaining: 3.05s
    83:	learn: 0.0614425	total: 15.1s	remaining: 2.87s
    84:	learn: 0.0614329	total: 15.2s	remaining: 2.69s
    85:	learn: 0.0614156	total: 15.4s	remaining: 2.51s
    86:	learn: 0.0614016	total: 15.6s	remaining: 2.33s
    87:	learn: 0.0613866	total: 15.8s	remaining: 2.15s
    88:	learn: 0.0613759	total: 15.9s	remaining: 1.97s
    89:	learn: 0.0613613	total: 16.1s	remaining: 1.79s
    90:	learn: 0.0613472	total: 16.3s	remaining: 1.61s
    91:	learn: 0.0613375	total: 16.5s	remaining: 1.43s
    92:	learn: 0.0613182	total: 16.6s	remaining: 1.25s
    93:	learn: 0.0613041	total: 16.8s	remaining: 1.07s
    94:	learn: 0.0612886	total: 17s	remaining: 895ms
    95:	learn: 0.0612750	total: 17.2s	remaining: 717ms
    96:	learn: 0.0612601	total: 17.4s	remaining: 537ms
    97:	learn: 0.0612524	total: 17.6s	remaining: 358ms
    98:	learn: 0.0612424	total: 17.7s	remaining: 179ms
    99:	learn: 0.0612326	total: 17.9s	remaining: 0us
    Trained model nº 13/27. Iterations: 100Depth: 8Learning rate: 0.05
    0:	learn: 0.5926721	total: 287ms	remaining: 57.2s
    1:	learn: 0.5060639	total: 574ms	remaining: 56.9s
    2:	learn: 0.4296100	total: 857ms	remaining: 56.3s
    3:	learn: 0.3650039	total: 1.15s	remaining: 56.3s
    4:	learn: 0.3187505	total: 1.38s	remaining: 53.7s
    5:	learn: 0.2727445	total: 1.64s	remaining: 53.1s
    6:	learn: 0.2361745	total: 1.92s	remaining: 52.9s
    7:	learn: 0.2083013	total: 2.19s	remaining: 52.6s
    8:	learn: 0.1843583	total: 2.47s	remaining: 52.4s
    9:	learn: 0.1652011	total: 2.75s	remaining: 52.3s
    10:	learn: 0.1500258	total: 3.03s	remaining: 52s
    11:	learn: 0.1374133	total: 3.3s	remaining: 51.6s
    12:	learn: 0.1260781	total: 3.57s	remaining: 51.4s
    13:	learn: 0.1168321	total: 3.82s	remaining: 50.8s
    14:	learn: 0.1092126	total: 4.08s	remaining: 50.3s
    15:	learn: 0.1027471	total: 4.36s	remaining: 50.1s
    16:	learn: 0.0971917	total: 4.66s	remaining: 50.1s
    17:	learn: 0.0926318	total: 4.95s	remaining: 50s
    18:	learn: 0.0887233	total: 5.23s	remaining: 49.8s
    19:	learn: 0.0852969	total: 5.5s	remaining: 49.5s
    20:	learn: 0.0824125	total: 5.81s	remaining: 49.5s
    21:	learn: 0.0799004	total: 6.1s	remaining: 49.4s
    22:	learn: 0.0780581	total: 6.41s	remaining: 49.3s
    23:	learn: 0.0762085	total: 6.71s	remaining: 49.2s
    24:	learn: 0.0746578	total: 7s	remaining: 49s
    25:	learn: 0.0731872	total: 7.29s	remaining: 48.8s
    26:	learn: 0.0719326	total: 7.59s	remaining: 48.6s
    27:	learn: 0.0708713	total: 7.87s	remaining: 48.4s
    28:	learn: 0.0699483	total: 8.16s	remaining: 48.1s
    29:	learn: 0.0690709	total: 8.43s	remaining: 47.8s
    30:	learn: 0.0683317	total: 8.72s	remaining: 47.5s
    31:	learn: 0.0676993	total: 9s	remaining: 47.2s
    32:	learn: 0.0671278	total: 9.28s	remaining: 47s
    33:	learn: 0.0666199	total: 9.57s	remaining: 46.7s
    34:	learn: 0.0661547	total: 9.85s	remaining: 46.4s
    35:	learn: 0.0657422	total: 10.1s	remaining: 46s
    36:	learn: 0.0653662	total: 10.4s	remaining: 45.7s
    37:	learn: 0.0650357	total: 10.6s	remaining: 45.3s
    38:	learn: 0.0647312	total: 10.9s	remaining: 45s
    39:	learn: 0.0645109	total: 11.2s	remaining: 44.8s
    40:	learn: 0.0642643	total: 11.5s	remaining: 44.5s
    41:	learn: 0.0640424	total: 11.8s	remaining: 44.2s
    42:	learn: 0.0638480	total: 12s	remaining: 43.9s
    43:	learn: 0.0636896	total: 12.3s	remaining: 43.7s
    44:	learn: 0.0635276	total: 12.6s	remaining: 43.4s
    45:	learn: 0.0633734	total: 12.9s	remaining: 43.2s
    46:	learn: 0.0632464	total: 13.2s	remaining: 42.9s
    47:	learn: 0.0631228	total: 13.4s	remaining: 42.5s
    48:	learn: 0.0630190	total: 13.7s	remaining: 42.2s
    49:	learn: 0.0629263	total: 14s	remaining: 41.9s
    50:	learn: 0.0628393	total: 14.2s	remaining: 41.6s
    51:	learn: 0.0627514	total: 14.5s	remaining: 41.3s
    52:	learn: 0.0626828	total: 14.8s	remaining: 41s
    53:	learn: 0.0625193	total: 15.1s	remaining: 40.7s
    54:	learn: 0.0624599	total: 15.3s	remaining: 40.4s
    55:	learn: 0.0623984	total: 15.6s	remaining: 40.1s
    56:	learn: 0.0623360	total: 15.9s	remaining: 39.9s
    57:	learn: 0.0622975	total: 16.2s	remaining: 39.5s
    58:	learn: 0.0622520	total: 16.5s	remaining: 39.4s
    59:	learn: 0.0622050	total: 16.8s	remaining: 39.1s
    60:	learn: 0.0621650	total: 17s	remaining: 38.8s
    61:	learn: 0.0621308	total: 17.3s	remaining: 38.5s
    62:	learn: 0.0620943	total: 17.6s	remaining: 38.2s
    63:	learn: 0.0620656	total: 17.9s	remaining: 37.9s
    64:	learn: 0.0620391	total: 18.1s	remaining: 37.7s
    65:	learn: 0.0620087	total: 18.4s	remaining: 37.3s
    66:	learn: 0.0619785	total: 18.6s	remaining: 37s
    67:	learn: 0.0619607	total: 18.9s	remaining: 36.8s
    68:	learn: 0.0619293	total: 19.2s	remaining: 36.5s
    69:	learn: 0.0618993	total: 19.5s	remaining: 36.2s
    70:	learn: 0.0618728	total: 19.8s	remaining: 35.9s
    71:	learn: 0.0617893	total: 20.1s	remaining: 35.7s
    72:	learn: 0.0617764	total: 20.3s	remaining: 35.4s
    73:	learn: 0.0617540	total: 20.6s	remaining: 35.1s
    74:	learn: 0.0617391	total: 20.9s	remaining: 34.8s
    75:	learn: 0.0617168	total: 21.2s	remaining: 34.5s
    76:	learn: 0.0616916	total: 21.5s	remaining: 34.3s
    77:	learn: 0.0616720	total: 21.8s	remaining: 34s
    78:	learn: 0.0616452	total: 22s	remaining: 33.8s
    79:	learn: 0.0616237	total: 22.3s	remaining: 33.4s
    80:	learn: 0.0616033	total: 22.6s	remaining: 33.1s
    81:	learn: 0.0615907	total: 22.8s	remaining: 32.8s
    82:	learn: 0.0615761	total: 23.1s	remaining: 32.5s
    83:	learn: 0.0615605	total: 23.4s	remaining: 32.3s
    84:	learn: 0.0615443	total: 23.6s	remaining: 32s
    85:	learn: 0.0614909	total: 23.9s	remaining: 31.7s
    86:	learn: 0.0614587	total: 24.2s	remaining: 31.4s
    87:	learn: 0.0614152	total: 24.4s	remaining: 31.1s
    88:	learn: 0.0614083	total: 24.7s	remaining: 30.8s
    89:	learn: 0.0613941	total: 25s	remaining: 30.5s
    90:	learn: 0.0613830	total: 25.3s	remaining: 30.3s
    91:	learn: 0.0613693	total: 25.5s	remaining: 30s
    92:	learn: 0.0613584	total: 25.8s	remaining: 29.7s
    93:	learn: 0.0613394	total: 26.1s	remaining: 29.4s
    94:	learn: 0.0613266	total: 26.3s	remaining: 29.1s
    95:	learn: 0.0613128	total: 26.6s	remaining: 28.8s
    96:	learn: 0.0613036	total: 26.9s	remaining: 28.6s
    97:	learn: 0.0612887	total: 27.1s	remaining: 28.3s
    98:	learn: 0.0612747	total: 27.5s	remaining: 28s
    99:	learn: 0.0612619	total: 27.7s	remaining: 27.7s
    100:	learn: 0.0612516	total: 28s	remaining: 27.5s
    101:	learn: 0.0612413	total: 28.3s	remaining: 27.2s
    102:	learn: 0.0612288	total: 28.5s	remaining: 26.9s
    103:	learn: 0.0612182	total: 28.8s	remaining: 26.6s
    104:	learn: 0.0612095	total: 29.1s	remaining: 26.3s
    105:	learn: 0.0612015	total: 29.4s	remaining: 26.1s
    106:	learn: 0.0611925	total: 29.7s	remaining: 25.8s
    107:	learn: 0.0611850	total: 29.9s	remaining: 25.5s
    108:	learn: 0.0611746	total: 30.2s	remaining: 25.2s
    109:	learn: 0.0611662	total: 30.5s	remaining: 24.9s
    110:	learn: 0.0611536	total: 30.8s	remaining: 24.7s
    111:	learn: 0.0611421	total: 31.1s	remaining: 24.4s
    112:	learn: 0.0611348	total: 31.3s	remaining: 24.1s
    113:	learn: 0.0611294	total: 31.6s	remaining: 23.8s
    114:	learn: 0.0611248	total: 31.9s	remaining: 23.5s
    115:	learn: 0.0611170	total: 32.1s	remaining: 23.3s
    116:	learn: 0.0611091	total: 32.4s	remaining: 23s
    117:	learn: 0.0610964	total: 32.7s	remaining: 22.7s
    118:	learn: 0.0610849	total: 33s	remaining: 22.4s
    119:	learn: 0.0610758	total: 33.2s	remaining: 22.2s
    120:	learn: 0.0610668	total: 33.5s	remaining: 21.9s
    121:	learn: 0.0610616	total: 33.8s	remaining: 21.6s
    122:	learn: 0.0610513	total: 34.1s	remaining: 21.3s
    123:	learn: 0.0610413	total: 34.3s	remaining: 21s
    124:	learn: 0.0610351	total: 34.6s	remaining: 20.8s
    125:	learn: 0.0610272	total: 34.9s	remaining: 20.5s
    126:	learn: 0.0610226	total: 35.2s	remaining: 20.2s
    127:	learn: 0.0610185	total: 35.5s	remaining: 20s
    128:	learn: 0.0610100	total: 35.8s	remaining: 19.7s
    129:	learn: 0.0610027	total: 36s	remaining: 19.4s
    130:	learn: 0.0609938	total: 36.3s	remaining: 19.1s
    131:	learn: 0.0609866	total: 36.6s	remaining: 18.8s
    132:	learn: 0.0609772	total: 36.9s	remaining: 18.6s
    133:	learn: 0.0609670	total: 37.1s	remaining: 18.3s
    134:	learn: 0.0609400	total: 37.4s	remaining: 18s
    135:	learn: 0.0609299	total: 37.7s	remaining: 17.7s
    136:	learn: 0.0609275	total: 38s	remaining: 17.5s
    137:	learn: 0.0609206	total: 38.3s	remaining: 17.2s
    138:	learn: 0.0609011	total: 38.6s	remaining: 16.9s
    139:	learn: 0.0608953	total: 38.9s	remaining: 16.7s
    140:	learn: 0.0608825	total: 39.1s	remaining: 16.4s
    141:	learn: 0.0608781	total: 39.4s	remaining: 16.1s
    142:	learn: 0.0608700	total: 39.7s	remaining: 15.8s
    143:	learn: 0.0608602	total: 40s	remaining: 15.5s
    144:	learn: 0.0608507	total: 40.3s	remaining: 15.3s
    145:	learn: 0.0608462	total: 40.5s	remaining: 15s
    146:	learn: 0.0608362	total: 40.8s	remaining: 14.7s
    147:	learn: 0.0608285	total: 41.1s	remaining: 14.4s
    148:	learn: 0.0608244	total: 41.4s	remaining: 14.2s
    149:	learn: 0.0608140	total: 41.6s	remaining: 13.9s
    150:	learn: 0.0608085	total: 41.9s	remaining: 13.6s
    151:	learn: 0.0607960	total: 42.2s	remaining: 13.3s
    152:	learn: 0.0607915	total: 42.4s	remaining: 13s
    153:	learn: 0.0607865	total: 42.7s	remaining: 12.8s
    154:	learn: 0.0607820	total: 43s	remaining: 12.5s
    155:	learn: 0.0607763	total: 43.3s	remaining: 12.2s
    156:	learn: 0.0607707	total: 43.5s	remaining: 11.9s
    157:	learn: 0.0607641	total: 43.8s	remaining: 11.6s
    158:	learn: 0.0607598	total: 44.1s	remaining: 11.4s
    159:	learn: 0.0607543	total: 44.4s	remaining: 11.1s
    160:	learn: 0.0607489	total: 44.6s	remaining: 10.8s
    161:	learn: 0.0607387	total: 44.9s	remaining: 10.5s
    162:	learn: 0.0607339	total: 45.1s	remaining: 10.2s
    163:	learn: 0.0607283	total: 45.4s	remaining: 9.97s
    164:	learn: 0.0607200	total: 45.7s	remaining: 9.7s
    165:	learn: 0.0607128	total: 46s	remaining: 9.42s
    166:	learn: 0.0607037	total: 46.3s	remaining: 9.15s
    167:	learn: 0.0606914	total: 46.6s	remaining: 8.87s
    168:	learn: 0.0606858	total: 46.9s	remaining: 8.6s
    169:	learn: 0.0606805	total: 47.2s	remaining: 8.32s
    170:	learn: 0.0606702	total: 47.4s	remaining: 8.04s
    171:	learn: 0.0606640	total: 47.7s	remaining: 7.77s
    172:	learn: 0.0606591	total: 48s	remaining: 7.49s
    173:	learn: 0.0606590	total: 48.1s	remaining: 7.18s
    174:	learn: 0.0606555	total: 48.3s	remaining: 6.9s
    175:	learn: 0.0606523	total: 48.6s	remaining: 6.63s
    176:	learn: 0.0606466	total: 48.9s	remaining: 6.36s
    177:	learn: 0.0606392	total: 49.2s	remaining: 6.08s
    178:	learn: 0.0606359	total: 49.5s	remaining: 5.8s
    179:	learn: 0.0606301	total: 49.7s	remaining: 5.52s
    180:	learn: 0.0606254	total: 50s	remaining: 5.25s
    181:	learn: 0.0606210	total: 50.3s	remaining: 4.97s
    182:	learn: 0.0606167	total: 50.5s	remaining: 4.69s
    183:	learn: 0.0606097	total: 50.8s	remaining: 4.42s
    184:	learn: 0.0606026	total: 51.1s	remaining: 4.15s
    185:	learn: 0.0605960	total: 51.4s	remaining: 3.87s
    186:	learn: 0.0605898	total: 51.7s	remaining: 3.59s
    187:	learn: 0.0605825	total: 51.9s	remaining: 3.31s
    188:	learn: 0.0605667	total: 52.2s	remaining: 3.04s
    189:	learn: 0.0605600	total: 52.5s	remaining: 2.76s
    190:	learn: 0.0605554	total: 52.7s	remaining: 2.48s
    191:	learn: 0.0605504	total: 53s	remaining: 2.21s
    192:	learn: 0.0605439	total: 53.3s	remaining: 1.93s
    193:	learn: 0.0605373	total: 53.5s	remaining: 1.66s
    194:	learn: 0.0605304	total: 53.8s	remaining: 1.38s
    195:	learn: 0.0605303	total: 54s	remaining: 1.1s
    196:	learn: 0.0605270	total: 54.3s	remaining: 826ms
    197:	learn: 0.0605223	total: 54.6s	remaining: 551ms
    198:	learn: 0.0605191	total: 54.8s	remaining: 276ms
    199:	learn: 0.0605156	total: 55.1s	remaining: 0us
    Trained model nº 14/27. Iterations: 200Depth: 8Learning rate: 0.05
    0:	learn: 0.5926721	total: 286ms	remaining: 1m 25s
    1:	learn: 0.5060639	total: 548ms	remaining: 1m 21s
    2:	learn: 0.4296100	total: 827ms	remaining: 1m 21s
    3:	learn: 0.3650039	total: 1.11s	remaining: 1m 22s
    4:	learn: 0.3187505	total: 1.35s	remaining: 1m 19s
    5:	learn: 0.2727445	total: 1.63s	remaining: 1m 20s
    6:	learn: 0.2361745	total: 1.9s	remaining: 1m 19s
    7:	learn: 0.2083013	total: 2.19s	remaining: 1m 19s
    8:	learn: 0.1843583	total: 2.49s	remaining: 1m 20s
    9:	learn: 0.1652011	total: 2.79s	remaining: 1m 20s
    10:	learn: 0.1500258	total: 3.06s	remaining: 1m 20s
    11:	learn: 0.1374133	total: 3.35s	remaining: 1m 20s
    12:	learn: 0.1260781	total: 3.64s	remaining: 1m 20s
    13:	learn: 0.1168321	total: 3.91s	remaining: 1m 19s
    14:	learn: 0.1092126	total: 4.2s	remaining: 1m 19s
    15:	learn: 0.1027471	total: 4.47s	remaining: 1m 19s
    16:	learn: 0.0971917	total: 4.75s	remaining: 1m 18s
    17:	learn: 0.0926318	total: 5.02s	remaining: 1m 18s
    18:	learn: 0.0887233	total: 5.29s	remaining: 1m 18s
    19:	learn: 0.0852969	total: 5.57s	remaining: 1m 18s
    20:	learn: 0.0824125	total: 5.86s	remaining: 1m 17s
    21:	learn: 0.0799004	total: 6.13s	remaining: 1m 17s
    22:	learn: 0.0780581	total: 6.41s	remaining: 1m 17s
    23:	learn: 0.0762085	total: 6.69s	remaining: 1m 16s
    24:	learn: 0.0746578	total: 6.97s	remaining: 1m 16s
    25:	learn: 0.0731872	total: 7.24s	remaining: 1m 16s
    26:	learn: 0.0719326	total: 7.51s	remaining: 1m 15s
    27:	learn: 0.0708713	total: 7.78s	remaining: 1m 15s
    28:	learn: 0.0699483	total: 8.07s	remaining: 1m 15s
    29:	learn: 0.0690709	total: 8.35s	remaining: 1m 15s
    30:	learn: 0.0683317	total: 8.63s	remaining: 1m 14s
    31:	learn: 0.0676993	total: 8.9s	remaining: 1m 14s
    32:	learn: 0.0671278	total: 9.17s	remaining: 1m 14s
    33:	learn: 0.0666199	total: 9.47s	remaining: 1m 14s
    34:	learn: 0.0661547	total: 9.76s	remaining: 1m 13s
    35:	learn: 0.0657422	total: 10s	remaining: 1m 13s
    36:	learn: 0.0653662	total: 10.3s	remaining: 1m 13s
    37:	learn: 0.0650357	total: 10.6s	remaining: 1m 13s
    38:	learn: 0.0647312	total: 10.9s	remaining: 1m 13s
    39:	learn: 0.0645109	total: 11.2s	remaining: 1m 12s
    40:	learn: 0.0642643	total: 11.5s	remaining: 1m 12s
    41:	learn: 0.0640424	total: 11.7s	remaining: 1m 12s
    42:	learn: 0.0638480	total: 12s	remaining: 1m 11s
    43:	learn: 0.0636896	total: 12.3s	remaining: 1m 11s
    44:	learn: 0.0635276	total: 12.5s	remaining: 1m 11s
    45:	learn: 0.0633734	total: 12.8s	remaining: 1m 10s
    46:	learn: 0.0632464	total: 13.1s	remaining: 1m 10s
    47:	learn: 0.0631228	total: 13.4s	remaining: 1m 10s
    48:	learn: 0.0630190	total: 13.6s	remaining: 1m 9s
    49:	learn: 0.0629263	total: 13.9s	remaining: 1m 9s
    50:	learn: 0.0628393	total: 14.2s	remaining: 1m 9s
    51:	learn: 0.0627514	total: 14.4s	remaining: 1m 8s
    52:	learn: 0.0626828	total: 14.7s	remaining: 1m 8s
    53:	learn: 0.0625193	total: 15s	remaining: 1m 8s
    54:	learn: 0.0624599	total: 15.3s	remaining: 1m 7s
    55:	learn: 0.0623984	total: 15.5s	remaining: 1m 7s
    56:	learn: 0.0623360	total: 15.8s	remaining: 1m 7s
    57:	learn: 0.0622975	total: 16.1s	remaining: 1m 7s
    58:	learn: 0.0622520	total: 16.4s	remaining: 1m 6s
    59:	learn: 0.0622050	total: 16.7s	remaining: 1m 6s
    60:	learn: 0.0621650	total: 16.9s	remaining: 1m 6s
    61:	learn: 0.0621308	total: 17.2s	remaining: 1m 6s
    62:	learn: 0.0620943	total: 17.5s	remaining: 1m 5s
    63:	learn: 0.0620656	total: 17.8s	remaining: 1m 5s
    64:	learn: 0.0620391	total: 18s	remaining: 1m 5s
    65:	learn: 0.0620087	total: 18.3s	remaining: 1m 4s
    66:	learn: 0.0619785	total: 18.5s	remaining: 1m 4s
    67:	learn: 0.0619607	total: 18.8s	remaining: 1m 4s
    68:	learn: 0.0619293	total: 19.1s	remaining: 1m 3s
    69:	learn: 0.0618993	total: 19.4s	remaining: 1m 3s
    70:	learn: 0.0618728	total: 19.7s	remaining: 1m 3s
    71:	learn: 0.0617893	total: 20s	remaining: 1m 3s
    72:	learn: 0.0617764	total: 20.2s	remaining: 1m 2s
    73:	learn: 0.0617540	total: 20.5s	remaining: 1m 2s
    74:	learn: 0.0617391	total: 20.8s	remaining: 1m 2s
    75:	learn: 0.0617168	total: 21.1s	remaining: 1m 2s
    76:	learn: 0.0616916	total: 21.3s	remaining: 1m 1s
    77:	learn: 0.0616720	total: 21.7s	remaining: 1m 1s
    78:	learn: 0.0616452	total: 22s	remaining: 1m 1s
    79:	learn: 0.0616237	total: 22.3s	remaining: 1m 1s
    80:	learn: 0.0616033	total: 22.6s	remaining: 1m
    81:	learn: 0.0615907	total: 22.8s	remaining: 1m
    82:	learn: 0.0615761	total: 23.1s	remaining: 1m
    83:	learn: 0.0615605	total: 23.4s	remaining: 1m
    84:	learn: 0.0615443	total: 23.6s	remaining: 59.8s
    85:	learn: 0.0614909	total: 23.9s	remaining: 59.5s
    86:	learn: 0.0614587	total: 24.2s	remaining: 59.2s
    87:	learn: 0.0614152	total: 24.5s	remaining: 58.9s
    88:	learn: 0.0614083	total: 24.7s	remaining: 58.6s
    89:	learn: 0.0613941	total: 25s	remaining: 58.4s
    90:	learn: 0.0613830	total: 25.3s	remaining: 58.2s
    91:	learn: 0.0613693	total: 25.6s	remaining: 57.9s
    92:	learn: 0.0613584	total: 25.9s	remaining: 57.6s
    93:	learn: 0.0613394	total: 26.2s	remaining: 57.3s
    94:	learn: 0.0613266	total: 26.5s	remaining: 57.1s
    95:	learn: 0.0613128	total: 26.7s	remaining: 56.8s
    96:	learn: 0.0613036	total: 27s	remaining: 56.5s
    97:	learn: 0.0612887	total: 27.3s	remaining: 56.2s
    98:	learn: 0.0612747	total: 27.6s	remaining: 56s
    99:	learn: 0.0612619	total: 27.9s	remaining: 55.7s
    100:	learn: 0.0612516	total: 28.1s	remaining: 55.5s
    101:	learn: 0.0612413	total: 28.4s	remaining: 55.2s
    102:	learn: 0.0612288	total: 28.7s	remaining: 54.9s
    103:	learn: 0.0612182	total: 29s	remaining: 54.7s
    104:	learn: 0.0612095	total: 29.3s	remaining: 54.4s
    105:	learn: 0.0612015	total: 29.6s	remaining: 54.2s
    106:	learn: 0.0611925	total: 29.9s	remaining: 53.9s
    107:	learn: 0.0611850	total: 30.2s	remaining: 53.6s
    108:	learn: 0.0611746	total: 30.5s	remaining: 53.4s
    109:	learn: 0.0611662	total: 30.8s	remaining: 53.2s
    110:	learn: 0.0611536	total: 31.1s	remaining: 52.9s
    111:	learn: 0.0611421	total: 31.4s	remaining: 52.6s
    112:	learn: 0.0611348	total: 31.6s	remaining: 52.3s
    113:	learn: 0.0611294	total: 31.9s	remaining: 52.1s
    114:	learn: 0.0611248	total: 32.2s	remaining: 51.8s
    115:	learn: 0.0611170	total: 32.5s	remaining: 51.5s
    116:	learn: 0.0611091	total: 32.8s	remaining: 51.2s
    117:	learn: 0.0610964	total: 33s	remaining: 51s
    118:	learn: 0.0610849	total: 33.3s	remaining: 50.7s
    119:	learn: 0.0610758	total: 33.6s	remaining: 50.4s
    120:	learn: 0.0610668	total: 33.9s	remaining: 50.1s
    121:	learn: 0.0610616	total: 34.2s	remaining: 49.9s
    122:	learn: 0.0610513	total: 34.5s	remaining: 49.6s
    123:	learn: 0.0610413	total: 34.7s	remaining: 49.3s
    124:	learn: 0.0610351	total: 35s	remaining: 49s
    125:	learn: 0.0610272	total: 35.3s	remaining: 48.8s
    126:	learn: 0.0610226	total: 35.6s	remaining: 48.5s
    127:	learn: 0.0610185	total: 35.9s	remaining: 48.3s
    128:	learn: 0.0610100	total: 36.2s	remaining: 48s
    129:	learn: 0.0610027	total: 36.4s	remaining: 47.7s
    130:	learn: 0.0609938	total: 36.7s	remaining: 47.4s
    131:	learn: 0.0609866	total: 37s	remaining: 47.1s
    132:	learn: 0.0609772	total: 37.3s	remaining: 46.8s
    133:	learn: 0.0609670	total: 37.6s	remaining: 46.6s
    134:	learn: 0.0609400	total: 37.9s	remaining: 46.3s
    135:	learn: 0.0609299	total: 38.2s	remaining: 46s
    136:	learn: 0.0609275	total: 38.4s	remaining: 45.7s
    137:	learn: 0.0609206	total: 38.7s	remaining: 45.5s
    138:	learn: 0.0609011	total: 39s	remaining: 45.2s
    139:	learn: 0.0608953	total: 39.3s	remaining: 44.9s
    140:	learn: 0.0608825	total: 39.6s	remaining: 44.6s
    141:	learn: 0.0608781	total: 39.9s	remaining: 44.4s
    142:	learn: 0.0608700	total: 40.2s	remaining: 44.1s
    143:	learn: 0.0608602	total: 40.4s	remaining: 43.8s
    144:	learn: 0.0608507	total: 40.7s	remaining: 43.5s
    145:	learn: 0.0608462	total: 41s	remaining: 43.3s
    146:	learn: 0.0608362	total: 41.3s	remaining: 43s
    147:	learn: 0.0608285	total: 41.6s	remaining: 42.7s
    148:	learn: 0.0608244	total: 41.9s	remaining: 42.5s
    149:	learn: 0.0608140	total: 42.2s	remaining: 42.2s
    150:	learn: 0.0608085	total: 42.5s	remaining: 41.9s
    151:	learn: 0.0607960	total: 42.8s	remaining: 41.6s
    152:	learn: 0.0607915	total: 43s	remaining: 41.4s
    153:	learn: 0.0607865	total: 43.3s	remaining: 41.1s
    154:	learn: 0.0607820	total: 43.6s	remaining: 40.8s
    155:	learn: 0.0607763	total: 43.9s	remaining: 40.5s
    156:	learn: 0.0607707	total: 44.1s	remaining: 40.2s
    157:	learn: 0.0607641	total: 44.4s	remaining: 39.9s
    158:	learn: 0.0607598	total: 44.7s	remaining: 39.6s
    159:	learn: 0.0607543	total: 45s	remaining: 39.4s
    160:	learn: 0.0607489	total: 45.2s	remaining: 39.1s
    161:	learn: 0.0607387	total: 45.5s	remaining: 38.8s
    162:	learn: 0.0607339	total: 45.8s	remaining: 38.5s
    163:	learn: 0.0607283	total: 46s	remaining: 38.2s
    164:	learn: 0.0607200	total: 46.3s	remaining: 37.9s
    165:	learn: 0.0607128	total: 46.6s	remaining: 37.6s
    166:	learn: 0.0607037	total: 46.9s	remaining: 37.3s
    167:	learn: 0.0606914	total: 47.2s	remaining: 37.1s
    168:	learn: 0.0606858	total: 47.4s	remaining: 36.8s
    169:	learn: 0.0606805	total: 47.7s	remaining: 36.5s
    170:	learn: 0.0606702	total: 48s	remaining: 36.2s
    171:	learn: 0.0606640	total: 48.2s	remaining: 35.9s
    172:	learn: 0.0606591	total: 48.5s	remaining: 35.6s
    173:	learn: 0.0606590	total: 48.6s	remaining: 35.2s
    174:	learn: 0.0606555	total: 48.9s	remaining: 34.9s
    175:	learn: 0.0606523	total: 49.1s	remaining: 34.6s
    176:	learn: 0.0606466	total: 49.4s	remaining: 34.3s
    177:	learn: 0.0606392	total: 49.7s	remaining: 34.1s
    178:	learn: 0.0606359	total: 50s	remaining: 33.8s
    179:	learn: 0.0606301	total: 50.2s	remaining: 33.5s
    180:	learn: 0.0606254	total: 50.5s	remaining: 33.2s
    181:	learn: 0.0606210	total: 50.7s	remaining: 32.9s
    182:	learn: 0.0606167	total: 51s	remaining: 32.6s
    183:	learn: 0.0606097	total: 51.2s	remaining: 32.3s
    184:	learn: 0.0606026	total: 51.5s	remaining: 32s
    185:	learn: 0.0605960	total: 51.8s	remaining: 31.8s
    186:	learn: 0.0605898	total: 52.1s	remaining: 31.5s
    187:	learn: 0.0605825	total: 52.4s	remaining: 31.2s
    188:	learn: 0.0605667	total: 52.6s	remaining: 30.9s
    189:	learn: 0.0605600	total: 52.9s	remaining: 30.6s
    190:	learn: 0.0605554	total: 53.2s	remaining: 30.3s
    191:	learn: 0.0605504	total: 53.4s	remaining: 30s
    192:	learn: 0.0605439	total: 53.7s	remaining: 29.8s
    193:	learn: 0.0605373	total: 54s	remaining: 29.5s
    194:	learn: 0.0605304	total: 54.2s	remaining: 29.2s
    195:	learn: 0.0605303	total: 54.4s	remaining: 28.9s
    196:	learn: 0.0605270	total: 54.7s	remaining: 28.6s
    197:	learn: 0.0605223	total: 55s	remaining: 28.3s
    198:	learn: 0.0605191	total: 55.3s	remaining: 28s
    199:	learn: 0.0605156	total: 55.5s	remaining: 27.8s
    200:	learn: 0.0605088	total: 55.8s	remaining: 27.5s
    201:	learn: 0.0605058	total: 56s	remaining: 27.2s
    202:	learn: 0.0605002	total: 56.3s	remaining: 26.9s
    203:	learn: 0.0604962	total: 56.6s	remaining: 26.6s
    204:	learn: 0.0604834	total: 56.9s	remaining: 26.4s
    205:	learn: 0.0604799	total: 57.2s	remaining: 26.1s
    206:	learn: 0.0604761	total: 57.4s	remaining: 25.8s
    207:	learn: 0.0604739	total: 57.7s	remaining: 25.5s
    208:	learn: 0.0604697	total: 57.9s	remaining: 25.2s
    209:	learn: 0.0604652	total: 58.2s	remaining: 25s
    210:	learn: 0.0604598	total: 58.5s	remaining: 24.7s
    211:	learn: 0.0604564	total: 58.8s	remaining: 24.4s
    212:	learn: 0.0604530	total: 59.1s	remaining: 24.1s
    213:	learn: 0.0604497	total: 59.3s	remaining: 23.8s
    214:	learn: 0.0604441	total: 59.6s	remaining: 23.6s
    215:	learn: 0.0604405	total: 59.9s	remaining: 23.3s
    216:	learn: 0.0604334	total: 1m	remaining: 23s
    217:	learn: 0.0604279	total: 1m	remaining: 22.7s
    218:	learn: 0.0604235	total: 1m	remaining: 22.4s
    219:	learn: 0.0604162	total: 1m	remaining: 22.2s
    220:	learn: 0.0604107	total: 1m 1s	remaining: 21.9s
    221:	learn: 0.0604035	total: 1m 1s	remaining: 21.6s
    222:	learn: 0.0603972	total: 1m 1s	remaining: 21.3s
    223:	learn: 0.0603913	total: 1m 2s	remaining: 21s
    224:	learn: 0.0603832	total: 1m 2s	remaining: 20.8s
    225:	learn: 0.0603799	total: 1m 2s	remaining: 20.5s
    226:	learn: 0.0603765	total: 1m 2s	remaining: 20.2s
    227:	learn: 0.0603741	total: 1m 3s	remaining: 19.9s
    228:	learn: 0.0603720	total: 1m 3s	remaining: 19.6s
    229:	learn: 0.0603656	total: 1m 3s	remaining: 19.4s
    230:	learn: 0.0603611	total: 1m 3s	remaining: 19.1s
    231:	learn: 0.0603578	total: 1m 4s	remaining: 18.8s
    232:	learn: 0.0603571	total: 1m 4s	remaining: 18.5s
    233:	learn: 0.0603545	total: 1m 4s	remaining: 18.2s
    234:	learn: 0.0603435	total: 1m 4s	remaining: 17.9s
    235:	learn: 0.0603389	total: 1m 5s	remaining: 17.7s
    236:	learn: 0.0603341	total: 1m 5s	remaining: 17.4s
    237:	learn: 0.0603262	total: 1m 5s	remaining: 17.1s
    238:	learn: 0.0603223	total: 1m 5s	remaining: 16.8s
    239:	learn: 0.0603125	total: 1m 6s	remaining: 16.6s
    240:	learn: 0.0603096	total: 1m 6s	remaining: 16.3s
    241:	learn: 0.0603059	total: 1m 6s	remaining: 16s
    242:	learn: 0.0602970	total: 1m 7s	remaining: 15.7s
    243:	learn: 0.0602951	total: 1m 7s	remaining: 15.4s
    244:	learn: 0.0602909	total: 1m 7s	remaining: 15.2s
    245:	learn: 0.0602882	total: 1m 7s	remaining: 14.9s
    246:	learn: 0.0602815	total: 1m 8s	remaining: 14.6s
    247:	learn: 0.0602790	total: 1m 8s	remaining: 14.3s
    248:	learn: 0.0602790	total: 1m 8s	remaining: 14s
    249:	learn: 0.0602733	total: 1m 8s	remaining: 13.7s
    250:	learn: 0.0602687	total: 1m 9s	remaining: 13.5s
    251:	learn: 0.0602636	total: 1m 9s	remaining: 13.2s
    252:	learn: 0.0602557	total: 1m 9s	remaining: 12.9s
    253:	learn: 0.0602524	total: 1m 9s	remaining: 12.7s
    254:	learn: 0.0602482	total: 1m 10s	remaining: 12.4s
    255:	learn: 0.0602429	total: 1m 10s	remaining: 12.1s
    256:	learn: 0.0602385	total: 1m 10s	remaining: 11.8s
    257:	learn: 0.0602349	total: 1m 10s	remaining: 11.6s
    258:	learn: 0.0602306	total: 1m 11s	remaining: 11.3s
    259:	learn: 0.0602266	total: 1m 11s	remaining: 11s
    260:	learn: 0.0602242	total: 1m 11s	remaining: 10.7s
    261:	learn: 0.0602183	total: 1m 12s	remaining: 10.5s
    262:	learn: 0.0602131	total: 1m 12s	remaining: 10.2s
    263:	learn: 0.0602062	total: 1m 12s	remaining: 9.91s
    264:	learn: 0.0602035	total: 1m 12s	remaining: 9.63s
    265:	learn: 0.0601995	total: 1m 13s	remaining: 9.36s
    266:	learn: 0.0601972	total: 1m 13s	remaining: 9.08s
    267:	learn: 0.0601899	total: 1m 13s	remaining: 8.8s
    268:	learn: 0.0601857	total: 1m 14s	remaining: 8.53s
    269:	learn: 0.0601823	total: 1m 14s	remaining: 8.25s
    270:	learn: 0.0601806	total: 1m 14s	remaining: 7.97s
    271:	learn: 0.0601761	total: 1m 14s	remaining: 7.7s
    272:	learn: 0.0601743	total: 1m 15s	remaining: 7.42s
    273:	learn: 0.0601679	total: 1m 15s	remaining: 7.15s
    274:	learn: 0.0601670	total: 1m 15s	remaining: 6.87s
    275:	learn: 0.0601579	total: 1m 15s	remaining: 6.6s
    276:	learn: 0.0601554	total: 1m 16s	remaining: 6.32s
    277:	learn: 0.0601477	total: 1m 16s	remaining: 6.05s
    278:	learn: 0.0601446	total: 1m 16s	remaining: 5.77s
    279:	learn: 0.0601401	total: 1m 16s	remaining: 5.5s
    280:	learn: 0.0601382	total: 1m 17s	remaining: 5.22s
    281:	learn: 0.0601378	total: 1m 17s	remaining: 4.94s
    282:	learn: 0.0601337	total: 1m 17s	remaining: 4.67s
    283:	learn: 0.0601296	total: 1m 17s	remaining: 4.39s
    284:	learn: 0.0601237	total: 1m 18s	remaining: 4.12s
    285:	learn: 0.0601192	total: 1m 18s	remaining: 3.84s
    286:	learn: 0.0601132	total: 1m 18s	remaining: 3.57s
    287:	learn: 0.0601081	total: 1m 18s	remaining: 3.29s
    288:	learn: 0.0601049	total: 1m 19s	remaining: 3.02s
    289:	learn: 0.0601013	total: 1m 19s	remaining: 2.74s
    290:	learn: 0.0600952	total: 1m 19s	remaining: 2.47s
    291:	learn: 0.0600905	total: 1m 20s	remaining: 2.2s
    292:	learn: 0.0600871	total: 1m 20s	remaining: 1.92s
    293:	learn: 0.0600817	total: 1m 20s	remaining: 1.65s
    294:	learn: 0.0600776	total: 1m 21s	remaining: 1.37s
    295:	learn: 0.0600741	total: 1m 21s	remaining: 1.1s
    296:	learn: 0.0600695	total: 1m 21s	remaining: 824ms
    297:	learn: 0.0600659	total: 1m 21s	remaining: 549ms
    298:	learn: 0.0600632	total: 1m 22s	remaining: 274ms
    299:	learn: 0.0600597	total: 1m 22s	remaining: 0us
    Trained model nº 15/27. Iterations: 300Depth: 8Learning rate: 0.05
    0:	learn: 0.5908777	total: 229ms	remaining: 22.6s
    1:	learn: 0.5079519	total: 450ms	remaining: 22s
    2:	learn: 0.4342613	total: 671ms	remaining: 21.7s
    3:	learn: 0.3688887	total: 882ms	remaining: 21.2s
    4:	learn: 0.3232199	total: 1000ms	remaining: 19s
    5:	learn: 0.2783083	total: 1.21s	remaining: 18.9s
    6:	learn: 0.2446067	total: 1.43s	remaining: 19s
    7:	learn: 0.2187214	total: 1.62s	remaining: 18.7s
    8:	learn: 0.1948354	total: 1.82s	remaining: 18.4s
    9:	learn: 0.1753584	total: 2.04s	remaining: 18.3s
    10:	learn: 0.1585682	total: 2.25s	remaining: 18.2s
    11:	learn: 0.1429196	total: 2.47s	remaining: 18.1s
    12:	learn: 0.1323849	total: 2.68s	remaining: 17.9s
    13:	learn: 0.1212496	total: 2.9s	remaining: 17.8s
    14:	learn: 0.1122666	total: 3.11s	remaining: 17.6s
    15:	learn: 0.1047483	total: 3.32s	remaining: 17.4s
    16:	learn: 0.0987502	total: 3.52s	remaining: 17.2s
    17:	learn: 0.0935327	total: 3.76s	remaining: 17.1s
    18:	learn: 0.0892144	total: 3.97s	remaining: 16.9s
    19:	learn: 0.0854992	total: 4.19s	remaining: 16.8s
    20:	learn: 0.0823699	total: 4.41s	remaining: 16.6s
    21:	learn: 0.0797742	total: 4.64s	remaining: 16.4s
    22:	learn: 0.0775493	total: 4.87s	remaining: 16.3s
    23:	learn: 0.0755986	total: 5.09s	remaining: 16.1s
    24:	learn: 0.0738797	total: 5.3s	remaining: 15.9s
    25:	learn: 0.0725158	total: 5.52s	remaining: 15.7s
    26:	learn: 0.0712269	total: 5.73s	remaining: 15.5s
    27:	learn: 0.0701639	total: 5.94s	remaining: 15.3s
    28:	learn: 0.0690367	total: 6.16s	remaining: 15.1s
    29:	learn: 0.0682510	total: 6.39s	remaining: 14.9s
    30:	learn: 0.0675754	total: 6.61s	remaining: 14.7s
    31:	learn: 0.0669141	total: 6.83s	remaining: 14.5s
    32:	learn: 0.0663716	total: 7.05s	remaining: 14.3s
    33:	learn: 0.0658672	total: 7.26s	remaining: 14.1s
    34:	learn: 0.0654137	total: 7.51s	remaining: 13.9s
    35:	learn: 0.0650155	total: 7.72s	remaining: 13.7s
    36:	learn: 0.0646445	total: 7.95s	remaining: 13.5s
    37:	learn: 0.0643075	total: 8.17s	remaining: 13.3s
    38:	learn: 0.0640651	total: 8.39s	remaining: 13.1s
    39:	learn: 0.0638308	total: 8.62s	remaining: 12.9s
    40:	learn: 0.0636053	total: 8.84s	remaining: 12.7s
    41:	learn: 0.0633770	total: 9.05s	remaining: 12.5s
    42:	learn: 0.0632045	total: 9.26s	remaining: 12.3s
    43:	learn: 0.0630196	total: 9.5s	remaining: 12.1s
    44:	learn: 0.0628740	total: 9.74s	remaining: 11.9s
    45:	learn: 0.0627555	total: 9.97s	remaining: 11.7s
    46:	learn: 0.0626604	total: 10.2s	remaining: 11.5s
    47:	learn: 0.0625317	total: 10.5s	remaining: 11.3s
    48:	learn: 0.0624329	total: 10.7s	remaining: 11.2s
    49:	learn: 0.0623461	total: 10.9s	remaining: 10.9s
    50:	learn: 0.0622629	total: 11.1s	remaining: 10.7s
    51:	learn: 0.0621812	total: 11.4s	remaining: 10.5s
    52:	learn: 0.0621161	total: 11.6s	remaining: 10.3s
    53:	learn: 0.0619863	total: 11.8s	remaining: 10.1s
    54:	learn: 0.0619110	total: 12.1s	remaining: 9.87s
    55:	learn: 0.0618468	total: 12.3s	remaining: 9.65s
    56:	learn: 0.0617852	total: 12.5s	remaining: 9.44s
    57:	learn: 0.0617419	total: 12.8s	remaining: 9.24s
    58:	learn: 0.0616908	total: 13s	remaining: 9.03s
    59:	learn: 0.0616366	total: 13.2s	remaining: 8.83s
    60:	learn: 0.0615970	total: 13.5s	remaining: 8.62s
    61:	learn: 0.0615568	total: 13.7s	remaining: 8.41s
    62:	learn: 0.0615186	total: 13.9s	remaining: 8.19s
    63:	learn: 0.0614735	total: 14.2s	remaining: 7.97s
    64:	learn: 0.0614223	total: 14.4s	remaining: 7.75s
    65:	learn: 0.0613793	total: 14.6s	remaining: 7.54s
    66:	learn: 0.0613540	total: 14.9s	remaining: 7.32s
    67:	learn: 0.0613284	total: 15.1s	remaining: 7.11s
    68:	learn: 0.0612972	total: 15.3s	remaining: 6.89s
    69:	learn: 0.0612712	total: 15.6s	remaining: 6.68s
    70:	learn: 0.0612478	total: 15.9s	remaining: 6.48s
    71:	learn: 0.0612243	total: 16.1s	remaining: 6.26s
    72:	learn: 0.0611962	total: 16.3s	remaining: 6.04s
    73:	learn: 0.0611712	total: 16.6s	remaining: 5.83s
    74:	learn: 0.0611499	total: 16.8s	remaining: 5.61s
    75:	learn: 0.0611299	total: 17.1s	remaining: 5.39s
    76:	learn: 0.0610928	total: 17.3s	remaining: 5.17s
    77:	learn: 0.0610653	total: 17.5s	remaining: 4.94s
    78:	learn: 0.0610391	total: 17.8s	remaining: 4.72s
    79:	learn: 0.0610151	total: 18s	remaining: 4.5s
    80:	learn: 0.0609856	total: 18.2s	remaining: 4.28s
    81:	learn: 0.0609595	total: 18.5s	remaining: 4.06s
    82:	learn: 0.0609468	total: 18.7s	remaining: 3.83s
    83:	learn: 0.0609241	total: 18.9s	remaining: 3.61s
    84:	learn: 0.0609104	total: 19.2s	remaining: 3.38s
    85:	learn: 0.0608940	total: 19.4s	remaining: 3.15s
    86:	learn: 0.0608818	total: 19.6s	remaining: 2.93s
    87:	learn: 0.0608649	total: 19.9s	remaining: 2.71s
    88:	learn: 0.0608023	total: 20.1s	remaining: 2.49s
    89:	learn: 0.0607879	total: 20.4s	remaining: 2.26s
    90:	learn: 0.0607694	total: 20.6s	remaining: 2.04s
    91:	learn: 0.0607434	total: 20.8s	remaining: 1.81s
    92:	learn: 0.0607300	total: 21.1s	remaining: 1.59s
    93:	learn: 0.0607121	total: 21.3s	remaining: 1.36s
    94:	learn: 0.0606927	total: 21.5s	remaining: 1.13s
    95:	learn: 0.0606666	total: 21.8s	remaining: 907ms
    96:	learn: 0.0606538	total: 22s	remaining: 680ms
    97:	learn: 0.0606378	total: 22.2s	remaining: 453ms
    98:	learn: 0.0606112	total: 22.5s	remaining: 227ms
    99:	learn: 0.0606009	total: 22.7s	remaining: 0us
    Trained model nº 16/27. Iterations: 100Depth: 10Learning rate: 0.05
    0:	learn: 0.5874604	total: 358ms	remaining: 1m 11s
    1:	learn: 0.4965186	total: 664ms	remaining: 1m 5s
    2:	learn: 0.4240445	total: 832ms	remaining: 54.6s
    3:	learn: 0.3616264	total: 1.21s	remaining: 59.1s
    4:	learn: 0.3178060	total: 1.32s	remaining: 51.6s
    5:	learn: 0.2736807	total: 1.66s	remaining: 53.8s
    6:	learn: 0.2375899	total: 1.88s	remaining: 51.8s
    7:	learn: 0.2082300	total: 2.26s	remaining: 54.3s
    8:	learn: 0.1856276	total: 2.59s	remaining: 55s
    9:	learn: 0.1658431	total: 2.92s	remaining: 55.5s
    10:	learn: 0.1497862	total: 3.25s	remaining: 55.8s
    11:	learn: 0.1364801	total: 3.59s	remaining: 56.2s
    12:	learn: 0.1255410	total: 3.94s	remaining: 56.6s
    13:	learn: 0.1163558	total: 4.28s	remaining: 56.9s
    14:	learn: 0.1082342	total: 4.67s	remaining: 57.6s
    15:	learn: 0.1009861	total: 5.04s	remaining: 58s
    16:	learn: 0.0957680	total: 5.4s	remaining: 58.1s
    17:	learn: 0.0912677	total: 5.76s	remaining: 58.2s
    18:	learn: 0.0874582	total: 6.13s	remaining: 58.3s
    19:	learn: 0.0841861	total: 6.47s	remaining: 58.2s
    20:	learn: 0.0814063	total: 6.78s	remaining: 57.8s
    21:	learn: 0.0789782	total: 7.15s	remaining: 57.8s
    22:	learn: 0.0769172	total: 7.49s	remaining: 57.6s
    23:	learn: 0.0751318	total: 7.84s	remaining: 57.5s
    24:	learn: 0.0735669	total: 8.17s	remaining: 57.2s
    25:	learn: 0.0722194	total: 8.54s	remaining: 57.2s
    26:	learn: 0.0710454	total: 8.89s	remaining: 57s
    27:	learn: 0.0700026	total: 9.25s	remaining: 56.8s
    28:	learn: 0.0690991	total: 9.59s	remaining: 56.5s
    29:	learn: 0.0683152	total: 9.93s	remaining: 56.3s
    30:	learn: 0.0675418	total: 10.3s	remaining: 56.1s
    31:	learn: 0.0669189	total: 10.6s	remaining: 55.7s
    32:	learn: 0.0663822	total: 10.9s	remaining: 55.4s
    33:	learn: 0.0657086	total: 11.3s	remaining: 55.1s
    34:	learn: 0.0652995	total: 11.6s	remaining: 54.7s
    35:	learn: 0.0649154	total: 12s	remaining: 54.6s
    36:	learn: 0.0645838	total: 12.3s	remaining: 54.3s
    37:	learn: 0.0642922	total: 12.7s	remaining: 54.1s
    38:	learn: 0.0640224	total: 13s	remaining: 53.8s
    39:	learn: 0.0637963	total: 13.4s	remaining: 53.5s
    40:	learn: 0.0635757	total: 13.7s	remaining: 53.2s
    41:	learn: 0.0633810	total: 14.1s	remaining: 52.9s
    42:	learn: 0.0632043	total: 14.4s	remaining: 52.5s
    43:	learn: 0.0630526	total: 14.7s	remaining: 52.1s
    44:	learn: 0.0628748	total: 15.1s	remaining: 51.9s
    45:	learn: 0.0627533	total: 15.4s	remaining: 51.5s
    46:	learn: 0.0626465	total: 15.7s	remaining: 51.1s
    47:	learn: 0.0625420	total: 16s	remaining: 50.8s
    48:	learn: 0.0623633	total: 16.4s	remaining: 50.4s
    49:	learn: 0.0622635	total: 16.7s	remaining: 50.1s
    50:	learn: 0.0621766	total: 17s	remaining: 49.8s
    51:	learn: 0.0621133	total: 17.4s	remaining: 49.4s
    52:	learn: 0.0620519	total: 17.7s	remaining: 49s
    53:	learn: 0.0619690	total: 18s	remaining: 48.7s
    54:	learn: 0.0618969	total: 18.3s	remaining: 48.4s
    55:	learn: 0.0618450	total: 18.7s	remaining: 48.1s
    56:	learn: 0.0617809	total: 19s	remaining: 47.8s
    57:	learn: 0.0617369	total: 19.4s	remaining: 47.4s
    58:	learn: 0.0616869	total: 19.7s	remaining: 47.1s
    59:	learn: 0.0616363	total: 20s	remaining: 46.8s
    60:	learn: 0.0615971	total: 20.4s	remaining: 46.4s
    61:	learn: 0.0615615	total: 20.7s	remaining: 46.1s
    62:	learn: 0.0615171	total: 21.1s	remaining: 45.8s
    63:	learn: 0.0614273	total: 21.4s	remaining: 45.6s
    64:	learn: 0.0613947	total: 21.8s	remaining: 45.2s
    65:	learn: 0.0613588	total: 22.1s	remaining: 44.9s
    66:	learn: 0.0613326	total: 22.5s	remaining: 44.6s
    67:	learn: 0.0613049	total: 22.8s	remaining: 44.2s
    68:	learn: 0.0612809	total: 23.1s	remaining: 43.9s
    69:	learn: 0.0612559	total: 23.5s	remaining: 43.6s
    70:	learn: 0.0612317	total: 23.8s	remaining: 43.3s
    71:	learn: 0.0612105	total: 24.2s	remaining: 43.1s
    72:	learn: 0.0611906	total: 24.6s	remaining: 42.7s
    73:	learn: 0.0611661	total: 24.9s	remaining: 42.5s
    74:	learn: 0.0611414	total: 25.3s	remaining: 42.1s
    75:	learn: 0.0611152	total: 25.6s	remaining: 41.8s
    76:	learn: 0.0610918	total: 26s	remaining: 41.6s
    77:	learn: 0.0610532	total: 26.4s	remaining: 41.3s
    78:	learn: 0.0610114	total: 26.8s	remaining: 41s
    79:	learn: 0.0609903	total: 27.1s	remaining: 40.6s
    80:	learn: 0.0609659	total: 27.4s	remaining: 40.3s
    81:	learn: 0.0609423	total: 27.8s	remaining: 40s
    82:	learn: 0.0609020	total: 28.2s	remaining: 39.7s
    83:	learn: 0.0608904	total: 28.5s	remaining: 39.4s
    84:	learn: 0.0608695	total: 28.9s	remaining: 39.1s
    85:	learn: 0.0608473	total: 29.3s	remaining: 38.8s
    86:	learn: 0.0608316	total: 29.6s	remaining: 38.4s
    87:	learn: 0.0608193	total: 29.9s	remaining: 38.1s
    88:	learn: 0.0608058	total: 30.3s	remaining: 37.8s
    89:	learn: 0.0607735	total: 30.7s	remaining: 37.5s
    90:	learn: 0.0607591	total: 31s	remaining: 37.1s
    91:	learn: 0.0607370	total: 31.4s	remaining: 36.8s
    92:	learn: 0.0607258	total: 31.7s	remaining: 36.5s
    93:	learn: 0.0607040	total: 32.1s	remaining: 36.2s
    94:	learn: 0.0606794	total: 32.4s	remaining: 35.9s
    95:	learn: 0.0606696	total: 32.8s	remaining: 35.5s
    96:	learn: 0.0606558	total: 33.1s	remaining: 35.2s
    97:	learn: 0.0606321	total: 33.5s	remaining: 34.9s
    98:	learn: 0.0606087	total: 33.8s	remaining: 34.5s
    99:	learn: 0.0606018	total: 34.2s	remaining: 34.2s
    100:	learn: 0.0605864	total: 34.6s	remaining: 33.9s
    101:	learn: 0.0605605	total: 34.9s	remaining: 33.5s
    102:	learn: 0.0605402	total: 35.3s	remaining: 33.2s
    103:	learn: 0.0605314	total: 35.6s	remaining: 32.9s
    104:	learn: 0.0605159	total: 36s	remaining: 32.5s
    105:	learn: 0.0604993	total: 36.3s	remaining: 32.2s
    106:	learn: 0.0604841	total: 36.6s	remaining: 31.8s
    107:	learn: 0.0604775	total: 37s	remaining: 31.5s
    108:	learn: 0.0604570	total: 37.4s	remaining: 31.2s
    109:	learn: 0.0604463	total: 37.7s	remaining: 30.9s
    110:	learn: 0.0604356	total: 38s	remaining: 30.5s
    111:	learn: 0.0604254	total: 38.3s	remaining: 30.1s
    112:	learn: 0.0604049	total: 38.7s	remaining: 29.8s
    113:	learn: 0.0603864	total: 39.1s	remaining: 29.5s
    114:	learn: 0.0603674	total: 39.5s	remaining: 29.2s
    115:	learn: 0.0603549	total: 39.8s	remaining: 28.8s
    116:	learn: 0.0603337	total: 40.2s	remaining: 28.5s
    117:	learn: 0.0603236	total: 40.6s	remaining: 28.2s
    118:	learn: 0.0603034	total: 40.9s	remaining: 27.8s
    119:	learn: 0.0602912	total: 41.2s	remaining: 27.5s
    120:	learn: 0.0602756	total: 41.6s	remaining: 27.2s
    121:	learn: 0.0602663	total: 42s	remaining: 26.8s
    122:	learn: 0.0602607	total: 42.3s	remaining: 26.5s
    123:	learn: 0.0602496	total: 42.7s	remaining: 26.2s
    124:	learn: 0.0602414	total: 43s	remaining: 25.8s
    125:	learn: 0.0602259	total: 43.3s	remaining: 25.4s
    126:	learn: 0.0602198	total: 43.7s	remaining: 25.1s
    127:	learn: 0.0602096	total: 44s	remaining: 24.8s
    128:	learn: 0.0602015	total: 44.4s	remaining: 24.4s
    129:	learn: 0.0601937	total: 44.8s	remaining: 24.1s
    130:	learn: 0.0601829	total: 45.1s	remaining: 23.8s
    131:	learn: 0.0601696	total: 45.4s	remaining: 23.4s
    132:	learn: 0.0601580	total: 45.8s	remaining: 23.1s
    133:	learn: 0.0601430	total: 46.1s	remaining: 22.7s
    134:	learn: 0.0601243	total: 46.5s	remaining: 22.4s
    135:	learn: 0.0601175	total: 46.8s	remaining: 22s
    136:	learn: 0.0601047	total: 47.1s	remaining: 21.7s
    137:	learn: 0.0600863	total: 47.5s	remaining: 21.3s
    138:	learn: 0.0600720	total: 47.8s	remaining: 21s
    139:	learn: 0.0600614	total: 48.1s	remaining: 20.6s
    140:	learn: 0.0600459	total: 48.4s	remaining: 20.3s
    141:	learn: 0.0600286	total: 48.8s	remaining: 19.9s
    142:	learn: 0.0600091	total: 49.2s	remaining: 19.6s
    143:	learn: 0.0599963	total: 49.5s	remaining: 19.3s
    144:	learn: 0.0599871	total: 49.9s	remaining: 18.9s
    145:	learn: 0.0599748	total: 50.2s	remaining: 18.6s
    146:	learn: 0.0599673	total: 50.6s	remaining: 18.2s
    147:	learn: 0.0599488	total: 50.9s	remaining: 17.9s
    148:	learn: 0.0599388	total: 51.3s	remaining: 17.6s
    149:	learn: 0.0599291	total: 51.7s	remaining: 17.2s
    150:	learn: 0.0599223	total: 52s	remaining: 16.9s
    151:	learn: 0.0599152	total: 52.3s	remaining: 16.5s
    152:	learn: 0.0599006	total: 52.7s	remaining: 16.2s
    153:	learn: 0.0598900	total: 53s	remaining: 15.8s
    154:	learn: 0.0598779	total: 53.3s	remaining: 15.5s
    155:	learn: 0.0598635	total: 53.7s	remaining: 15.1s
    156:	learn: 0.0598541	total: 54s	remaining: 14.8s
    157:	learn: 0.0598463	total: 54.4s	remaining: 14.5s
    158:	learn: 0.0598260	total: 54.7s	remaining: 14.1s
    159:	learn: 0.0598171	total: 55.1s	remaining: 13.8s
    160:	learn: 0.0598080	total: 55.4s	remaining: 13.4s
    161:	learn: 0.0598028	total: 55.7s	remaining: 13.1s
    162:	learn: 0.0597970	total: 56.1s	remaining: 12.7s
    163:	learn: 0.0597878	total: 56.4s	remaining: 12.4s
    164:	learn: 0.0597791	total: 56.8s	remaining: 12s
    165:	learn: 0.0597682	total: 57.1s	remaining: 11.7s
    166:	learn: 0.0597544	total: 57.4s	remaining: 11.4s
    167:	learn: 0.0597391	total: 57.8s	remaining: 11s
    168:	learn: 0.0597254	total: 58.2s	remaining: 10.7s
    169:	learn: 0.0597178	total: 58.5s	remaining: 10.3s
    170:	learn: 0.0597099	total: 58.8s	remaining: 9.98s
    171:	learn: 0.0596988	total: 59.2s	remaining: 9.63s
    172:	learn: 0.0596859	total: 59.5s	remaining: 9.29s
    173:	learn: 0.0596740	total: 59.9s	remaining: 8.95s
    174:	learn: 0.0596629	total: 1m	remaining: 8.6s
    175:	learn: 0.0596539	total: 1m	remaining: 8.25s
    176:	learn: 0.0596455	total: 1m	remaining: 7.91s
    177:	learn: 0.0596304	total: 1m 1s	remaining: 7.56s
    178:	learn: 0.0596246	total: 1m 1s	remaining: 7.21s
    179:	learn: 0.0596175	total: 1m 1s	remaining: 6.87s
    180:	learn: 0.0596102	total: 1m 2s	remaining: 6.53s
    181:	learn: 0.0596039	total: 1m 2s	remaining: 6.18s
    182:	learn: 0.0595957	total: 1m 2s	remaining: 5.84s
    183:	learn: 0.0595844	total: 1m 3s	remaining: 5.49s
    184:	learn: 0.0595656	total: 1m 3s	remaining: 5.15s
    185:	learn: 0.0595566	total: 1m 3s	remaining: 4.8s
    186:	learn: 0.0595371	total: 1m 4s	remaining: 4.46s
    187:	learn: 0.0595207	total: 1m 4s	remaining: 4.12s
    188:	learn: 0.0595147	total: 1m 4s	remaining: 3.77s
    189:	learn: 0.0595032	total: 1m 5s	remaining: 3.43s
    190:	learn: 0.0594918	total: 1m 5s	remaining: 3.09s
    191:	learn: 0.0594812	total: 1m 5s	remaining: 2.74s
    192:	learn: 0.0594741	total: 1m 6s	remaining: 2.4s
    193:	learn: 0.0594691	total: 1m 6s	remaining: 2.06s
    194:	learn: 0.0594611	total: 1m 6s	remaining: 1.72s
    195:	learn: 0.0594534	total: 1m 7s	remaining: 1.37s
    196:	learn: 0.0594436	total: 1m 7s	remaining: 1.03s
    197:	learn: 0.0594323	total: 1m 7s	remaining: 687ms
    198:	learn: 0.0594248	total: 1m 8s	remaining: 343ms
    199:	learn: 0.0594170	total: 1m 8s	remaining: 0us
    Trained model nº 17/27. Iterations: 200Depth: 10Learning rate: 0.05
    0:	learn: 0.5874604	total: 343ms	remaining: 1m 42s
    1:	learn: 0.4965186	total: 645ms	remaining: 1m 36s
    2:	learn: 0.4240445	total: 808ms	remaining: 1m 20s
    3:	learn: 0.3616264	total: 1.18s	remaining: 1m 27s
    4:	learn: 0.3178060	total: 1.29s	remaining: 1m 16s
    5:	learn: 0.2736807	total: 1.62s	remaining: 1m 19s
    6:	learn: 0.2375899	total: 1.84s	remaining: 1m 17s
    7:	learn: 0.2082300	total: 2.19s	remaining: 1m 19s
    8:	learn: 0.1856276	total: 2.51s	remaining: 1m 21s
    9:	learn: 0.1658431	total: 2.83s	remaining: 1m 22s
    10:	learn: 0.1497862	total: 3.15s	remaining: 1m 22s
    11:	learn: 0.1364801	total: 3.49s	remaining: 1m 23s
    12:	learn: 0.1255410	total: 3.84s	remaining: 1m 24s
    13:	learn: 0.1163558	total: 4.18s	remaining: 1m 25s
    14:	learn: 0.1082342	total: 4.52s	remaining: 1m 25s
    15:	learn: 0.1009861	total: 4.86s	remaining: 1m 26s
    16:	learn: 0.0957680	total: 5.17s	remaining: 1m 26s
    17:	learn: 0.0912677	total: 5.49s	remaining: 1m 26s
    18:	learn: 0.0874582	total: 5.83s	remaining: 1m 26s
    19:	learn: 0.0841861	total: 6.16s	remaining: 1m 26s
    20:	learn: 0.0814063	total: 6.47s	remaining: 1m 25s
    21:	learn: 0.0789782	total: 6.82s	remaining: 1m 26s
    22:	learn: 0.0769172	total: 7.14s	remaining: 1m 26s
    23:	learn: 0.0751318	total: 7.48s	remaining: 1m 26s
    24:	learn: 0.0735669	total: 7.79s	remaining: 1m 25s
    25:	learn: 0.0722194	total: 8.13s	remaining: 1m 25s
    26:	learn: 0.0710454	total: 8.46s	remaining: 1m 25s
    27:	learn: 0.0700026	total: 8.84s	remaining: 1m 25s
    28:	learn: 0.0690991	total: 9.18s	remaining: 1m 25s
    29:	learn: 0.0683152	total: 9.52s	remaining: 1m 25s
    30:	learn: 0.0675418	total: 9.87s	remaining: 1m 25s
    31:	learn: 0.0669189	total: 10.2s	remaining: 1m 25s
    32:	learn: 0.0663822	total: 10.5s	remaining: 1m 24s
    33:	learn: 0.0657086	total: 10.8s	remaining: 1m 24s
    34:	learn: 0.0652995	total: 11.1s	remaining: 1m 24s
    35:	learn: 0.0649154	total: 11.5s	remaining: 1m 24s
    36:	learn: 0.0645838	total: 11.8s	remaining: 1m 23s
    37:	learn: 0.0642922	total: 12.1s	remaining: 1m 23s
    38:	learn: 0.0640224	total: 12.5s	remaining: 1m 23s
    39:	learn: 0.0637963	total: 12.8s	remaining: 1m 23s
    40:	learn: 0.0635757	total: 13.2s	remaining: 1m 23s
    41:	learn: 0.0633810	total: 13.5s	remaining: 1m 22s
    42:	learn: 0.0632043	total: 13.8s	remaining: 1m 22s
    43:	learn: 0.0630526	total: 14.2s	remaining: 1m 22s
    44:	learn: 0.0628748	total: 14.5s	remaining: 1m 22s
    45:	learn: 0.0627533	total: 14.8s	remaining: 1m 21s
    46:	learn: 0.0626465	total: 15.2s	remaining: 1m 21s
    47:	learn: 0.0625420	total: 15.5s	remaining: 1m 21s
    48:	learn: 0.0623633	total: 15.8s	remaining: 1m 21s
    49:	learn: 0.0622635	total: 16.2s	remaining: 1m 20s
    50:	learn: 0.0621766	total: 16.5s	remaining: 1m 20s
    51:	learn: 0.0621133	total: 16.8s	remaining: 1m 20s
    52:	learn: 0.0620519	total: 17.1s	remaining: 1m 19s
    53:	learn: 0.0619690	total: 17.5s	remaining: 1m 19s
    54:	learn: 0.0618969	total: 17.8s	remaining: 1m 19s
    55:	learn: 0.0618450	total: 18.1s	remaining: 1m 19s
    56:	learn: 0.0617809	total: 18.5s	remaining: 1m 18s
    57:	learn: 0.0617369	total: 18.8s	remaining: 1m 18s
    58:	learn: 0.0616869	total: 19.1s	remaining: 1m 18s
    59:	learn: 0.0616363	total: 19.5s	remaining: 1m 17s
    60:	learn: 0.0615971	total: 19.8s	remaining: 1m 17s
    61:	learn: 0.0615615	total: 20.1s	remaining: 1m 17s
    62:	learn: 0.0615171	total: 20.4s	remaining: 1m 16s
    63:	learn: 0.0614273	total: 20.8s	remaining: 1m 16s
    64:	learn: 0.0613947	total: 21.1s	remaining: 1m 16s
    65:	learn: 0.0613588	total: 21.5s	remaining: 1m 16s
    66:	learn: 0.0613326	total: 21.8s	remaining: 1m 15s
    67:	learn: 0.0613049	total: 22.2s	remaining: 1m 15s
    68:	learn: 0.0612809	total: 22.5s	remaining: 1m 15s
    69:	learn: 0.0612559	total: 22.8s	remaining: 1m 15s
    70:	learn: 0.0612317	total: 23.2s	remaining: 1m 14s
    71:	learn: 0.0612105	total: 23.6s	remaining: 1m 14s
    72:	learn: 0.0611906	total: 23.9s	remaining: 1m 14s
    73:	learn: 0.0611661	total: 24.3s	remaining: 1m 14s
    74:	learn: 0.0611414	total: 24.6s	remaining: 1m 13s
    75:	learn: 0.0611152	total: 25s	remaining: 1m 13s
    76:	learn: 0.0610918	total: 25.4s	remaining: 1m 13s
    77:	learn: 0.0610532	total: 25.7s	remaining: 1m 13s
    78:	learn: 0.0610114	total: 26.1s	remaining: 1m 12s
    79:	learn: 0.0609903	total: 26.4s	remaining: 1m 12s
    80:	learn: 0.0609659	total: 26.7s	remaining: 1m 12s
    81:	learn: 0.0609423	total: 27.1s	remaining: 1m 12s
    82:	learn: 0.0609020	total: 27.4s	remaining: 1m 11s
    83:	learn: 0.0608904	total: 27.8s	remaining: 1m 11s
    84:	learn: 0.0608695	total: 28.2s	remaining: 1m 11s
    85:	learn: 0.0608473	total: 28.6s	remaining: 1m 11s
    86:	learn: 0.0608316	total: 28.9s	remaining: 1m 10s
    87:	learn: 0.0608193	total: 29.3s	remaining: 1m 10s
    88:	learn: 0.0608058	total: 29.7s	remaining: 1m 10s
    89:	learn: 0.0607735	total: 30.1s	remaining: 1m 10s
    90:	learn: 0.0607591	total: 30.5s	remaining: 1m 9s
    91:	learn: 0.0607370	total: 30.8s	remaining: 1m 9s
    92:	learn: 0.0607258	total: 31.2s	remaining: 1m 9s
    93:	learn: 0.0607040	total: 31.6s	remaining: 1m 9s
    94:	learn: 0.0606794	total: 32s	remaining: 1m 8s
    95:	learn: 0.0606696	total: 32.4s	remaining: 1m 8s
    96:	learn: 0.0606558	total: 32.7s	remaining: 1m 8s
    97:	learn: 0.0606321	total: 33.1s	remaining: 1m 8s
    98:	learn: 0.0606087	total: 33.4s	remaining: 1m 7s
    99:	learn: 0.0606018	total: 33.8s	remaining: 1m 7s
    100:	learn: 0.0605864	total: 34.2s	remaining: 1m 7s
    101:	learn: 0.0605605	total: 34.6s	remaining: 1m 7s
    102:	learn: 0.0605402	total: 35s	remaining: 1m 6s
    103:	learn: 0.0605314	total: 35.3s	remaining: 1m 6s
    104:	learn: 0.0605159	total: 35.7s	remaining: 1m 6s
    105:	learn: 0.0604993	total: 36.1s	remaining: 1m 6s
    106:	learn: 0.0604841	total: 36.4s	remaining: 1m 5s
    107:	learn: 0.0604775	total: 36.8s	remaining: 1m 5s
    108:	learn: 0.0604570	total: 37.2s	remaining: 1m 5s
    109:	learn: 0.0604463	total: 37.5s	remaining: 1m 4s
    110:	learn: 0.0604356	total: 37.9s	remaining: 1m 4s
    111:	learn: 0.0604254	total: 38.2s	remaining: 1m 4s
    112:	learn: 0.0604049	total: 38.5s	remaining: 1m 3s
    113:	learn: 0.0603864	total: 38.9s	remaining: 1m 3s
    114:	learn: 0.0603674	total: 39.3s	remaining: 1m 3s
    115:	learn: 0.0603549	total: 39.7s	remaining: 1m 2s
    116:	learn: 0.0603337	total: 40s	remaining: 1m 2s
    117:	learn: 0.0603236	total: 40.4s	remaining: 1m 2s
    118:	learn: 0.0603034	total: 40.7s	remaining: 1m 1s
    119:	learn: 0.0602912	total: 41.1s	remaining: 1m 1s
    120:	learn: 0.0602756	total: 41.5s	remaining: 1m 1s
    121:	learn: 0.0602663	total: 41.8s	remaining: 1m 1s
    122:	learn: 0.0602607	total: 42.2s	remaining: 1m
    123:	learn: 0.0602496	total: 42.6s	remaining: 1m
    124:	learn: 0.0602414	total: 42.9s	remaining: 1m
    125:	learn: 0.0602259	total: 43.3s	remaining: 59.8s
    126:	learn: 0.0602198	total: 43.7s	remaining: 59.5s
    127:	learn: 0.0602096	total: 44.1s	remaining: 59.2s
    128:	learn: 0.0602015	total: 44.5s	remaining: 59s
    129:	learn: 0.0601937	total: 44.9s	remaining: 58.7s
    130:	learn: 0.0601829	total: 45.3s	remaining: 58.4s
    131:	learn: 0.0601696	total: 45.6s	remaining: 58.1s
    132:	learn: 0.0601580	total: 46s	remaining: 57.7s
    133:	learn: 0.0601430	total: 46.3s	remaining: 57.4s
    134:	learn: 0.0601243	total: 46.7s	remaining: 57.1s
    135:	learn: 0.0601175	total: 47s	remaining: 56.7s
    136:	learn: 0.0601047	total: 47.4s	remaining: 56.4s
    137:	learn: 0.0600863	total: 47.8s	remaining: 56.1s
    138:	learn: 0.0600720	total: 48.1s	remaining: 55.7s
    139:	learn: 0.0600614	total: 48.4s	remaining: 55.3s
    140:	learn: 0.0600459	total: 48.7s	remaining: 54.9s
    141:	learn: 0.0600286	total: 49.2s	remaining: 54.7s
    142:	learn: 0.0600091	total: 49.6s	remaining: 54.4s
    143:	learn: 0.0599963	total: 50s	remaining: 54.1s
    144:	learn: 0.0599871	total: 50.3s	remaining: 53.8s
    145:	learn: 0.0599748	total: 50.7s	remaining: 53.5s
    146:	learn: 0.0599673	total: 51.1s	remaining: 53.2s
    147:	learn: 0.0599488	total: 51.5s	remaining: 52.9s
    148:	learn: 0.0599388	total: 51.8s	remaining: 52.5s
    149:	learn: 0.0599291	total: 52.2s	remaining: 52.2s
    150:	learn: 0.0599223	total: 52.6s	remaining: 51.9s
    151:	learn: 0.0599152	total: 52.9s	remaining: 51.6s
    152:	learn: 0.0599006	total: 53.3s	remaining: 51.2s
    153:	learn: 0.0598900	total: 53.7s	remaining: 50.9s
    154:	learn: 0.0598779	total: 54s	remaining: 50.5s
    155:	learn: 0.0598635	total: 54.4s	remaining: 50.2s
    156:	learn: 0.0598541	total: 54.7s	remaining: 49.8s
    157:	learn: 0.0598463	total: 55.1s	remaining: 49.5s
    158:	learn: 0.0598260	total: 55.4s	remaining: 49.1s
    159:	learn: 0.0598171	total: 55.8s	remaining: 48.8s
    160:	learn: 0.0598080	total: 56.1s	remaining: 48.5s
    161:	learn: 0.0598028	total: 56.5s	remaining: 48.1s
    162:	learn: 0.0597970	total: 56.8s	remaining: 47.8s
    163:	learn: 0.0597878	total: 57.2s	remaining: 47.4s
    164:	learn: 0.0597791	total: 57.6s	remaining: 47.1s
    165:	learn: 0.0597682	total: 57.9s	remaining: 46.8s
    166:	learn: 0.0597544	total: 58.3s	remaining: 46.4s
    167:	learn: 0.0597391	total: 58.6s	remaining: 46.1s
    168:	learn: 0.0597254	total: 59s	remaining: 45.7s
    169:	learn: 0.0597178	total: 59.3s	remaining: 45.4s
    170:	learn: 0.0597099	total: 59.7s	remaining: 45s
    171:	learn: 0.0596988	total: 1m	remaining: 44.7s
    172:	learn: 0.0596859	total: 1m	remaining: 44.3s
    173:	learn: 0.0596740	total: 1m	remaining: 44s
    174:	learn: 0.0596629	total: 1m 1s	remaining: 43.6s
    175:	learn: 0.0596539	total: 1m 1s	remaining: 43.3s
    176:	learn: 0.0596455	total: 1m 1s	remaining: 42.9s
    177:	learn: 0.0596304	total: 1m 2s	remaining: 42.6s
    178:	learn: 0.0596246	total: 1m 2s	remaining: 42.3s
    179:	learn: 0.0596175	total: 1m 2s	remaining: 41.9s
    180:	learn: 0.0596102	total: 1m 3s	remaining: 41.6s
    181:	learn: 0.0596039	total: 1m 3s	remaining: 41.3s
    182:	learn: 0.0595957	total: 1m 4s	remaining: 41s
    183:	learn: 0.0595844	total: 1m 4s	remaining: 40.6s
    184:	learn: 0.0595656	total: 1m 4s	remaining: 40.3s
    185:	learn: 0.0595566	total: 1m 5s	remaining: 39.9s
    186:	learn: 0.0595371	total: 1m 5s	remaining: 39.6s
    187:	learn: 0.0595207	total: 1m 5s	remaining: 39.3s
    188:	learn: 0.0595147	total: 1m 6s	remaining: 38.9s
    189:	learn: 0.0595032	total: 1m 6s	remaining: 38.6s
    190:	learn: 0.0594918	total: 1m 6s	remaining: 38.2s
    191:	learn: 0.0594812	total: 1m 7s	remaining: 37.9s
    192:	learn: 0.0594741	total: 1m 7s	remaining: 37.5s
    193:	learn: 0.0594691	total: 1m 8s	remaining: 37.2s
    194:	learn: 0.0594611	total: 1m 8s	remaining: 36.9s
    195:	learn: 0.0594534	total: 1m 8s	remaining: 36.5s
    196:	learn: 0.0594436	total: 1m 9s	remaining: 36.2s
    197:	learn: 0.0594323	total: 1m 9s	remaining: 35.8s
    198:	learn: 0.0594248	total: 1m 9s	remaining: 35.5s
    199:	learn: 0.0594170	total: 1m 10s	remaining: 35.1s
    200:	learn: 0.0594135	total: 1m 10s	remaining: 34.8s
    201:	learn: 0.0594028	total: 1m 10s	remaining: 34.4s
    202:	learn: 0.0593941	total: 1m 11s	remaining: 34.1s
    203:	learn: 0.0593853	total: 1m 11s	remaining: 33.7s
    204:	learn: 0.0593808	total: 1m 11s	remaining: 33.4s
    205:	learn: 0.0593719	total: 1m 12s	remaining: 33s
    206:	learn: 0.0593650	total: 1m 12s	remaining: 32.7s
    207:	learn: 0.0593550	total: 1m 13s	remaining: 32.3s
    208:	learn: 0.0593491	total: 1m 13s	remaining: 32s
    209:	learn: 0.0593423	total: 1m 13s	remaining: 31.6s
    210:	learn: 0.0593342	total: 1m 14s	remaining: 31.2s
    211:	learn: 0.0593308	total: 1m 14s	remaining: 30.9s
    212:	learn: 0.0593191	total: 1m 14s	remaining: 30.5s
    213:	learn: 0.0593140	total: 1m 15s	remaining: 30.2s
    214:	learn: 0.0593075	total: 1m 15s	remaining: 29.8s
    215:	learn: 0.0593039	total: 1m 15s	remaining: 29.5s
    216:	learn: 0.0592957	total: 1m 16s	remaining: 29.1s
    217:	learn: 0.0592891	total: 1m 16s	remaining: 28.8s
    218:	learn: 0.0592848	total: 1m 16s	remaining: 28.4s
    219:	learn: 0.0592743	total: 1m 17s	remaining: 28s
    220:	learn: 0.0592642	total: 1m 17s	remaining: 27.7s
    221:	learn: 0.0592639	total: 1m 17s	remaining: 27.3s
    222:	learn: 0.0592585	total: 1m 18s	remaining: 26.9s
    223:	learn: 0.0592549	total: 1m 18s	remaining: 26.6s
    224:	learn: 0.0592454	total: 1m 18s	remaining: 26.2s
    225:	learn: 0.0592419	total: 1m 19s	remaining: 25.9s
    226:	learn: 0.0592328	total: 1m 19s	remaining: 25.5s
    227:	learn: 0.0592246	total: 1m 19s	remaining: 25.2s
    228:	learn: 0.0592203	total: 1m 20s	remaining: 24.8s
    229:	learn: 0.0592145	total: 1m 20s	remaining: 24.5s
    230:	learn: 0.0592059	total: 1m 20s	remaining: 24.1s
    231:	learn: 0.0591987	total: 1m 21s	remaining: 23.8s
    232:	learn: 0.0591887	total: 1m 21s	remaining: 23.4s
    233:	learn: 0.0591843	total: 1m 21s	remaining: 23.1s
    234:	learn: 0.0591742	total: 1m 22s	remaining: 22.7s
    235:	learn: 0.0591614	total: 1m 22s	remaining: 22.4s
    236:	learn: 0.0591557	total: 1m 22s	remaining: 22s
    237:	learn: 0.0591488	total: 1m 23s	remaining: 21.6s
    238:	learn: 0.0591422	total: 1m 23s	remaining: 21.3s
    239:	learn: 0.0591339	total: 1m 23s	remaining: 20.9s
    240:	learn: 0.0591244	total: 1m 24s	remaining: 20.6s
    241:	learn: 0.0591156	total: 1m 24s	remaining: 20.2s
    242:	learn: 0.0591122	total: 1m 24s	remaining: 19.9s
    243:	learn: 0.0591085	total: 1m 25s	remaining: 19.5s
    244:	learn: 0.0591015	total: 1m 25s	remaining: 19.2s
    245:	learn: 0.0590949	total: 1m 25s	remaining: 18.8s
    246:	learn: 0.0590923	total: 1m 26s	remaining: 18.5s
    247:	learn: 0.0590801	total: 1m 26s	remaining: 18.1s
    248:	learn: 0.0590707	total: 1m 26s	remaining: 17.8s
    249:	learn: 0.0590638	total: 1m 27s	remaining: 17.4s
    250:	learn: 0.0590561	total: 1m 27s	remaining: 17.1s
    251:	learn: 0.0590513	total: 1m 27s	remaining: 16.7s
    252:	learn: 0.0590470	total: 1m 28s	remaining: 16.4s
    253:	learn: 0.0590422	total: 1m 28s	remaining: 16s
    254:	learn: 0.0590332	total: 1m 28s	remaining: 15.7s
    255:	learn: 0.0590246	total: 1m 29s	remaining: 15.3s
    256:	learn: 0.0590150	total: 1m 29s	remaining: 15s
    257:	learn: 0.0590083	total: 1m 29s	remaining: 14.6s
    258:	learn: 0.0590043	total: 1m 30s	remaining: 14.3s
    259:	learn: 0.0589990	total: 1m 30s	remaining: 13.9s
    260:	learn: 0.0589960	total: 1m 30s	remaining: 13.6s
    261:	learn: 0.0589873	total: 1m 31s	remaining: 13.2s
    262:	learn: 0.0589747	total: 1m 31s	remaining: 12.9s
    263:	learn: 0.0589610	total: 1m 31s	remaining: 12.5s
    264:	learn: 0.0589447	total: 1m 32s	remaining: 12.2s
    265:	learn: 0.0589384	total: 1m 32s	remaining: 11.9s
    266:	learn: 0.0589352	total: 1m 33s	remaining: 11.5s
    267:	learn: 0.0589247	total: 1m 33s	remaining: 11.2s
    268:	learn: 0.0589184	total: 1m 33s	remaining: 10.8s
    269:	learn: 0.0589085	total: 1m 34s	remaining: 10.5s
    270:	learn: 0.0589047	total: 1m 34s	remaining: 10.1s
    271:	learn: 0.0588978	total: 1m 34s	remaining: 9.76s
    272:	learn: 0.0588914	total: 1m 35s	remaining: 9.4s
    273:	learn: 0.0588851	total: 1m 35s	remaining: 9.06s
    274:	learn: 0.0588781	total: 1m 35s	remaining: 8.71s
    275:	learn: 0.0588720	total: 1m 36s	remaining: 8.36s
    276:	learn: 0.0588690	total: 1m 36s	remaining: 8.01s
    277:	learn: 0.0588638	total: 1m 36s	remaining: 7.66s
    278:	learn: 0.0588554	total: 1m 37s	remaining: 7.32s
    279:	learn: 0.0588479	total: 1m 37s	remaining: 6.97s
    280:	learn: 0.0588430	total: 1m 37s	remaining: 6.62s
    281:	learn: 0.0588384	total: 1m 38s	remaining: 6.26s
    282:	learn: 0.0588363	total: 1m 38s	remaining: 5.92s
    283:	learn: 0.0588307	total: 1m 38s	remaining: 5.57s
    284:	learn: 0.0588260	total: 1m 39s	remaining: 5.22s
    285:	learn: 0.0588191	total: 1m 39s	remaining: 4.87s
    286:	learn: 0.0588125	total: 1m 39s	remaining: 4.53s
    287:	learn: 0.0588040	total: 1m 40s	remaining: 4.18s
    288:	learn: 0.0587977	total: 1m 40s	remaining: 3.83s
    289:	learn: 0.0587873	total: 1m 40s	remaining: 3.48s
    290:	learn: 0.0587829	total: 1m 41s	remaining: 3.13s
    291:	learn: 0.0587728	total: 1m 41s	remaining: 2.79s
    292:	learn: 0.0587594	total: 1m 42s	remaining: 2.44s
    293:	learn: 0.0587501	total: 1m 42s	remaining: 2.09s
    294:	learn: 0.0587425	total: 1m 42s	remaining: 1.74s
    295:	learn: 0.0587285	total: 1m 43s	remaining: 1.39s
    296:	learn: 0.0587233	total: 1m 43s	remaining: 1.04s
    297:	learn: 0.0587154	total: 1m 43s	remaining: 696ms
    298:	learn: 0.0587112	total: 1m 44s	remaining: 348ms
    299:	learn: 0.0587036	total: 1m 44s	remaining: 0us
    Trained model nº 18/27. Iterations: 300Depth: 10Learning rate: 0.05
    0:	learn: 0.5062796	total: 169ms	remaining: 16.7s
    1:	learn: 0.3635625	total: 322ms	remaining: 15.8s
    2:	learn: 0.2790112	total: 482ms	remaining: 15.6s
    3:	learn: 0.2163357	total: 645ms	remaining: 15.5s
    4:	learn: 0.1761711	total: 810ms	remaining: 15.4s
    5:	learn: 0.1453769	total: 986ms	remaining: 15.4s
    6:	learn: 0.1224737	total: 1.16s	remaining: 15.4s
    7:	learn: 0.1075807	total: 1.28s	remaining: 14.8s
    8:	learn: 0.0948237	total: 1.44s	remaining: 14.6s
    9:	learn: 0.0863867	total: 1.59s	remaining: 14.4s
    10:	learn: 0.0816316	total: 1.75s	remaining: 14.2s
    11:	learn: 0.0770182	total: 1.91s	remaining: 14s
    12:	learn: 0.0744387	total: 2.06s	remaining: 13.8s
    13:	learn: 0.0715663	total: 2.22s	remaining: 13.6s
    14:	learn: 0.0698181	total: 2.37s	remaining: 13.4s
    15:	learn: 0.0683959	total: 2.53s	remaining: 13.3s
    16:	learn: 0.0672321	total: 2.68s	remaining: 13.1s
    17:	learn: 0.0664944	total: 2.84s	remaining: 12.9s
    18:	learn: 0.0657608	total: 2.99s	remaining: 12.8s
    19:	learn: 0.0651507	total: 3.15s	remaining: 12.6s
    20:	learn: 0.0645686	total: 3.31s	remaining: 12.4s
    21:	learn: 0.0639546	total: 3.47s	remaining: 12.3s
    22:	learn: 0.0637070	total: 3.62s	remaining: 12.1s
    23:	learn: 0.0634488	total: 3.78s	remaining: 12s
    24:	learn: 0.0633218	total: 3.93s	remaining: 11.8s
    25:	learn: 0.0630896	total: 4.09s	remaining: 11.6s
    26:	learn: 0.0629238	total: 4.24s	remaining: 11.5s
    27:	learn: 0.0628407	total: 4.4s	remaining: 11.3s
    28:	learn: 0.0627308	total: 4.55s	remaining: 11.1s
    29:	learn: 0.0626335	total: 4.71s	remaining: 11s
    30:	learn: 0.0625828	total: 4.87s	remaining: 10.8s
    31:	learn: 0.0624641	total: 5.02s	remaining: 10.7s
    32:	learn: 0.0623752	total: 5.18s	remaining: 10.5s
    33:	learn: 0.0623402	total: 5.35s	remaining: 10.4s
    34:	learn: 0.0622707	total: 5.5s	remaining: 10.2s
    35:	learn: 0.0622026	total: 5.66s	remaining: 10.1s
    36:	learn: 0.0620094	total: 5.82s	remaining: 9.92s
    37:	learn: 0.0619877	total: 5.98s	remaining: 9.76s
    38:	learn: 0.0619638	total: 6.15s	remaining: 9.62s
    39:	learn: 0.0619420	total: 6.3s	remaining: 9.45s
    40:	learn: 0.0619036	total: 6.46s	remaining: 9.3s
    41:	learn: 0.0618839	total: 6.63s	remaining: 9.16s
    42:	learn: 0.0618435	total: 6.79s	remaining: 9s
    43:	learn: 0.0618083	total: 6.95s	remaining: 8.84s
    44:	learn: 0.0617900	total: 7.11s	remaining: 8.69s
    45:	learn: 0.0617716	total: 7.27s	remaining: 8.53s
    46:	learn: 0.0617418	total: 7.43s	remaining: 8.38s
    47:	learn: 0.0617109	total: 7.6s	remaining: 8.24s
    48:	learn: 0.0617002	total: 7.78s	remaining: 8.1s
    49:	learn: 0.0616810	total: 7.95s	remaining: 7.95s
    50:	learn: 0.0616650	total: 8.11s	remaining: 7.79s
    51:	learn: 0.0616485	total: 8.27s	remaining: 7.63s
    52:	learn: 0.0615890	total: 8.43s	remaining: 7.47s
    53:	learn: 0.0615764	total: 8.58s	remaining: 7.31s
    54:	learn: 0.0615659	total: 8.75s	remaining: 7.16s
    55:	learn: 0.0615510	total: 8.91s	remaining: 7s
    56:	learn: 0.0615319	total: 9.08s	remaining: 6.85s
    57:	learn: 0.0615208	total: 9.24s	remaining: 6.69s
    58:	learn: 0.0615088	total: 9.39s	remaining: 6.53s
    59:	learn: 0.0615031	total: 9.55s	remaining: 6.37s
    60:	learn: 0.0614894	total: 9.72s	remaining: 6.21s
    61:	learn: 0.0614729	total: 9.88s	remaining: 6.05s
    62:	learn: 0.0614620	total: 10s	remaining: 5.89s
    63:	learn: 0.0614490	total: 10.2s	remaining: 5.73s
    64:	learn: 0.0614375	total: 10.3s	remaining: 5.57s
    65:	learn: 0.0614238	total: 10.5s	remaining: 5.42s
    66:	learn: 0.0614101	total: 10.7s	remaining: 5.25s
    67:	learn: 0.0614003	total: 10.8s	remaining: 5.09s
    68:	learn: 0.0613556	total: 11s	remaining: 4.94s
    69:	learn: 0.0613428	total: 11.2s	remaining: 4.78s
    70:	learn: 0.0613369	total: 11.3s	remaining: 4.62s
    71:	learn: 0.0613251	total: 11.5s	remaining: 4.46s
    72:	learn: 0.0613130	total: 11.6s	remaining: 4.3s
    73:	learn: 0.0613052	total: 11.8s	remaining: 4.14s
    74:	learn: 0.0612568	total: 11.9s	remaining: 3.98s
    75:	learn: 0.0612366	total: 12.1s	remaining: 3.81s
    76:	learn: 0.0612225	total: 12.2s	remaining: 3.65s
    77:	learn: 0.0612143	total: 12.4s	remaining: 3.49s
    78:	learn: 0.0612062	total: 12.5s	remaining: 3.33s
    79:	learn: 0.0611914	total: 12.7s	remaining: 3.17s
    80:	learn: 0.0611826	total: 12.9s	remaining: 3.02s
    81:	learn: 0.0611756	total: 13s	remaining: 2.86s
    82:	learn: 0.0611680	total: 13.2s	remaining: 2.7s
    83:	learn: 0.0611612	total: 13.3s	remaining: 2.54s
    84:	learn: 0.0611543	total: 13.5s	remaining: 2.38s
    85:	learn: 0.0611450	total: 13.6s	remaining: 2.22s
    86:	learn: 0.0611375	total: 13.8s	remaining: 2.06s
    87:	learn: 0.0611304	total: 13.9s	remaining: 1.9s
    88:	learn: 0.0611246	total: 14.1s	remaining: 1.74s
    89:	learn: 0.0611194	total: 14.2s	remaining: 1.58s
    90:	learn: 0.0611165	total: 14.4s	remaining: 1.43s
    91:	learn: 0.0611102	total: 14.6s	remaining: 1.27s
    92:	learn: 0.0611029	total: 14.8s	remaining: 1.11s
    93:	learn: 0.0610907	total: 14.9s	remaining: 954ms
    94:	learn: 0.0610806	total: 15.1s	remaining: 795ms
    95:	learn: 0.0610769	total: 15.3s	remaining: 636ms
    96:	learn: 0.0610705	total: 15.4s	remaining: 477ms
    97:	learn: 0.0610662	total: 15.6s	remaining: 318ms
    98:	learn: 0.0610606	total: 15.7s	remaining: 159ms
    99:	learn: 0.0610455	total: 15.9s	remaining: 0us
    Trained model nº 19/27. Iterations: 100Depth: 6Learning rate: 0.1
    0:	learn: 0.5042661	total: 214ms	remaining: 42.7s
    1:	learn: 0.3693048	total: 432ms	remaining: 42.7s
    2:	learn: 0.2745350	total: 636ms	remaining: 41.8s
    3:	learn: 0.2165385	total: 866ms	remaining: 42.4s
    4:	learn: 0.1729344	total: 1.09s	remaining: 42.7s
    5:	learn: 0.1436341	total: 1.33s	remaining: 42.9s
    6:	learn: 0.1248890	total: 1.54s	remaining: 42.5s
    7:	learn: 0.1087769	total: 1.77s	remaining: 42.4s
    8:	learn: 0.0953987	total: 2s	remaining: 42.5s
    9:	learn: 0.0879656	total: 2.23s	remaining: 42.3s
    10:	learn: 0.0831718	total: 2.45s	remaining: 42.1s
    11:	learn: 0.0780844	total: 2.67s	remaining: 41.8s
    12:	learn: 0.0756458	total: 2.83s	remaining: 40.7s
    13:	learn: 0.0724599	total: 3.06s	remaining: 40.6s
    14:	learn: 0.0706491	total: 3.28s	remaining: 40.4s
    15:	learn: 0.0684151	total: 3.51s	remaining: 40.3s
    16:	learn: 0.0672456	total: 3.72s	remaining: 40.1s
    17:	learn: 0.0665162	total: 3.94s	remaining: 39.8s
    18:	learn: 0.0657933	total: 4.15s	remaining: 39.6s
    19:	learn: 0.0652833	total: 4.37s	remaining: 39.3s
    20:	learn: 0.0647583	total: 4.6s	remaining: 39.2s
    21:	learn: 0.0642916	total: 4.81s	remaining: 38.9s
    22:	learn: 0.0640125	total: 5.04s	remaining: 38.8s
    23:	learn: 0.0636357	total: 5.25s	remaining: 38.5s
    24:	learn: 0.0633818	total: 5.48s	remaining: 38.4s
    25:	learn: 0.0632414	total: 5.71s	remaining: 38.3s
    26:	learn: 0.0631142	total: 5.96s	remaining: 38.2s
    27:	learn: 0.0630314	total: 6.16s	remaining: 37.8s
    28:	learn: 0.0629191	total: 6.39s	remaining: 37.7s
    29:	learn: 0.0627590	total: 6.63s	remaining: 37.6s
    30:	learn: 0.0626677	total: 6.87s	remaining: 37.4s
    31:	learn: 0.0625806	total: 7.11s	remaining: 37.3s
    32:	learn: 0.0624836	total: 7.33s	remaining: 37.1s
    33:	learn: 0.0623721	total: 7.56s	remaining: 36.9s
    34:	learn: 0.0623383	total: 7.78s	remaining: 36.7s
    35:	learn: 0.0621072	total: 8s	remaining: 36.4s
    36:	learn: 0.0620779	total: 8.22s	remaining: 36.2s
    37:	learn: 0.0620555	total: 8.41s	remaining: 35.9s
    38:	learn: 0.0620347	total: 8.64s	remaining: 35.7s
    39:	learn: 0.0620030	total: 8.85s	remaining: 35.4s
    40:	learn: 0.0619325	total: 9.07s	remaining: 35.2s
    41:	learn: 0.0619029	total: 9.3s	remaining: 35s
    42:	learn: 0.0618823	total: 9.53s	remaining: 34.8s
    43:	learn: 0.0618631	total: 9.78s	remaining: 34.7s
    44:	learn: 0.0618031	total: 9.98s	remaining: 34.4s
    45:	learn: 0.0617623	total: 10.2s	remaining: 34.1s
    46:	learn: 0.0617490	total: 10.4s	remaining: 33.9s
    47:	learn: 0.0617014	total: 10.6s	remaining: 33.6s
    48:	learn: 0.0616768	total: 10.9s	remaining: 33.4s
    49:	learn: 0.0616601	total: 11.1s	remaining: 33.2s
    50:	learn: 0.0616494	total: 11.3s	remaining: 33s
    51:	learn: 0.0616232	total: 11.5s	remaining: 32.7s
    52:	learn: 0.0615942	total: 11.7s	remaining: 32.4s
    53:	learn: 0.0615838	total: 11.9s	remaining: 32.2s
    54:	learn: 0.0615535	total: 12.1s	remaining: 32s
    55:	learn: 0.0615367	total: 12.4s	remaining: 31.8s
    56:	learn: 0.0615232	total: 12.6s	remaining: 31.6s
    57:	learn: 0.0615093	total: 12.8s	remaining: 31.3s
    58:	learn: 0.0615005	total: 13s	remaining: 31.1s
    59:	learn: 0.0614651	total: 13.3s	remaining: 31s
    60:	learn: 0.0614521	total: 13.5s	remaining: 30.8s
    61:	learn: 0.0614391	total: 13.8s	remaining: 30.6s
    62:	learn: 0.0614235	total: 13.9s	remaining: 30.3s
    63:	learn: 0.0614135	total: 14.1s	remaining: 30.1s
    64:	learn: 0.0614040	total: 14.4s	remaining: 29.9s
    65:	learn: 0.0613917	total: 14.6s	remaining: 29.7s
    66:	learn: 0.0613834	total: 14.9s	remaining: 29.5s
    67:	learn: 0.0613147	total: 15.1s	remaining: 29.3s
    68:	learn: 0.0613036	total: 15.3s	remaining: 29.1s
    69:	learn: 0.0612914	total: 15.6s	remaining: 28.9s
    70:	learn: 0.0612849	total: 15.8s	remaining: 28.7s
    71:	learn: 0.0612719	total: 16s	remaining: 28.4s
    72:	learn: 0.0612660	total: 16.2s	remaining: 28.2s
    73:	learn: 0.0612587	total: 16.5s	remaining: 28s
    74:	learn: 0.0612483	total: 16.7s	remaining: 27.8s
    75:	learn: 0.0612385	total: 16.9s	remaining: 27.6s
    76:	learn: 0.0612274	total: 17.1s	remaining: 27.4s
    77:	learn: 0.0612214	total: 17.3s	remaining: 27.1s
    78:	learn: 0.0612123	total: 17.6s	remaining: 26.9s
    79:	learn: 0.0612006	total: 17.8s	remaining: 26.7s
    80:	learn: 0.0611906	total: 18s	remaining: 26.5s
    81:	learn: 0.0611830	total: 18.2s	remaining: 26.2s
    82:	learn: 0.0611747	total: 18.5s	remaining: 26s
    83:	learn: 0.0611669	total: 18.7s	remaining: 25.8s
    84:	learn: 0.0611638	total: 18.9s	remaining: 25.6s
    85:	learn: 0.0611591	total: 19.2s	remaining: 25.4s
    86:	learn: 0.0611551	total: 19.4s	remaining: 25.2s
    87:	learn: 0.0611512	total: 19.6s	remaining: 25s
    88:	learn: 0.0611435	total: 19.9s	remaining: 24.8s
    89:	learn: 0.0611302	total: 20.1s	remaining: 24.6s
    90:	learn: 0.0611182	total: 20.3s	remaining: 24.3s
    91:	learn: 0.0611143	total: 20.5s	remaining: 24.1s
    92:	learn: 0.0611018	total: 20.8s	remaining: 23.9s
    93:	learn: 0.0610943	total: 21s	remaining: 23.7s
    94:	learn: 0.0610866	total: 21.3s	remaining: 23.5s
    95:	learn: 0.0610819	total: 21.5s	remaining: 23.3s
    96:	learn: 0.0610762	total: 21.7s	remaining: 23s
    97:	learn: 0.0610688	total: 21.9s	remaining: 22.8s
    98:	learn: 0.0610653	total: 22.1s	remaining: 22.6s
    99:	learn: 0.0610619	total: 22.4s	remaining: 22.4s
    100:	learn: 0.0610542	total: 22.6s	remaining: 22.1s
    101:	learn: 0.0610504	total: 22.8s	remaining: 21.9s
    102:	learn: 0.0610472	total: 23.1s	remaining: 21.7s
    103:	learn: 0.0610426	total: 23.3s	remaining: 21.5s
    104:	learn: 0.0610385	total: 23.5s	remaining: 21.3s
    105:	learn: 0.0610336	total: 23.7s	remaining: 21.1s
    106:	learn: 0.0610268	total: 24s	remaining: 20.8s
    107:	learn: 0.0610184	total: 24.2s	remaining: 20.6s
    108:	learn: 0.0610083	total: 24.4s	remaining: 20.4s
    109:	learn: 0.0610028	total: 24.6s	remaining: 20.1s
    110:	learn: 0.0609825	total: 24.8s	remaining: 19.9s
    111:	learn: 0.0609802	total: 25.1s	remaining: 19.7s
    112:	learn: 0.0609753	total: 25.3s	remaining: 19.5s
    113:	learn: 0.0609654	total: 25.5s	remaining: 19.3s
    114:	learn: 0.0609628	total: 25.7s	remaining: 19s
    115:	learn: 0.0609564	total: 25.9s	remaining: 18.8s
    116:	learn: 0.0609416	total: 26.1s	remaining: 18.5s
    117:	learn: 0.0609380	total: 26.3s	remaining: 18.3s
    118:	learn: 0.0609325	total: 26.5s	remaining: 18.1s
    119:	learn: 0.0609272	total: 26.8s	remaining: 17.9s
    120:	learn: 0.0609244	total: 27s	remaining: 17.6s
    121:	learn: 0.0609206	total: 27.2s	remaining: 17.4s
    122:	learn: 0.0609168	total: 27.4s	remaining: 17.2s
    123:	learn: 0.0609116	total: 27.6s	remaining: 16.9s
    124:	learn: 0.0609004	total: 27.9s	remaining: 16.7s
    125:	learn: 0.0608935	total: 28.1s	remaining: 16.5s
    126:	learn: 0.0608844	total: 28.3s	remaining: 16.3s
    127:	learn: 0.0608746	total: 28.6s	remaining: 16.1s
    128:	learn: 0.0608745	total: 28.7s	remaining: 15.8s
    129:	learn: 0.0608715	total: 28.9s	remaining: 15.6s
    130:	learn: 0.0608688	total: 29.1s	remaining: 15.3s
    131:	learn: 0.0608635	total: 29.3s	remaining: 15.1s
    132:	learn: 0.0608600	total: 29.5s	remaining: 14.9s
    133:	learn: 0.0608557	total: 29.8s	remaining: 14.7s
    134:	learn: 0.0608526	total: 30s	remaining: 14.4s
    135:	learn: 0.0608487	total: 30.2s	remaining: 14.2s
    136:	learn: 0.0608449	total: 30.4s	remaining: 14s
    137:	learn: 0.0608414	total: 30.6s	remaining: 13.8s
    138:	learn: 0.0608342	total: 30.8s	remaining: 13.5s
    139:	learn: 0.0608300	total: 31.1s	remaining: 13.3s
    140:	learn: 0.0608257	total: 31.3s	remaining: 13.1s
    141:	learn: 0.0608231	total: 31.5s	remaining: 12.9s
    142:	learn: 0.0608178	total: 31.7s	remaining: 12.6s
    143:	learn: 0.0608151	total: 31.9s	remaining: 12.4s
    144:	learn: 0.0608114	total: 32.2s	remaining: 12.2s
    145:	learn: 0.0608038	total: 32.4s	remaining: 12s
    146:	learn: 0.0608003	total: 32.6s	remaining: 11.8s
    147:	learn: 0.0607974	total: 32.8s	remaining: 11.5s
    148:	learn: 0.0607922	total: 33.1s	remaining: 11.3s
    149:	learn: 0.0607884	total: 33.3s	remaining: 11.1s
    150:	learn: 0.0607843	total: 33.5s	remaining: 10.9s
    151:	learn: 0.0607819	total: 33.7s	remaining: 10.6s
    152:	learn: 0.0607775	total: 33.9s	remaining: 10.4s
    153:	learn: 0.0607742	total: 34.2s	remaining: 10.2s
    154:	learn: 0.0607682	total: 34.4s	remaining: 9.98s
    155:	learn: 0.0607629	total: 34.6s	remaining: 9.77s
    156:	learn: 0.0607555	total: 34.9s	remaining: 9.54s
    157:	learn: 0.0607490	total: 35.1s	remaining: 9.32s
    158:	learn: 0.0607459	total: 35.3s	remaining: 9.09s
    159:	learn: 0.0607426	total: 35.5s	remaining: 8.86s
    160:	learn: 0.0607402	total: 35.7s	remaining: 8.64s
    161:	learn: 0.0607375	total: 35.9s	remaining: 8.42s
    162:	learn: 0.0607372	total: 36.1s	remaining: 8.19s
    163:	learn: 0.0607345	total: 36.3s	remaining: 7.96s
    164:	learn: 0.0607267	total: 36.5s	remaining: 7.74s
    165:	learn: 0.0607239	total: 36.7s	remaining: 7.52s
    166:	learn: 0.0607220	total: 36.9s	remaining: 7.29s
    167:	learn: 0.0607177	total: 37.1s	remaining: 7.07s
    168:	learn: 0.0607136	total: 37.3s	remaining: 6.85s
    169:	learn: 0.0607095	total: 37.6s	remaining: 6.63s
    170:	learn: 0.0607068	total: 37.8s	remaining: 6.41s
    171:	learn: 0.0607034	total: 38s	remaining: 6.19s
    172:	learn: 0.0606973	total: 38.2s	remaining: 5.97s
    173:	learn: 0.0606945	total: 38.5s	remaining: 5.75s
    174:	learn: 0.0606931	total: 38.7s	remaining: 5.53s
    175:	learn: 0.0606884	total: 38.9s	remaining: 5.31s
    176:	learn: 0.0606853	total: 39.2s	remaining: 5.09s
    177:	learn: 0.0606812	total: 39.4s	remaining: 4.87s
    178:	learn: 0.0606785	total: 39.6s	remaining: 4.65s
    179:	learn: 0.0606759	total: 39.8s	remaining: 4.43s
    180:	learn: 0.0606692	total: 40.1s	remaining: 4.21s
    181:	learn: 0.0606636	total: 40.3s	remaining: 3.99s
    182:	learn: 0.0606598	total: 40.5s	remaining: 3.77s
    183:	learn: 0.0606539	total: 40.7s	remaining: 3.54s
    184:	learn: 0.0606472	total: 40.9s	remaining: 3.32s
    185:	learn: 0.0606449	total: 41.2s	remaining: 3.1s
    186:	learn: 0.0606399	total: 41.4s	remaining: 2.88s
    187:	learn: 0.0606351	total: 41.6s	remaining: 2.65s
    188:	learn: 0.0606312	total: 41.8s	remaining: 2.43s
    189:	learn: 0.0606261	total: 42s	remaining: 2.21s
    190:	learn: 0.0606238	total: 42.2s	remaining: 1.99s
    191:	learn: 0.0606217	total: 42.4s	remaining: 1.77s
    192:	learn: 0.0606160	total: 42.6s	remaining: 1.55s
    193:	learn: 0.0606084	total: 42.9s	remaining: 1.32s
    194:	learn: 0.0606041	total: 43.1s	remaining: 1.1s
    195:	learn: 0.0605995	total: 43.3s	remaining: 884ms
    196:	learn: 0.0605950	total: 43.5s	remaining: 663ms
    197:	learn: 0.0605586	total: 43.8s	remaining: 442ms
    198:	learn: 0.0605545	total: 44s	remaining: 221ms
    199:	learn: 0.0605514	total: 44.2s	remaining: 0us
    Trained model nº 20/27. Iterations: 200Depth: 6Learning rate: 0.1
    0:	learn: 0.5042661	total: 211ms	remaining: 1m 3s
    1:	learn: 0.3693048	total: 413ms	remaining: 1m 1s
    2:	learn: 0.2745350	total: 618ms	remaining: 1m 1s
    3:	learn: 0.2165385	total: 854ms	remaining: 1m 3s
    4:	learn: 0.1729344	total: 1.07s	remaining: 1m 3s
    5:	learn: 0.1436341	total: 1.3s	remaining: 1m 3s
    6:	learn: 0.1248890	total: 1.51s	remaining: 1m 3s
    7:	learn: 0.1087769	total: 1.74s	remaining: 1m 3s
    8:	learn: 0.0953987	total: 1.99s	remaining: 1m 4s
    9:	learn: 0.0879656	total: 2.22s	remaining: 1m 4s
    10:	learn: 0.0831718	total: 2.46s	remaining: 1m 4s
    11:	learn: 0.0780844	total: 2.69s	remaining: 1m 4s
    12:	learn: 0.0756458	total: 2.85s	remaining: 1m 2s
    13:	learn: 0.0724599	total: 3.09s	remaining: 1m 3s
    14:	learn: 0.0706491	total: 3.33s	remaining: 1m 3s
    15:	learn: 0.0684151	total: 3.57s	remaining: 1m 3s
    16:	learn: 0.0672456	total: 3.81s	remaining: 1m 3s
    17:	learn: 0.0665162	total: 4.03s	remaining: 1m 3s
    18:	learn: 0.0657933	total: 4.25s	remaining: 1m 2s
    19:	learn: 0.0652833	total: 4.5s	remaining: 1m 2s
    20:	learn: 0.0647583	total: 4.74s	remaining: 1m 2s
    21:	learn: 0.0642916	total: 4.96s	remaining: 1m 2s
    22:	learn: 0.0640125	total: 5.21s	remaining: 1m 2s
    23:	learn: 0.0636357	total: 5.45s	remaining: 1m 2s
    24:	learn: 0.0633818	total: 5.71s	remaining: 1m 2s
    25:	learn: 0.0632414	total: 5.95s	remaining: 1m 2s
    26:	learn: 0.0631142	total: 6.19s	remaining: 1m 2s
    27:	learn: 0.0630314	total: 6.42s	remaining: 1m 2s
    28:	learn: 0.0629191	total: 6.65s	remaining: 1m 2s
    29:	learn: 0.0627590	total: 6.89s	remaining: 1m 1s
    30:	learn: 0.0626677	total: 7.12s	remaining: 1m 1s
    31:	learn: 0.0625806	total: 7.37s	remaining: 1m 1s
    32:	learn: 0.0624836	total: 7.6s	remaining: 1m 1s
    33:	learn: 0.0623721	total: 7.84s	remaining: 1m 1s
    34:	learn: 0.0623383	total: 8.07s	remaining: 1m 1s
    35:	learn: 0.0621072	total: 8.29s	remaining: 1m
    36:	learn: 0.0620779	total: 8.53s	remaining: 1m
    37:	learn: 0.0620555	total: 8.74s	remaining: 1m
    38:	learn: 0.0620347	total: 8.98s	remaining: 1m
    39:	learn: 0.0620030	total: 9.21s	remaining: 59.9s
    40:	learn: 0.0619325	total: 9.45s	remaining: 59.7s
    41:	learn: 0.0619029	total: 9.72s	remaining: 59.7s
    42:	learn: 0.0618823	total: 10s	remaining: 59.8s
    43:	learn: 0.0618631	total: 10.3s	remaining: 59.7s
    44:	learn: 0.0618031	total: 10.5s	remaining: 59.4s
    45:	learn: 0.0617623	total: 10.7s	remaining: 59.1s
    46:	learn: 0.0617490	total: 11s	remaining: 59s
    47:	learn: 0.0617014	total: 11.2s	remaining: 58.7s
    48:	learn: 0.0616768	total: 11.4s	remaining: 58.6s
    49:	learn: 0.0616601	total: 11.7s	remaining: 58.4s
    50:	learn: 0.0616494	total: 11.9s	remaining: 58.2s
    51:	learn: 0.0616232	total: 12.2s	remaining: 58s
    52:	learn: 0.0615942	total: 12.4s	remaining: 57.7s
    53:	learn: 0.0615838	total: 12.6s	remaining: 57.4s
    54:	learn: 0.0615535	total: 12.8s	remaining: 57.2s
    55:	learn: 0.0615367	total: 13.1s	remaining: 57s
    56:	learn: 0.0615232	total: 13.3s	remaining: 56.8s
    57:	learn: 0.0615093	total: 13.6s	remaining: 56.6s
    58:	learn: 0.0615005	total: 13.8s	remaining: 56.4s
    59:	learn: 0.0614651	total: 14.1s	remaining: 56.3s
    60:	learn: 0.0614521	total: 14.3s	remaining: 56s
    61:	learn: 0.0614391	total: 14.5s	remaining: 55.8s
    62:	learn: 0.0614235	total: 14.7s	remaining: 55.4s
    63:	learn: 0.0614135	total: 15s	remaining: 55.1s
    64:	learn: 0.0614040	total: 15.2s	remaining: 54.9s
    65:	learn: 0.0613917	total: 15.4s	remaining: 54.7s
    66:	learn: 0.0613834	total: 15.7s	remaining: 54.4s
    67:	learn: 0.0613147	total: 15.9s	remaining: 54.2s
    68:	learn: 0.0613036	total: 16.1s	remaining: 54s
    69:	learn: 0.0612914	total: 16.4s	remaining: 53.8s
    70:	learn: 0.0612849	total: 16.6s	remaining: 53.5s
    71:	learn: 0.0612719	total: 16.8s	remaining: 53.2s
    72:	learn: 0.0612660	total: 17s	remaining: 53s
    73:	learn: 0.0612587	total: 17.3s	remaining: 52.7s
    74:	learn: 0.0612483	total: 17.5s	remaining: 52.4s
    75:	learn: 0.0612385	total: 17.7s	remaining: 52.2s
    76:	learn: 0.0612274	total: 17.9s	remaining: 51.9s
    77:	learn: 0.0612214	total: 18.1s	remaining: 51.6s
    78:	learn: 0.0612123	total: 18.4s	remaining: 51.4s
    79:	learn: 0.0612006	total: 18.6s	remaining: 51.2s
    80:	learn: 0.0611906	total: 18.8s	remaining: 50.9s
    81:	learn: 0.0611830	total: 19.1s	remaining: 50.7s
    82:	learn: 0.0611747	total: 19.3s	remaining: 50.4s
    83:	learn: 0.0611669	total: 19.5s	remaining: 50.2s
    84:	learn: 0.0611638	total: 19.8s	remaining: 50.1s
    85:	learn: 0.0611591	total: 20s	remaining: 49.9s
    86:	learn: 0.0611551	total: 20.3s	remaining: 49.6s
    87:	learn: 0.0611512	total: 20.5s	remaining: 49.4s
    88:	learn: 0.0611435	total: 20.7s	remaining: 49.1s
    89:	learn: 0.0611302	total: 20.9s	remaining: 48.9s
    90:	learn: 0.0611182	total: 21.2s	remaining: 48.6s
    91:	learn: 0.0611143	total: 21.4s	remaining: 48.4s
    92:	learn: 0.0611018	total: 21.6s	remaining: 48.2s
    93:	learn: 0.0610943	total: 21.9s	remaining: 48s
    94:	learn: 0.0610866	total: 22.1s	remaining: 47.7s
    95:	learn: 0.0610819	total: 22.3s	remaining: 47.4s
    96:	learn: 0.0610762	total: 22.5s	remaining: 47.2s
    97:	learn: 0.0610688	total: 22.8s	remaining: 46.9s
    98:	learn: 0.0610653	total: 23s	remaining: 46.7s
    99:	learn: 0.0610619	total: 23.2s	remaining: 46.5s
    100:	learn: 0.0610542	total: 23.5s	remaining: 46.2s
    101:	learn: 0.0610504	total: 23.7s	remaining: 46s
    102:	learn: 0.0610472	total: 23.9s	remaining: 45.8s
    103:	learn: 0.0610426	total: 24.2s	remaining: 45.5s
    104:	learn: 0.0610385	total: 24.4s	remaining: 45.3s
    105:	learn: 0.0610336	total: 24.6s	remaining: 45.1s
    106:	learn: 0.0610268	total: 24.9s	remaining: 44.8s
    107:	learn: 0.0610184	total: 25.1s	remaining: 44.6s
    108:	learn: 0.0610083	total: 25.3s	remaining: 44.4s
    109:	learn: 0.0610028	total: 25.6s	remaining: 44.1s
    110:	learn: 0.0609825	total: 25.8s	remaining: 44s
    111:	learn: 0.0609802	total: 26.1s	remaining: 43.7s
    112:	learn: 0.0609753	total: 26.3s	remaining: 43.5s
    113:	learn: 0.0609654	total: 26.5s	remaining: 43.3s
    114:	learn: 0.0609628	total: 26.7s	remaining: 42.9s
    115:	learn: 0.0609564	total: 26.9s	remaining: 42.7s
    116:	learn: 0.0609416	total: 27.1s	remaining: 42.4s
    117:	learn: 0.0609380	total: 27.4s	remaining: 42.2s
    118:	learn: 0.0609325	total: 27.6s	remaining: 42s
    119:	learn: 0.0609272	total: 27.8s	remaining: 41.7s
    120:	learn: 0.0609244	total: 28s	remaining: 41.5s
    121:	learn: 0.0609206	total: 28.3s	remaining: 41.3s
    122:	learn: 0.0609168	total: 28.5s	remaining: 41s
    123:	learn: 0.0609116	total: 28.7s	remaining: 40.7s
    124:	learn: 0.0609004	total: 28.9s	remaining: 40.5s
    125:	learn: 0.0608935	total: 29.1s	remaining: 40.2s
    126:	learn: 0.0608844	total: 29.4s	remaining: 40s
    127:	learn: 0.0608746	total: 29.6s	remaining: 39.8s
    128:	learn: 0.0608745	total: 29.7s	remaining: 39.4s
    129:	learn: 0.0608715	total: 30s	remaining: 39.2s
    130:	learn: 0.0608688	total: 30.2s	remaining: 38.9s
    131:	learn: 0.0608635	total: 30.4s	remaining: 38.7s
    132:	learn: 0.0608600	total: 30.6s	remaining: 38.4s
    133:	learn: 0.0608557	total: 30.8s	remaining: 38.2s
    134:	learn: 0.0608526	total: 31s	remaining: 37.9s
    135:	learn: 0.0608487	total: 31.3s	remaining: 37.7s
    136:	learn: 0.0608449	total: 31.5s	remaining: 37.4s
    137:	learn: 0.0608414	total: 31.7s	remaining: 37.2s
    138:	learn: 0.0608342	total: 31.9s	remaining: 36.9s
    139:	learn: 0.0608300	total: 32.1s	remaining: 36.7s
    140:	learn: 0.0608257	total: 32.3s	remaining: 36.5s
    141:	learn: 0.0608231	total: 32.5s	remaining: 36.2s
    142:	learn: 0.0608178	total: 32.8s	remaining: 36s
    143:	learn: 0.0608151	total: 33s	remaining: 35.7s
    144:	learn: 0.0608114	total: 33.2s	remaining: 35.5s
    145:	learn: 0.0608038	total: 33.4s	remaining: 35.3s
    146:	learn: 0.0608003	total: 33.6s	remaining: 35s
    147:	learn: 0.0607974	total: 33.9s	remaining: 34.8s
    148:	learn: 0.0607922	total: 34.1s	remaining: 34.6s
    149:	learn: 0.0607884	total: 34.3s	remaining: 34.3s
    150:	learn: 0.0607843	total: 34.6s	remaining: 34.1s
    151:	learn: 0.0607819	total: 34.7s	remaining: 33.8s
    152:	learn: 0.0607775	total: 35s	remaining: 33.6s
    153:	learn: 0.0607742	total: 35.2s	remaining: 33.4s
    154:	learn: 0.0607682	total: 35.4s	remaining: 33.2s
    155:	learn: 0.0607629	total: 35.7s	remaining: 32.9s
    156:	learn: 0.0607555	total: 35.9s	remaining: 32.7s
    157:	learn: 0.0607490	total: 36.1s	remaining: 32.5s
    158:	learn: 0.0607459	total: 36.3s	remaining: 32.2s
    159:	learn: 0.0607426	total: 36.5s	remaining: 31.9s
    160:	learn: 0.0607402	total: 36.7s	remaining: 31.7s
    161:	learn: 0.0607375	total: 37s	remaining: 31.5s
    162:	learn: 0.0607372	total: 37.1s	remaining: 31.2s
    163:	learn: 0.0607345	total: 37.3s	remaining: 31s
    164:	learn: 0.0607267	total: 37.6s	remaining: 30.8s
    165:	learn: 0.0607239	total: 37.8s	remaining: 30.5s
    166:	learn: 0.0607220	total: 38s	remaining: 30.3s
    167:	learn: 0.0607177	total: 38.2s	remaining: 30s
    168:	learn: 0.0607136	total: 38.5s	remaining: 29.8s
    169:	learn: 0.0607095	total: 38.7s	remaining: 29.6s
    170:	learn: 0.0607068	total: 38.8s	remaining: 29.3s
    171:	learn: 0.0607034	total: 39.1s	remaining: 29.1s
    172:	learn: 0.0606973	total: 39.3s	remaining: 28.9s
    173:	learn: 0.0606945	total: 39.6s	remaining: 28.6s
    174:	learn: 0.0606931	total: 39.8s	remaining: 28.4s
    175:	learn: 0.0606884	total: 40s	remaining: 28.2s
    176:	learn: 0.0606853	total: 40.2s	remaining: 28s
    177:	learn: 0.0606812	total: 40.5s	remaining: 27.7s
    178:	learn: 0.0606785	total: 40.7s	remaining: 27.5s
    179:	learn: 0.0606759	total: 40.9s	remaining: 27.3s
    180:	learn: 0.0606692	total: 41.2s	remaining: 27.1s
    181:	learn: 0.0606636	total: 41.4s	remaining: 26.9s
    182:	learn: 0.0606598	total: 41.7s	remaining: 26.6s
    183:	learn: 0.0606539	total: 41.8s	remaining: 26.4s
    184:	learn: 0.0606472	total: 42.1s	remaining: 26.1s
    185:	learn: 0.0606449	total: 42.3s	remaining: 25.9s
    186:	learn: 0.0606399	total: 42.5s	remaining: 25.7s
    187:	learn: 0.0606351	total: 42.7s	remaining: 25.5s
    188:	learn: 0.0606312	total: 42.9s	remaining: 25.2s
    189:	learn: 0.0606261	total: 43.2s	remaining: 25s
    190:	learn: 0.0606238	total: 43.4s	remaining: 24.8s
    191:	learn: 0.0606217	total: 43.6s	remaining: 24.5s
    192:	learn: 0.0606160	total: 43.8s	remaining: 24.3s
    193:	learn: 0.0606084	total: 44.1s	remaining: 24.1s
    194:	learn: 0.0606041	total: 44.3s	remaining: 23.9s
    195:	learn: 0.0605995	total: 44.5s	remaining: 23.6s
    196:	learn: 0.0605950	total: 44.8s	remaining: 23.4s
    197:	learn: 0.0605586	total: 45s	remaining: 23.2s
    198:	learn: 0.0605545	total: 45.2s	remaining: 23s
    199:	learn: 0.0605514	total: 45.5s	remaining: 22.7s
    200:	learn: 0.0605461	total: 45.7s	remaining: 22.5s
    201:	learn: 0.0605414	total: 45.9s	remaining: 22.3s
    202:	learn: 0.0605386	total: 46.1s	remaining: 22s
    203:	learn: 0.0605340	total: 46.4s	remaining: 21.8s
    204:	learn: 0.0605302	total: 46.6s	remaining: 21.6s
    205:	learn: 0.0605271	total: 46.8s	remaining: 21.4s
    206:	learn: 0.0605259	total: 47s	remaining: 21.1s
    207:	learn: 0.0605224	total: 47.3s	remaining: 20.9s
    208:	learn: 0.0605164	total: 47.5s	remaining: 20.7s
    209:	learn: 0.0605120	total: 47.7s	remaining: 20.5s
    210:	learn: 0.0605091	total: 48s	remaining: 20.2s
    211:	learn: 0.0605064	total: 48.2s	remaining: 20s
    212:	learn: 0.0605031	total: 48.4s	remaining: 19.8s
    213:	learn: 0.0604999	total: 48.6s	remaining: 19.5s
    214:	learn: 0.0604977	total: 48.8s	remaining: 19.3s
    215:	learn: 0.0604959	total: 49s	remaining: 19.1s
    216:	learn: 0.0604902	total: 49.3s	remaining: 18.8s
    217:	learn: 0.0604876	total: 49.5s	remaining: 18.6s
    218:	learn: 0.0604824	total: 49.7s	remaining: 18.4s
    219:	learn: 0.0604775	total: 49.9s	remaining: 18.2s
    220:	learn: 0.0604740	total: 50.1s	remaining: 17.9s
    221:	learn: 0.0604688	total: 50.4s	remaining: 17.7s
    222:	learn: 0.0604632	total: 50.6s	remaining: 17.5s
    223:	learn: 0.0604592	total: 50.8s	remaining: 17.2s
    224:	learn: 0.0604552	total: 51.1s	remaining: 17s
    225:	learn: 0.0604517	total: 51.3s	remaining: 16.8s
    226:	learn: 0.0604415	total: 51.5s	remaining: 16.6s
    227:	learn: 0.0604388	total: 51.8s	remaining: 16.3s
    228:	learn: 0.0604357	total: 52s	remaining: 16.1s
    229:	learn: 0.0604317	total: 52.2s	remaining: 15.9s
    230:	learn: 0.0604271	total: 52.5s	remaining: 15.7s
    231:	learn: 0.0604254	total: 52.7s	remaining: 15.4s
    232:	learn: 0.0604239	total: 52.9s	remaining: 15.2s
    233:	learn: 0.0604196	total: 53.1s	remaining: 15s
    234:	learn: 0.0604174	total: 53.3s	remaining: 14.8s
    235:	learn: 0.0604136	total: 53.6s	remaining: 14.5s
    236:	learn: 0.0604118	total: 53.8s	remaining: 14.3s
    237:	learn: 0.0604094	total: 54s	remaining: 14.1s
    238:	learn: 0.0604052	total: 54.2s	remaining: 13.8s
    239:	learn: 0.0604015	total: 54.4s	remaining: 13.6s
    240:	learn: 0.0603947	total: 54.7s	remaining: 13.4s
    241:	learn: 0.0603927	total: 54.9s	remaining: 13.2s
    242:	learn: 0.0603893	total: 55.1s	remaining: 12.9s
    243:	learn: 0.0603722	total: 55.4s	remaining: 12.7s
    244:	learn: 0.0603687	total: 55.6s	remaining: 12.5s
    245:	learn: 0.0603646	total: 55.8s	remaining: 12.3s
    246:	learn: 0.0603627	total: 56s	remaining: 12s
    247:	learn: 0.0603580	total: 56.2s	remaining: 11.8s
    248:	learn: 0.0603542	total: 56.5s	remaining: 11.6s
    249:	learn: 0.0603512	total: 56.7s	remaining: 11.3s
    250:	learn: 0.0603493	total: 56.9s	remaining: 11.1s
    251:	learn: 0.0603474	total: 57.1s	remaining: 10.9s
    252:	learn: 0.0603440	total: 57.4s	remaining: 10.7s
    253:	learn: 0.0603406	total: 57.6s	remaining: 10.4s
    254:	learn: 0.0603379	total: 57.8s	remaining: 10.2s
    255:	learn: 0.0603342	total: 58s	remaining: 9.97s
    256:	learn: 0.0603323	total: 58.3s	remaining: 9.75s
    257:	learn: 0.0603234	total: 58.5s	remaining: 9.52s
    258:	learn: 0.0603214	total: 58.7s	remaining: 9.29s
    259:	learn: 0.0603189	total: 59s	remaining: 9.07s
    260:	learn: 0.0603151	total: 59.2s	remaining: 8.84s
    261:	learn: 0.0603096	total: 59.4s	remaining: 8.62s
    262:	learn: 0.0603071	total: 59.6s	remaining: 8.39s
    263:	learn: 0.0603053	total: 59.9s	remaining: 8.16s
    264:	learn: 0.0603018	total: 1m	remaining: 7.93s
    265:	learn: 0.0602982	total: 1m	remaining: 7.71s
    266:	learn: 0.0602942	total: 1m	remaining: 7.48s
    267:	learn: 0.0602895	total: 1m	remaining: 7.26s
    268:	learn: 0.0602864	total: 1m 1s	remaining: 7.04s
    269:	learn: 0.0602810	total: 1m 1s	remaining: 6.81s
    270:	learn: 0.0602778	total: 1m 1s	remaining: 6.59s
    271:	learn: 0.0602751	total: 1m 1s	remaining: 6.36s
    272:	learn: 0.0602725	total: 1m 1s	remaining: 6.13s
    273:	learn: 0.0602690	total: 1m 2s	remaining: 5.9s
    274:	learn: 0.0602632	total: 1m 2s	remaining: 5.67s
    275:	learn: 0.0602585	total: 1m 2s	remaining: 5.45s
    276:	learn: 0.0602544	total: 1m 2s	remaining: 5.22s
    277:	learn: 0.0602520	total: 1m 3s	remaining: 4.99s
    278:	learn: 0.0602496	total: 1m 3s	remaining: 4.77s
    279:	learn: 0.0602487	total: 1m 3s	remaining: 4.54s
    280:	learn: 0.0602468	total: 1m 3s	remaining: 4.31s
    281:	learn: 0.0602452	total: 1m 4s	remaining: 4.08s
    282:	learn: 0.0602426	total: 1m 4s	remaining: 3.86s
    283:	learn: 0.0602411	total: 1m 4s	remaining: 3.63s
    284:	learn: 0.0602365	total: 1m 4s	remaining: 3.4s
    285:	learn: 0.0602329	total: 1m 4s	remaining: 3.17s
    286:	learn: 0.0602269	total: 1m 5s	remaining: 2.95s
    287:	learn: 0.0602233	total: 1m 5s	remaining: 2.72s
    288:	learn: 0.0602171	total: 1m 5s	remaining: 2.5s
    289:	learn: 0.0602148	total: 1m 5s	remaining: 2.27s
    290:	learn: 0.0602100	total: 1m 6s	remaining: 2.04s
    291:	learn: 0.0602082	total: 1m 6s	remaining: 1.81s
    292:	learn: 0.0602063	total: 1m 6s	remaining: 1.59s
    293:	learn: 0.0602012	total: 1m 6s	remaining: 1.36s
    294:	learn: 0.0601974	total: 1m 6s	remaining: 1.13s
    295:	learn: 0.0601935	total: 1m 7s	remaining: 908ms
    296:	learn: 0.0601895	total: 1m 7s	remaining: 681ms
    297:	learn: 0.0601877	total: 1m 7s	remaining: 454ms
    298:	learn: 0.0601855	total: 1m 7s	remaining: 227ms
    299:	learn: 0.0601811	total: 1m 8s	remaining: 0us
    Trained model nº 21/27. Iterations: 300Depth: 6Learning rate: 0.1
    0:	learn: 0.5010775	total: 198ms	remaining: 19.6s
    1:	learn: 0.3557770	total: 383ms	remaining: 18.8s
    2:	learn: 0.2658318	total: 567ms	remaining: 18.3s
    3:	learn: 0.2027384	total: 740ms	remaining: 17.8s
    4:	learn: 0.1606472	total: 914ms	remaining: 17.4s
    5:	learn: 0.1324986	total: 1.09s	remaining: 17.2s
    6:	learn: 0.1135172	total: 1.29s	remaining: 17.1s
    7:	learn: 0.1002754	total: 1.49s	remaining: 17.2s
    8:	learn: 0.0908929	total: 1.69s	remaining: 17.1s
    9:	learn: 0.0840925	total: 1.88s	remaining: 16.9s
    10:	learn: 0.0791413	total: 2.06s	remaining: 16.7s
    11:	learn: 0.0748461	total: 2.25s	remaining: 16.5s
    12:	learn: 0.0721446	total: 2.43s	remaining: 16.2s
    13:	learn: 0.0699592	total: 2.6s	remaining: 16s
    14:	learn: 0.0683607	total: 2.76s	remaining: 15.7s
    15:	learn: 0.0671871	total: 2.96s	remaining: 15.5s
    16:	learn: 0.0662658	total: 3.14s	remaining: 15.3s
    17:	learn: 0.0654208	total: 3.33s	remaining: 15.2s
    18:	learn: 0.0648389	total: 3.51s	remaining: 15s
    19:	learn: 0.0643473	total: 3.68s	remaining: 14.7s
    20:	learn: 0.0639732	total: 3.85s	remaining: 14.5s
    21:	learn: 0.0636442	total: 4.02s	remaining: 14.2s
    22:	learn: 0.0634199	total: 4.18s	remaining: 14s
    23:	learn: 0.0631678	total: 4.37s	remaining: 13.8s
    24:	learn: 0.0629987	total: 4.53s	remaining: 13.6s
    25:	learn: 0.0626492	total: 4.7s	remaining: 13.4s
    26:	learn: 0.0625135	total: 4.88s	remaining: 13.2s
    27:	learn: 0.0622758	total: 5.04s	remaining: 13s
    28:	learn: 0.0621776	total: 5.21s	remaining: 12.8s
    29:	learn: 0.0620927	total: 5.38s	remaining: 12.6s
    30:	learn: 0.0620173	total: 5.55s	remaining: 12.4s
    31:	learn: 0.0619552	total: 5.73s	remaining: 12.2s
    32:	learn: 0.0619065	total: 5.91s	remaining: 12s
    33:	learn: 0.0618551	total: 6.11s	remaining: 11.9s
    34:	learn: 0.0618293	total: 6.28s	remaining: 11.7s
    35:	learn: 0.0617853	total: 6.47s	remaining: 11.5s
    36:	learn: 0.0617380	total: 6.64s	remaining: 11.3s
    37:	learn: 0.0617109	total: 6.83s	remaining: 11.1s
    38:	learn: 0.0616680	total: 7.01s	remaining: 11s
    39:	learn: 0.0616334	total: 7.19s	remaining: 10.8s
    40:	learn: 0.0616157	total: 7.39s	remaining: 10.6s
    41:	learn: 0.0615997	total: 7.56s	remaining: 10.4s
    42:	learn: 0.0615745	total: 7.74s	remaining: 10.3s
    43:	learn: 0.0615480	total: 7.93s	remaining: 10.1s
    44:	learn: 0.0615102	total: 8.11s	remaining: 9.92s
    45:	learn: 0.0614855	total: 8.3s	remaining: 9.75s
    46:	learn: 0.0614583	total: 8.49s	remaining: 9.58s
    47:	learn: 0.0614410	total: 8.69s	remaining: 9.41s
    48:	learn: 0.0613994	total: 8.88s	remaining: 9.24s
    49:	learn: 0.0613787	total: 9.07s	remaining: 9.07s
    50:	learn: 0.0613614	total: 9.25s	remaining: 8.89s
    51:	learn: 0.0613234	total: 9.43s	remaining: 8.71s
    52:	learn: 0.0613071	total: 9.6s	remaining: 8.51s
    53:	learn: 0.0612573	total: 9.77s	remaining: 8.32s
    54:	learn: 0.0612409	total: 9.95s	remaining: 8.14s
    55:	learn: 0.0612283	total: 10.1s	remaining: 7.96s
    56:	learn: 0.0611839	total: 10.3s	remaining: 7.79s
    57:	learn: 0.0611594	total: 10.5s	remaining: 7.62s
    58:	learn: 0.0611467	total: 10.7s	remaining: 7.45s
    59:	learn: 0.0611300	total: 10.9s	remaining: 7.27s
    60:	learn: 0.0611067	total: 11.1s	remaining: 7.09s
    61:	learn: 0.0610843	total: 11.3s	remaining: 6.92s
    62:	learn: 0.0610742	total: 11.5s	remaining: 6.75s
    63:	learn: 0.0610534	total: 11.7s	remaining: 6.57s
    64:	learn: 0.0610317	total: 11.9s	remaining: 6.38s
    65:	learn: 0.0609997	total: 12s	remaining: 6.2s
    66:	learn: 0.0609797	total: 12.2s	remaining: 6.01s
    67:	learn: 0.0609658	total: 12.4s	remaining: 5.83s
    68:	learn: 0.0609449	total: 12.6s	remaining: 5.65s
    69:	learn: 0.0609160	total: 12.8s	remaining: 5.48s
    70:	learn: 0.0609010	total: 13s	remaining: 5.3s
    71:	learn: 0.0608824	total: 13.2s	remaining: 5.13s
    72:	learn: 0.0608474	total: 13.4s	remaining: 4.95s
    73:	learn: 0.0608368	total: 13.6s	remaining: 4.77s
    74:	learn: 0.0608234	total: 13.8s	remaining: 4.59s
    75:	learn: 0.0608095	total: 14s	remaining: 4.41s
    76:	learn: 0.0607997	total: 14.1s	remaining: 4.22s
    77:	learn: 0.0607888	total: 14.3s	remaining: 4.04s
    78:	learn: 0.0607723	total: 14.5s	remaining: 3.85s
    79:	learn: 0.0607609	total: 14.7s	remaining: 3.67s
    80:	learn: 0.0607514	total: 14.9s	remaining: 3.49s
    81:	learn: 0.0607351	total: 15s	remaining: 3.3s
    82:	learn: 0.0607104	total: 15.2s	remaining: 3.12s
    83:	learn: 0.0606953	total: 15.4s	remaining: 2.93s
    84:	learn: 0.0606840	total: 15.6s	remaining: 2.75s
    85:	learn: 0.0606744	total: 15.8s	remaining: 2.56s
    86:	learn: 0.0606616	total: 15.9s	remaining: 2.38s
    87:	learn: 0.0606340	total: 16.1s	remaining: 2.2s
    88:	learn: 0.0606267	total: 16.3s	remaining: 2.01s
    89:	learn: 0.0606183	total: 16.5s	remaining: 1.83s
    90:	learn: 0.0606119	total: 16.6s	remaining: 1.65s
    91:	learn: 0.0606069	total: 16.8s	remaining: 1.46s
    92:	learn: 0.0605996	total: 17s	remaining: 1.28s
    93:	learn: 0.0605914	total: 17.2s	remaining: 1.1s
    94:	learn: 0.0605758	total: 17.4s	remaining: 914ms
    95:	learn: 0.0605758	total: 17.5s	remaining: 729ms
    96:	learn: 0.0605626	total: 17.7s	remaining: 547ms
    97:	learn: 0.0605406	total: 17.9s	remaining: 365ms
    98:	learn: 0.0605325	total: 18.1s	remaining: 182ms
    99:	learn: 0.0605100	total: 18.2s	remaining: 0us
    Trained model nº 22/27. Iterations: 100Depth: 8Learning rate: 0.1
    0:	learn: 0.5040680	total: 309ms	remaining: 1m 1s
    1:	learn: 0.3747597	total: 592ms	remaining: 58.6s
    2:	learn: 0.2774131	total: 865ms	remaining: 56.8s
    3:	learn: 0.2150157	total: 1.14s	remaining: 55.8s
    4:	learn: 0.1701417	total: 1.41s	remaining: 54.9s
    5:	learn: 0.1383840	total: 1.66s	remaining: 53.6s
    6:	learn: 0.1168210	total: 1.92s	remaining: 53.1s
    7:	learn: 0.1019590	total: 2.19s	remaining: 52.6s
    8:	learn: 0.0915974	total: 2.45s	remaining: 52s
    9:	learn: 0.0845304	total: 2.72s	remaining: 51.7s
    10:	learn: 0.0784835	total: 3s	remaining: 51.5s
    11:	learn: 0.0748310	total: 3.26s	remaining: 51s
    12:	learn: 0.0721101	total: 3.51s	remaining: 50.5s
    13:	learn: 0.0699696	total: 3.78s	remaining: 50.2s
    14:	learn: 0.0682439	total: 4.03s	remaining: 49.7s
    15:	learn: 0.0669627	total: 4.3s	remaining: 49.4s
    16:	learn: 0.0660660	total: 4.55s	remaining: 49s
    17:	learn: 0.0653093	total: 4.81s	remaining: 48.6s
    18:	learn: 0.0646638	total: 5.08s	remaining: 48.4s
    19:	learn: 0.0641881	total: 5.34s	remaining: 48.1s
    20:	learn: 0.0638147	total: 5.62s	remaining: 47.9s
    21:	learn: 0.0635175	total: 5.88s	remaining: 47.6s
    22:	learn: 0.0632533	total: 6.16s	remaining: 47.4s
    23:	learn: 0.0628616	total: 6.42s	remaining: 47.1s
    24:	learn: 0.0626834	total: 6.69s	remaining: 46.8s
    25:	learn: 0.0625484	total: 6.95s	remaining: 46.5s
    26:	learn: 0.0624232	total: 7.2s	remaining: 46.2s
    27:	learn: 0.0623159	total: 7.47s	remaining: 45.9s
    28:	learn: 0.0622386	total: 7.75s	remaining: 45.7s
    29:	learn: 0.0621623	total: 8.02s	remaining: 45.5s
    30:	learn: 0.0621043	total: 8.29s	remaining: 45.2s
    31:	learn: 0.0620557	total: 8.57s	remaining: 45s
    32:	learn: 0.0619992	total: 8.84s	remaining: 44.7s
    33:	learn: 0.0619421	total: 9.09s	remaining: 44.4s
    34:	learn: 0.0618981	total: 9.34s	remaining: 44s
    35:	learn: 0.0618618	total: 9.61s	remaining: 43.8s
    36:	learn: 0.0618257	total: 9.89s	remaining: 43.6s
    37:	learn: 0.0617920	total: 10.2s	remaining: 43.3s
    38:	learn: 0.0617557	total: 10.4s	remaining: 43.1s
    39:	learn: 0.0615805	total: 10.7s	remaining: 42.9s
    40:	learn: 0.0614751	total: 11s	remaining: 42.5s
    41:	learn: 0.0614405	total: 11.2s	remaining: 42.2s
    42:	learn: 0.0614175	total: 11.5s	remaining: 41.9s
    43:	learn: 0.0613893	total: 11.8s	remaining: 41.7s
    44:	learn: 0.0613711	total: 12.1s	remaining: 41.5s
    45:	learn: 0.0613417	total: 12.3s	remaining: 41.3s
    46:	learn: 0.0613230	total: 12.6s	remaining: 41.1s
    47:	learn: 0.0613024	total: 12.9s	remaining: 40.8s
    48:	learn: 0.0612747	total: 13.2s	remaining: 40.6s
    49:	learn: 0.0612471	total: 13.5s	remaining: 40.4s
    50:	learn: 0.0612237	total: 13.7s	remaining: 40.1s
    51:	learn: 0.0611830	total: 14s	remaining: 39.9s
    52:	learn: 0.0611641	total: 14.3s	remaining: 39.6s
    53:	learn: 0.0611483	total: 14.6s	remaining: 39.3s
    54:	learn: 0.0611283	total: 14.8s	remaining: 39s
    55:	learn: 0.0610708	total: 15.1s	remaining: 38.8s
    56:	learn: 0.0610570	total: 15.3s	remaining: 38.5s
    57:	learn: 0.0610221	total: 15.6s	remaining: 38.2s
    58:	learn: 0.0610037	total: 15.9s	remaining: 38s
    59:	learn: 0.0609944	total: 16.2s	remaining: 37.7s
    60:	learn: 0.0609778	total: 16.5s	remaining: 37.5s
    61:	learn: 0.0609634	total: 16.7s	remaining: 37.2s
    62:	learn: 0.0609510	total: 17s	remaining: 36.9s
    63:	learn: 0.0609444	total: 17.2s	remaining: 36.6s
    64:	learn: 0.0609314	total: 17.5s	remaining: 36.4s
    65:	learn: 0.0609072	total: 17.8s	remaining: 36.1s
    66:	learn: 0.0608940	total: 18.1s	remaining: 35.9s
    67:	learn: 0.0608804	total: 18.3s	remaining: 35.6s
    68:	learn: 0.0608720	total: 18.6s	remaining: 35.4s
    69:	learn: 0.0608441	total: 18.9s	remaining: 35.1s
    70:	learn: 0.0608272	total: 19.2s	remaining: 34.8s
    71:	learn: 0.0608176	total: 19.4s	remaining: 34.5s
    72:	learn: 0.0608017	total: 19.7s	remaining: 34.3s
    73:	learn: 0.0607751	total: 19.9s	remaining: 34s
    74:	learn: 0.0607684	total: 20.2s	remaining: 33.6s
    75:	learn: 0.0607529	total: 20.4s	remaining: 33.4s
    76:	learn: 0.0607423	total: 20.7s	remaining: 33.1s
    77:	learn: 0.0607237	total: 21s	remaining: 32.8s
    78:	learn: 0.0607130	total: 21.2s	remaining: 32.5s
    79:	learn: 0.0607021	total: 21.5s	remaining: 32.2s
    80:	learn: 0.0606876	total: 21.8s	remaining: 32s
    81:	learn: 0.0606760	total: 22s	remaining: 31.7s
    82:	learn: 0.0606676	total: 22.3s	remaining: 31.4s
    83:	learn: 0.0606561	total: 22.5s	remaining: 31.1s
    84:	learn: 0.0606443	total: 22.8s	remaining: 30.9s
    85:	learn: 0.0606349	total: 23.1s	remaining: 30.6s
    86:	learn: 0.0606223	total: 23.4s	remaining: 30.4s
    87:	learn: 0.0606129	total: 23.7s	remaining: 30.1s
    88:	learn: 0.0606042	total: 24s	remaining: 29.9s
    89:	learn: 0.0605979	total: 24.2s	remaining: 29.6s
    90:	learn: 0.0605829	total: 24.5s	remaining: 29.3s
    91:	learn: 0.0605723	total: 24.8s	remaining: 29.1s
    92:	learn: 0.0605589	total: 25s	remaining: 28.8s
    93:	learn: 0.0605458	total: 25.3s	remaining: 28.6s
    94:	learn: 0.0605416	total: 25.6s	remaining: 28.3s
    95:	learn: 0.0605277	total: 25.8s	remaining: 28s
    96:	learn: 0.0605208	total: 26.1s	remaining: 27.7s
    97:	learn: 0.0605033	total: 26.3s	remaining: 27.4s
    98:	learn: 0.0604935	total: 26.6s	remaining: 27.1s
    99:	learn: 0.0604835	total: 26.9s	remaining: 26.9s
    100:	learn: 0.0604715	total: 27.1s	remaining: 26.6s
    101:	learn: 0.0604623	total: 27.4s	remaining: 26.3s
    102:	learn: 0.0604447	total: 27.7s	remaining: 26.1s
    103:	learn: 0.0604409	total: 28s	remaining: 25.8s
    104:	learn: 0.0604222	total: 28.2s	remaining: 25.6s
    105:	learn: 0.0604105	total: 28.5s	remaining: 25.3s
    106:	learn: 0.0604032	total: 28.8s	remaining: 25s
    107:	learn: 0.0603919	total: 29.1s	remaining: 24.8s
    108:	learn: 0.0603847	total: 29.3s	remaining: 24.5s
    109:	learn: 0.0603725	total: 29.6s	remaining: 24.2s
    110:	learn: 0.0603649	total: 29.9s	remaining: 24s
    111:	learn: 0.0603541	total: 30.2s	remaining: 23.7s
    112:	learn: 0.0603468	total: 30.4s	remaining: 23.4s
    113:	learn: 0.0603375	total: 30.7s	remaining: 23.1s
    114:	learn: 0.0603358	total: 30.9s	remaining: 22.8s
    115:	learn: 0.0603304	total: 31.1s	remaining: 22.6s
    116:	learn: 0.0603140	total: 31.4s	remaining: 22.3s
    117:	learn: 0.0603066	total: 31.7s	remaining: 22s
    118:	learn: 0.0603008	total: 31.9s	remaining: 21.7s
    119:	learn: 0.0602946	total: 32.2s	remaining: 21.5s
    120:	learn: 0.0602816	total: 32.5s	remaining: 21.2s
    121:	learn: 0.0602742	total: 32.8s	remaining: 21s
    122:	learn: 0.0602708	total: 33s	remaining: 20.7s
    123:	learn: 0.0602596	total: 33.3s	remaining: 20.4s
    124:	learn: 0.0602538	total: 33.6s	remaining: 20.1s
    125:	learn: 0.0602485	total: 33.8s	remaining: 19.9s
    126:	learn: 0.0602462	total: 34.1s	remaining: 19.6s
    127:	learn: 0.0602374	total: 34.4s	remaining: 19.3s
    128:	learn: 0.0602281	total: 34.6s	remaining: 19.1s
    129:	learn: 0.0602168	total: 34.9s	remaining: 18.8s
    130:	learn: 0.0602083	total: 35.2s	remaining: 18.5s
    131:	learn: 0.0602049	total: 35.4s	remaining: 18.3s
    132:	learn: 0.0601992	total: 35.7s	remaining: 18s
    133:	learn: 0.0601906	total: 36s	remaining: 17.7s
    134:	learn: 0.0601906	total: 36.2s	remaining: 17.4s
    135:	learn: 0.0601775	total: 36.4s	remaining: 17.1s
    136:	learn: 0.0601697	total: 36.7s	remaining: 16.9s
    137:	learn: 0.0601629	total: 37s	remaining: 16.6s
    138:	learn: 0.0601558	total: 37.3s	remaining: 16.4s
    139:	learn: 0.0601434	total: 37.5s	remaining: 16.1s
    140:	learn: 0.0601378	total: 37.8s	remaining: 15.8s
    141:	learn: 0.0601250	total: 38.1s	remaining: 15.5s
    142:	learn: 0.0601163	total: 38.3s	remaining: 15.3s
    143:	learn: 0.0601102	total: 38.6s	remaining: 15s
    144:	learn: 0.0601018	total: 38.9s	remaining: 14.8s
    145:	learn: 0.0600967	total: 39.2s	remaining: 14.5s
    146:	learn: 0.0600927	total: 39.4s	remaining: 14.2s
    147:	learn: 0.0600772	total: 39.7s	remaining: 13.9s
    148:	learn: 0.0600720	total: 40s	remaining: 13.7s
    149:	learn: 0.0600670	total: 40.2s	remaining: 13.4s
    150:	learn: 0.0600505	total: 40.5s	remaining: 13.1s
    151:	learn: 0.0600301	total: 40.8s	remaining: 12.9s
    152:	learn: 0.0600252	total: 41.1s	remaining: 12.6s
    153:	learn: 0.0600143	total: 41.3s	remaining: 12.3s
    154:	learn: 0.0600082	total: 41.6s	remaining: 12.1s
    155:	learn: 0.0600003	total: 41.9s	remaining: 11.8s
    156:	learn: 0.0599913	total: 42.1s	remaining: 11.5s
    157:	learn: 0.0599848	total: 42.4s	remaining: 11.3s
    158:	learn: 0.0599803	total: 42.7s	remaining: 11s
    159:	learn: 0.0599701	total: 42.9s	remaining: 10.7s
    160:	learn: 0.0599651	total: 43.2s	remaining: 10.5s
    161:	learn: 0.0599600	total: 43.4s	remaining: 10.2s
    162:	learn: 0.0599523	total: 43.7s	remaining: 9.92s
    163:	learn: 0.0599466	total: 44s	remaining: 9.65s
    164:	learn: 0.0599375	total: 44.2s	remaining: 9.38s
    165:	learn: 0.0599319	total: 44.5s	remaining: 9.11s
    166:	learn: 0.0599277	total: 44.7s	remaining: 8.84s
    167:	learn: 0.0599263	total: 45s	remaining: 8.56s
    168:	learn: 0.0599181	total: 45.3s	remaining: 8.3s
    169:	learn: 0.0599105	total: 45.5s	remaining: 8.04s
    170:	learn: 0.0599015	total: 45.8s	remaining: 7.77s
    171:	learn: 0.0598914	total: 46.1s	remaining: 7.5s
    172:	learn: 0.0598840	total: 46.3s	remaining: 7.23s
    173:	learn: 0.0598783	total: 46.6s	remaining: 6.96s
    174:	learn: 0.0598689	total: 46.9s	remaining: 6.7s
    175:	learn: 0.0598600	total: 47.1s	remaining: 6.43s
    176:	learn: 0.0598546	total: 47.4s	remaining: 6.16s
    177:	learn: 0.0598389	total: 47.7s	remaining: 5.89s
    178:	learn: 0.0598347	total: 47.9s	remaining: 5.62s
    179:	learn: 0.0598267	total: 48.1s	remaining: 5.35s
    180:	learn: 0.0598216	total: 48.4s	remaining: 5.08s
    181:	learn: 0.0598171	total: 48.7s	remaining: 4.81s
    182:	learn: 0.0598134	total: 48.9s	remaining: 4.54s
    183:	learn: 0.0598034	total: 49.2s	remaining: 4.28s
    184:	learn: 0.0597998	total: 49.5s	remaining: 4.01s
    185:	learn: 0.0597934	total: 49.7s	remaining: 3.74s
    186:	learn: 0.0597886	total: 50s	remaining: 3.47s
    187:	learn: 0.0597753	total: 50.2s	remaining: 3.21s
    188:	learn: 0.0597581	total: 50.5s	remaining: 2.94s
    189:	learn: 0.0597441	total: 50.8s	remaining: 2.67s
    190:	learn: 0.0597337	total: 51s	remaining: 2.4s
    191:	learn: 0.0597306	total: 51.2s	remaining: 2.13s
    192:	learn: 0.0597233	total: 51.5s	remaining: 1.87s
    193:	learn: 0.0597147	total: 51.7s	remaining: 1.6s
    194:	learn: 0.0597098	total: 52s	remaining: 1.33s
    195:	learn: 0.0597013	total: 52.2s	remaining: 1.07s
    196:	learn: 0.0596952	total: 52.5s	remaining: 800ms
    197:	learn: 0.0596855	total: 52.8s	remaining: 533ms
    198:	learn: 0.0596721	total: 53.1s	remaining: 267ms
    199:	learn: 0.0596705	total: 53.3s	remaining: 0us
    Trained model nº 23/27. Iterations: 200Depth: 8Learning rate: 0.1
    0:	learn: 0.5040680	total: 265ms	remaining: 1m 19s
    1:	learn: 0.3747597	total: 532ms	remaining: 1m 19s
    2:	learn: 0.2774131	total: 805ms	remaining: 1m 19s
    3:	learn: 0.2150157	total: 1.08s	remaining: 1m 19s
    4:	learn: 0.1701417	total: 1.34s	remaining: 1m 19s
    5:	learn: 0.1383840	total: 1.59s	remaining: 1m 18s
    6:	learn: 0.1168210	total: 1.86s	remaining: 1m 17s
    7:	learn: 0.1019590	total: 2.12s	remaining: 1m 17s
    8:	learn: 0.0915974	total: 2.38s	remaining: 1m 17s
    9:	learn: 0.0845304	total: 2.66s	remaining: 1m 17s
    10:	learn: 0.0784835	total: 2.94s	remaining: 1m 17s
    11:	learn: 0.0748310	total: 3.19s	remaining: 1m 16s
    12:	learn: 0.0721101	total: 3.45s	remaining: 1m 16s
    13:	learn: 0.0699696	total: 3.71s	remaining: 1m 15s
    14:	learn: 0.0682439	total: 3.96s	remaining: 1m 15s
    15:	learn: 0.0669627	total: 4.22s	remaining: 1m 15s
    16:	learn: 0.0660660	total: 4.47s	remaining: 1m 14s
    17:	learn: 0.0653093	total: 4.72s	remaining: 1m 14s
    18:	learn: 0.0646638	total: 5.02s	remaining: 1m 14s
    19:	learn: 0.0641881	total: 5.3s	remaining: 1m 14s
    20:	learn: 0.0638147	total: 5.57s	remaining: 1m 14s
    21:	learn: 0.0635175	total: 5.83s	remaining: 1m 13s
    22:	learn: 0.0632533	total: 6.12s	remaining: 1m 13s
    23:	learn: 0.0628616	total: 6.39s	remaining: 1m 13s
    24:	learn: 0.0626834	total: 6.67s	remaining: 1m 13s
    25:	learn: 0.0625484	total: 6.93s	remaining: 1m 13s
    26:	learn: 0.0624232	total: 7.18s	remaining: 1m 12s
    27:	learn: 0.0623159	total: 7.45s	remaining: 1m 12s
    28:	learn: 0.0622386	total: 7.74s	remaining: 1m 12s
    29:	learn: 0.0621623	total: 8.03s	remaining: 1m 12s
    30:	learn: 0.0621043	total: 8.29s	remaining: 1m 11s
    31:	learn: 0.0620557	total: 8.57s	remaining: 1m 11s
    32:	learn: 0.0619992	total: 8.84s	remaining: 1m 11s
    33:	learn: 0.0619421	total: 9.12s	remaining: 1m 11s
    34:	learn: 0.0618981	total: 9.39s	remaining: 1m 11s
    35:	learn: 0.0618618	total: 9.65s	remaining: 1m 10s
    36:	learn: 0.0618257	total: 9.92s	remaining: 1m 10s
    37:	learn: 0.0617920	total: 10.2s	remaining: 1m 10s
    38:	learn: 0.0617557	total: 10.5s	remaining: 1m 10s
    39:	learn: 0.0615805	total: 10.8s	remaining: 1m 9s
    40:	learn: 0.0614751	total: 11s	remaining: 1m 9s
    41:	learn: 0.0614405	total: 11.3s	remaining: 1m 9s
    42:	learn: 0.0614175	total: 11.6s	remaining: 1m 9s
    43:	learn: 0.0613893	total: 11.8s	remaining: 1m 8s
    44:	learn: 0.0613711	total: 12.1s	remaining: 1m 8s
    45:	learn: 0.0613417	total: 12.4s	remaining: 1m 8s
    46:	learn: 0.0613230	total: 12.6s	remaining: 1m 8s
    47:	learn: 0.0613024	total: 12.9s	remaining: 1m 7s
    48:	learn: 0.0612747	total: 13.2s	remaining: 1m 7s
    49:	learn: 0.0612471	total: 13.5s	remaining: 1m 7s
    50:	learn: 0.0612237	total: 13.8s	remaining: 1m 7s
    51:	learn: 0.0611830	total: 14s	remaining: 1m 6s
    52:	learn: 0.0611641	total: 14.3s	remaining: 1m 6s
    53:	learn: 0.0611483	total: 14.6s	remaining: 1m 6s
    54:	learn: 0.0611283	total: 14.8s	remaining: 1m 6s
    55:	learn: 0.0610708	total: 15.1s	remaining: 1m 5s
    56:	learn: 0.0610570	total: 15.4s	remaining: 1m 5s
    57:	learn: 0.0610221	total: 15.7s	remaining: 1m 5s
    58:	learn: 0.0610037	total: 16s	remaining: 1m 5s
    59:	learn: 0.0609944	total: 16.3s	remaining: 1m 5s
    60:	learn: 0.0609778	total: 16.6s	remaining: 1m 5s
    61:	learn: 0.0609634	total: 16.9s	remaining: 1m 4s
    62:	learn: 0.0609510	total: 17.1s	remaining: 1m 4s
    63:	learn: 0.0609444	total: 17.4s	remaining: 1m 4s
    64:	learn: 0.0609314	total: 17.7s	remaining: 1m 3s
    65:	learn: 0.0609072	total: 18s	remaining: 1m 3s
    66:	learn: 0.0608940	total: 18.3s	remaining: 1m 3s
    67:	learn: 0.0608804	total: 18.6s	remaining: 1m 3s
    68:	learn: 0.0608720	total: 18.9s	remaining: 1m 3s
    69:	learn: 0.0608441	total: 19.1s	remaining: 1m 2s
    70:	learn: 0.0608272	total: 19.4s	remaining: 1m 2s
    71:	learn: 0.0608176	total: 19.7s	remaining: 1m 2s
    72:	learn: 0.0608017	total: 20s	remaining: 1m 2s
    73:	learn: 0.0607751	total: 20.3s	remaining: 1m 1s
    74:	learn: 0.0607684	total: 20.5s	remaining: 1m 1s
    75:	learn: 0.0607529	total: 20.8s	remaining: 1m 1s
    76:	learn: 0.0607423	total: 21.1s	remaining: 1m
    77:	learn: 0.0607237	total: 21.3s	remaining: 1m
    78:	learn: 0.0607130	total: 21.6s	remaining: 1m
    79:	learn: 0.0607021	total: 21.9s	remaining: 1m
    80:	learn: 0.0606876	total: 22.2s	remaining: 59.9s
    81:	learn: 0.0606760	total: 22.4s	remaining: 59.6s
    82:	learn: 0.0606676	total: 22.7s	remaining: 59.4s
    83:	learn: 0.0606561	total: 23s	remaining: 59.1s
    84:	learn: 0.0606443	total: 23.3s	remaining: 58.9s
    85:	learn: 0.0606349	total: 23.6s	remaining: 58.6s
    86:	learn: 0.0606223	total: 23.8s	remaining: 58.4s
    87:	learn: 0.0606129	total: 24.1s	remaining: 58.1s
    88:	learn: 0.0606042	total: 24.4s	remaining: 57.9s
    89:	learn: 0.0605979	total: 24.7s	remaining: 57.6s
    90:	learn: 0.0605829	total: 24.9s	remaining: 57.3s
    91:	learn: 0.0605723	total: 25.2s	remaining: 57.1s
    92:	learn: 0.0605589	total: 25.5s	remaining: 56.8s
    93:	learn: 0.0605458	total: 25.8s	remaining: 56.5s
    94:	learn: 0.0605416	total: 26.1s	remaining: 56.2s
    95:	learn: 0.0605277	total: 26.3s	remaining: 56s
    96:	learn: 0.0605208	total: 26.6s	remaining: 55.6s
    97:	learn: 0.0605033	total: 26.9s	remaining: 55.4s
    98:	learn: 0.0604935	total: 27.1s	remaining: 55.1s
    99:	learn: 0.0604835	total: 27.4s	remaining: 54.8s
    100:	learn: 0.0604715	total: 27.7s	remaining: 54.5s
    101:	learn: 0.0604623	total: 27.9s	remaining: 54.2s
    102:	learn: 0.0604447	total: 28.2s	remaining: 54s
    103:	learn: 0.0604409	total: 28.5s	remaining: 53.7s
    104:	learn: 0.0604222	total: 28.8s	remaining: 53.4s
    105:	learn: 0.0604105	total: 29s	remaining: 53.2s
    106:	learn: 0.0604032	total: 29.3s	remaining: 52.9s
    107:	learn: 0.0603919	total: 29.6s	remaining: 52.7s
    108:	learn: 0.0603847	total: 29.9s	remaining: 52.4s
    109:	learn: 0.0603725	total: 30.2s	remaining: 52.2s
    110:	learn: 0.0603649	total: 30.5s	remaining: 51.9s
    111:	learn: 0.0603541	total: 30.8s	remaining: 51.7s
    112:	learn: 0.0603468	total: 31.1s	remaining: 51.4s
    113:	learn: 0.0603375	total: 31.3s	remaining: 51.1s
    114:	learn: 0.0603358	total: 31.5s	remaining: 50.7s
    115:	learn: 0.0603304	total: 31.8s	remaining: 50.5s
    116:	learn: 0.0603140	total: 32.1s	remaining: 50.2s
    117:	learn: 0.0603066	total: 32.4s	remaining: 49.9s
    118:	learn: 0.0603008	total: 32.6s	remaining: 49.7s
    119:	learn: 0.0602946	total: 32.9s	remaining: 49.4s
    120:	learn: 0.0602816	total: 33.2s	remaining: 49.2s
    121:	learn: 0.0602742	total: 33.5s	remaining: 48.9s
    122:	learn: 0.0602708	total: 33.8s	remaining: 48.6s
    123:	learn: 0.0602596	total: 34.1s	remaining: 48.4s
    124:	learn: 0.0602538	total: 34.3s	remaining: 48.1s
    125:	learn: 0.0602485	total: 34.6s	remaining: 47.8s
    126:	learn: 0.0602462	total: 34.8s	remaining: 47.5s
    127:	learn: 0.0602374	total: 35.1s	remaining: 47.2s
    128:	learn: 0.0602281	total: 35.4s	remaining: 46.9s
    129:	learn: 0.0602168	total: 35.7s	remaining: 46.6s
    130:	learn: 0.0602083	total: 35.9s	remaining: 46.4s
    131:	learn: 0.0602049	total: 36.2s	remaining: 46.1s
    132:	learn: 0.0601992	total: 36.5s	remaining: 45.8s
    133:	learn: 0.0601906	total: 36.8s	remaining: 45.6s
    134:	learn: 0.0601906	total: 37s	remaining: 45.2s
    135:	learn: 0.0601775	total: 37.2s	remaining: 44.9s
    136:	learn: 0.0601697	total: 37.5s	remaining: 44.6s
    137:	learn: 0.0601629	total: 37.8s	remaining: 44.3s
    138:	learn: 0.0601558	total: 38s	remaining: 44.1s
    139:	learn: 0.0601434	total: 38.3s	remaining: 43.8s
    140:	learn: 0.0601378	total: 38.6s	remaining: 43.5s
    141:	learn: 0.0601250	total: 38.9s	remaining: 43.2s
    142:	learn: 0.0601163	total: 39.1s	remaining: 43s
    143:	learn: 0.0601102	total: 39.4s	remaining: 42.7s
    144:	learn: 0.0601018	total: 39.7s	remaining: 42.4s
    145:	learn: 0.0600967	total: 40s	remaining: 42.2s
    146:	learn: 0.0600927	total: 40.3s	remaining: 41.9s
    147:	learn: 0.0600772	total: 40.6s	remaining: 41.7s
    148:	learn: 0.0600720	total: 40.9s	remaining: 41.4s
    149:	learn: 0.0600670	total: 41.1s	remaining: 41.1s
    150:	learn: 0.0600505	total: 41.4s	remaining: 40.9s
    151:	learn: 0.0600301	total: 41.7s	remaining: 40.6s
    152:	learn: 0.0600252	total: 42s	remaining: 40.3s
    153:	learn: 0.0600143	total: 42.2s	remaining: 40s
    154:	learn: 0.0600082	total: 42.5s	remaining: 39.7s
    155:	learn: 0.0600003	total: 42.8s	remaining: 39.5s
    156:	learn: 0.0599913	total: 43s	remaining: 39.2s
    157:	learn: 0.0599848	total: 43.3s	remaining: 38.9s
    158:	learn: 0.0599803	total: 43.5s	remaining: 38.6s
    159:	learn: 0.0599701	total: 43.8s	remaining: 38.3s
    160:	learn: 0.0599651	total: 44s	remaining: 38s
    161:	learn: 0.0599600	total: 44.3s	remaining: 37.7s
    162:	learn: 0.0599523	total: 44.6s	remaining: 37.5s
    163:	learn: 0.0599466	total: 44.8s	remaining: 37.2s
    164:	learn: 0.0599375	total: 45.1s	remaining: 36.9s
    165:	learn: 0.0599319	total: 45.3s	remaining: 36.6s
    166:	learn: 0.0599277	total: 45.6s	remaining: 36.3s
    167:	learn: 0.0599263	total: 45.8s	remaining: 36s
    168:	learn: 0.0599181	total: 46.1s	remaining: 35.7s
    169:	learn: 0.0599105	total: 46.4s	remaining: 35.5s
    170:	learn: 0.0599015	total: 46.7s	remaining: 35.2s
    171:	learn: 0.0598914	total: 46.9s	remaining: 34.9s
    172:	learn: 0.0598840	total: 47.2s	remaining: 34.6s
    173:	learn: 0.0598783	total: 47.4s	remaining: 34.4s
    174:	learn: 0.0598689	total: 47.7s	remaining: 34.1s
    175:	learn: 0.0598600	total: 48s	remaining: 33.8s
    176:	learn: 0.0598546	total: 48.2s	remaining: 33.5s
    177:	learn: 0.0598389	total: 48.5s	remaining: 33.2s
    178:	learn: 0.0598347	total: 48.7s	remaining: 32.9s
    179:	learn: 0.0598267	total: 49s	remaining: 32.7s
    180:	learn: 0.0598216	total: 49.3s	remaining: 32.4s
    181:	learn: 0.0598171	total: 49.5s	remaining: 32.1s
    182:	learn: 0.0598134	total: 49.8s	remaining: 31.8s
    183:	learn: 0.0598034	total: 50.1s	remaining: 31.6s
    184:	learn: 0.0597998	total: 50.4s	remaining: 31.3s
    185:	learn: 0.0597934	total: 50.6s	remaining: 31s
    186:	learn: 0.0597886	total: 50.9s	remaining: 30.7s
    187:	learn: 0.0597753	total: 51.1s	remaining: 30.5s
    188:	learn: 0.0597581	total: 51.4s	remaining: 30.2s
    189:	learn: 0.0597441	total: 51.7s	remaining: 29.9s
    190:	learn: 0.0597337	total: 51.9s	remaining: 29.6s
    191:	learn: 0.0597306	total: 52.1s	remaining: 29.3s
    192:	learn: 0.0597233	total: 52.4s	remaining: 29s
    193:	learn: 0.0597147	total: 52.6s	remaining: 28.8s
    194:	learn: 0.0597098	total: 52.9s	remaining: 28.5s
    195:	learn: 0.0597013	total: 53.2s	remaining: 28.2s
    196:	learn: 0.0596952	total: 53.4s	remaining: 27.9s
    197:	learn: 0.0596855	total: 53.7s	remaining: 27.7s
    198:	learn: 0.0596721	total: 54s	remaining: 27.4s
    199:	learn: 0.0596705	total: 54.2s	remaining: 27.1s
    200:	learn: 0.0596626	total: 54.5s	remaining: 26.9s
    201:	learn: 0.0596484	total: 54.8s	remaining: 26.6s
    202:	learn: 0.0596420	total: 55.1s	remaining: 26.3s
    203:	learn: 0.0596312	total: 55.3s	remaining: 26s
    204:	learn: 0.0596200	total: 55.6s	remaining: 25.8s
    205:	learn: 0.0596069	total: 55.9s	remaining: 25.5s
    206:	learn: 0.0596019	total: 56.1s	remaining: 25.2s
    207:	learn: 0.0595908	total: 56.4s	remaining: 24.9s
    208:	learn: 0.0595849	total: 56.7s	remaining: 24.7s
    209:	learn: 0.0595803	total: 56.9s	remaining: 24.4s
    210:	learn: 0.0595732	total: 57.2s	remaining: 24.1s
    211:	learn: 0.0595662	total: 57.5s	remaining: 23.9s
    212:	learn: 0.0595565	total: 57.7s	remaining: 23.6s
    213:	learn: 0.0595532	total: 58s	remaining: 23.3s
    214:	learn: 0.0595405	total: 58.2s	remaining: 23s
    215:	learn: 0.0595311	total: 58.5s	remaining: 22.8s
    216:	learn: 0.0595235	total: 58.8s	remaining: 22.5s
    217:	learn: 0.0595123	total: 59.1s	remaining: 22.2s
    218:	learn: 0.0595059	total: 59.3s	remaining: 21.9s
    219:	learn: 0.0594980	total: 59.6s	remaining: 21.7s
    220:	learn: 0.0594904	total: 59.9s	remaining: 21.4s
    221:	learn: 0.0594845	total: 1m	remaining: 21.1s
    222:	learn: 0.0594750	total: 1m	remaining: 20.8s
    223:	learn: 0.0594713	total: 1m	remaining: 20.6s
    224:	learn: 0.0594651	total: 1m	remaining: 20.3s
    225:	learn: 0.0594587	total: 1m 1s	remaining: 20s
    226:	learn: 0.0594536	total: 1m 1s	remaining: 19.8s
    227:	learn: 0.0594500	total: 1m 1s	remaining: 19.5s
    228:	learn: 0.0594378	total: 1m 1s	remaining: 19.2s
    229:	learn: 0.0594262	total: 1m 2s	remaining: 18.9s
    230:	learn: 0.0594180	total: 1m 2s	remaining: 18.7s
    231:	learn: 0.0594124	total: 1m 2s	remaining: 18.4s
    232:	learn: 0.0594027	total: 1m 3s	remaining: 18.1s
    233:	learn: 0.0593984	total: 1m 3s	remaining: 17.9s
    234:	learn: 0.0593925	total: 1m 3s	remaining: 17.6s
    235:	learn: 0.0593886	total: 1m 3s	remaining: 17.3s
    236:	learn: 0.0593838	total: 1m 4s	remaining: 17s
    237:	learn: 0.0593783	total: 1m 4s	remaining: 16.8s
    238:	learn: 0.0593717	total: 1m 4s	remaining: 16.5s
    239:	learn: 0.0593695	total: 1m 4s	remaining: 16.2s
    240:	learn: 0.0593634	total: 1m 5s	remaining: 16s
    241:	learn: 0.0593573	total: 1m 5s	remaining: 15.7s
    242:	learn: 0.0593487	total: 1m 5s	remaining: 15.4s
    243:	learn: 0.0593372	total: 1m 5s	remaining: 15.1s
    244:	learn: 0.0593309	total: 1m 6s	remaining: 14.9s
    245:	learn: 0.0593219	total: 1m 6s	remaining: 14.6s
    246:	learn: 0.0593161	total: 1m 6s	remaining: 14.3s
    247:	learn: 0.0593112	total: 1m 7s	remaining: 14.1s
    248:	learn: 0.0593035	total: 1m 7s	remaining: 13.8s
    249:	learn: 0.0592961	total: 1m 7s	remaining: 13.5s
    250:	learn: 0.0592885	total: 1m 7s	remaining: 13.2s
    251:	learn: 0.0592800	total: 1m 8s	remaining: 13s
    252:	learn: 0.0592734	total: 1m 8s	remaining: 12.7s
    253:	learn: 0.0592665	total: 1m 8s	remaining: 12.4s
    254:	learn: 0.0592594	total: 1m 8s	remaining: 12.2s
    255:	learn: 0.0592535	total: 1m 9s	remaining: 11.9s
    256:	learn: 0.0592479	total: 1m 9s	remaining: 11.6s
    257:	learn: 0.0592388	total: 1m 9s	remaining: 11.3s
    258:	learn: 0.0592305	total: 1m 9s	remaining: 11.1s
    259:	learn: 0.0592219	total: 1m 10s	remaining: 10.8s
    260:	learn: 0.0592142	total: 1m 10s	remaining: 10.5s
    261:	learn: 0.0592058	total: 1m 10s	remaining: 10.3s
    262:	learn: 0.0592011	total: 1m 10s	remaining: 9.98s
    263:	learn: 0.0591969	total: 1m 11s	remaining: 9.71s
    264:	learn: 0.0591886	total: 1m 11s	remaining: 9.44s
    265:	learn: 0.0591842	total: 1m 11s	remaining: 9.17s
    266:	learn: 0.0591816	total: 1m 12s	remaining: 8.9s
    267:	learn: 0.0591710	total: 1m 12s	remaining: 8.63s
    268:	learn: 0.0591633	total: 1m 12s	remaining: 8.36s
    269:	learn: 0.0591549	total: 1m 12s	remaining: 8.09s
    270:	learn: 0.0591512	total: 1m 13s	remaining: 7.82s
    271:	learn: 0.0591480	total: 1m 13s	remaining: 7.55s
    272:	learn: 0.0591436	total: 1m 13s	remaining: 7.28s
    273:	learn: 0.0591388	total: 1m 13s	remaining: 7.01s
    274:	learn: 0.0591273	total: 1m 14s	remaining: 6.74s
    275:	learn: 0.0591251	total: 1m 14s	remaining: 6.47s
    276:	learn: 0.0591204	total: 1m 14s	remaining: 6.2s
    277:	learn: 0.0591154	total: 1m 14s	remaining: 5.93s
    278:	learn: 0.0591131	total: 1m 15s	remaining: 5.66s
    279:	learn: 0.0591119	total: 1m 15s	remaining: 5.39s
    280:	learn: 0.0591062	total: 1m 15s	remaining: 5.12s
    281:	learn: 0.0591030	total: 1m 15s	remaining: 4.84s
    282:	learn: 0.0590926	total: 1m 16s	remaining: 4.58s
    283:	learn: 0.0590811	total: 1m 16s	remaining: 4.31s
    284:	learn: 0.0590734	total: 1m 16s	remaining: 4.04s
    285:	learn: 0.0590683	total: 1m 16s	remaining: 3.77s
    286:	learn: 0.0590562	total: 1m 17s	remaining: 3.5s
    287:	learn: 0.0590524	total: 1m 17s	remaining: 3.23s
    288:	learn: 0.0590437	total: 1m 17s	remaining: 2.96s
    289:	learn: 0.0590390	total: 1m 18s	remaining: 2.69s
    290:	learn: 0.0590332	total: 1m 18s	remaining: 2.42s
    291:	learn: 0.0590299	total: 1m 18s	remaining: 2.15s
    292:	learn: 0.0590264	total: 1m 18s	remaining: 1.88s
    293:	learn: 0.0590210	total: 1m 19s	remaining: 1.61s
    294:	learn: 0.0590100	total: 1m 19s	remaining: 1.34s
    295:	learn: 0.0590040	total: 1m 19s	remaining: 1.08s
    296:	learn: 0.0590016	total: 1m 19s	remaining: 807ms
    297:	learn: 0.0589990	total: 1m 20s	remaining: 538ms
    298:	learn: 0.0589968	total: 1m 20s	remaining: 269ms
    299:	learn: 0.0589900	total: 1m 20s	remaining: 0us
    Trained model nº 24/27. Iterations: 300Depth: 8Learning rate: 0.1
    0:	learn: 0.5010363	total: 247ms	remaining: 24.5s
    1:	learn: 0.3728707	total: 466ms	remaining: 22.9s
    2:	learn: 0.2757066	total: 681ms	remaining: 22s
    3:	learn: 0.2024066	total: 897ms	remaining: 21.5s
    4:	learn: 0.1609722	total: 1.03s	remaining: 19.6s
    5:	learn: 0.1314675	total: 1.25s	remaining: 19.6s
    6:	learn: 0.1126885	total: 1.48s	remaining: 19.7s
    7:	learn: 0.0989038	total: 1.7s	remaining: 19.5s
    8:	learn: 0.0898129	total: 1.93s	remaining: 19.5s
    9:	learn: 0.0834256	total: 2.15s	remaining: 19.4s
    10:	learn: 0.0784164	total: 2.38s	remaining: 19.2s
    11:	learn: 0.0743813	total: 2.61s	remaining: 19.1s
    12:	learn: 0.0716915	total: 2.83s	remaining: 18.9s
    13:	learn: 0.0696917	total: 3.05s	remaining: 18.7s
    14:	learn: 0.0680134	total: 3.28s	remaining: 18.6s
    15:	learn: 0.0666494	total: 3.48s	remaining: 18.3s
    16:	learn: 0.0657724	total: 3.7s	remaining: 18.1s
    17:	learn: 0.0649136	total: 3.92s	remaining: 17.9s
    18:	learn: 0.0644451	total: 4.13s	remaining: 17.6s
    19:	learn: 0.0638588	total: 4.35s	remaining: 17.4s
    20:	learn: 0.0634764	total: 4.59s	remaining: 17.3s
    21:	learn: 0.0631372	total: 4.83s	remaining: 17.1s
    22:	learn: 0.0628541	total: 5.06s	remaining: 16.9s
    23:	learn: 0.0625510	total: 5.28s	remaining: 16.7s
    24:	learn: 0.0623198	total: 5.5s	remaining: 16.5s
    25:	learn: 0.0621770	total: 5.72s	remaining: 16.3s
    26:	learn: 0.0620237	total: 5.94s	remaining: 16.1s
    27:	learn: 0.0619056	total: 6.16s	remaining: 15.8s
    28:	learn: 0.0617894	total: 6.39s	remaining: 15.6s
    29:	learn: 0.0616392	total: 6.64s	remaining: 15.5s
    30:	learn: 0.0615165	total: 6.88s	remaining: 15.3s
    31:	learn: 0.0614355	total: 7.1s	remaining: 15.1s
    32:	learn: 0.0613789	total: 7.32s	remaining: 14.9s
    33:	learn: 0.0612952	total: 7.54s	remaining: 14.6s
    34:	learn: 0.0612440	total: 7.79s	remaining: 14.5s
    35:	learn: 0.0612159	total: 8.03s	remaining: 14.3s
    36:	learn: 0.0611175	total: 8.28s	remaining: 14.1s
    37:	learn: 0.0610393	total: 8.52s	remaining: 13.9s
    38:	learn: 0.0610011	total: 8.75s	remaining: 13.7s
    39:	learn: 0.0609481	total: 8.97s	remaining: 13.5s
    40:	learn: 0.0608992	total: 9.19s	remaining: 13.2s
    41:	learn: 0.0608550	total: 9.43s	remaining: 13s
    42:	learn: 0.0608063	total: 9.65s	remaining: 12.8s
    43:	learn: 0.0607600	total: 9.88s	remaining: 12.6s
    44:	learn: 0.0607257	total: 10.1s	remaining: 12.3s
    45:	learn: 0.0606951	total: 10.3s	remaining: 12.1s
    46:	learn: 0.0606668	total: 10.5s	remaining: 11.9s
    47:	learn: 0.0606349	total: 10.8s	remaining: 11.7s
    48:	learn: 0.0606057	total: 11s	remaining: 11.4s
    49:	learn: 0.0605822	total: 11.2s	remaining: 11.2s
    50:	learn: 0.0605611	total: 11.4s	remaining: 11s
    51:	learn: 0.0605216	total: 11.7s	remaining: 10.8s
    52:	learn: 0.0604908	total: 11.9s	remaining: 10.6s
    53:	learn: 0.0604644	total: 12.1s	remaining: 10.3s
    54:	learn: 0.0604325	total: 12.3s	remaining: 10.1s
    55:	learn: 0.0603929	total: 12.6s	remaining: 9.89s
    56:	learn: 0.0603593	total: 12.8s	remaining: 9.68s
    57:	learn: 0.0603301	total: 13.1s	remaining: 9.46s
    58:	learn: 0.0603001	total: 13.3s	remaining: 9.25s
    59:	learn: 0.0602750	total: 13.6s	remaining: 9.05s
    60:	learn: 0.0602509	total: 13.8s	remaining: 8.84s
    61:	learn: 0.0602251	total: 14.1s	remaining: 8.62s
    62:	learn: 0.0602057	total: 14.3s	remaining: 8.41s
    63:	learn: 0.0601799	total: 14.6s	remaining: 8.19s
    64:	learn: 0.0601629	total: 14.8s	remaining: 7.97s
    65:	learn: 0.0601346	total: 15s	remaining: 7.74s
    66:	learn: 0.0601069	total: 15.2s	remaining: 7.5s
    67:	learn: 0.0600905	total: 15.5s	remaining: 7.27s
    68:	learn: 0.0600744	total: 15.7s	remaining: 7.05s
    69:	learn: 0.0600361	total: 15.9s	remaining: 6.82s
    70:	learn: 0.0600037	total: 16.2s	remaining: 6.6s
    71:	learn: 0.0599693	total: 16.4s	remaining: 6.38s
    72:	learn: 0.0599501	total: 16.6s	remaining: 6.15s
    73:	learn: 0.0599269	total: 16.9s	remaining: 5.92s
    74:	learn: 0.0599052	total: 17.1s	remaining: 5.69s
    75:	learn: 0.0598932	total: 17.3s	remaining: 5.46s
    76:	learn: 0.0598746	total: 17.5s	remaining: 5.23s
    77:	learn: 0.0598488	total: 17.7s	remaining: 5s
    78:	learn: 0.0598312	total: 17.9s	remaining: 4.77s
    79:	learn: 0.0598052	total: 18.1s	remaining: 4.54s
    80:	learn: 0.0597741	total: 18.4s	remaining: 4.31s
    81:	learn: 0.0597586	total: 18.6s	remaining: 4.08s
    82:	learn: 0.0597422	total: 18.8s	remaining: 3.85s
    83:	learn: 0.0597270	total: 19s	remaining: 3.62s
    84:	learn: 0.0597021	total: 19.2s	remaining: 3.39s
    85:	learn: 0.0596551	total: 19.5s	remaining: 3.17s
    86:	learn: 0.0596219	total: 19.7s	remaining: 2.94s
    87:	learn: 0.0595882	total: 19.9s	remaining: 2.72s
    88:	learn: 0.0595613	total: 20.1s	remaining: 2.49s
    89:	learn: 0.0595482	total: 20.4s	remaining: 2.26s
    90:	learn: 0.0595316	total: 20.6s	remaining: 2.04s
    91:	learn: 0.0595052	total: 20.8s	remaining: 1.81s
    92:	learn: 0.0594911	total: 21s	remaining: 1.58s
    93:	learn: 0.0594765	total: 21.2s	remaining: 1.36s
    94:	learn: 0.0594575	total: 21.5s	remaining: 1.13s
    95:	learn: 0.0594145	total: 21.7s	remaining: 904ms
    96:	learn: 0.0593974	total: 21.9s	remaining: 678ms
    97:	learn: 0.0593821	total: 22.1s	remaining: 451ms
    98:	learn: 0.0593596	total: 22.3s	remaining: 226ms
    99:	learn: 0.0593453	total: 22.6s	remaining: 0us
    Trained model nº 25/27. Iterations: 100Depth: 10Learning rate: 0.1
    0:	learn: 0.4951117	total: 365ms	remaining: 1m 12s
    1:	learn: 0.3561883	total: 664ms	remaining: 1m 5s
    2:	learn: 0.2699763	total: 832ms	remaining: 54.6s
    3:	learn: 0.2049447	total: 1.18s	remaining: 58s
    4:	learn: 0.1634708	total: 1.52s	remaining: 59.5s
    5:	learn: 0.1331380	total: 1.86s	remaining: 1m
    6:	learn: 0.1125367	total: 2.2s	remaining: 1m
    7:	learn: 0.0989998	total: 2.51s	remaining: 1m
    8:	learn: 0.0895771	total: 2.82s	remaining: 59.9s
    9:	learn: 0.0827062	total: 3.15s	remaining: 59.9s
    10:	learn: 0.0777571	total: 3.48s	remaining: 59.9s
    11:	learn: 0.0740450	total: 3.82s	remaining: 59.9s
    12:	learn: 0.0712642	total: 4.15s	remaining: 59.8s
    13:	learn: 0.0691942	total: 4.48s	remaining: 59.5s
    14:	learn: 0.0676235	total: 4.79s	remaining: 59.1s
    15:	learn: 0.0664100	total: 5.11s	remaining: 58.7s
    16:	learn: 0.0655007	total: 5.43s	remaining: 58.5s
    17:	learn: 0.0645760	total: 5.77s	remaining: 58.4s
    18:	learn: 0.0640169	total: 6.09s	remaining: 58s
    19:	learn: 0.0635818	total: 6.41s	remaining: 57.7s
    20:	learn: 0.0632151	total: 6.75s	remaining: 57.5s
    21:	learn: 0.0629174	total: 7.08s	remaining: 57.3s
    22:	learn: 0.0626751	total: 7.41s	remaining: 57s
    23:	learn: 0.0624629	total: 7.75s	remaining: 56.9s
    24:	learn: 0.0623201	total: 8.09s	remaining: 56.6s
    25:	learn: 0.0621606	total: 8.43s	remaining: 56.4s
    26:	learn: 0.0620484	total: 8.75s	remaining: 56.1s
    27:	learn: 0.0619428	total: 9.08s	remaining: 55.8s
    28:	learn: 0.0618412	total: 9.43s	remaining: 55.6s
    29:	learn: 0.0617519	total: 9.77s	remaining: 55.3s
    30:	learn: 0.0616745	total: 10.1s	remaining: 55.1s
    31:	learn: 0.0614788	total: 10.5s	remaining: 54.9s
    32:	learn: 0.0613502	total: 10.8s	remaining: 54.4s
    33:	learn: 0.0613081	total: 11.1s	remaining: 54.2s
    34:	learn: 0.0612533	total: 11.4s	remaining: 53.9s
    35:	learn: 0.0612133	total: 11.8s	remaining: 53.7s
    36:	learn: 0.0611289	total: 12.1s	remaining: 53.3s
    37:	learn: 0.0610914	total: 12.4s	remaining: 53s
    38:	learn: 0.0610511	total: 12.8s	remaining: 52.7s
    39:	learn: 0.0609967	total: 13.1s	remaining: 52.5s
    40:	learn: 0.0609653	total: 13.5s	remaining: 52.2s
    41:	learn: 0.0609229	total: 13.8s	remaining: 51.9s
    42:	learn: 0.0608856	total: 14.2s	remaining: 51.7s
    43:	learn: 0.0608384	total: 14.5s	remaining: 51.4s
    44:	learn: 0.0607970	total: 14.9s	remaining: 51.2s
    45:	learn: 0.0607700	total: 15.2s	remaining: 50.9s
    46:	learn: 0.0607082	total: 15.6s	remaining: 50.7s
    47:	learn: 0.0606778	total: 15.9s	remaining: 50.4s
    48:	learn: 0.0606479	total: 16.3s	remaining: 50.1s
    49:	learn: 0.0605957	total: 16.6s	remaining: 49.8s
    50:	learn: 0.0605808	total: 16.9s	remaining: 49.5s
    51:	learn: 0.0605373	total: 17.3s	remaining: 49.2s
    52:	learn: 0.0604943	total: 17.6s	remaining: 48.9s
    53:	learn: 0.0604527	total: 18s	remaining: 48.6s
    54:	learn: 0.0604267	total: 18.3s	remaining: 48.3s
    55:	learn: 0.0603771	total: 18.7s	remaining: 48s
    56:	learn: 0.0603503	total: 19s	remaining: 47.6s
    57:	learn: 0.0603087	total: 19.3s	remaining: 47.3s
    58:	learn: 0.0602807	total: 19.7s	remaining: 47s
    59:	learn: 0.0602511	total: 20s	remaining: 46.7s
    60:	learn: 0.0602155	total: 20.3s	remaining: 46.4s
    61:	learn: 0.0602006	total: 20.7s	remaining: 46s
    62:	learn: 0.0601804	total: 21s	remaining: 45.7s
    63:	learn: 0.0601590	total: 21.3s	remaining: 45.3s
    64:	learn: 0.0601348	total: 21.6s	remaining: 45s
    65:	learn: 0.0601152	total: 22s	remaining: 44.7s
    66:	learn: 0.0600999	total: 22.3s	remaining: 44.3s
    67:	learn: 0.0600676	total: 22.6s	remaining: 43.9s
    68:	learn: 0.0600450	total: 23s	remaining: 43.6s
    69:	learn: 0.0600209	total: 23.3s	remaining: 43.3s
    70:	learn: 0.0599987	total: 23.7s	remaining: 43s
    71:	learn: 0.0599786	total: 24s	remaining: 42.7s
    72:	learn: 0.0599415	total: 24.3s	remaining: 42.3s
    73:	learn: 0.0599168	total: 24.7s	remaining: 42s
    74:	learn: 0.0599013	total: 25s	remaining: 41.7s
    75:	learn: 0.0598905	total: 25.4s	remaining: 41.4s
    76:	learn: 0.0598737	total: 25.7s	remaining: 41s
    77:	learn: 0.0598393	total: 26s	remaining: 40.7s
    78:	learn: 0.0598250	total: 26.4s	remaining: 40.4s
    79:	learn: 0.0598019	total: 26.7s	remaining: 40.1s
    80:	learn: 0.0597875	total: 27s	remaining: 39.7s
    81:	learn: 0.0597729	total: 27.4s	remaining: 39.4s
    82:	learn: 0.0597616	total: 27.7s	remaining: 39s
    83:	learn: 0.0597423	total: 28s	remaining: 38.7s
    84:	learn: 0.0597088	total: 28.3s	remaining: 38.3s
    85:	learn: 0.0596975	total: 28.6s	remaining: 37.9s
    86:	learn: 0.0596791	total: 29s	remaining: 37.6s
    87:	learn: 0.0596622	total: 29.3s	remaining: 37.3s
    88:	learn: 0.0596562	total: 29.6s	remaining: 36.9s
    89:	learn: 0.0596450	total: 30s	remaining: 36.6s
    90:	learn: 0.0596447	total: 30.2s	remaining: 36.1s
    91:	learn: 0.0596269	total: 30.5s	remaining: 35.8s
    92:	learn: 0.0596195	total: 30.8s	remaining: 35.5s
    93:	learn: 0.0595901	total: 31.1s	remaining: 35.1s
    94:	learn: 0.0595667	total: 31.5s	remaining: 34.8s
    95:	learn: 0.0595569	total: 31.8s	remaining: 34.5s
    96:	learn: 0.0595332	total: 32.2s	remaining: 34.2s
    97:	learn: 0.0595281	total: 32.4s	remaining: 33.8s
    98:	learn: 0.0595141	total: 32.8s	remaining: 33.5s
    99:	learn: 0.0595014	total: 33.1s	remaining: 33.1s
    100:	learn: 0.0594890	total: 33.5s	remaining: 32.8s
    101:	learn: 0.0594704	total: 33.8s	remaining: 32.5s
    102:	learn: 0.0594541	total: 34.2s	remaining: 32.2s
    103:	learn: 0.0594268	total: 34.5s	remaining: 31.8s
    104:	learn: 0.0594145	total: 34.9s	remaining: 31.5s
    105:	learn: 0.0594038	total: 35.2s	remaining: 31.2s
    106:	learn: 0.0593764	total: 35.6s	remaining: 30.9s
    107:	learn: 0.0593597	total: 35.9s	remaining: 30.6s
    108:	learn: 0.0593568	total: 36.3s	remaining: 30.3s
    109:	learn: 0.0593457	total: 36.7s	remaining: 30s
    110:	learn: 0.0593259	total: 37.1s	remaining: 29.7s
    111:	learn: 0.0593038	total: 37.4s	remaining: 29.4s
    112:	learn: 0.0592809	total: 37.8s	remaining: 29.1s
    113:	learn: 0.0592639	total: 38.1s	remaining: 28.8s
    114:	learn: 0.0592473	total: 38.5s	remaining: 28.4s
    115:	learn: 0.0592213	total: 38.8s	remaining: 28.1s
    116:	learn: 0.0592093	total: 39.2s	remaining: 27.8s
    117:	learn: 0.0591947	total: 39.5s	remaining: 27.5s
    118:	learn: 0.0591875	total: 39.9s	remaining: 27.1s
    119:	learn: 0.0591677	total: 40.2s	remaining: 26.8s
    120:	learn: 0.0591486	total: 40.6s	remaining: 26.5s
    121:	learn: 0.0591273	total: 41s	remaining: 26.2s
    122:	learn: 0.0591189	total: 41.3s	remaining: 25.8s
    123:	learn: 0.0591076	total: 41.7s	remaining: 25.5s
    124:	learn: 0.0591010	total: 42s	remaining: 25.2s
    125:	learn: 0.0590890	total: 42.4s	remaining: 24.9s
    126:	learn: 0.0590708	total: 42.8s	remaining: 24.6s
    127:	learn: 0.0590619	total: 43.2s	remaining: 24.3s
    128:	learn: 0.0590515	total: 43.5s	remaining: 23.9s
    129:	learn: 0.0590434	total: 43.8s	remaining: 23.6s
    130:	learn: 0.0590278	total: 44.2s	remaining: 23.3s
    131:	learn: 0.0590200	total: 44.6s	remaining: 23s
    132:	learn: 0.0590006	total: 44.9s	remaining: 22.6s
    133:	learn: 0.0589898	total: 45.3s	remaining: 22.3s
    134:	learn: 0.0589776	total: 45.7s	remaining: 22s
    135:	learn: 0.0589677	total: 46s	remaining: 21.6s
    136:	learn: 0.0589511	total: 46.4s	remaining: 21.3s
    137:	learn: 0.0589377	total: 46.7s	remaining: 21s
    138:	learn: 0.0589197	total: 47.1s	remaining: 20.7s
    139:	learn: 0.0588928	total: 47.5s	remaining: 20.3s
    140:	learn: 0.0588863	total: 47.8s	remaining: 20s
    141:	learn: 0.0588683	total: 48.1s	remaining: 19.7s
    142:	learn: 0.0588574	total: 48.5s	remaining: 19.3s
    143:	learn: 0.0588456	total: 48.9s	remaining: 19s
    144:	learn: 0.0588303	total: 49.2s	remaining: 18.7s
    145:	learn: 0.0588073	total: 49.6s	remaining: 18.3s
    146:	learn: 0.0587980	total: 50s	remaining: 18s
    147:	learn: 0.0587769	total: 50.3s	remaining: 17.7s
    148:	learn: 0.0587582	total: 50.7s	remaining: 17.3s
    149:	learn: 0.0587446	total: 51s	remaining: 17s
    150:	learn: 0.0587371	total: 51.4s	remaining: 16.7s
    151:	learn: 0.0587264	total: 51.7s	remaining: 16.3s
    152:	learn: 0.0587078	total: 52s	remaining: 16s
    153:	learn: 0.0586811	total: 52.4s	remaining: 15.6s
    154:	learn: 0.0586653	total: 52.7s	remaining: 15.3s
    155:	learn: 0.0586505	total: 53.1s	remaining: 15s
    156:	learn: 0.0586315	total: 53.4s	remaining: 14.6s
    157:	learn: 0.0586067	total: 53.7s	remaining: 14.3s
    158:	learn: 0.0585898	total: 54.1s	remaining: 13.9s
    159:	learn: 0.0585781	total: 54.4s	remaining: 13.6s
    160:	learn: 0.0585568	total: 54.8s	remaining: 13.3s
    161:	learn: 0.0585396	total: 55.1s	remaining: 12.9s
    162:	learn: 0.0585230	total: 55.5s	remaining: 12.6s
    163:	learn: 0.0585105	total: 55.8s	remaining: 12.3s
    164:	learn: 0.0584985	total: 56.2s	remaining: 11.9s
    165:	learn: 0.0584945	total: 56.5s	remaining: 11.6s
    166:	learn: 0.0584872	total: 56.9s	remaining: 11.2s
    167:	learn: 0.0584729	total: 57.1s	remaining: 10.9s
    168:	learn: 0.0584635	total: 57.5s	remaining: 10.5s
    169:	learn: 0.0584403	total: 57.8s	remaining: 10.2s
    170:	learn: 0.0584218	total: 58.2s	remaining: 9.87s
    171:	learn: 0.0583910	total: 58.6s	remaining: 9.53s
    172:	learn: 0.0583770	total: 58.9s	remaining: 9.19s
    173:	learn: 0.0583583	total: 59.2s	remaining: 8.85s
    174:	learn: 0.0583372	total: 59.6s	remaining: 8.51s
    175:	learn: 0.0583262	total: 59.9s	remaining: 8.17s
    176:	learn: 0.0583191	total: 1m	remaining: 7.83s
    177:	learn: 0.0582998	total: 1m	remaining: 7.5s
    178:	learn: 0.0582853	total: 1m 1s	remaining: 7.16s
    179:	learn: 0.0582779	total: 1m 1s	remaining: 6.83s
    180:	learn: 0.0582582	total: 1m 1s	remaining: 6.49s
    181:	learn: 0.0582406	total: 1m 2s	remaining: 6.14s
    182:	learn: 0.0582310	total: 1m 2s	remaining: 5.81s
    183:	learn: 0.0582147	total: 1m 2s	remaining: 5.47s
    184:	learn: 0.0582067	total: 1m 3s	remaining: 5.13s
    185:	learn: 0.0581986	total: 1m 3s	remaining: 4.79s
    186:	learn: 0.0581841	total: 1m 3s	remaining: 4.45s
    187:	learn: 0.0581710	total: 1m 4s	remaining: 4.11s
    188:	learn: 0.0581488	total: 1m 4s	remaining: 3.77s
    189:	learn: 0.0581351	total: 1m 5s	remaining: 3.42s
    190:	learn: 0.0581283	total: 1m 5s	remaining: 3.08s
    191:	learn: 0.0581194	total: 1m 5s	remaining: 2.74s
    192:	learn: 0.0581014	total: 1m 6s	remaining: 2.4s
    193:	learn: 0.0580928	total: 1m 6s	remaining: 2.06s
    194:	learn: 0.0580811	total: 1m 6s	remaining: 1.72s
    195:	learn: 0.0580690	total: 1m 7s	remaining: 1.37s
    196:	learn: 0.0580447	total: 1m 7s	remaining: 1.03s
    197:	learn: 0.0580332	total: 1m 7s	remaining: 687ms
    198:	learn: 0.0580222	total: 1m 8s	remaining: 343ms
    199:	learn: 0.0580139	total: 1m 8s	remaining: 0us
    Trained model nº 26/27. Iterations: 200Depth: 10Learning rate: 0.1
    0:	learn: 0.4951117	total: 363ms	remaining: 1m 48s
    1:	learn: 0.3561883	total: 675ms	remaining: 1m 40s
    2:	learn: 0.2699763	total: 852ms	remaining: 1m 24s
    3:	learn: 0.2049447	total: 1.22s	remaining: 1m 30s
    4:	learn: 0.1634708	total: 1.58s	remaining: 1m 33s
    5:	learn: 0.1331380	total: 1.93s	remaining: 1m 34s
    6:	learn: 0.1125367	total: 2.29s	remaining: 1m 35s
    7:	learn: 0.0989998	total: 2.63s	remaining: 1m 36s
    8:	learn: 0.0895771	total: 2.97s	remaining: 1m 36s
    9:	learn: 0.0827062	total: 3.32s	remaining: 1m 36s
    10:	learn: 0.0777571	total: 3.67s	remaining: 1m 36s
    11:	learn: 0.0740450	total: 4.04s	remaining: 1m 36s
    12:	learn: 0.0712642	total: 4.39s	remaining: 1m 37s
    13:	learn: 0.0691942	total: 4.75s	remaining: 1m 36s
    14:	learn: 0.0676235	total: 5.08s	remaining: 1m 36s
    15:	learn: 0.0664100	total: 5.4s	remaining: 1m 35s
    16:	learn: 0.0655007	total: 5.75s	remaining: 1m 35s
    17:	learn: 0.0645760	total: 6.09s	remaining: 1m 35s
    18:	learn: 0.0640169	total: 6.42s	remaining: 1m 35s
    19:	learn: 0.0635818	total: 6.77s	remaining: 1m 34s
    20:	learn: 0.0632151	total: 7.12s	remaining: 1m 34s
    21:	learn: 0.0629174	total: 7.49s	remaining: 1m 34s
    22:	learn: 0.0626751	total: 7.84s	remaining: 1m 34s
    23:	learn: 0.0624629	total: 8.21s	remaining: 1m 34s
    24:	learn: 0.0623201	total: 8.57s	remaining: 1m 34s
    25:	learn: 0.0621606	total: 8.94s	remaining: 1m 34s
    26:	learn: 0.0620484	total: 9.28s	remaining: 1m 33s
    27:	learn: 0.0619428	total: 9.63s	remaining: 1m 33s
    28:	learn: 0.0618412	total: 9.98s	remaining: 1m 33s
    29:	learn: 0.0617519	total: 10.4s	remaining: 1m 33s
    30:	learn: 0.0616745	total: 10.7s	remaining: 1m 33s
    31:	learn: 0.0614788	total: 11.1s	remaining: 1m 33s
    32:	learn: 0.0613502	total: 11.4s	remaining: 1m 32s
    33:	learn: 0.0613081	total: 11.8s	remaining: 1m 32s
    34:	learn: 0.0612533	total: 12.1s	remaining: 1m 31s
    35:	learn: 0.0612133	total: 12.5s	remaining: 1m 31s
    36:	learn: 0.0611289	total: 12.8s	remaining: 1m 31s
    37:	learn: 0.0610914	total: 13.2s	remaining: 1m 30s
    38:	learn: 0.0610511	total: 13.5s	remaining: 1m 30s
    39:	learn: 0.0609967	total: 13.9s	remaining: 1m 30s
    40:	learn: 0.0609653	total: 14.3s	remaining: 1m 30s
    41:	learn: 0.0609229	total: 14.6s	remaining: 1m 29s
    42:	learn: 0.0608856	total: 15s	remaining: 1m 29s
    43:	learn: 0.0608384	total: 15.4s	remaining: 1m 29s
    44:	learn: 0.0607970	total: 15.8s	remaining: 1m 29s
    45:	learn: 0.0607700	total: 16.1s	remaining: 1m 29s
    46:	learn: 0.0607082	total: 16.5s	remaining: 1m 28s
    47:	learn: 0.0606778	total: 16.9s	remaining: 1m 28s
    48:	learn: 0.0606479	total: 17.2s	remaining: 1m 28s
    49:	learn: 0.0605957	total: 17.6s	remaining: 1m 28s
    50:	learn: 0.0605808	total: 18s	remaining: 1m 27s
    51:	learn: 0.0605373	total: 18.4s	remaining: 1m 27s
    52:	learn: 0.0604943	total: 18.7s	remaining: 1m 27s
    53:	learn: 0.0604527	total: 19.1s	remaining: 1m 27s
    54:	learn: 0.0604267	total: 19.5s	remaining: 1m 26s
    55:	learn: 0.0603771	total: 19.8s	remaining: 1m 26s
    56:	learn: 0.0603503	total: 20.2s	remaining: 1m 26s
    57:	learn: 0.0603087	total: 20.5s	remaining: 1m 25s
    58:	learn: 0.0602807	total: 20.9s	remaining: 1m 25s
    59:	learn: 0.0602511	total: 21.3s	remaining: 1m 25s
    60:	learn: 0.0602155	total: 21.6s	remaining: 1m 24s
    61:	learn: 0.0602006	total: 22s	remaining: 1m 24s
    62:	learn: 0.0601804	total: 22.4s	remaining: 1m 24s
    63:	learn: 0.0601590	total: 22.7s	remaining: 1m 23s
    64:	learn: 0.0601348	total: 23s	remaining: 1m 23s
    65:	learn: 0.0601152	total: 23.4s	remaining: 1m 22s
    66:	learn: 0.0600999	total: 23.7s	remaining: 1m 22s
    67:	learn: 0.0600676	total: 24s	remaining: 1m 21s
    68:	learn: 0.0600450	total: 24.3s	remaining: 1m 21s
    69:	learn: 0.0600209	total: 24.7s	remaining: 1m 21s
    70:	learn: 0.0599987	total: 25.1s	remaining: 1m 20s
    71:	learn: 0.0599786	total: 25.4s	remaining: 1m 20s
    72:	learn: 0.0599415	total: 25.8s	remaining: 1m 20s
    73:	learn: 0.0599168	total: 26.1s	remaining: 1m 19s
    74:	learn: 0.0599013	total: 26.5s	remaining: 1m 19s
    75:	learn: 0.0598905	total: 26.8s	remaining: 1m 19s
    76:	learn: 0.0598737	total: 27.1s	remaining: 1m 18s
    77:	learn: 0.0598393	total: 27.5s	remaining: 1m 18s
    78:	learn: 0.0598250	total: 27.8s	remaining: 1m 17s
    79:	learn: 0.0598019	total: 28.2s	remaining: 1m 17s
    80:	learn: 0.0597875	total: 28.5s	remaining: 1m 17s
    81:	learn: 0.0597729	total: 28.9s	remaining: 1m 16s
    82:	learn: 0.0597616	total: 29.2s	remaining: 1m 16s
    83:	learn: 0.0597423	total: 29.6s	remaining: 1m 16s
    84:	learn: 0.0597088	total: 29.9s	remaining: 1m 15s
    85:	learn: 0.0596975	total: 30.2s	remaining: 1m 15s
    86:	learn: 0.0596791	total: 30.6s	remaining: 1m 14s
    87:	learn: 0.0596622	total: 30.9s	remaining: 1m 14s
    88:	learn: 0.0596562	total: 31.3s	remaining: 1m 14s
    89:	learn: 0.0596450	total: 31.6s	remaining: 1m 13s
    90:	learn: 0.0596447	total: 31.8s	remaining: 1m 13s
    91:	learn: 0.0596269	total: 32.1s	remaining: 1m 12s
    92:	learn: 0.0596195	total: 32.5s	remaining: 1m 12s
    93:	learn: 0.0595901	total: 32.8s	remaining: 1m 11s
    94:	learn: 0.0595667	total: 33.2s	remaining: 1m 11s
    95:	learn: 0.0595569	total: 33.5s	remaining: 1m 11s
    96:	learn: 0.0595332	total: 33.9s	remaining: 1m 10s
    97:	learn: 0.0595281	total: 34.1s	remaining: 1m 10s
    98:	learn: 0.0595141	total: 34.5s	remaining: 1m 10s
    99:	learn: 0.0595014	total: 34.8s	remaining: 1m 9s
    100:	learn: 0.0594890	total: 35.2s	remaining: 1m 9s
    101:	learn: 0.0594704	total: 35.5s	remaining: 1m 8s
    102:	learn: 0.0594541	total: 35.9s	remaining: 1m 8s
    103:	learn: 0.0594268	total: 36.2s	remaining: 1m 8s
    104:	learn: 0.0594145	total: 36.6s	remaining: 1m 7s
    105:	learn: 0.0594038	total: 36.9s	remaining: 1m 7s
    106:	learn: 0.0593764	total: 37.2s	remaining: 1m 7s
    107:	learn: 0.0593597	total: 37.6s	remaining: 1m 6s
    108:	learn: 0.0593568	total: 37.9s	remaining: 1m 6s
    109:	learn: 0.0593457	total: 38.3s	remaining: 1m 6s
    110:	learn: 0.0593259	total: 38.6s	remaining: 1m 5s
    111:	learn: 0.0593038	total: 38.9s	remaining: 1m 5s
    112:	learn: 0.0592809	total: 39.3s	remaining: 1m 4s
    113:	learn: 0.0592639	total: 39.6s	remaining: 1m 4s
    114:	learn: 0.0592473	total: 39.9s	remaining: 1m 4s
    115:	learn: 0.0592213	total: 40.2s	remaining: 1m 3s
    116:	learn: 0.0592093	total: 40.6s	remaining: 1m 3s
    117:	learn: 0.0591947	total: 40.9s	remaining: 1m 3s
    118:	learn: 0.0591875	total: 41.2s	remaining: 1m 2s
    119:	learn: 0.0591677	total: 41.6s	remaining: 1m 2s
    120:	learn: 0.0591486	total: 41.9s	remaining: 1m 2s
    121:	learn: 0.0591273	total: 42.3s	remaining: 1m 1s
    122:	learn: 0.0591189	total: 42.6s	remaining: 1m 1s
    123:	learn: 0.0591076	total: 42.9s	remaining: 1m
    124:	learn: 0.0591010	total: 43.3s	remaining: 1m
    125:	learn: 0.0590890	total: 43.7s	remaining: 1m
    126:	learn: 0.0590708	total: 44s	remaining: 60s
    127:	learn: 0.0590619	total: 44.4s	remaining: 59.6s
    128:	learn: 0.0590515	total: 44.7s	remaining: 59.3s
    129:	learn: 0.0590434	total: 45.1s	remaining: 58.9s
    130:	learn: 0.0590278	total: 45.4s	remaining: 58.6s
    131:	learn: 0.0590200	total: 45.8s	remaining: 58.3s
    132:	learn: 0.0590006	total: 46.1s	remaining: 57.9s
    133:	learn: 0.0589898	total: 46.5s	remaining: 57.6s
    134:	learn: 0.0589776	total: 46.8s	remaining: 57.2s
    135:	learn: 0.0589677	total: 47.1s	remaining: 56.8s
    136:	learn: 0.0589511	total: 47.5s	remaining: 56.5s
    137:	learn: 0.0589377	total: 47.9s	remaining: 56.2s
    138:	learn: 0.0589197	total: 48.2s	remaining: 55.9s
    139:	learn: 0.0588928	total: 48.6s	remaining: 55.5s
    140:	learn: 0.0588863	total: 48.9s	remaining: 55.1s
    141:	learn: 0.0588683	total: 49.2s	remaining: 54.7s
    142:	learn: 0.0588574	total: 49.6s	remaining: 54.4s
    143:	learn: 0.0588456	total: 49.9s	remaining: 54.1s
    144:	learn: 0.0588303	total: 50.2s	remaining: 53.7s
    145:	learn: 0.0588073	total: 50.6s	remaining: 53.4s
    146:	learn: 0.0587980	total: 51s	remaining: 53.1s
    147:	learn: 0.0587769	total: 51.3s	remaining: 52.7s
    148:	learn: 0.0587582	total: 51.7s	remaining: 52.4s
    149:	learn: 0.0587446	total: 52s	remaining: 52s
    150:	learn: 0.0587371	total: 52.3s	remaining: 51.6s
    151:	learn: 0.0587264	total: 52.7s	remaining: 51.3s
    152:	learn: 0.0587078	total: 53s	remaining: 50.9s
    153:	learn: 0.0586811	total: 53.3s	remaining: 50.6s
    154:	learn: 0.0586653	total: 53.7s	remaining: 50.2s
    155:	learn: 0.0586505	total: 54s	remaining: 49.8s
    156:	learn: 0.0586315	total: 54.3s	remaining: 49.5s
    157:	learn: 0.0586067	total: 54.7s	remaining: 49.1s
    158:	learn: 0.0585898	total: 55s	remaining: 48.8s
    159:	learn: 0.0585781	total: 55.4s	remaining: 48.5s
    160:	learn: 0.0585568	total: 55.7s	remaining: 48.1s
    161:	learn: 0.0585396	total: 56.1s	remaining: 47.8s
    162:	learn: 0.0585230	total: 56.4s	remaining: 47.4s
    163:	learn: 0.0585105	total: 56.8s	remaining: 47.1s
    164:	learn: 0.0584985	total: 57.1s	remaining: 46.7s
    165:	learn: 0.0584945	total: 57.5s	remaining: 46.4s
    166:	learn: 0.0584872	total: 57.8s	remaining: 46s
    167:	learn: 0.0584729	total: 58s	remaining: 45.6s
    168:	learn: 0.0584635	total: 58.4s	remaining: 45.3s
    169:	learn: 0.0584403	total: 58.7s	remaining: 44.9s
    170:	learn: 0.0584218	total: 59.1s	remaining: 44.6s
    171:	learn: 0.0583910	total: 59.5s	remaining: 44.3s
    172:	learn: 0.0583770	total: 59.8s	remaining: 43.9s
    173:	learn: 0.0583583	total: 1m	remaining: 43.5s
    174:	learn: 0.0583372	total: 1m	remaining: 43.2s
    175:	learn: 0.0583262	total: 1m	remaining: 42.9s
    176:	learn: 0.0583191	total: 1m 1s	remaining: 42.5s
    177:	learn: 0.0582998	total: 1m 1s	remaining: 42.2s
    178:	learn: 0.0582853	total: 1m 1s	remaining: 41.9s
    179:	learn: 0.0582779	total: 1m 2s	remaining: 41.5s
    180:	learn: 0.0582582	total: 1m 2s	remaining: 41.2s
    181:	learn: 0.0582406	total: 1m 2s	remaining: 40.8s
    182:	learn: 0.0582310	total: 1m 3s	remaining: 40.5s
    183:	learn: 0.0582147	total: 1m 3s	remaining: 40.1s
    184:	learn: 0.0582067	total: 1m 3s	remaining: 39.7s
    185:	learn: 0.0581986	total: 1m 4s	remaining: 39.4s
    186:	learn: 0.0581841	total: 1m 4s	remaining: 39s
    187:	learn: 0.0581710	total: 1m 4s	remaining: 38.7s
    188:	learn: 0.0581488	total: 1m 5s	remaining: 38.4s
    189:	learn: 0.0581351	total: 1m 5s	remaining: 38s
    190:	learn: 0.0581283	total: 1m 6s	remaining: 37.7s
    191:	learn: 0.0581194	total: 1m 6s	remaining: 37.3s
    192:	learn: 0.0581014	total: 1m 6s	remaining: 37s
    193:	learn: 0.0580928	total: 1m 7s	remaining: 36.6s
    194:	learn: 0.0580811	total: 1m 7s	remaining: 36.3s
    195:	learn: 0.0580690	total: 1m 7s	remaining: 35.9s
    196:	learn: 0.0580447	total: 1m 8s	remaining: 35.6s
    197:	learn: 0.0580332	total: 1m 8s	remaining: 35.2s
    198:	learn: 0.0580222	total: 1m 8s	remaining: 34.9s
    199:	learn: 0.0580139	total: 1m 9s	remaining: 34.5s
    200:	learn: 0.0580022	total: 1m 9s	remaining: 34.2s
    201:	learn: 0.0579859	total: 1m 9s	remaining: 33.8s
    202:	learn: 0.0579655	total: 1m 10s	remaining: 33.5s
    203:	learn: 0.0579461	total: 1m 10s	remaining: 33.2s
    204:	learn: 0.0579354	total: 1m 10s	remaining: 32.8s
    205:	learn: 0.0579204	total: 1m 11s	remaining: 32.5s
    206:	learn: 0.0579132	total: 1m 11s	remaining: 32.1s
    207:	learn: 0.0579069	total: 1m 11s	remaining: 31.8s
    208:	learn: 0.0578936	total: 1m 12s	remaining: 31.4s
    209:	learn: 0.0578750	total: 1m 12s	remaining: 31.1s
    210:	learn: 0.0578529	total: 1m 12s	remaining: 30.8s
    211:	learn: 0.0578377	total: 1m 13s	remaining: 30.4s
    212:	learn: 0.0578308	total: 1m 13s	remaining: 30.1s
    213:	learn: 0.0578233	total: 1m 13s	remaining: 29.7s
    214:	learn: 0.0578143	total: 1m 14s	remaining: 29.4s
    215:	learn: 0.0578105	total: 1m 14s	remaining: 29s
    216:	learn: 0.0577904	total: 1m 14s	remaining: 28.7s
    217:	learn: 0.0577743	total: 1m 15s	remaining: 28.3s
    218:	learn: 0.0577683	total: 1m 15s	remaining: 28s
    219:	learn: 0.0577493	total: 1m 16s	remaining: 27.6s
    220:	learn: 0.0577278	total: 1m 16s	remaining: 27.3s
    221:	learn: 0.0577127	total: 1m 16s	remaining: 26.9s
    222:	learn: 0.0577032	total: 1m 17s	remaining: 26.6s
    223:	learn: 0.0576911	total: 1m 17s	remaining: 26.3s
    224:	learn: 0.0576785	total: 1m 17s	remaining: 25.9s
    225:	learn: 0.0576623	total: 1m 18s	remaining: 25.6s
    226:	learn: 0.0576436	total: 1m 18s	remaining: 25.2s
    227:	learn: 0.0576344	total: 1m 18s	remaining: 24.9s
    228:	learn: 0.0576227	total: 1m 19s	remaining: 24.5s
    229:	learn: 0.0576184	total: 1m 19s	remaining: 24.2s
    230:	learn: 0.0576084	total: 1m 19s	remaining: 23.8s
    231:	learn: 0.0576005	total: 1m 20s	remaining: 23.5s
    232:	learn: 0.0575896	total: 1m 20s	remaining: 23.2s
    233:	learn: 0.0575809	total: 1m 20s	remaining: 22.8s
    234:	learn: 0.0575783	total: 1m 21s	remaining: 22.5s
    235:	learn: 0.0575621	total: 1m 21s	remaining: 22.1s
    236:	learn: 0.0575497	total: 1m 21s	remaining: 21.7s
    237:	learn: 0.0575428	total: 1m 22s	remaining: 21.4s
    238:	learn: 0.0575383	total: 1m 22s	remaining: 21.1s
    239:	learn: 0.0575296	total: 1m 22s	remaining: 20.7s
    240:	learn: 0.0575173	total: 1m 23s	remaining: 20.4s
    241:	learn: 0.0575063	total: 1m 23s	remaining: 20s
    242:	learn: 0.0574952	total: 1m 23s	remaining: 19.7s
    243:	learn: 0.0574812	total: 1m 24s	remaining: 19.3s
    244:	learn: 0.0574720	total: 1m 24s	remaining: 19s
    245:	learn: 0.0574514	total: 1m 24s	remaining: 18.6s
    246:	learn: 0.0574423	total: 1m 25s	remaining: 18.3s
    247:	learn: 0.0574301	total: 1m 25s	remaining: 17.9s
    248:	learn: 0.0574176	total: 1m 25s	remaining: 17.6s
    249:	learn: 0.0574063	total: 1m 26s	remaining: 17.3s
    250:	learn: 0.0573982	total: 1m 26s	remaining: 16.9s
    251:	learn: 0.0573886	total: 1m 26s	remaining: 16.6s
    252:	learn: 0.0573806	total: 1m 27s	remaining: 16.2s
    253:	learn: 0.0573655	total: 1m 27s	remaining: 15.9s
    254:	learn: 0.0573511	total: 1m 27s	remaining: 15.5s
    255:	learn: 0.0573325	total: 1m 28s	remaining: 15.2s
    256:	learn: 0.0573241	total: 1m 28s	remaining: 14.8s
    257:	learn: 0.0573179	total: 1m 29s	remaining: 14.5s
    258:	learn: 0.0573147	total: 1m 29s	remaining: 14.1s
    259:	learn: 0.0573063	total: 1m 29s	remaining: 13.8s
    260:	learn: 0.0572983	total: 1m 30s	remaining: 13.5s
    261:	learn: 0.0572913	total: 1m 30s	remaining: 13.1s
    262:	learn: 0.0572854	total: 1m 30s	remaining: 12.8s
    263:	learn: 0.0572776	total: 1m 31s	remaining: 12.4s
    264:	learn: 0.0572631	total: 1m 31s	remaining: 12.1s
    265:	learn: 0.0572588	total: 1m 31s	remaining: 11.7s
    266:	learn: 0.0572531	total: 1m 32s	remaining: 11.4s
    267:	learn: 0.0572436	total: 1m 32s	remaining: 11s
    268:	learn: 0.0572341	total: 1m 32s	remaining: 10.7s
    269:	learn: 0.0572267	total: 1m 33s	remaining: 10.3s
    270:	learn: 0.0572166	total: 1m 33s	remaining: 10s
    271:	learn: 0.0572110	total: 1m 33s	remaining: 9.65s
    272:	learn: 0.0572078	total: 1m 34s	remaining: 9.31s
    273:	learn: 0.0571934	total: 1m 34s	remaining: 8.96s
    274:	learn: 0.0571816	total: 1m 34s	remaining: 8.62s
    275:	learn: 0.0571713	total: 1m 35s	remaining: 8.27s
    276:	learn: 0.0571666	total: 1m 35s	remaining: 7.93s
    277:	learn: 0.0571621	total: 1m 35s	remaining: 7.58s
    278:	learn: 0.0571502	total: 1m 36s	remaining: 7.24s
    279:	learn: 0.0571455	total: 1m 36s	remaining: 6.89s
    280:	learn: 0.0571329	total: 1m 36s	remaining: 6.55s
    281:	learn: 0.0571225	total: 1m 37s	remaining: 6.21s
    282:	learn: 0.0571147	total: 1m 37s	remaining: 5.86s
    283:	learn: 0.0570981	total: 1m 37s	remaining: 5.52s
    284:	learn: 0.0570894	total: 1m 38s	remaining: 5.17s
    285:	learn: 0.0570782	total: 1m 38s	remaining: 4.83s
    286:	learn: 0.0570621	total: 1m 38s	remaining: 4.48s
    287:	learn: 0.0570501	total: 1m 39s	remaining: 4.14s
    288:	learn: 0.0570364	total: 1m 39s	remaining: 3.79s
    289:	learn: 0.0570289	total: 1m 39s	remaining: 3.44s
    290:	learn: 0.0570158	total: 1m 40s	remaining: 3.1s
    291:	learn: 0.0570084	total: 1m 40s	remaining: 2.75s
    292:	learn: 0.0569940	total: 1m 40s	remaining: 2.41s
    293:	learn: 0.0569834	total: 1m 41s	remaining: 2.06s
    294:	learn: 0.0569729	total: 1m 41s	remaining: 1.72s
    295:	learn: 0.0569562	total: 1m 41s	remaining: 1.38s
    296:	learn: 0.0569369	total: 1m 42s	remaining: 1.03s
    297:	learn: 0.0569284	total: 1m 42s	remaining: 689ms
    298:	learn: 0.0569235	total: 1m 43s	remaining: 345ms
    299:	learn: 0.0569123	total: 1m 43s	remaining: 0us
    Trained model nº 27/27. Iterations: 300Depth: 10Learning rate: 0.1



```python
# We create a dictionary -> to data frame of the metrics of our grid search
hyper_search = pd.DataFrame({
    "Iterations": hyper_iterations,
    "Depth": hyper_depths,
    "Learning rate": hyper_lr,
    "ROC AUC Train": roc_scores_train,
    "PR AUC Train": pr_scores_train,
    "ROC AUC Val": roc_scores_val,
    "PR AUC Val": pr_scores_val})

```


```python
hyper_search.sort_values(by = "PR AUC Val", ascending = False)
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
      <th>Iterations</th>
      <th>Depth</th>
      <th>Learning rate</th>
      <th>ROC AUC Train</th>
      <th>PR AUC Train</th>
      <th>ROC AUC Val</th>
      <th>PR AUC Val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>100</td>
      <td>6</td>
      <td>0.10</td>
      <td>0.844489</td>
      <td>0.217333</td>
      <td>0.846711</td>
      <td>0.203752</td>
    </tr>
    <tr>
      <th>13</th>
      <td>200</td>
      <td>8</td>
      <td>0.05</td>
      <td>0.846405</td>
      <td>0.232359</td>
      <td>0.846042</td>
      <td>0.203281</td>
    </tr>
    <tr>
      <th>11</th>
      <td>300</td>
      <td>6</td>
      <td>0.05</td>
      <td>0.845562</td>
      <td>0.222988</td>
      <td>0.845337</td>
      <td>0.203170</td>
    </tr>
    <tr>
      <th>10</th>
      <td>200</td>
      <td>6</td>
      <td>0.05</td>
      <td>0.843946</td>
      <td>0.216978</td>
      <td>0.845314</td>
      <td>0.202769</td>
    </tr>
    <tr>
      <th>12</th>
      <td>100</td>
      <td>8</td>
      <td>0.05</td>
      <td>0.843196</td>
      <td>0.217298</td>
      <td>0.846390</td>
      <td>0.202072</td>
    </tr>
    <tr>
      <th>21</th>
      <td>100</td>
      <td>8</td>
      <td>0.10</td>
      <td>0.846537</td>
      <td>0.231929</td>
      <td>0.846658</td>
      <td>0.201967</td>
    </tr>
    <tr>
      <th>14</th>
      <td>300</td>
      <td>8</td>
      <td>0.05</td>
      <td>0.848731</td>
      <td>0.241686</td>
      <td>0.845584</td>
      <td>0.201794</td>
    </tr>
    <tr>
      <th>15</th>
      <td>100</td>
      <td>10</td>
      <td>0.05</td>
      <td>0.846841</td>
      <td>0.235169</td>
      <td>0.845838</td>
      <td>0.201509</td>
    </tr>
    <tr>
      <th>16</th>
      <td>200</td>
      <td>10</td>
      <td>0.05</td>
      <td>0.851528</td>
      <td>0.258252</td>
      <td>0.845943</td>
      <td>0.201120</td>
    </tr>
    <tr>
      <th>8</th>
      <td>300</td>
      <td>10</td>
      <td>0.01</td>
      <td>0.843007</td>
      <td>0.220242</td>
      <td>0.845291</td>
      <td>0.200637</td>
    </tr>
    <tr>
      <th>24</th>
      <td>100</td>
      <td>10</td>
      <td>0.10</td>
      <td>0.853234</td>
      <td>0.258276</td>
      <td>0.847333</td>
      <td>0.199950</td>
    </tr>
    <tr>
      <th>19</th>
      <td>200</td>
      <td>6</td>
      <td>0.10</td>
      <td>0.846732</td>
      <td>0.227935</td>
      <td>0.845485</td>
      <td>0.199715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>300</td>
      <td>8</td>
      <td>0.01</td>
      <td>0.841974</td>
      <td>0.207906</td>
      <td>0.844903</td>
      <td>0.198991</td>
    </tr>
    <tr>
      <th>17</th>
      <td>300</td>
      <td>10</td>
      <td>0.05</td>
      <td>0.855415</td>
      <td>0.272773</td>
      <td>0.845026</td>
      <td>0.198568</td>
    </tr>
    <tr>
      <th>20</th>
      <td>300</td>
      <td>6</td>
      <td>0.10</td>
      <td>0.848278</td>
      <td>0.236306</td>
      <td>0.843887</td>
      <td>0.197848</td>
    </tr>
    <tr>
      <th>9</th>
      <td>100</td>
      <td>6</td>
      <td>0.05</td>
      <td>0.840146</td>
      <td>0.205096</td>
      <td>0.843230</td>
      <td>0.197681</td>
    </tr>
    <tr>
      <th>22</th>
      <td>200</td>
      <td>8</td>
      <td>0.10</td>
      <td>0.850704</td>
      <td>0.249350</td>
      <td>0.845402</td>
      <td>0.197339</td>
    </tr>
    <tr>
      <th>7</th>
      <td>200</td>
      <td>10</td>
      <td>0.01</td>
      <td>0.840540</td>
      <td>0.210980</td>
      <td>0.843509</td>
      <td>0.196132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200</td>
      <td>8</td>
      <td>0.01</td>
      <td>0.840634</td>
      <td>0.201243</td>
      <td>0.843903</td>
      <td>0.194009</td>
    </tr>
    <tr>
      <th>23</th>
      <td>300</td>
      <td>8</td>
      <td>0.10</td>
      <td>0.853623</td>
      <td>0.264995</td>
      <td>0.843456</td>
      <td>0.193929</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300</td>
      <td>6</td>
      <td>0.01</td>
      <td>0.839880</td>
      <td>0.198099</td>
      <td>0.843766</td>
      <td>0.193261</td>
    </tr>
    <tr>
      <th>25</th>
      <td>200</td>
      <td>10</td>
      <td>0.10</td>
      <td>0.858973</td>
      <td>0.287500</td>
      <td>0.843962</td>
      <td>0.191292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>200</td>
      <td>6</td>
      <td>0.01</td>
      <td>0.838959</td>
      <td>0.193575</td>
      <td>0.842649</td>
      <td>0.188996</td>
    </tr>
    <tr>
      <th>6</th>
      <td>100</td>
      <td>10</td>
      <td>0.01</td>
      <td>0.839319</td>
      <td>0.197323</td>
      <td>0.843932</td>
      <td>0.187239</td>
    </tr>
    <tr>
      <th>26</th>
      <td>300</td>
      <td>10</td>
      <td>0.10</td>
      <td>0.864176</td>
      <td>0.310399</td>
      <td>0.841785</td>
      <td>0.187003</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>8</td>
      <td>0.01</td>
      <td>0.837441</td>
      <td>0.192520</td>
      <td>0.842513</td>
      <td>0.185447</td>
    </tr>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>6</td>
      <td>0.01</td>
      <td>0.838362</td>
      <td>0.187541</td>
      <td>0.840940</td>
      <td>0.182169</td>
    </tr>
  </tbody>
</table>
</div>



## Insights
The best parameters for catboost are:

Iterations = 100, Depth = 6, Learning rate = 0.10

## XGBoost (Work in progress)

Grid search, random search? Let's explore random search this time. ()


```python

```

## Comparison of all our models
We will only compare the different models with the best hyperparameters
- Baseline Model based on global popularity
- Logistic Regression with Ridge regularization: C = 1e-6
- Random Forest: n_trees = 50
- Gradient Boosting Trees: number of estimators = 100, max_depth = 5
- Catoobst: iterations = 100, depth = 6, learning rate = 0.1
- XGBoost: 


```python
# We initialize the subplots
fig1, ax1 = plt.subplots(1, 2, figsize=(14,7))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(1, 2, figsize = (14,7))
fig2.suptitle("Validation metrics")


# RIDGE    
ridge = make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l2", C=1e-6)
)
ridge.fit(X_train[train_cols], y_train)

train_proba = ridge.predict_proba(X_train[train_cols])[:,1]
plot_metrics(f"LR; C=1e-6", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = ridge.predict_proba(X_val[train_cols])[:, 1]
plot_metrics(f"LR; C=1e-6", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

# RANDOM FOREST

rf = RandomForestClassifier(n_estimators = 50) 
rf.fit(X_train[train_cols], y_train)
train_proba = rf.predict_proba(X_train[train_cols])[:, 1]
plot_metrics(f"RF; N_Trees={n}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = rf.predict_proba(X_val[train_cols])[:, 1]
plot_metrics(f"RF; N_Trees={n}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

# GRADIENT BOOSTING TREES
gbt = GradientBoostingClassifier(
        n_estimators = 100,
        max_depth = 5,
        learning_rate = 1
)
        
gbt.fit(X_train[train_cols], y_train)

train_proba = gbt.predict_proba(X_train[train_cols])[:, 1]
plot_metrics(f"GBT; N_estimators={100}, Depth={5}, lr={1}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = gbt.predict_proba(X_val[train_cols])[:, 1]
plot_metrics(f"GBT; N_estimators={100}, Depth={5}, lr={1}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))


# CATBOOST

cbc = CatBoostClassifier(
    learning_rate = 0.1, depth = 6, iterations = 100
)

cbc.fit(X_train[catboost_cols], y_train, categorical_cols)

train_proba = cbc.predict_proba(X_train[catboost_cols])[:, 1]   
plot_metrics(f"CatBoost; N_iterations={100}, Depth={6}, lr={0.1}", y_pred = train_proba, y_test = train_df[label_col], figure = (fig1, ax1))


val_proba = cbc.predict_proba(X_val[catboost_cols])[:, 1]
plot_metrics(f"CatBoost; N_iterations={100}, Depth={6}, lr={0.1}", y_pred = val_proba, y_test = val_df[label_col], figure = (fig2, ax2))

# BASELINE

plot_metrics("Popularity baseline", y_pred=val_df["global_popularity"], y_test=val_df[label_col])

```

    0:	learn: 0.5062796	total: 234ms	remaining: 23.2s
    1:	learn: 0.3635625	total: 396ms	remaining: 19.4s
    2:	learn: 0.2790112	total: 590ms	remaining: 19.1s
    3:	learn: 0.2163357	total: 751ms	remaining: 18s
    4:	learn: 0.1761711	total: 915ms	remaining: 17.4s
    5:	learn: 0.1453769	total: 1.1s	remaining: 17.2s
    6:	learn: 0.1224737	total: 1.26s	remaining: 16.8s
    7:	learn: 0.1075807	total: 1.39s	remaining: 16s
    8:	learn: 0.0948237	total: 1.55s	remaining: 15.7s
    9:	learn: 0.0863867	total: 1.71s	remaining: 15.4s
    10:	learn: 0.0816316	total: 1.88s	remaining: 15.2s
    11:	learn: 0.0770182	total: 2.03s	remaining: 14.9s
    12:	learn: 0.0744387	total: 2.19s	remaining: 14.7s
    13:	learn: 0.0715663	total: 2.35s	remaining: 14.5s
    14:	learn: 0.0698181	total: 2.52s	remaining: 14.3s
    15:	learn: 0.0683959	total: 2.68s	remaining: 14.1s
    16:	learn: 0.0672321	total: 2.85s	remaining: 13.9s
    17:	learn: 0.0664944	total: 3.01s	remaining: 13.7s
    18:	learn: 0.0657608	total: 3.17s	remaining: 13.5s
    19:	learn: 0.0651507	total: 3.33s	remaining: 13.3s
    20:	learn: 0.0645686	total: 3.49s	remaining: 13.1s
    21:	learn: 0.0639546	total: 3.65s	remaining: 12.9s
    22:	learn: 0.0637070	total: 3.82s	remaining: 12.8s
    23:	learn: 0.0634488	total: 3.98s	remaining: 12.6s
    24:	learn: 0.0633218	total: 4.13s	remaining: 12.4s
    25:	learn: 0.0630896	total: 4.29s	remaining: 12.2s
    26:	learn: 0.0629238	total: 4.43s	remaining: 12s
    27:	learn: 0.0628407	total: 4.59s	remaining: 11.8s
    28:	learn: 0.0627308	total: 4.76s	remaining: 11.7s
    29:	learn: 0.0626335	total: 4.92s	remaining: 11.5s
    30:	learn: 0.0625828	total: 5.08s	remaining: 11.3s
    31:	learn: 0.0624641	total: 5.23s	remaining: 11.1s
    32:	learn: 0.0623752	total: 5.4s	remaining: 11s
    33:	learn: 0.0623402	total: 5.57s	remaining: 10.8s
    34:	learn: 0.0622707	total: 5.73s	remaining: 10.6s
    35:	learn: 0.0622026	total: 5.89s	remaining: 10.5s
    36:	learn: 0.0620094	total: 6.05s	remaining: 10.3s
    37:	learn: 0.0619877	total: 6.21s	remaining: 10.1s
    38:	learn: 0.0619638	total: 6.38s	remaining: 9.98s
    39:	learn: 0.0619420	total: 6.54s	remaining: 9.81s
    40:	learn: 0.0619036	total: 6.7s	remaining: 9.64s
    41:	learn: 0.0618839	total: 6.87s	remaining: 9.48s
    42:	learn: 0.0618435	total: 7.03s	remaining: 9.32s
    43:	learn: 0.0618083	total: 7.19s	remaining: 9.14s
    44:	learn: 0.0617900	total: 7.34s	remaining: 8.98s
    45:	learn: 0.0617716	total: 7.51s	remaining: 8.81s
    46:	learn: 0.0617418	total: 7.67s	remaining: 8.65s
    47:	learn: 0.0617109	total: 7.83s	remaining: 8.48s
    48:	learn: 0.0617002	total: 7.99s	remaining: 8.31s
    49:	learn: 0.0616810	total: 8.15s	remaining: 8.15s
    50:	learn: 0.0616650	total: 8.31s	remaining: 7.98s
    51:	learn: 0.0616485	total: 8.47s	remaining: 7.82s
    52:	learn: 0.0615890	total: 8.64s	remaining: 7.66s
    53:	learn: 0.0615764	total: 8.81s	remaining: 7.5s
    54:	learn: 0.0615659	total: 8.98s	remaining: 7.34s
    55:	learn: 0.0615510	total: 9.14s	remaining: 7.18s
    56:	learn: 0.0615319	total: 9.3s	remaining: 7.02s
    57:	learn: 0.0615208	total: 9.46s	remaining: 6.85s
    58:	learn: 0.0615088	total: 9.61s	remaining: 6.68s
    59:	learn: 0.0615031	total: 9.77s	remaining: 6.51s
    60:	learn: 0.0614894	total: 9.93s	remaining: 6.35s
    61:	learn: 0.0614729	total: 10.1s	remaining: 6.19s
    62:	learn: 0.0614620	total: 10.3s	remaining: 6.03s
    63:	learn: 0.0614490	total: 10.4s	remaining: 5.86s
    64:	learn: 0.0614375	total: 10.6s	remaining: 5.7s
    65:	learn: 0.0614238	total: 10.8s	remaining: 5.54s
    66:	learn: 0.0614101	total: 10.9s	remaining: 5.38s
    67:	learn: 0.0614003	total: 11.1s	remaining: 5.21s
    68:	learn: 0.0613556	total: 11.2s	remaining: 5.05s
    69:	learn: 0.0613428	total: 11.4s	remaining: 4.88s
    70:	learn: 0.0613369	total: 11.6s	remaining: 4.72s
    71:	learn: 0.0613251	total: 11.7s	remaining: 4.55s
    72:	learn: 0.0613130	total: 11.9s	remaining: 4.39s
    73:	learn: 0.0613052	total: 12s	remaining: 4.23s
    74:	learn: 0.0612568	total: 12.2s	remaining: 4.06s
    75:	learn: 0.0612366	total: 12.3s	remaining: 3.89s
    76:	learn: 0.0612225	total: 12.5s	remaining: 3.73s
    77:	learn: 0.0612143	total: 12.6s	remaining: 3.57s
    78:	learn: 0.0612062	total: 12.8s	remaining: 3.4s
    79:	learn: 0.0611914	total: 13s	remaining: 3.24s
    80:	learn: 0.0611826	total: 13.1s	remaining: 3.08s
    81:	learn: 0.0611756	total: 13.3s	remaining: 2.92s
    82:	learn: 0.0611680	total: 13.4s	remaining: 2.75s
    83:	learn: 0.0611612	total: 13.6s	remaining: 2.59s
    84:	learn: 0.0611543	total: 13.8s	remaining: 2.43s
    85:	learn: 0.0611450	total: 13.9s	remaining: 2.27s
    86:	learn: 0.0611375	total: 14.1s	remaining: 2.11s
    87:	learn: 0.0611304	total: 14.3s	remaining: 1.94s
    88:	learn: 0.0611246	total: 14.4s	remaining: 1.78s
    89:	learn: 0.0611194	total: 14.6s	remaining: 1.62s
    90:	learn: 0.0611165	total: 14.7s	remaining: 1.46s
    91:	learn: 0.0611102	total: 14.9s	remaining: 1.29s
    92:	learn: 0.0611029	total: 15s	remaining: 1.13s
    93:	learn: 0.0610907	total: 15.2s	remaining: 970ms
    94:	learn: 0.0610806	total: 15.4s	remaining: 809ms
    95:	learn: 0.0610769	total: 15.5s	remaining: 647ms
    96:	learn: 0.0610705	total: 15.7s	remaining: 485ms
    97:	learn: 0.0610662	total: 15.9s	remaining: 324ms
    98:	learn: 0.0610606	total: 16s	remaining: 162ms
    99:	learn: 0.0610455	total: 16.2s	remaining: 0us



    
![png](push_notifications_project_files/push_notifications_project_59_1.png)
    



    
![png](push_notifications_project_files/push_notifications_project_59_2.png)
    



    
![png](push_notifications_project_files/push_notifications_project_59_3.png)
    


## Insights
- Our best model is CatBoost with iterations = 100, depth = 6, learning rate = 0.1
- We got a PR-AUC of 0.20 and a ROC-AUC of 0.85

We save our CatBoost model as a .joblib file


```python
# Create the folder 'models'
models_folder = 'models'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)
```


```python
# Full file path where we will save our best performing model
file_path = os.path.join(models_folder, 'CatBoost.joblib')
```


```python
# save the model using joblib
joblib.dump(cbc, file_path)

# Confirm it has been saved
print(f"Model saved in {file_path}")
```

    Model saved in models/CatBoost.joblib



```python
'''file_path = 'models/CatBoost.joblib'

# Load the model from the file
loaded_model = joblib.load(file_path)

# Now, 'loaded_model' contains the model previously saved

# Use example:
train_prediction = loaded_model.predict(train_df[catboost_cols])

print("Prediction result:", train_prediction)'''
```




    'file_path = \'models/CatBoost.joblib\'\n\n# Load the model from the file\nloaded_model = joblib.load(file_path)\n\n# Now, \'loaded_model\' contains the model previously saved\n\n# Use example:\ntrain_prediction = loaded_model.predict(train_df[catboost_cols])\n\nprint("Prediction result:", train_prediction)'




```python

```
