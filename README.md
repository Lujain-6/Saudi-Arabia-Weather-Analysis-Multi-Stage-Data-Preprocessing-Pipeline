## 🛠️ Data Preprocessing Pipeline

This section covers the data cleaning, feature engineering, and preparation steps applied to the Saudi Arabia Weather dataset before feeding it into the machine learning models.

### 1. Feature Drop & Dimensionality Reduction
After extracting essential temporal features (Year, Month, Day), redundant or non-informative columns were removed. For instance, the `minute` column (which contained only zero values) and the original `date` column were dropped to streamline the dataset:

```python
# Dropping redundant and non-informative features
columns_to_drop = ['date', 'minute']
raw_data = raw_data.drop(columns=[col for col in columns_to_drop if col in raw_data.columns])


### 2. Feature Scaling (Normalization)To prevent features with larger magnitudes from dominating the model training, MinMaxScaler was applied to scale all numerical variables (Temperature, Wind Speed, Humidity, Barometer, and Visibility) into a uniform range of $[0, 1]$:Pythonfrom sklearn.preprocessing import MinMaxScaler

# Scaling numerical features to a [0, 1] range
numerical_cols = ['temp', 'wind', 'humidity', 'barometer', 'visibility']
scaler = MinMaxScaler()

raw_data[numerical_cols] = scaler.fit_transform(raw_data[numerical_cols])


### 3. Dataset SplittingFinally, the dataset was split into independent features ($X$) and the target variable ($y$ - rain). The data was then partitioned into 80% for training and 20% for testing to ensure reliable model evaluation:Pythonfrom sklearn.model_selection import train_test_split

# Splitting the features and target variable
X = raw_data.drop(columns=['rain'])
y = raw_data['rain']

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
