import streamlit as st
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# page header
st.set_page_config(page_title='Model Training', page_icon=':books:', layout='wide')

# title
st.title('Model Training App')
st.markdown('This app allows you to train and evaluate different regression models.')

# load data set
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

# preprocessing
if st.checkbox('Select columns for analysis'):
    selected_columns = st.multiselect('Select columns:', df.columns)
    df = df[selected_columns]

if st.checkbox('Remove rows with missing values'):
    df.dropna(inplace=True)

# encode category columns
cat_cols = list(df.select_dtypes(include=['object']).columns)
if len(cat_cols) > 0:
    encoder = ce.BinaryEncoder(cols=cat_cols)
    df = encoder.fit_transform(df)

# choose target column
target_col = st.selectbox('Select target column:', df.columns)

# train model and show results
model_type = st.selectbox('Select a model:', ['Linear Regression', 'Random Forest'])
test_size = st.slider('Select test size:', 0.1, 0.5, 0.2, 0.1)
X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
model = LinearRegression()
if st.button('Train Model'):

    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Random Forest':
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    st.write('R-squared:', r2_score(y_test, y_pred))

    # графики
    st.write('Actual vs Predicted:')
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.line_chart(results_df)

    st.write('Feature Importances:')
    if model_type == 'Random Forest':
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    else:
        importances = pd.Series(model.coef_, index=X_train.columns)
    importances.sort_values(ascending=False, inplace=True)
    st.bar_chart(importances)

    # add result table
    y_pred = model.predict(X_test)
    results = pd.concat([y_test.reset_index(drop=True), pd.Series(y_pred)], axis=1)
    st.write('Prediction Results:')
    st.dataframe(results)

# author info
st.markdown("""
*Created by Polina (https://github.com/shimolina-polina)*
""")
