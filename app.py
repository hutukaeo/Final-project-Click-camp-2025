import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv('sustainable_waste_management_dataset_2024.csv')

selected_features = ['recyclable_kg', 'organic_kg', 'collection_capacity_kg', 'temp_c']
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

st.write("MSE: ", mean_squared_error(Y_test, Y_pred))
st.write("R squared: ", r2_score(Y_test, Y_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label=' Prediction Line')
ax.set_xlabel('Actual Waste (kg) (Y_test)')
ax.set_ylabel('Predicted Waste (kg) (Y_pred)')
ax.set_title('Predicted vs. Actual Waste (kg)')
ax.legend()
ax.grid(True)

st.pyplot(fig)