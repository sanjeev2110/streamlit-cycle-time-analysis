import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title of the app
st.title("Cycle Time Data Analysis")

# File uploader
uploaded_file = st.file_uploader("plc_cycle_time_day", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    data['Time'] = pd.to_datetime(data['Time'])
    data['value_sec'] = data['value'] / 1000
    data['Hour'] = data['Time'].dt.hour

    # Descriptive Analysis
    mean_cycle_time = data['value_sec'].mean()
    median_cycle_time = data['value_sec'].median()
    std_dev_cycle_time = data['value_sec'].std()

    st.write(f"Mean Cycle Time: {mean_cycle_time}")
    st.write(f"Median Cycle Time: {median_cycle_time}")
    st.write(f"Standard Deviation of Cycle Time: {std_dev_cycle_time}")

    # Visualize Data
    st.write("### Cycle Time Distribution")
    fig, ax = plt.subplots()
    ax.hist(data['value_sec'], bins=20, alpha=0.7)
    ax.set_title('Cycle Time Distribution')
    ax.set_xlabel('Cycle Time (seconds)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("### Cycle Time Over the Day")
    fig, ax = plt.subplots()
    data.set_index('Time', inplace=True)
    data['value_sec'].plot(ax=ax)
    ax.set_title('Cycle Time Over the Day')
    ax.set_xlabel('Time')
    ax.set_ylabel('Cycle Time (seconds)')
    st.pyplot(fig)

    # Predictive Analysis
    X = data[['Hour']]
    y = data['value_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse}")

    st.write("### Predicted vs Actual Cycle Time")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_title('Predicted vs Actual Cycle Time')
    ax.set_xlabel('Actual Cycle Time (seconds)')
    ax.set_ylabel('Predicted Cycle Time (seconds)')
    st.pyplot(fig)
