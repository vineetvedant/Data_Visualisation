import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the data
@st.cache_data
def load_data():
    df = pd.read_json("jsondata.json")
    return df

df = load_data()

# Handle missing values and adjust data types
df['end_year'] = pd.to_numeric(df['end_year'], errors='coerce').fillna(0).astype(int)
df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce').fillna(0)
df['relevance'] = pd.to_numeric(df['relevance'], errors='coerce').fillna(0)
df['likelihood'] = pd.to_numeric(df['likelihood'], errors='coerce').fillna(0)
df['topic'] = df['topic'].fillna('Unknown')
df['sector'] = df['sector'].fillna('Unknown')
df['region'] = df['region'].fillna('Unknown')
df['pestle'] = df['pestle'].fillna('Unknown')
df['source'] = df['source'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Encode categorical variables for the model
label_encoders = {}
for column in ['topic', 'sector', 'region', 'pestle', 'source', 'country']:
    le = LabelEncoder()
    df[column + '_encoded'] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Prepare data for training the model
model_data = df[['intensity', 'relevance', 'end_year', 'sector_encoded', 'region_encoded', 'pestle_encoded', 'topic_encoded', 'source_encoded', 'country_encoded', 'likelihood']].dropna()

# Ensure all data is numeric and drop rows with invalid data
model_data = model_data.replace([np.inf, -np.inf, ''], np.nan).dropna()

# Split the data into training and testing sets
X = model_data.drop('likelihood', axis=1)
y = model_data['likelihood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Display model accuracy
st.write(f"Model Mean Absolute Error: {mae:.2f}")

# Sidebar filters
st.sidebar.header("Filters")

# End Year filter
end_years = sorted(df['end_year'].dropna().unique())
end_years = [year for year in end_years if year != 0]
selected_end_year = st.sidebar.selectbox("Select End Year", ["All"] + [str(year) for year in end_years])

# Topic filter
topics = sorted(label_encoders['topic'].classes_)
selected_topic = st.sidebar.selectbox("Select Topic", ["All"] + topics)

# Sector filter
sectors = sorted(label_encoders['sector'].classes_)
selected_sector = st.sidebar.selectbox("Select Sector", ["All"] + sectors)

# Region filter
regions = sorted(label_encoders['region'].classes_)
selected_region = st.sidebar.selectbox("Select Region", ["All"] + regions)

# PESTLE filter
pestles = sorted(label_encoders['pestle'].classes_)
selected_pestle = st.sidebar.selectbox("Select PESTLE", ["All"] + pestles)

# Source filter
sources = sorted(label_encoders['source'].classes_)
selected_source = st.sidebar.selectbox("Select Source", ["All"] + sources)

# Country filter
countries = sorted(label_encoders['country'].classes_)
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

# Apply filters
if selected_end_year != "All":
    df = df[df['end_year'] == int(selected_end_year)]
if selected_topic != "All":
    df = df[df['topic'] == selected_topic]
if selected_sector != "All":
    df = df[df['sector'] == selected_sector]
if selected_region != "All":
    df = df[df['region'] == selected_region]
if selected_pestle != "All":
    df = df[df['pestle'] == selected_pestle]
if selected_source != "All":
    df = df[df['source'] == selected_source]
if selected_country != "All":
    df = df[df['country'] == selected_country]

# Display filtered data information
st.write("Number of rows after filtering:", len(df))

# Main content
st.title("Data Visualization Dashboard")

# 1. Intensity vs Likelihood scatter plot
st.subheader("Intensity vs Likelihood")
if not df.empty:
    fig = px.scatter(df, x="intensity", y="likelihood", hover_data=['title'], color="topic")
    st.plotly_chart(fig)
else:
    st.write("No data available for Intensity vs Likelihood.")

# 2. Relevance Over Time line plot
st.subheader("Relevance Over Time")
if not df.empty:
    fig = px.line(df, x="added", y="relevance", color="topic", title="Relevance Over Time")
    st.plotly_chart(fig)
else:
    st.write("No data available for Relevance Over Time.")

# 3. Region Distribution pie chart
st.subheader("Region Distribution")
if not df.empty:
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    fig = px.pie(region_counts, values='Count', names='Region', title="Region Distribution")
    st.plotly_chart(fig)
else:
    st.write("No data available for Region Distribution.")

# 4. Top 10 Topics bar chart
st.subheader("Top 10 Topics")
if not df.empty and 'topic' in df.columns:
    topic_counts = df['topic'].value_counts().nlargest(10).reset_index()
    topic_counts.columns = ['Topic', 'Count']
    fig = px.bar(topic_counts, x='Topic', y='Count', title="Top 10 Topics")
    st.plotly_chart(fig)
else:
    st.write("No data available for Top 10 Topics.")

# 5. PESTLE analysis bar chart
st.subheader("PESTLE Analysis")
if not df.empty and 'pestle' in df.columns:
    pestle_counts = df['pestle'].value_counts().reset_index()
    pestle_counts.columns = ['PESTLE Category', 'Count']
    fig = px.bar(pestle_counts, x='PESTLE Category', y='Count', title="PESTLE Analysis")
    st.plotly_chart(fig)
else:
    st.write("No data available for PESTLE Analysis.")

# 6. Heatmap for Correlation between Numerical Variables
st.subheader("Correlation Heatmap")
if not df.empty:
    corr_matrix = df[['intensity', 'relevance', 'likelihood']].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis'))
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig)
else:
    st.write("No data available for Correlation Heatmap.")

# Prediction Section
st.subheader("Predict Likelihood")
intensity = st.number_input("Intensity", min_value=0, max_value=100, value=10)
relevance = st.number_input("Relevance", min_value=0, max_value=10, value=5)
end_year = st.number_input("End Year", min_value=0, max_value=2100, value=2025)
sector = st.selectbox("Sector", list(label_encoders['sector'].classes_))
region = st.selectbox("Region", list(label_encoders['region'].classes_))
pestle = st.selectbox("PESTLE", list(label_encoders['pestle'].classes_))
topic = st.selectbox("Topic", list(label_encoders['topic'].classes_))
source = st.selectbox("Source", list(label_encoders['source'].classes_))
country = st.selectbox("Country", list(label_encoders['country'].classes_))

# Encode input values
encoded_values = [
    intensity,
    relevance,
    end_year,
    label_encoders['sector'].transform([sector])[0],
    label_encoders['region'].transform([region])[0],
    label_encoders['pestle'].transform([pestle])[0],
    label_encoders['topic'].transform([topic])[0],
    label_encoders['source'].transform([source])[0],
    label_encoders['country'].transform([country])[0]
]

# Make prediction
predicted_likelihood = model.predict([encoded_values])[0]
st.write(f"Predicted Likelihood: {predicted_likelihood:.2f}")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df)
