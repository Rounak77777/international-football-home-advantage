import streamlit as st
import pandas as pd


url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
df=pd.read_csv(url)
df["result"] = df.apply(
    lambda x: "Neutral" if x["neutral"] 
    else "Home Win" if x.home_score > x.away_score
    else "Away Win" if x.home_score < x.away_score
    else "Draw",
    axis=1
) #axis=1, so that the function is applied to row by row, axis=0 would make it column by column

st.title("Home advantage in international football matches")

st.write("Do home teams have a statistical advantage in international football matches? Let's find out!")

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
year = st.sidebar.slider("Select year",int(df.year.min()),int(df.year.max()),2000) 
filtered_df = df[df.year == year]
results = filtered_df["result"].value_counts()

st.subheader("Dataset")
st.write(df)

percent=df["result"].value_counts(normalize=True) * 100
st.subheader("Percentage of match outcomes")
st.write(percent)

st.subheader("Match Results Distribution")
st.bar_chart(results)




