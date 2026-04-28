import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import binomtest
import plotly.express as px

st.set_page_config(
    page_title="Home Advantage Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the data: Download once, store in memory
@st.cache_data 
def load_data():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    
    # Drop neutral venues 
    df = df[df["neutral"] == False].copy()
    
    df["goal_difference"] = df["home_score"] - df["away_score"] 

    # Vectorization
    conditions = [
        (df["home_score"] > df["away_score"]),
        (df["home_score"] < df["away_score"])
    ]
    choices = ["Home Win", "Away Win"]
    df["result"] = np.select(conditions, choices, default="Draw")
    
    return df

df = load_data()

st.title("Analyzing Home Advantage in International Football")
st.write("Do home teams have a statistical advantage in international football matches? Let's prove it mathematically.")

# Control Panel 
st.divider()
st.subheader("Analysis Parameter")

# column to contain the slider 
control_col, empty_col = st.columns([1, 1])

with control_col:
    year = st.slider(
        "Select a Year to Analyze:", 
        min_value=int(df.year.min()), 
        max_value=int(df.year.max()), 
        value=2000,
        help="Slide to filter the statistical test and charts below by a specific year."
    )

st.divider()

# filtering data using selected year
filtered_df = df[df.year == year]

# Hiding the Clutter: Putting raw data in an expander
with st.expander(f"View Raw Dataset for {year}"):
    st.write("*Note: Neutral venues are strictly excluded to maintain analytical integrity.*")
    st.dataframe(filtered_df.head(10)) 

st.subheader(f"Match Analysis for {year}")

results = filtered_df["result"].value_counts()

# side-by-side layout
col1, col2 = st.columns(2)

with col1:
    pie_data = pd.DataFrame({'Outcome': results.index, 'Count': results.values})
    # Using a colorblind-safe palette (Blue, Vermilion, Grey)
    fig_pie = px.pie(pie_data, values='Count', names='Outcome', hole=0.4, 
                     title="Win Proportion",
                     color='Outcome', color_discrete_map={"Home Win": "#0072B2", "Away Win": "#D55E00", "Draw": "#7f7f7f"})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(filtered_df, x="goal_difference", nbins=20, 
                            title="Goal Difference Distribution",
                            color_discrete_sequence=['#0072B2']) # Matched to the Home Win color
    fig_hist.update_layout(xaxis_title="Home Team Goal Difference", yaxis_title="Number of Matches")
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Statistical Proof: Is the advantage real?")

# Isolating only decisive matches
decisive_games = filtered_df[filtered_df["result"].isin(["Home Win", "Away Win"])]

n_trials = len(decisive_games)
n_home_wins = len(decisive_games[decisive_games["result"] == "Home Win"])

# Exact binomial test (Right-tailed)
if n_trials > 0:
    test_result = binomtest(n_home_wins, n_trials, p=0.5, alternative='greater')
    p_val = test_result.pvalue
    
    # KPI cards
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Decisive Matches", f"{n_trials:,}")
    metric_col2.metric("Home Wins", f"{n_home_wins:,}")
    metric_col3.metric("p-value", f"{p_val}")
    
    # Interpretation logic
    alpha = 0.05
    if p_val < alpha:
        st.success("Result: We reject the null hypothesis. The home advantage is mathematically significant at the 5% level.")
    else:
        st.error("Result: We fail to reject the null hypothesis. There is not enough evidence to prove a significant home advantage.")
else:
    st.write("Not enough data to run the test for this year.")

st.divider()
st.subheader("Macro Trend: Is Home Advantage Dying?")
st.write("Let's look at the home win percentage across the entire century.")

# Grouping global dataset by year and calculating percentage of home wins
yearly_stats = df.groupby('year')['result'].value_counts(normalize=True).unstack().fillna(0)
yearly_stats['Home Win %'] = yearly_stats['Home Win'] * 100

# Interactive line chart
fig_line = px.line(yearly_stats, x=yearly_stats.index, y='Home Win %', 
                   title="Home Win Percentage Over Time",
                   labels={'year': 'Year', 'Home Win %': 'Home Win %'})
# Trendline styling
fig_line.update_traces(mode="lines+markers", line_color="#0072B2")
st.plotly_chart(fig_line, use_container_width=True)
st.info("""
**Analytical Interpretation:** Historically, the home win percentage has experienced a gradual, undeniable decline. 
In the early to mid 20th century, home advantage was massive due to grueling, multi-week travel conditions for away teams, lack of standardized refereeing, and highly hostile crowds influencing subjective officiating. 

As the game modernized with chartered flights, sports science, professionalized global referee standards, and the introduction of technology like VAR (Video Assistant Referee), the environmental friction of playing away has been heavily mitigated. The math reflects this reality: the gap between home and away performance is systematically shrinking.
""")