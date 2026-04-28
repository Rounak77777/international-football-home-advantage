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

# Cache data: Download once, store in memory
@st.cache_data 
def load_data():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    
    # Dropping neutral venues 
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

# Title & Introduction
st.title("Analyzing Home Advantage in International Football")
st.write("Do home teams have a statistical advantage in international football matches? Let's prove it mathematically.")

st.divider()
st.subheader("⚙️ Analysis Parameters")

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

filtered_df = df[df.year == year]

with st.expander(f"View Raw Dataset for {year}"):
    st.write("*Note: Neutral venues are strictly excluded to maintain analytical integrity.*")
    st.dataframe(filtered_df.head(10)) 

# Descriptive Analysis
st.subheader(f"Match Analysis for {year}")
results = filtered_df["result"].value_counts()

col1, col2 = st.columns(2)
with col1:
    pie_data = pd.DataFrame({'Outcome': results.index, 'Count': results.values})
    fig_pie = px.pie(pie_data, values='Count', names='Outcome', hole=0.4, 
                     title="Win Proportion",
                     color='Outcome', color_discrete_map={"Home Win": "#0072B2", "Away Win": "#D55E00", "Draw": "#7f7f7f"})
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(filtered_df, x="goal_difference", nbins=20, 
                            title="Goal Difference Distribution",
                            color_discrete_sequence=['#0072B2']) 
    fig_hist.update_layout(xaxis_title="Home Team Goal Difference", yaxis_title="Number of Matches")
    st.plotly_chart(fig_hist, use_container_width=True)

# Hypothesis Testing
st.subheader("Statistical Proof: Is the advantage real?")

decisive_games = filtered_df[filtered_df["result"].isin(["Home Win", "Away Win"])]
n_trials = len(decisive_games)
n_home_wins = len(decisive_games[decisive_games["result"] == "Home Win"])

if n_trials > 0:
    test_result = binomtest(n_home_wins, n_trials, p=0.5, alternative='greater')
    p_val = test_result.pvalue
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Decisive Matches", f"{n_trials:,}")
    metric_col2.metric("Home Wins", f"{n_home_wins:,}")
    formatted_p = f"{p_val:.5f}" if p_val >= 0.00001 else f"{p_val:.5e}"
    metric_col3.metric("p-value", formatted_p)
    
    alpha = 0.05
    if p_val < alpha:
        st.success("Result: We reject the null hypothesis. The home advantage is mathematically significant at the 5% level.")
    else:
        st.error("Result: We fail to reject the null hypothesis. There is not enough evidence to prove a significant home advantage.")
else:
    st.write("Not enough data to run the test for this year.")

# Team & Factor Analysis
st.divider()
st.subheader(f"Team-Specific Variance: Top Home Teams ({year})")
st.write("Does the home advantage apply equally to everyone, or do specific teams dominate their home turf?")

home_counts = filtered_df['home_team'].value_counts()
valid_teams = home_counts[home_counts >= 5].index 

if len(valid_teams) > 0:
    team_df = filtered_df[filtered_df['home_team'].isin(valid_teams)]
    team_stats = team_df.groupby('home_team')['result'].apply(lambda x: (x == 'Home Win').mean() * 100).reset_index()
    team_stats.columns = ['Team', 'Home Win %']
    team_stats = team_stats.sort_values(by='Home Win %', ascending=True).tail(10) 
    
    fig_teams = px.bar(team_stats, x='Home Win %', y='Team', orientation='h', 
                       title="Top 10 Home Win Percentages (Minimum 5 Games)", 
                       color_discrete_sequence=['#0072B2'])
    
    fig_teams.update_layout(
        xaxis_title="Home Win %", 
        yaxis_title="Team",
        xaxis=dict(range=[0, 100]) 
    )
    st.plotly_chart(fig_teams, use_container_width=True)
    
    st.info("""
    **Analytical Interpretation:** This chart demonstrates that "Home Advantage" is not a uniform constant. Global averages often mask extreme variance driven by underlying team quality, hostile crowd sizes, and geographical anomalies. 
    
    For instance, elite squads playing in massive stadiums, or teams playing in extreme high altitude environments (like Bolivia at Estadio Hernando Siles), extract a vastly higher mathematical advantage from their home environment than average teams. Furthermore, by enforcing a strict minimum threshold of $N \ge 5$ home games, we deliberately filter out statistical noise, preventing a team that simply won 1 out of 1 home games from falsely appearing as a 100% dominant outlier.
    """)
else:
    st.warning("Not enough data to calculate reliable team-specific win rates for this year. (Requires teams to have played at least 5 home games).")

st.divider()
st.subheader(f"Factor Analysis: The Impact of Match Stakes ({year})")
st.write("We lack environmental data, but we can test a psychological factor: do home teams maintain their advantage when the pressure is high?")

factor_df = filtered_df.copy()
factor_df['match_importance'] = np.where(factor_df['tournament'] == 'Friendly', 'Friendly', 'Competitive')

importance_stats = factor_df.groupby('match_importance')['result'].apply(lambda x: (x == 'Home Win').mean() * 100).reset_index()
importance_stats.columns = ['Match Type', 'Home Win %']

fig_factor = px.bar(importance_stats, x='Match Type', y='Home Win %',
                    title="Home Advantage: Friendlies vs. Competitive Matches",
                    color='Match Type',
                    color_discrete_map={'Friendly': '#7f7f7f', 'Competitive': '#D55E00'}) 
                    
fig_factor.update_layout(yaxis=dict(range=[0, 100]), yaxis_title="Home Win %")
st.plotly_chart(fig_factor, use_container_width=True)

st.info("""
**Analytical Interpretation:** Because the raw dataset lacks environmental variables (like altitude or attendance figures), we must engineer features from the available dimensions. By classifying the 'tournament' column into binary states ('Friendly' vs. 'Competitive'), we test if pressure acts as a confounding variable. 

If the home win rate drops during competitive matches, it suggests that when the stakes are absolute, the psychological weight of playing away is mitigated by the away team's increased tactical discipline and motivation. If it rises, it implies hostile tournament crowds amplify the pressure on visiting squads.
""")

# Macro Trend Analysis
st.divider()
st.subheader("Macro Trend: Is Home Advantage Dying?")
st.write("Let's look at the home win percentage across the entire century.")

yearly_stats = df.groupby('year')['result'].value_counts(normalize=True).unstack().fillna(0)
yearly_stats['Home Win %'] = yearly_stats['Home Win'] * 100

fig_line = px.line(yearly_stats, x=yearly_stats.index, y='Home Win %', 
                   title="Home Win Percentage Over Time",
                   labels={'year': 'Year', 'Home Win %': 'Home Win %'})
fig_line.update_traces(mode="lines+markers", line_color="#0072B2")
st.plotly_chart(fig_line, use_container_width=True)

st.info("""
**Analytical Interpretation:** Historically, the home win percentage has experienced a gradual, undeniable decline. 
In the early to mid 20th century, home advantage was massive due to grueling, multi-week travel conditions for away teams, lack of standardized refereeing, and highly hostile crowds influencing subjective officiating. 

As the game modernized with chartered flights, sports science, professionalized global referee standards, and the introduction of technology like VAR (Video Assistant Referee), the environmental friction of playing away has been heavily mitigated. The math reflects this reality: the gap between home and away performance is systematically shrinking.
""")