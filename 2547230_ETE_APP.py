import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try: from nltk.corpus import stopwords as nltk_stopwords
except Exception: nltk_stopwords = None

st.set_page_config(page_title="GATEWAYS-2025 Fest Insights Dashboard", layout="wide", page_icon="bar_chart")

STOPWORDS = {"and", "the", "to", "of", "in", "on", "for", "very", "good", "well", "event", "session", "slight", "needs"}
POS_WORDS = {"excellent", "creative", "engaging", "informative", "challenging", "useful", "structured", "organized", "practical", "learning", "exposure", "fun", "interesting"}
NEG_WORDS = {"improvement", "timing", "needs"}

def get_stopwords() -> set:
    try: return set(nltk_stopwords.words("english")) | STOPWORDS
    except: return STOPWORDS

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "C5-FestDataset - fest_dataset.csv")
    df.columns = df.columns.str.strip()
    df[["Rating", "Amount Paid"]] = df[["Rating", "Amount Paid"]].apply(pd.to_numeric, errors="coerce")
    for c in df.columns.intersection(["College", "Place", "State", "Event Name", "Event Type", "Feedback on Fest"]):
        df[c] = df[c].astype(str).str.strip()
    return df.dropna(subset=["Rating", "Amount Paid"])

def tokenize_feedback(series: pd.Series) -> pd.Series:
    words = series.fillna("").str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.split().explode()
    return words[(words.str.len() > 2) & ~words.isin(get_stopwords())].dropna()

def classify_feedback(text: str) -> str:
    tokens = set(re.findall(r"[a-z]+", str(text).lower()))
    pos, neg = len(tokens & POS_WORDS), len(tokens & NEG_WORDS)
    return "Positive" if pos > neg else ("Needs Improvement" if neg > pos else "Neutral")

def build_sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filter Panel"); st.sidebar.caption("Use filters to drill down.")
    return {
        **{k: st.sidebar.multiselect(col, sorted(df[col].dropna().unique()), default=sorted(df[col].dropna().unique()))
           for col, k in [("State", "states"), ("College", "colleges"), ("Event Type", "event_types"), ("Event Name", "events")]},
        "rating_range": st.sidebar.slider("Rating Range", 1, 5, (1, 5))
    }

def filter_data(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    filtered = df[df["State"].isin(f["states"]) & df["College"].isin(f["colleges"]) & 
                  df["Event Type"].isin(f["event_types"]) & df["Event Name"].isin(f["events"]) & 
                  df["Rating"].between(*f["rating_range"])].copy()
    filtered["Feedback Sentiment"] = filtered["Feedback on Fest"].apply(classify_feedback)
    return filtered

def render_kpis(df: pd.DataFrame) -> None:
    cols, metrics = st.columns(5), [("Participants", len(df)), ("Colleges", df["College"].nunique()), 
               ("States", df["State"].nunique()), ("Avg Rating", f"{df['Rating'].mean() if len(df) else 0:.1f}/5"), 
               ("Revenue", f"₹ {df['Amount Paid'].sum():,.0f}")]
    for c, (l, v) in zip(cols, metrics): c.metric(l, v)

def plot_participation_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    ev_counts = df["Event Name"].value_counts().reset_index(name="Participants")
    col_counts = df["College"].value_counts().reset_index(name="Participants")
    
    c1.plotly_chart(px.bar(ev_counts, x="Event Name", y="Participants", color="Participants", color_continuous_scale="Teal", title="Event-wise Participation"), use_container_width=True)
    c2.plotly_chart(px.bar(col_counts, x="Participants", y="College", orientation="h", color="Participants", color_continuous_scale="Teal", title="College-wise Participation").update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)

    st.markdown("---\n### Participation Summary")
    if not df.empty:
        st.write(f"**Event Popularity:** **{ev_counts.iloc[0]['Event Name']}** is the most sought-after event, capturing **{ev_counts.iloc[0]['Participants']}** active registrations.")
        st.write(f"**Institutional Engagement:** **{col_counts.iloc[0]['College']}** shows highest engagement with **{col_counts.iloc[0]['Participants']}** participants.")

    st.markdown("---")
    c3, c4 = st.columns(2)
    type_counts = df["Event Type"].value_counts().reset_index(name="Participants")
    c3.plotly_chart(px.pie(type_counts, values="Participants", names="Event Type", hole=0.45, title="Participation Split by Event Type", color_discrete_sequence=px.colors.sequential.Teal), use_container_width=True)

    rev_ev = df.groupby("Event Name", as_index=False)["Amount Paid"].sum().sort_values("Amount Paid", ascending=False)
    c4.plotly_chart(px.bar(rev_ev, x="Event Name", y="Amount Paid", color="Amount Paid", color_continuous_scale="YlGnBu", title="Revenue by Event"), use_container_width=True)

    st.markdown("---\n### Financial & Format Summary")
    if not df.empty:
        st.write(f"**Event Formats:** The **{type_counts.iloc[0]['Event Type']}** format is the crowd's favorite, drawing **{type_counts.iloc[0]['Participants']}** attendees.")
        st.write(f"**Financial Driver:** **{rev_ev.iloc[0]['Event Name']}** generated the peak financial return at **₹{rev_ev.iloc[0]['Amount Paid']:,.0f}**, making up **{rev_ev.iloc[0]['Amount Paid']/max(1, rev_ev['Amount Paid'].sum())*100:.1f}%** of revenue.")

def plot_state_map(df: pd.DataFrame) -> None:
    counts = df["State"].value_counts().reset_index(name="Participants")
    counts["State"] = counts["State"].replace({"Tamil Nadu": "Tamilnadu", "Telangana": "Telengana", "Chhattisgarh": "Chhattishgarh"})
    shape_path = Path(__file__).parent / "India-State-and-Country-Shapefile-Updated-Jan-2020-master/India_State_Boundary.shp"
    
    try:
        import geopandas as gpd, matplotlib.pyplot as plt
        map_df = gpd.read_file(shape_path).merge(counts, left_on="State_Name", right_on="State", how="left").fillna({"Participants": 0})
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        map_df[map_df["Participants"] == 0].plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.5)
        map_df[map_df["Participants"] > 0].plot(ax=ax, column="Participants", cmap="YlGnBu", linewidth=0.8, edgecolor="gray", legend=True, legend_kwds={"shrink": 0.5})
        ax.set_axis_off(); st.pyplot(fig)

        st.markdown("---\n### Regional Analytics")
        if not df.empty:
            tc = df["State"].value_counts().max()
            st.write(f"**Dominant Region:** **{df['State'].mode()[0]}** leads with **{tc}** participants (**{tc/len(df)*100:.1f}%**).")
            st.write("**Expansion Scope:** Gray states have zero participation, indicating potential for marketing.")
            st.markdown("#### State-wise Participant Counts")
            st.dataframe(counts[["State", "Participants"]], use_container_width=True, hide_index=True)
    except Exception as e: st.error(f"Map error: {e}")

def plot_feedback_analysis(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    rat_dist = df["Rating"].value_counts().reset_index(name="Count").sort_values("Rating")
    c1.plotly_chart(px.bar(rat_dist, x="Rating", y="Count", text="Count", color="Rating", color_continuous_scale="Teal", title="Rating Distribution").update_traces(textposition="outside"), use_container_width=True)

    avg_rat = df.groupby("Event Name", as_index=False)["Rating"].mean().sort_values("Rating", ascending=False)
    c2.plotly_chart(px.bar(avg_rat, x="Event Name", y="Rating", color="Rating", color_continuous_scale="YlGnBu", title="Average Rating by Event").update_layout(yaxis_range=[0, 5]), use_container_width=True)

    st.markdown("---\n### Overall Rating Trends")
    if not df.empty:
        st.write(f"**Rating Breakdown:** The majority of participants provided positive scores, maintaining strong average ratings across the board.")
        st.write(f"**Top Performing Event(s):** **{avg_rat.iloc[0]['Event Name']}** leads with an outstanding average rating of **{avg_rat.iloc[0]['Rating']:.1f}/5**.")

    st.markdown("---")
    c3, c4 = st.columns(2)
    sent_counts = df["Feedback Sentiment"].value_counts().reset_index(name="Count")
    c3.plotly_chart(px.pie(sent_counts, values="Count", names="Feedback Sentiment", title="Feedback Sentiment Split", hole=0.35, color="Feedback Sentiment", color_discrete_map={"Positive": "#1C818D", "Neutral": "#41B6C4", "Needs Improvement": "#EDF8B1"}).update_traces(marker=dict(line=dict(color='#FFFFFF', width=2))), use_container_width=True)

    terms = tokenize_feedback(df["Feedback on Fest"]).value_counts().head(12).reset_index()
    terms.columns = ["Term", "Frequency"]
    c4.plotly_chart(px.bar(terms, x="Frequency", y="Term", orientation="h", color="Frequency", color_continuous_scale="Teal", title="Most Frequent Feedback Terms").update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)

    st.markdown("---\n### Sentiment & Qualitative Feedback")
    if not df.empty:
        st.write(f"**Sentiment Overview:** Visualized text analytics indicate that **{sent_counts.iloc[0]['Feedback Sentiment']}** responses dominate the feedback pool with **{sent_counts.iloc[0]['Count']}** entries.")
        st.write(f"**Recurring Themes:** The most frequently mentioned keyword is **'{terms.iloc[0]['Term']}'**, appearing **{terms.iloc[0]['Frequency']}** times prominently in the student reviews.")

def show_key_insights(df: pd.DataFrame) -> None:
    if df.empty: return st.warning("No data available.")
    st.markdown("### Key Highlights")
    card = lambda t: f'<div style="background-color:#F4FBFB;padding:15px;border-radius:5px;margin-bottom:15px;border-left:5px solid #1C818D;"><span style="color:#004D40;font-size:16px;">{t}</span></div>'
    c1, c2 = st.columns(2)
    c1.markdown(card(f"<b>Highest participation</b> from <b>{df['State'].mode()[0]}</b>.") + card(f"<b>Most popular event</b> is <b>{df['Event Name'].mode()[0]}</b>."), unsafe_allow_html=True)
    c2.markdown(card(f"<b>Most active college</b> is <b>{df['College'].mode()[0]}</b>.") + card(f"<b>Best-rated event (avg)</b> is <b>{df.groupby('Event Name')['Rating'].mean().idxmax()}</b>."), unsafe_allow_html=True)

def show_feedback_samples(df: pd.DataFrame) -> None:
    if df.empty: return
    st.markdown("### Sample Participant Feedback\n")
    c1, c2, c3, c4 = st.columns(4)
    sents = c1.multiselect("Filter by Sentiment:", ["Positive", "Neutral", "Needs Improvement"])
    events = c2.multiselect("Filter by Event:", sorted(df["Event Name"].unique()))
    ratings = c3.multiselect("Filter by Rating:", sorted(df["Rating"].unique()))
    query = c4.text_input("Search Keywords:")
    
    view = df.copy()
    if sents: view = view[view["Feedback Sentiment"].isin(sents)]
    if events: view = view[view["Event Name"].isin(events)]
    if ratings: view = view[view["Rating"].isin(ratings)]
    if query: view = view[view["Feedback on Fest"].str.contains(query, case=False, na=False)]
    
    st.dataframe(view[["Student Name", "College", "Event Name", "Rating", "Feedback on Fest", "Feedback Sentiment"]].head(15), use_container_width=True, hide_index=True)

def main() -> None:
    st.markdown('<style>[data-testid="stSidebar"] { background-color: #F4FBFB; }</style>', unsafe_allow_html=True)
    st.title("GATEWAYS-2025 National Level Fest Dashboard")
    st.caption("Interactive analytics for participation trends, feedback interpretation, and organizer decision support.")
    df = load_data()
    f_df = filter_data(df, build_sidebar_filters(df))
    if f_df.empty: return st.error("No records match current filters.")
    
    render_kpis(f_df)
    st.markdown("---"); show_key_insights(f_df)
    t1, t2, t3 = st.tabs(["Participation Analysis", "State-wise India Map", "Feedback and Ratings"])
    with t1: st.subheader("Participation Trends"); plot_participation_charts(f_df)
    with t2: st.subheader("State-wise Participants in India"); plot_state_map(f_df)
    with t3: st.subheader("Feedback and Rating Analysis"); plot_feedback_analysis(f_df); st.markdown("---"); show_feedback_samples(f_df)

if __name__ == "__main__": main()
