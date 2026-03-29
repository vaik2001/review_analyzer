import streamlit as st
from google_play_scraper import reviews, Sort
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Revolut Review Intelligence",
    page_icon="🏦",
    layout="wide"
)

# Load AI Model
@st.cache_resource
def load_model():
    return pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        truncation=True,
        max_length=512
    )

sentiment_model = load_model()

# Scrape Reviews
@st.cache_data(ttl=3600)
def scrape_reviews(count=500):
    result, _ = reviews(
        'com.revolut.revolut',
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        count=count
    )
    df = pd.DataFrame(result)
    df = df[['content', 'score', 'at', 'thumbsUpCount']]
    df = df[df['content'].str.len() > 10]
    df['at'] = pd.to_datetime(df['at'])
    df['date'] = df['at'].dt.date
    return df

# Categorize Reviews
def categorize_review(text):
    text = text.lower()
    if any(w in text for w in ['account', 'frozen', 'blocked', 'closed',
                                'suspended', 'login', 'password', 'access',
                                'verify', 'verification', 'code', 'log in',
                                'scan', 'biometric', 'face', 'passport',
                                'kyc', 'document', 'identity', 'pending']):
        return 'Account & Access'
    elif any(w in text for w in ['transfer', 'send', 'money', 'payment',
                                  'transaction', 'fee', 'charge', 'cost',
                                  'card', 'refund', 'chargeback', 'dispute']):
        return 'Payments & Money'
    elif any(w in text for w in ['crash', 'bug', 'update', 'freeze',
                                  'fix', 'broken', 'open', 'fail', 'slow',
                                  'laggy', 'device', 'notification',
                                  'design', 'confusing', 'interface']):
        return 'App Performance'
    elif any(w in text for w in ['support', 'service', 'help', 'response',
                                  'chat', 'contact', 'agent', 'wait',
                                  'hours', 'reply', 'bot']):
        return 'Customer Support'
    elif any(w in text for w in ['scam', 'fraud', 'stolen', 'fake',
                                  'privacy', 'data', 'security', 'gdpr']):
        return 'Security & Trust'
    elif any(w in text for w in ['invest', 'stocks', 'crypto', 'bitcoin',
                                  'fx', 'currency', 'feature', 'referral']):
        return 'Features & Products'
    else:
        return 'Other'

# Recommendations
recommendations_map = {
    'Account & Access':    'Escalate to account recovery team immediately',
    'Payments & Money':    'Review payment flow and dispute resolution process',
    'App Performance':     'Flag to engineering team for immediate investigation',
    'Customer Support':    'Review support SLA and escalation procedures',
    'Security & Trust':    'Escalate to security and fraud team urgently',
    'Features & Products': 'Log in product backlog for next sprint review',
    'Other':               'Manual review required by product team'
}

# Analyze Sentiment
def analyze_sentiment(df):
    sentiments = []
    for review in df['content'].tolist():
        try:
            result = sentiment_model(review[:512])
            sentiments.append(result[0]['label'].lower())
        except:
            sentiments.append('neutral')
    df['sentiment'] = sentiments
    df['category'] = df['content'].apply(categorize_review)
    return df

# Dynamic Priority
def get_dynamic_priority(rank):
    if rank == 0:
        return 'P1 — Critical', '#E24B4A', '#FFF0F0'
    elif rank == 1:
        return 'P2 — High', '#EF9F27', '#FFFBF0'
    elif rank == 2:
        return 'P3 — Medium', '#378ADD', '#F0F5FF'
    else:
        return 'P4 — Low', '#888780', '#F8F8F8'

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem'>
    <h1 style='font-size:32px;font-weight:600;
    color:#2C2C2A;letter-spacing:-0.5px'>
        Revolut Review Intelligence
    </h1>
    <p style='color:#888780; font-size:15px; margin-top:4px'>
        Live AI-powered product dashboard — Google Play Store
    </p>
    <div style='background:#F8F8F8; padding:16px 24px;
    border-radius:8px; max-width:680px; margin:1rem auto;
    border:0.5px solid #D3D1C7'>
        <p style='color:#444441;font-size:14px;margin:0;line-height:1.7'>
        A real-time product intelligence tool that monitors Revolut app reviews,
        categorizes complaints by priority and surfaces actionable insights
        for product managers — refreshed every hour.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Refresh Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()

# Load Data
with st.spinner("Fetching latest Revolut reviews from Google Play..."):
    df = scrape_reviews(500)

with st.spinner("Running AI sentiment analysis..."):
    df = analyze_sentiment(df)

# Last updated
st.markdown(
    f"<p style='text-align:center;color:#888780;font-size:12px;'>"
    f"Last updated: {datetime.now().strftime('%d %B %Y, %I:%M %p')} "
    f"&nbsp;|&nbsp; {len(df)} reviews analysed</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Sentiment Overview
st.markdown("### Sentiment Overview")

sentiment_counts = df['sentiment'].value_counts()
total = len(df)

col1, col2, col3 = st.columns(3)

with col1:
    positive = sentiment_counts.get('positive', 0)
    st.markdown(f"""
    <div style='background:#5DCAA5;padding:20px;border-radius:8px;
    text-align:center'>
        <p style='color:white;font-size:26px;font-weight:600;margin:0'>
            {positive/total*100:.1f}%</p>
        <p style='color:white;margin:0;font-size:13px'>
            Positive &nbsp;|&nbsp; {positive:,} reviews</p>
    </div>""", unsafe_allow_html=True)

with col2:
    neutral = sentiment_counts.get('neutral', 0)
    st.markdown(f"""
    <div style='background:#EF9F27;padding:20px;border-radius:8px;
    text-align:center'>
        <p style='color:white;font-size:26px;font-weight:600;margin:0'>
            {neutral/total*100:.1f}%</p>
        <p style='color:white;margin:0;font-size:13px'>
            Neutral &nbsp;|&nbsp; {neutral:,} reviews</p>
    </div>""", unsafe_allow_html=True)

with col3:
    negative = sentiment_counts.get('negative', 0)
    st.markdown(f"""
    <div style='background:#E24B4A;padding:20px;border-radius:8px;
    text-align:center'>
        <p style='color:white;font-size:26px;font-weight:600;margin:0'>
            {negative/total*100:.1f}%</p>
        <p style='color:white;margin:0;font-size:13px'>
            Negative &nbsp;|&nbsp; {negative:,} reviews</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Rating Trend
st.markdown("### Rating Trend Over Time")
st.markdown(
    "<p style='color:#888780;font-size:13px;margin-top:-12px'>"
    "Daily average star rating based on most recent reviews</p>",
    unsafe_allow_html=True
)

daily_rating = (
    df.groupby('date')['score']
    .agg(['mean', 'count'])
    .reset_index()
    .rename(columns={'mean': 'avg_rating', 'count': 'review_count'})
)
daily_rating = daily_rating[daily_rating['review_count'] >= 2]
daily_rating['date'] = pd.to_datetime(daily_rating['date'])

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('#F8F8F8')

ax.plot(daily_rating['date'], daily_rating['avg_rating'],
        color='#378ADD', linewidth=2, zorder=3)
ax.fill_between(daily_rating['date'], daily_rating['avg_rating'],
                alpha=0.1, color='#378ADD')

overall_avg = daily_rating['avg_rating'].mean()
ax.axhline(y=overall_avg, color='#E24B4A', linestyle='--',
           linewidth=1.2, alpha=0.7,
           label=f'Average: {overall_avg:.2f}')

ax.scatter(daily_rating['date'], daily_rating['avg_rating'],
           color='#378ADD', s=30, zorder=4)

ax.set_ylim(1, 5.5)
ax.set_ylabel('Average Rating', fontsize=11, color='#444441')
ax.set_xlabel('Date', fontsize=11, color='#444441')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.xticks(rotation=45)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#D3D1C7')
ax.tick_params(colors='#444441')
ax.legend(fontsize=10)
ax.yaxis.grid(True, color='#D3D1C7', linewidth=0.8, alpha=0.7)
ax.set_axisbelow(True)

max_day = daily_rating.loc[daily_rating['avg_rating'].idxmax()]
min_day = daily_rating.loc[daily_rating['avg_rating'].idxmin()]

ax.annotate(f"Best: {max_day['avg_rating']:.1f}",
            xy=(max_day['date'], max_day['avg_rating']),
            xytext=(10, 10), textcoords='offset points',
            fontsize=9, color='#5DCAA5', fontweight='bold')
ax.annotate(f"Worst: {min_day['avg_rating']:.1f}",
            xy=(min_day['date'], min_day['avg_rating']),
            xytext=(10, -15), textcoords='offset points',
            fontsize=9, color='#E24B4A', fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Average Rating", f"{overall_avg:.2f} / 5.00")
with col2:
    st.metric("Best Day",
              f"{max_day['avg_rating']:.2f}",
              f"{max_day['date'].strftime('%d %b')}")
with col3:
    st.metric("Worst Day",
              f"{min_day['avg_rating']:.2f}",
              f"{min_day['date'].strftime('%d %b')}",
              delta_color="inverse")

st.markdown("---")

# Top Complaint
negative_df = df[df['sentiment'] == 'negative'].copy()
category_counts = negative_df['category'].value_counts()
top_category = category_counts.index[0]
top_count = category_counts.values[0]

st.markdown("### Primary Issue — Requires Immediate Attention")
st.markdown(f"""
<div style='background:#E24B4A;padding:20px;
border-radius:8px;text-align:center'>
    <p style='color:white;font-size:20px;
    font-weight:600;margin:0'>{top_category}</p>
    <p style='color:white;margin:4px 0 0;font-size:14px'>
        {top_count} complaints identified — P1 Critical
    </p>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# Complaint Breakdown
st.markdown("### Complaint Breakdown")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#E24B4A', '#EF9F27', '#378ADD',
              '#5DCAA5', '#7F77DD', '#D85A30', '#888780']
    bars = ax.barh(category_counts.index[::-1],
                   category_counts.values[::-1],
                   color=colors[:len(category_counts)],
                   edgecolor='white', height=0.6)
    for bar, val in zip(bars, category_counts.values[::-1]):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=10,
                fontweight='bold', color='#2C2C2A')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_xlabel('Number of Complaints', fontsize=11)
    ax.set_facecolor('#F8F8F8')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    negative_text = ' '.join(negative_df['content'].tolist())
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on',
                 'at', 'to', 'for', 'of', 'with', 'is', 'it', 'my',
                 'i', 'they', 'this', 'that', 'was', 'are', 'have',
                 'has', 'be', 'not', 'me', 'we', 'you', 'your',
                 'just', 'so', 'up', 'do', 'can', 'will', 'its',
                 'from', 'as', 'by', 'been', 'which', 'when'}
    wc = WordCloud(width=600, height=400,
                   background_color='white',
                   stopwords=stopwords,
                   colormap='Reds',
                   max_words=80).generate(negative_text)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.imshow(wc, interpolation='bilinear')
    ax2.axis('off')
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

# Dynamic Priority Board
st.markdown("### Product Priority Board")
st.markdown(
    "<p style='color:#888780;font-size:13px;margin-top:-12px'>"
    "Priorities are assigned dynamically based on real-time "
    "complaint volume</p>",
    unsafe_allow_html=True
)

sorted_categories = category_counts[
    category_counts.index != 'Other'
].reset_index()
sorted_categories.columns = ['category', 'count']
sorted_categories = sorted_categories.sort_values(
    'count', ascending=False).reset_index(drop=True)

col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

for i, row in sorted_categories.iterrows():
    cat = row['category']
    count = row['count']
    priority_label, priority_color, bg_color = get_dynamic_priority(i)
    rec = recommendations_map.get(cat, 'Manual review required')
    pct = count / len(negative_df) * 100
    col_index = min(i, 2)
    with columns[col_index]:
        st.markdown(f"""
        <div style='background:{bg_color};
        border-left:3px solid {priority_color};
        padding:12px 14px;border-radius:4px;margin-bottom:10px'>
            <p style='color:{priority_color};font-weight:600;
            font-size:12px;margin:0 0 4px'>{priority_label}</p>
            <p style='font-weight:600;font-size:14px;
            margin:0 0 4px;color:#2C2C2A'>{cat}</p>
            <p style='font-size:12px;color:#444441;margin:0 0 4px'>
                {count} complaints &nbsp;({pct:.1f}%)
            </p>
            <p style='font-size:11px;color:#888780;margin:0'>{rec}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# Most Upvoted Complaints
st.markdown("### Most Upvoted Complaints")
st.markdown(
    "<p style='color:#888780;font-size:13px;margin-top:-12px'>"
    "Reviews with the highest number of likes — "
    "indicating widespread user agreement</p>",
    unsafe_allow_html=True
)

top_complaints = negative_df.nlargest(5, 'thumbsUpCount')[
    ['content', 'score', 'thumbsUpCount', 'category']]

for _, row in top_complaints.iterrows():
    st.markdown(f"""
    <div style='background:#F8F8F8;padding:16px;border-radius:6px;
    margin-bottom:8px;border-left:3px solid #E24B4A'>
        <p style='margin:0;font-size:14px;color:#2C2C2A;line-height:1.6'>
            {row['content'][:200]}...
        </p>
        <p style='margin:6px 0 0;font-size:12px;color:#888780'>
            {row['score']} stars &nbsp;|&nbsp;
            {row['thumbsUpCount']} users found this helpful &nbsp;|&nbsp;
            {row['category']}
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Manual Review Analyzer
st.markdown("### Review Analyser")
st.markdown(
    "<p style='color:#888780;font-size:13px;margin-top:-12px'>"
    "Paste any Revolut review to instantly classify "
    "its sentiment, category and priority</p>",
    unsafe_allow_html=True
)

review_input = st.text_area(
    "Review text",
    placeholder="e.g. My account got frozen and I cannot access my money...",
    height=100,
    label_visibility="collapsed"
)

if st.button("Analyse Review", use_container_width=False):
    if review_input.strip() == "":
        st.warning("Please enter a review before analysing.")
    else:
        with st.spinner("Analysing..."):
            result = sentiment_model(review_input[:512])
            sentiment = result[0]['label'].lower()
            confidence = result[0]['score'] * 100
            category = categorize_review(review_input)

            cat_rank = list(sorted_categories['category']).index(category) \
                if category in list(sorted_categories['category']) else 3
            priority_label, priority_color, bg_color = \
                get_dynamic_priority(cat_rank)

            recommendation = recommendations_map.get(
                category, 'Manual review required')

            sentiment_colors = {
                'positive': '#5DCAA5',
                'negative': '#E24B4A',
                'neutral':  '#EF9F27'
            }
            sentiment_labels = {
                'positive': 'Positive',
                'negative': 'Negative',
                'neutral':  'Neutral'
            }

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div style='background:{sentiment_colors[sentiment]};
                padding:16px;border-radius:8px;text-align:center'>
                    <p style='color:white;font-weight:600;margin:0;
                    font-size:16px'>{sentiment_labels[sentiment]}</p>
                    <p style='color:white;margin:4px 0 0;font-size:12px'>
                        {confidence:.1f}% confidence
                    </p>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background:#F1EFE8;padding:16px;
                border-radius:8px;text-align:center;
                border:0.5px solid #D3D1C7'>
                    <p style='font-weight:600;margin:0;
                    font-size:16px;color:#2C2C2A'>{category}</p>
                    <p style='color:#888780;margin:4px 0 0;
                    font-size:12px'>Complaint category</p>
                </div>""", unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style='background:{priority_color};
                padding:16px;border-radius:8px;text-align:center'>
                    <p style='color:white;font-weight:600;margin:0;
                    font-size:16px'>{priority_label}</p>
                    <p style='color:white;margin:4px 0 0;
                    font-size:12px'>Current priority</p>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#F8F8F8;padding:14px;
            border-radius:6px;margin-top:12px;
            border-left:3px solid {priority_color}'>
                <p style='margin:0;font-size:13px;color:#444441'>
                    <b>Recommended action:</b> {recommendation}
                </p>
            </div>""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888780;font-size:11px;"
    "padding-bottom:1rem'>"
    "Built with Python, Hugging Face Transformers and Streamlit &nbsp;|&nbsp; "
    "Data sourced from Google Play Store &nbsp;|&nbsp; "
    "AI-powered product intelligence"
    "</div>",
    unsafe_allow_html=True
)
