import streamlit as st
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(
    page_title="Fintech Review Analyzer",
    page_icon="💳",
    layout="centered"
)

# Load AI model
@st.cache_resource
def load_model():
    return pipeline(
        'sentiment-analysis',
        model='cardiffnlp/twitter-roberta-base-sentiment-latest',
        truncation=True,
        max_length=512
    )

sentiment_model = load_model()

# Categorization function
def categorize_review(text):
    text = text.lower()
    if any(word in text for word in ['account', 'frozen', 'blocked',
                                      'closed', 'suspended', 'login',
                                      'password', 'access', 'verify',
                                      'verification', 'code', 'log in',
                                      'scan', 'biometric', 'face',
                                      'passport', 'kyc', 'document',
                                      'identity', 'pending', 'sign up']):
        return '🔴 Account & Access Issues'
    elif any(word in text for word in ['transfer', 'send', 'money',
                                        'payment', 'transaction', 'fee',
                                        'charge', 'expensive', 'cost',
                                        'card', 'delivery', 'refund',
                                        'chargeback', 'merchant', 'dispute']):
        return '💸 Payments & Money'
    elif any(word in text for word in ['crash', 'bug', 'update',
                                        'reinstall', 'freeze', 'fix',
                                        'broken', 'open', 'fail', 'slow',
                                        'laggy', 'compatible', 'device',
                                        'notification', 'design',
                                        'confusing', 'interface']):
        return '📱 App Performance'
    elif any(word in text for word in ['support', 'service', 'help',
                                        'response', 'chat', 'contact',
                                        'agent', 'chatbot', 'bot',
                                        'wait', 'hours', 'reply']):
        return '🎧 Customer Support'
    elif any(word in text for word in ['scam', 'fraud', 'stolen',
                                        'fake', 'cheat', 'privacy',
                                        'data', 'gdpr', 'security']):
        return '🔒 Security & Trust'
    elif any(word in text for word in ['invest', 'investment', 'stocks',
                                        'crypto', 'bitcoin', 'btc',
                                        'fx', 'currency', 'feature',
                                        'referral', 'spam', 'email']):
        return '📈 Features & Products'
    else:
        return '❓ Other'

# Priority and recommendation maps
priority_map = {
    '🔴 Account & Access Issues': ('P1 — Critical', '#E24B4A'),
    '💸 Payments & Money':        ('P2 — High',     '#EF9F27'),
    '📱 App Performance':         ('P3 — Medium',   '#378ADD'),
    '🎧 Customer Support':        ('P2 — High',     '#EF9F27'),
    '🔒 Security & Trust':        ('P1 — Critical', '#E24B4A'),
    '📈 Features & Products':     ('P3 — Medium',   '#378ADD'),
    '❓ Other':                   ('P4 — Low',      '#888780')
}

recommendations_map = {
    '🔴 Account & Access Issues': 'Escalate to account recovery team immediately',
    '💸 Payments & Money':        'Review payment flow and dispute resolution process',
    '📱 App Performance':         'Flag to engineering team for immediate bug fix',
    '🎧 Customer Support':        'Review support ticket SLA and chatbot accuracy',
    '🔒 Security & Trust':        'Escalate to security and fraud team urgently',
    '📈 Features & Products':     'Add to product backlog for next sprint review',
    '❓ Other':                   'Manual review required by product team'
}

sentiment_emoji = {
    'positive': '😊 Positive',
    'negative': '😠 Negative',
    'neutral':  '😐 Neutral'
}

sentiment_color = {
    'positive': '#5DCAA5',
    'negative': '#E24B4A',
    'neutral':  '#EF9F27'
}

# ── UI ──────────────────────────────────────────────────────

st.title("💳 Fintech Review Analyzer")
st.markdown("#### AI powered product intelligence for fintech apps")
st.markdown("---")

# App selector
app_name = st.selectbox(
    "Which app are you analyzing?",
    ["Revolut", "PayPal", "CashApp", "Monzo", "Wise",
     "Klarna", "Starling", "N26", "Other"]
)

st.markdown("---")

# Single review analysis
st.subheader("🔍 Analyze a Single Review")
review_input = st.text_area(
    "Paste a review here",
    placeholder="e.g. My account got frozen and I can't access my money...",
    height=120
)

if st.button("Analyze Review ↗"):
    if review_input.strip() == "":
        st.warning("Please paste a review first!")
    else:
        with st.spinner("AI is analyzing the review..."):

            # Run sentiment model
            result = sentiment_model(review_input[:512])
            sentiment = result[0]['label'].lower()
            confidence = result[0]['score'] * 100

            # Get category and priority
            category = categorize_review(review_input)
            priority, priority_color = priority_map[category]
            recommendation = recommendations_map[category]

            # Display results
            st.markdown("### Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Sentiment**")
                st.markdown(
                    f"<div style='background:{sentiment_color[sentiment]};"
                    f"padding:12px;border-radius:8px;text-align:center;"
                    f"color:white;font-weight:bold;font-size:16px'>"
                    f"{sentiment_emoji[sentiment]}<br>"
                    f"<small>{confidence:.1f}% confidence</small></div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("**Category**")
                st.markdown(
                    f"<div style='background:#F1EFE8;"
                    f"padding:12px;border-radius:8px;text-align:center;"
                    f"font-weight:bold;font-size:14px'>"
                    f"{category}</div>",
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown("**Priority**")
                st.markdown(
                    f"<div style='background:{priority_color};"
                    f"padding:12px;border-radius:8px;text-align:center;"
                    f"color:white;font-weight:bold;font-size:14px'>"
                    f"{priority}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("**📋 Recommendation for Product Team**")
            st.info(f"**{app_name}** — {recommendation}")

st.markdown("---")

# Bulk analysis
st.subheader("📊 Bulk Analysis")
st.markdown("Paste multiple reviews — one per line")

bulk_input = st.text_area(
    "Paste reviews here (one per line)",
    placeholder="Review 1...\nReview 2...\nReview 3...",
    height=200
)

if st.button("Analyze All Reviews ↗"):
    if bulk_input.strip() == "":
        st.warning("Please paste some reviews first!")
    else:
        reviews_list = [r.strip() for r in bulk_input.split('\n')
                        if r.strip() != ""]
        st.info(f"Analyzing {len(reviews_list)} reviews...")

        results = []
        progress = st.progress(0)

        for i, review in enumerate(reviews_list):
            result = sentiment_model(review[:512])
            sentiment = result[0]['label'].lower()
            category = categorize_review(review)
            priority, _ = priority_map[category]
            results.append({
                'Review': review[:80] + '...' if len(review) > 80 else review,
                'Sentiment': sentiment_emoji[sentiment],
                'Category': category,
                'Priority': priority
            })
            progress.progress((i + 1) / len(reviews_list))

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        # Summary chart
        st.markdown("### Category Breakdown")
        category_counts = df_results['Category'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#E24B4A', '#EF9F27', '#378ADD',
                  '#5DCAA5', '#7F77DD', '#D85A30', '#888780']
        ax.barh(category_counts.index[::-1],
                category_counts.values[::-1],
                color=colors[:len(category_counts)],
                edgecolor='white')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_xlabel('Number of Reviews')
        plt.tight_layout()
        st.pyplot(fig)

        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "review_analysis.csv",
            "text/csv"
        )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888780;font-size:12px'>"
    "Built with Python, Transformers & Streamlit | "
    "AI Powered Fintech Review Analysis</div>",
    unsafe_allow_html=True
)
