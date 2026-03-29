# Revolut Review Intelligence

A real-time AI-powered product intelligence dashboard that automatically 
scrapes Revolut's Google Play Store reviews, analyzes sentiment using 
Natural Language Processing and categorizes complaints by business priority.

## What it does
- Scrapes the 500 most recent Revolut reviews from Google Play Store
- Uses AI (Hugging Face Transformers) to classify each review as 
  positive, negative or neutral
- Automatically categorizes complaints into 6 business categories
- Assigns dynamic priorities based on real-time complaint volume
- Displays a live rating trend chart over time
- Surfaces the most upvoted complaints for product team review
- Allows manual analysis of any review instantly

## Built with
- Python
- Streamlit
- Hugging Face Transformers
- Google Play Scraper
- Pandas & Matplotlib

## Live Demo
https://reviewanalyzer-ampf3wruzuve62a6lkpvfd.streamlit.app/
