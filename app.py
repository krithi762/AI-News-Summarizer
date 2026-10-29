"""
app.py
------
Main Streamlit user interface entry point for the AI News Summarizer application.
Provides a modern, responsive, and professional dashboard for news aggregation,
article extraction, AI summarization, key point extraction, categorization,
keyword extraction, and database persistence.
"""

import os
import json
from typing import Any
import streamlit as st
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

import news_fetcher
import article_extractor
import summarizer
import classifier
import keyword_extractor
import duplicate_detector
import sentiment_analyzer
import trending_topics
import chat
import source_comparison
import database
import theme
import utils

# ---------------------------------------------------------------------------
# 1. Application Initialization & Page Config
# ---------------------------------------------------------------------------

# Initialize SQLite database on startup
database.init_database()

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme in session state before CSS injection
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Apply centralized professional CSS styling
utils.apply_custom_css()


# ---------------------------------------------------------------------------
# 2. Cached Fetchers (Performance Optimization)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_news(category: str, limit: int, source_mode: str) -> list[dict]:
    """
    Fetches news articles with a 10-minute cache TTL to optimize network usage
    and persists fetched articles in the local SQLite database.
    """
    return utils.fetch_news_with_source(category=category, limit=limit, source_mode=source_mode)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_extracted_article(url: str, source_hint: str | None) -> dict:
    """
    Extracts full article text with a 1-hour cache TTL to avoid redundant requests.
    """
    return article_extractor.extract_article(url=url, source_hint=source_hint)


# ---------------------------------------------------------------------------
# 3. Session State Management
# ---------------------------------------------------------------------------

if "active_article_url" not in st.session_state:
    st.session_state["active_article_url"] = None
if "active_article_source" not in st.session_state:
    st.session_state["active_article_source"] = None
if "active_tab_index" not in st.session_state:
    st.session_state["active_tab_index"] = 0
if "summaries" not in st.session_state:
    st.session_state["summaries"] = {}
if "key_points" not in st.session_state:
    st.session_state["key_points"] = {}
if "categories" not in st.session_state:
    st.session_state["categories"] = {}
if "keywords" not in st.session_state:
    st.session_state["keywords"] = {}
if "sentiments" not in st.session_state:
    st.session_state["sentiments"] = {}
if "chat_messages_by_url" not in st.session_state:
    st.session_state["chat_messages_by_url"] = {}
if "comparison_results" not in st.session_state:
    st.session_state["comparison_results"] = {}
if "related_news" not in st.session_state:
    st.session_state["related_news"] = {}
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "search_query_val" not in st.session_state:
    st.session_state["search_query_val"] = ""


# ---------------------------------------------------------------------------
# 4. Sidebar Controls & System Status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⚙️ News Controls</div>', unsafe_allow_html=True)

    # Category Selection
    category_options = ["All"] + news_fetcher.SUPPORTED_CATEGORIES
    selected_category = st.selectbox(
        "Category",
        options=category_options,
        index=1,  # Default to Technology
        help="Select a news topic to retrieve relevant headlines."
    )

    # Article Count Limit Selection
    article_limit = st.selectbox(
        "Number of Articles",
        options=[5, 10, 15, 20],
        index=1,  # Default to 10
        help="Select the maximum number of articles to retrieve."
    )

    # News Source Selection
    source_mode = st.selectbox(
        "News Source",
        options=["Automatic", "NewsAPI", "RSS"],
        index=0,
        help="Choose automatic fallback or force a specific news provider."
    )

    # Fetch Action Button
    fetch_clicked = st.button("🔄 Fetch Latest News", type="primary", use_container_width=True)

    st.markdown("---")

    # Database Status Section
    st.markdown('<div class="sidebar-section-title">💾 Database</div>', unsafe_allow_html=True)
    try:
        total_db_count = database.get_article_count()
        ai_processed_count = database.get_ai_processed_count()
        st.markdown(
            f"""
            <div class="db-status-card">
                <div style="display:flex; align-items:center; gap:6px; color:var(--success); font-weight:600; font-size:0.85rem; margin-bottom:4px;">
                    <span class="status-dot"></span> Connected
                </div>
                <div class="db-status-text">
                    <strong>Stored Articles:</strong> {total_db_count}<br>
                    <strong>AI Processed:</strong> {ai_processed_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.info("🟡 **Database:** Initializing...")

    st.markdown("---")

    # Sidebar AI Information
    st.markdown('<div class="sidebar-section-title">🤖 Features</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-features-list">
            <div class="sidebar-feature-item">✓ <strong>Summarization</strong> (Short/Med/Detail)</div>
            <div class="sidebar-feature-item">✓ <strong>Key Points</strong> (3/5/7 Bullets)</div>
            <div class="sidebar-feature-item">✓ <strong>Categorization</strong> (7 Domains)</div>
            <div class="sidebar-feature-item">✓ <strong>Keywords</strong> (Entities & Tags)</div>
            <div class="sidebar-feature-item">✓ <strong>Duplicate Detection</strong> (TF-IDF)</div>
            <div class="sidebar-feature-item">✓ <strong>Search & Filtering</strong> (SQLite)</div>
            <div class="sidebar-feature-item">✓ <strong>Analytics Dashboard</strong> (Plotly)</div>
            <div class="sidebar-feature-item">✓ <strong>Sentiment Analysis</strong> (Pos/Neu/Neg)</div>
            <div class="sidebar-feature-item">✓ <strong>Trending Topics</strong> (Recency + Freq)</div>
            <div class="sidebar-feature-item">✓ <strong>Article AI Chat</strong> (Grounded Q&A)</div>
            <div class="sidebar-feature-item">✓ <strong>Source Comparison</strong> (2–5 Publishers)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Connection Status Badges
    st.markdown('<div class="sidebar-section-title">🔌 Connection Status</div>', unsafe_allow_html=True)
    
    # NewsAPI Status
    news_api_key = os.getenv("NEWS_API_KEY", "").strip()
    if news_api_key and news_api_key != "your_news_api_key_here":
        st.caption("🟢 **NewsAPI:** Key Active")
    else:
        st.caption("🟡 **NewsAPI:** Using RSS Fallback Feeds")

    # Gemini API Status
    if summarizer.is_gemini_configured():
        model_name = summarizer.get_configured_model()
        st.caption(f"🟢 **Gemini AI:** Ready (`{model_name}`)")
    else:
        st.caption("🟡 **Gemini AI:** Unconfigured (Local Fallbacks)")


# ---------------------------------------------------------------------------
# 5. Main Header & Status Indicator & Theme Switcher
# ---------------------------------------------------------------------------

hdr_col1, hdr_col2 = st.columns([3.8, 1.4])
with hdr_col1:
    st.markdown('<div class="main-header-title">📰 AI News Summarizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-header-subtitle">AI-powered news collection, summarization, classification and insight extraction.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span> System Online</div>',
        unsafe_allow_html=True
    )
with hdr_col2:
    st.markdown("<div style='font-size:0.85rem; font-weight:600; margin-bottom:2px; color:var(--text-primary);'>Theme:</div>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "Select Theme",
        options=["☀️ Light", "🌙 Dark"],
        index=0 if st.session_state.get("theme", "dark") == "light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio_selector",
    )
    chosen_theme = "dark" if "Dark" in theme_choice else "light"
    if chosen_theme != st.session_state.get("theme", "dark"):
        st.session_state["theme"] = chosen_theme
        st.rerun()

# Clear cache if user requested new fetch
if fetch_clicked:
    get_cached_news.clear()
    st.session_state["active_article_url"] = None
    st.session_state["active_article_source"] = None


# ---------------------------------------------------------------------------
# 6. Fetch News & Compute Real Dashboard Metrics
# ---------------------------------------------------------------------------

with st.spinner("🔄 Fetching latest news..."):
    try:
        articles = get_cached_news(
            category=selected_category,
            limit=article_limit,
            source_mode=source_mode
        )
    except Exception as e:
        st.error(f"Unable to fetch news articles: {type(e).__name__}. Please try again.")
        articles = []

# Fetch real database metrics
try:
    total_stored = database.get_article_count()
    ai_processed = database.get_ai_processed_count()
except Exception:
    total_stored = 0
    ai_processed = 0

# Render Real Metrics Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(label="📰 Loaded Articles", value=len(articles))
with col_m2:
    st.metric(label="🤖 AI Processed", value=ai_processed)
with col_m3:
    st.metric(label="💾 Stored in Database", value=total_stored)
with col_m4:
    display_cat = selected_category if selected_category != "All" else "All Categories"
    st.metric(label="🗂 Active Category", value=display_cat)

st.markdown("---")


# ---------------------------------------------------------------------------
# 7. Navigation Tabs
# ---------------------------------------------------------------------------

tab_news, tab_search, tab_analysis, tab_history, tab_dashboard = st.tabs([
    "📰 Latest News",
    "🔍 Search & Filter News",
    "📄 Article Details & AI Insights",
    "💾 Stored Articles",
    "📊 Dashboard",
])


# ---------------------------------------------------------------------------
# TAB 1: Latest News Feed
# ---------------------------------------------------------------------------

with tab_news:
    st.markdown("## 📰 Latest News")

    if not articles:
        # Empty State
        st.info("### 📰 No News Loaded\n\nSelect a category and click **Fetch Latest News** to get started.")
    else:
        # Feed Information Banner
        provider_badge = articles[0].get("provider", "RSS") if articles else "RSS"
        st.markdown(
            f"""
            <div class="feed-info-banner">
                <strong>{selected_category}</strong> &nbsp;•&nbsp; <strong>{len(articles)} Articles</strong> &nbsp;•&nbsp; <span class="meta-pill meta-pill-provider">{provider_badge}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render Article Cards
        for idx, article in enumerate(articles, start=1):
            with st.container(border=True):
                col_card_img, col_card_info = st.columns([1, 2.8])

                # Article Image / Fallback Placeholder
                with col_card_img:
                    img_url = article.get("image_url")
                    rendered_img = False
                    if img_url:
                        try:
                            st.image(img_url, use_container_width=True)
                            rendered_img = True
                        except Exception:
                            rendered_img = False

                    if not rendered_img:
                        cat_icon = "📰"
                        st.markdown(
                            f"""
                            <div class="image-placeholder">
                                <span style="font-size:2rem; margin-bottom:4px;">{cat_icon}</span>
                                <span>{article.get('source') or 'News'}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Article Information & Actions
                with col_card_info:
                    # Title
                    art_title = article.get("title") or "Untitled Article"
                    st.markdown(f'<div class="article-card-header">{art_title}</div>', unsafe_allow_html=True)

                    # Metadata Line (Source • Date • Provider)
                    src_val = article.get("source") or "Unknown"
                    date_val = utils.format_date_human(article.get("published_at"))
                    prov_val = article.get("provider", "RSS")
                    author_val = f" • By {article.get('author')}" if article.get("author") else ""

                    st.markdown(
                        f"""
                        <div class="article-meta-line">
                            <span class="meta-pill">🏢 {src_val}{author_val}</span>
                            <span class="meta-pill">📅 {date_val}</span>
                            <span class="meta-pill meta-pill-provider">{prov_val}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Description
                    desc_text = utils.truncate_description(article.get("description"), max_chars=200)
                    st.write(desc_text)

                    # Action Buttons
                    btn_c1, btn_c2 = st.columns([1.3, 2.2])
                    with btn_c1:
                        if st.button("📄 Extract Article", key=f"feed_extract_{idx}_{hash(article.get('url', ''))}"):
                            st.session_state["active_article_url"] = article.get("url")
                            st.session_state["active_article_source"] = article.get("source")
                            st.rerun()
                    with btn_c2:
                        raw_url = article.get("url", "#")
                        st.link_button("🔗 Read Original", url=raw_url)


# ---------------------------------------------------------------------------
# TAB 2: Search & Filter News (Feature 11)
# ---------------------------------------------------------------------------

with tab_search:
    st.markdown("## 🔍 Search & Filter Stored News")
    st.caption("Search across titles, descriptions, full text, AI categories, and keywords stored in SQLite.")

    # Search Bar Input
    search_query_input = st.text_input(
        "Search Query",
        value=st.session_state.get("search_query_val", ""),
        placeholder="Search articles, topics, people, organizations (e.g. Artificial Intelligence, NASA, Cricket)...",
        label_visibility="collapsed",
    )

    # Filter Controls Row 1
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        search_category = st.selectbox(
            "Category",
            options=["All"] + news_fetcher.SUPPORTED_CATEGORIES,
            index=0,
            key="search_cat_select",
            help="Filter by article category."
        )
    with f_col2:
        stored_sources = database.get_distinct_sources()
        source_options = ["All Sources"] + stored_sources
        search_source = st.selectbox(
            "Source / Publisher",
            options=source_options,
            index=0,
            key="search_src_select",
            help="Filter by specific news source stored in the database."
        )
    with f_col3:
        search_sentiment = st.selectbox(
            "Sentiment",
            options=["All", "Positive", "Neutral", "Negative"],
            index=0,
            key="search_sentiment_select",
            help="Filter by article sentiment."
        )
    with f_col4:
        search_date_preset = st.selectbox(
            "Date Filter",
            options=["All Time", "Today", "Last 7 Days", "Last 30 Days", "Custom Range"],
            index=0,
            key="search_date_preset",
            help="Filter by publication date."
        )

    # Custom Date Range Inputs (Conditional)
    custom_start_date = None
    custom_end_date = None
    if search_date_preset == "Custom Range":
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            custom_start_date = st.date_input("From Date", key="search_from_date")
        with d_col2:
            custom_end_date = st.date_input("To Date", key="search_to_date")

    # Filter Controls Row 2
    s_col1, s_col2, s_col3 = st.columns([1.5, 1.5, 2])
    with s_col1:
        sort_by_selected = st.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First", "Most Relevant"],
            index=0,
            key="search_sort_select",
            help="Sort by publication recency or keyword relevance."
        )
    with s_col2:
        search_limit = st.selectbox(
            "Result Limit",
            options=[10, 20, 50, 100],
            index=1,  # Default: 20
            key="search_limit_select",
            help="Maximum number of articles to display."
        )
    with s_col3:
        st.write("")
        st.write("")
        b_c1, b_c2 = st.columns(2)
        with b_c1:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with b_c2:
            clear_filters_clicked = st.button("🔄 Clear Filters", use_container_width=True)

    if clear_filters_clicked:
        st.session_state["search_query_val"] = ""
        st.session_state["search_results"] = None
        st.rerun()

    # Map sort selection
    sort_key_map = {
        "Newest First": "newest",
        "Oldest First": "oldest",
        "Most Relevant": "relevance",
    }
    sort_key = sort_key_map.get(sort_by_selected, "newest")

    # Execute search query
    if search_clicked or st.session_state["search_results"] is None:
        st.session_state["search_query_val"] = search_query_input
        search_results = database.search_articles(
            query=search_query_input,
            category=search_category,
            source=search_source,
            sentiment=search_sentiment,
            date_filter=search_date_preset,
            start_date=str(custom_start_date) if custom_start_date else None,
            end_date=str(custom_end_date) if custom_end_date else None,
            sort_by=sort_key,
            limit=search_limit,
        )
        st.session_state["search_results"] = search_results

    results_list = st.session_state.get("search_results", [])

    st.markdown("---")

    # Display Results Feedback & Cards
    db_total = database.get_article_count()
    if db_total == 0:
        st.info("### 📰 No Stored News\n\nFetch some news first from the **📰 Latest News** tab. Your saved articles will appear here.")
    elif not results_list:
        st.info("### 🔍 No Results Found\n\nTry:\n- A different keyword\n- Another category\n- Another source\n- A wider date range")
    else:
        st.markdown(f"**Found {len(results_list)} matching article(s):**")
        for idx, art in enumerate(results_list, start=1):
            with st.container(border=True):
                col_c_img, col_c_info = st.columns([1, 2.8])

                with col_c_img:
                    img_url = art.get("image_url")
                    rendered_img = False
                    if img_url:
                        try:
                            st.image(img_url, use_container_width=True)
                            rendered_img = True
                        except Exception:
                            rendered_img = False

                    if not rendered_img:
                        st.markdown(
                            f"""
                            <div class="image-placeholder">
                                <span style="font-size:2rem; margin-bottom:4px;">📰</span>
                                <span>{art.get('source') or 'News'}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with col_c_info:
                    art_title = art.get("title") or "Untitled Article"
                    st.markdown(f'<div class="article-card-header">{art_title}</div>', unsafe_allow_html=True)

                    src_val = art.get("source") or "Unknown Source"
                    date_val = utils.format_date_human(art.get("published_at"))
                    cat_val = art.get("ai_category") or art.get("category") or "General"
                    author_val = f" • By {art.get('author')}" if art.get("author") else ""

                    st.markdown(
                        f"""
                        <div class="article-meta-line">
                            <span class="meta-pill">🏢 {src_val}{author_val}</span>
                            <span class="meta-pill">📅 {date_val}</span>
                            <span class="meta-pill">🏷️ {cat_val}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    desc_text = utils.truncate_description(art.get("description"), max_chars=180)
                    st.write(desc_text)

                    # Show badges for existing AI outputs
                    ai_badges = []
                    if art.get("summary"):
                        ai_badges.append("🤖 Summary ✓")
                    if art.get("key_points"):
                        ai_badges.append("🔑 Key Points ✓")
                    if art.get("ai_category"):
                        ai_badges.append("🏷️ AI Category ✓")
                    if art.get("keywords"):
                        ai_badges.append("🔖 Keywords ✓")
                    if art.get("sentiment"):
                        ai_badges.append(f"💭 {art.get('sentiment')} ✓")

                    if ai_badges:
                        badges_html = " ".join([f'<span class="meta-pill meta-pill-provider">{b}</span>' for b in ai_badges])
                        st.markdown(f'<div style="margin-bottom:10px;">{badges_html}</div>', unsafe_allow_html=True)

                    btn_r1, btn_r2 = st.columns([1.3, 2.2])
                    with btn_r1:
                        if st.button("📄 Open Article", key=f"search_open_{art.get('id')}_{idx}_{hash(art.get('url', ''))}"):
                            st.session_state["active_article_url"] = art.get("url")
                            st.session_state["active_article_source"] = art.get("source")
                            st.rerun()
                    with btn_r2:
                        raw_url = art.get("url", "#")
                        st.link_button("🔗 Read Original", url=raw_url)


# ---------------------------------------------------------------------------
# TAB 3: Extracted Article & AI Insights View
# ---------------------------------------------------------------------------

with tab_analysis:
    if not st.session_state["active_article_url"]:
        # Empty State
        st.markdown("## 📄 Select an Article")
        st.info("Choose an article from the **📰 Latest News** tab and click **📄 Extract Article** to analyze it.")
    else:
        active_url = st.session_state["active_article_url"]
        active_source = st.session_state["active_article_source"]

        # Article Header & Close Button
        col_hdr_title, col_hdr_btn = st.columns([5, 1.2])
        with col_hdr_title:
            st.markdown("## 📄 Article Details")
        with col_hdr_btn:
            if st.button("✖️ Close View", use_container_width=True):
                st.session_state["active_article_url"] = None
                st.session_state["active_article_source"] = None
                st.rerun()

        # Perform extraction with spinner
        with st.spinner("📄 Extracting article..."):
            extracted_data = get_cached_extracted_article(active_url, active_source)

        if extracted_data.get("success"):
            st.success("✓ Article extracted successfully")

            # Persist extracted article content in SQLite
            database.update_article_content(
                url=active_url,
                article_text=extracted_data.get("text", ""),
                word_count=extracted_data.get("word_count", 0),
                title=extracted_data.get("title"),
                author=extracted_data.get("author"),
                published_at=extracted_data.get("published_at"),
            )

            # Retrieve any previously stored AI results from SQLite
            stored_db_article = database.get_article_by_url(active_url)

            # Display Article Metadata
            art_title = extracted_data.get("title") or "Article Content"
            st.markdown(f"### {art_title}")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.caption(f"🏢 **Source:** {extracted_data.get('source') or 'Unknown'}")
            with m_col2:
                st.caption(f"✍️ **Author:** {extracted_data.get('author') or 'Not listed'}")
            with m_col3:
                st.caption(f"📅 **Published:** {utils.format_date_human(extracted_data.get('published_at'))}")
            with m_col4:
                word_count_val = extracted_data.get("word_count", 0)
                st.caption(f"📊 **Word Count:** {word_count_val:,} words")

            # Display Article Content in clean readable format
            st.markdown("#### Article Content")
            st.markdown(
                f"""
                <div class="article-reader-box">
                    {extracted_data.get('text', '').replace(chr(10), '<br>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 4: AI Summary Section
            # ---------------------------------------------------------------
            st.markdown("## 🤖 AI Summary")

            col_s_len, col_s_btn = st.columns([2, 2])
            with col_s_len:
                summary_length = st.selectbox(
                    "Summary Length",
                    options=["Short", "Medium", "Detailed"],
                    index=1,
                    key=f"summary_len_sel_{hash(active_url)}",
                    help="Short: 2–3 sentences | Medium: ~1 paragraph | Detailed: 2–4 paragraphs"
                )
            with col_s_btn:
                st.write("")
                st.write("")
                generate_summary_clicked = st.button(
                    "✨ Generate AI Summary",
                    type="primary",
                    key=f"btn_gen_sum_{hash(active_url)}"
                )

            summary_cache_key = f"{active_url}_{summary_length.lower()}"

            # Preload from SQLite if available
            if summary_cache_key not in st.session_state["summaries"] and stored_db_article:
                if stored_db_article.get("summary") and (stored_db_article.get("summary_length") or "").lower() == summary_length.lower():
                    st.session_state["summaries"][summary_cache_key] = {
                        "success": True,
                        "summary": stored_db_article["summary"],
                        "summary_type": summary_length.capitalize(),
                        "original_word_count": stored_db_article.get("word_count", 0),
                        "summary_word_count": len(stored_db_article["summary"].split()),
                        "error": None,
                    }

            if generate_summary_clicked:
                with st.spinner("🤖 Generating AI summary..."):
                    summary_output = summarizer.summarize_article(
                        article_text=extracted_data.get("text"),
                        summary_length=summary_length.lower()
                    )
                    st.session_state["summaries"][summary_cache_key] = summary_output
                    if summary_output.get("success") and summary_output.get("summary"):
                        database.save_summary(
                            url=active_url,
                            summary=summary_output["summary"],
                            summary_length=summary_length.lower(),
                        )

            # Display generated summary if available
            current_summary = st.session_state["summaries"].get(summary_cache_key)
            if current_summary:
                if current_summary.get("success"):
                    st.markdown(
                        f"""
                        <div class="ai-result-box">
                            <div style="font-size:1.02rem; margin-bottom:12px; line-height:1.6;">{current_summary.get('summary', '')}</div>
                            <div class="ai-result-meta">
                                <strong>Original:</strong> {current_summary.get('original_word_count', 0):,} words &nbsp;|&nbsp; 
                                <strong>Summary:</strong> {current_summary.get('summary_word_count', 0):,} words &nbsp;|&nbsp; 
                                <strong>Type:</strong> {current_summary.get('summary_type', 'Medium')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Unable to generate the summary. {current_summary.get('error', 'Please try again.')}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 5: AI Key Points Section
            # ---------------------------------------------------------------
            st.markdown("## 🔑 Key Points")

            col_k_num, col_k_btn = st.columns([2, 2])
            with col_k_num:
                num_points_selected = st.selectbox(
                    "Number of Points",
                    options=[3, 5, 7],
                    index=1,  # Default to 5
                    key=f"kp_count_sel_{hash(active_url)}",
                    help="Select how many factual key takeaways to extract."
                )
            with col_k_btn:
                st.write("")
                st.write("")
                generate_kp_clicked = st.button(
                    "✨ Generate Key Points",
                    type="primary",
                    key=f"btn_gen_kp_{hash(active_url)}"
                )

            kp_cache_key = f"{active_url}_{num_points_selected}"

            # Preload from SQLite if available
            if kp_cache_key not in st.session_state["key_points"] and stored_db_article:
                db_kps = stored_db_article.get("key_points")
                if db_kps and len(db_kps) == num_points_selected:
                    st.session_state["key_points"][kp_cache_key] = {
                        "success": True,
                        "key_points": db_kps,
                        "requested_points": num_points_selected,
                        "points_count": len(db_kps),
                        "original_word_count": stored_db_article.get("word_count", 0),
                        "error": None,
                    }

            if generate_kp_clicked:
                with st.spinner("🔑 Extracting key points..."):
                    kp_output = summarizer.extract_key_points(
                        article_text=extracted_data.get("text"),
                        number_of_points=num_points_selected
                    )
                    st.session_state["key_points"][kp_cache_key] = kp_output
                    if kp_output.get("success") and kp_output.get("key_points"):
                        database.save_key_points(
                            url=active_url,
                            key_points=kp_output["key_points"],
                        )

            # Display extracted key points
            current_kp = st.session_state["key_points"].get(kp_cache_key)
            if current_kp:
                if current_kp.get("success"):
                    for idx, pt in enumerate(current_kp.get("key_points", []), start=1):
                        st.markdown(
                            f"""
                            <div class="kp-card">
                                <span class="kp-badge">{idx:02d}</span> {pt}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.error(f"Unable to extract key points. {current_kp.get('error', 'Please try again.')}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 6 & 7: AI Category & Keywords (Side-by-Side Layout)
            # ---------------------------------------------------------------
            col_ai_cat, col_ai_kw = st.columns(2)

            # Feature 6: AI Category Section
            with col_ai_cat:
                st.markdown("## 🏷️ AI Category")
                classify_clicked = st.button(
                    "🔍 Classify Article",
                    type="primary",
                    key=f"btn_classify_{hash(active_url)}",
                    use_container_width=True
                )

                category_cache_key = f"{active_url}_category"

                # Preload from SQLite
                if category_cache_key not in st.session_state["categories"] and stored_db_article:
                    if stored_db_article.get("ai_category"):
                        st.session_state["categories"][category_cache_key] = {
                            "success": True,
                            "category": stored_db_article["ai_category"],
                            "confidence": stored_db_article.get("category_confidence"),
                            "method": stored_db_article.get("category_method", "gemini"),
                            "error": None,
                        }

                if classify_clicked:
                    with st.spinner("🏷️ Classifying article..."):
                        cat_output = classifier.classify_article(
                            title=art_title,
                            article_text=extracted_data.get("text")
                        )
                        st.session_state["categories"][category_cache_key] = cat_output
                        if cat_output.get("success") and cat_output.get("category"):
                            database.save_category(
                                url=active_url,
                                category=cat_output["category"],
                                confidence=cat_output.get("confidence"),
                                method=cat_output.get("method"),
                            )

                # Display category
                current_category = st.session_state["categories"].get(category_cache_key)
                if current_category:
                    if current_category.get("success"):
                        cat_name = current_category.get("category", "General")
                        conf_val = current_category.get("confidence")
                        conf_str = f"{int(conf_val * 100)}%" if conf_val is not None else "Rule-based"
                        method_label = "Gemini AI" if current_category.get("method") == "gemini" else "Keyword Fallback"

                        st.markdown(
                            f"""
                            <div class="ai-result-box" style="text-align:center;">
                                <div class="category-pill">🏷️ {cat_name}</div>
                                <div class="ai-result-meta" style="border-top:none; padding-top:0; margin-top:6px;">
                                    <strong>AI Confidence:</strong> {conf_str} &nbsp;|&nbsp; 
                                    <strong>Method:</strong> {method_label}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"Unable to classify article. {current_category.get('error', '')}")

            # Feature 7: AI Keywords Section
            with col_ai_kw:
                st.markdown("## 🔖 AI Keywords")
                col_kw_cnt, col_kw_act = st.columns([1.5, 2])
                with col_kw_cnt:
                    num_keywords_selected = st.selectbox(
                        "Count",
                        options=[5, 8, 10, 15],
                        index=1,  # Default to 8
                        key=f"kw_count_sel_{hash(active_url)}",
                        help="Select how many keywords to extract."
                    )
                with col_kw_act:
                    st.write("")
                    st.write("")
                    extract_kw_clicked = st.button(
                        "✨ Extract Keywords",
                        type="primary",
                        key=f"btn_extract_kw_{hash(active_url)}",
                        use_container_width=True
                    )

                kw_cache_key = f"{active_url}_{num_keywords_selected}"

                # Preload from SQLite
                if kw_cache_key not in st.session_state["keywords"] and stored_db_article:
                    db_kws = stored_db_article.get("keywords")
                    if db_kws and len(db_kws) == num_keywords_selected:
                        st.session_state["keywords"][kw_cache_key] = {
                            "success": True,
                            "keywords": db_kws,
                            "requested_count": num_keywords_selected,
                            "extracted_count": len(db_kws),
                            "method": "gemini",
                            "error": None,
                        }

                if extract_kw_clicked:
                    with st.spinner("🔖 Extracting keywords..."):
                        kw_output = keyword_extractor.extract_keywords(
                            article_text=extracted_data.get("text"),
                            number_of_keywords=num_keywords_selected
                        )
                        st.session_state["keywords"][kw_cache_key] = kw_output
                        if kw_output.get("success") and kw_output.get("keywords"):
                            database.save_keywords(
                                url=active_url,
                                keywords=kw_output["keywords"],
                                method=kw_output.get("method"),
                            )

                # Display keywords
                current_kw = st.session_state["keywords"].get(kw_cache_key)
                if current_kw:
                    if current_kw.get("success"):
                        kw_list = current_kw.get("keywords", [])
                        if kw_list:
                            chips_html = "".join([
                                f'<span class="keyword-chip">🏷️ {kw}</span>'
                                for kw in kw_list
                            ])
                            st.markdown(
                                f"""
                                <div class="ai-result-box">
                                    <div style="line-height:2.2;">{chips_html}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.write("*No distinct keywords found.*")
                    else:
                        st.error(f"Unable to extract keywords. {current_kw.get('error', '')}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 13: AI Sentiment Analysis Section
            # ---------------------------------------------------------------
            st.markdown("## 💭 AI Sentiment")
            st.caption("Analyze the dominant real-world tone (Positive, Neutral, or Negative) and reported outcome of the news story.")

            col_sent_btn, col_sent_spacer = st.columns([1.8, 2.2])
            with col_sent_btn:
                analyze_sentiment_clicked = st.button(
                    "✨ Analyze Sentiment",
                    type="primary",
                    key=f"btn_analyze_sentiment_{hash(active_url)}",
                    use_container_width=True
                )

            sentiment_cache_key = f"{active_url}_sentiment"

            # Preload from SQLite if available
            if sentiment_cache_key not in st.session_state["sentiments"] and stored_db_article:
                if stored_db_article.get("sentiment"):
                    st.session_state["sentiments"][sentiment_cache_key] = {
                        "success": True,
                        "sentiment": stored_db_article["sentiment"],
                        "confidence": stored_db_article.get("sentiment_confidence"),
                        "explanation": stored_db_article.get("sentiment_explanation"),
                        "method": stored_db_article.get("sentiment_method", "gemini"),
                        "error": None,
                    }

            if analyze_sentiment_clicked:
                with st.spinner("💭 Analyzing article sentiment..."):
                    sentiment_output = sentiment_analyzer.analyze_sentiment(
                        title=art_title,
                        article_text=extracted_data.get("text")
                    )
                    st.session_state["sentiments"][sentiment_cache_key] = sentiment_output
                    if sentiment_output.get("success") and sentiment_output.get("sentiment"):
                        database.save_sentiment(
                            url=active_url,
                            sentiment=sentiment_output["sentiment"],
                            confidence=sentiment_output.get("confidence"),
                            explanation=sentiment_output.get("explanation"),
                            method=sentiment_output.get("method"),
                        )

            # Display sentiment result
            current_sentiment = st.session_state["sentiments"].get(sentiment_cache_key)
            if current_sentiment:
                if current_sentiment.get("success"):
                    s_val = current_sentiment.get("sentiment", "Neutral")
                    conf_val = current_sentiment.get("confidence")
                    conf_str = f"{int(conf_val * 100)}%" if conf_val is not None else "Rule-based"
                    method_label = "Gemini AI" if current_sentiment.get("method") == "gemini" else "Local Fallback"
                    explanation_text = current_sentiment.get("explanation") or "Dominant overall tone determined from reported events."

                    if s_val == "Positive":
                        pill_html = '<span class="sentiment-pill-positive">🟢 Positive</span>'
                    elif s_val == "Negative":
                        pill_html = '<span class="sentiment-pill-negative">🔴 Negative</span>'
                    else:
                        pill_html = '<span class="sentiment-pill-neutral">⚪ Neutral</span>'

                    st.markdown(
                        f"""
                        <div class="ai-result-box">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
                                <div>{pill_html}</div>
                                <div style="font-size:0.85rem; color:var(--text-muted);">
                                    <strong>AI Confidence:</strong> {conf_str} &nbsp;|&nbsp; 
                                    <strong>Method:</strong> {method_label}
                                </div>
                            </div>
                            <div style="font-size:0.95rem; color:var(--text-secondary); line-height:1.6;">
                                {explanation_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"Unable to analyze sentiment. {current_sentiment.get('error', '')}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 10: Duplicate and Related News Detection Section
            # ---------------------------------------------------------------
            st.markdown("## 🔗 Related & Duplicate News")
            st.caption("Identify syndicated coverage, same-story duplicates, and closely related articles across publishers using local TF-IDF cosine similarity.")

            col_rn_num, col_rn_btn = st.columns([1.5, 2])
            with col_rn_num:
                related_limit = st.selectbox(
                    "Max Results",
                    options=[3, 5, 10],
                    index=1,  # Default to 5
                    key=f"related_limit_sel_{hash(active_url)}",
                    help="Select the maximum number of similar articles to retrieve."
                )
            with col_rn_btn:
                st.write("")
                st.write("")
                find_related_clicked = st.button(
                    "🔍 Find Related News",
                    type="primary",
                    key=f"btn_find_related_{hash(active_url)}",
                    use_container_width=True
                )

            related_cache_key = f"{active_url}_{related_limit}"

            if find_related_clicked:
                if not duplicate_detector.is_sklearn_available():
                    st.error("Related news detection is currently unavailable because the similarity module (scikit-learn) is not installed.")
                else:
                    with st.spinner("🔍 Finding similar news articles..."):
                        # Retrieve recent articles from SQLite database to compare against
                        candidate_pool = database.get_recent_articles(limit=100)

                        # Prepare current target article dictionary
                        current_target_article = {
                            "url": active_url,
                            "title": art_title,
                            "description": stored_db_article.get("description") if stored_db_article else "",
                            "article_text": extracted_data.get("text", ""),
                            "source": extracted_data.get("source"),
                            "published_at": extracted_data.get("published_at"),
                        }

                        similar_matches = duplicate_detector.find_similar_articles(
                            target_article=current_target_article,
                            candidate_articles=candidate_pool,
                            threshold=duplicate_detector.RELATED_THRESHOLD,
                            max_results=related_limit,
                        )
                        st.session_state["related_news"][related_cache_key] = similar_matches

            current_related = st.session_state["related_news"].get(related_cache_key)

            if current_related is not None:
                if not current_related:
                    st.info("### 🔗 No Related News Found\n\nNo sufficiently similar articles were found among the stored news.")
                else:
                    st.markdown(f"**Found {len(current_related)} similar / related article(s):**")
                    for match_item in current_related:
                        match_art = match_item["article"]
                        sim_pct = match_item["similarity_percent"]
                        rel_type = match_item["relationship"]

                        badge_class = "badge-duplicate" if "Duplicate" in rel_type else "badge-related"

                        with st.container(border=True):
                            r_c1, r_c2 = st.columns([3.5, 1.2])
                            with r_c1:
                                m_title = match_art.get("title") or "Untitled"
                                st.markdown(f"**{m_title}**")
                                m_src = match_art.get("source") or "Unknown"
                                m_date = utils.format_date_human(match_art.get("published_at"))
                                m_prov = match_art.get("provider", "RSS")

                                st.markdown(
                                    f"""
                                    <div style="margin-top:6px; margin-bottom:8px; display:flex; flex-wrap:wrap; gap:8px; align-items:center;">
                                        <span class="{badge_class}">{rel_type}</span>
                                        <span class="similarity-score-pill">📊 {sim_pct:.1f}% Text Similarity</span>
                                        <span class="meta-pill">🏢 {m_src}</span>
                                        <span class="meta-pill">📅 {m_date}</span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                if match_art.get("description"):
                                    st.caption(utils.truncate_description(match_art.get("description"), max_chars=160))

                            with r_c2:
                                st.write("")
                                if st.button("📄 Analyze This", key=f"btn_rel_analyze_{hash(match_art.get('url', ''))}"):
                                    st.session_state["active_article_url"] = match_art.get("url")
                                    st.session_state["active_article_source"] = match_art.get("source")
                                    st.rerun()

                                if match_art.get("url"):
                                    st.link_button("🔗 Original", url=match_art["url"], use_container_width=True)

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 16: Multi-Source News Comparison
            # ---------------------------------------------------------------
            st.markdown("## ⚖️ Multi-Source News Comparison")
            st.caption("Compare how different news publishers cover this event: uncover common facts, unique angles, framing differences, and sentiment.")

            # Retrieve candidate articles from SQLite to compare against
            all_db_candidates = database.get_recent_articles(limit=100)
            other_candidates = [a for a in all_db_candidates if a.get("url") != active_url]

            if len(other_candidates) == 0:
                st.info("No additional stored articles available to compare. Fetch more news from the **📰 Latest News** tab to enable multi-source comparison.")
            else:
                # Find related articles using Feature 10 duplicate_detector
                current_target_for_comp = {
                    "url": active_url,
                    "title": art_title,
                    "description": stored_db_article.get("description") if stored_db_article else "",
                    "article_text": extracted_data.get("text", ""),
                    "source": extracted_data.get("source"),
                    "published_at": extracted_data.get("published_at"),
                }

                similar_cands = []
                if duplicate_detector.is_sklearn_available():
                    similar_cands = duplicate_detector.find_similar_articles(
                        target_article=current_target_for_comp,
                        candidate_articles=other_candidates,
                        threshold=0.20,
                        max_results=10,
                    )

                # Prioritize similar candidates; if few found, fallback to recent database articles
                candidate_pool_for_selection: list[dict[str, Any]] = []
                if similar_cands:
                    for sc in similar_cands:
                        cand_art = sc["article"]
                        cand_art["_similarity"] = sc["similarity_percent"]
                        candidate_pool_for_selection.append(cand_art)
                else:
                    candidate_pool_for_selection = other_candidates[:8]

                # Options map for selection
                candidate_options_map: dict[str, dict[str, Any]] = {}
                for cand in candidate_pool_for_selection:
                    cand_src = cand.get("source") or "News Source"
                    cand_title = cand.get("title") or "Untitled"
                    sim_label = f" ({cand.get('_similarity', 0):.0f}% Match)" if "_similarity" in cand else ""
                    opt_label = f"{cand_src}: {cand_title[:75]}{'...' if len(cand_title) > 75 else ''}{sim_label}"
                    candidate_options_map[opt_label] = cand

                st.markdown(f"**Primary Article:** `{extracted_data.get('source') or 'Current Source'}` — *{art_title}*")

                selected_candidate_labels = st.multiselect(
                    "Select 1 to 4 additional source articles to compare against:",
                    options=list(candidate_options_map.keys()),
                    max_selections=4,
                    key=f"compare_multiselect_{hash(active_url)}",
                    help="Choose up to 4 other articles to compare with the current article (total 2 to 5 sources)."
                )

                col_comp_btn, col_comp_status = st.columns([2, 2])
                with col_comp_btn:
                    compare_clicked = st.button(
                        "⚖️ Compare Selected Sources",
                        type="primary",
                        key=f"btn_compare_sources_{hash(active_url)}",
                        use_container_width=True
                    )

                # Build comparison list
                selected_articles_for_comp = [
                    {
                        "url": active_url,
                        "title": art_title,
                        "source": extracted_data.get("source") or "Primary Source",
                        "article_text": extracted_data.get("text") or (stored_db_article.get("article_text") if stored_db_article else ""),
                        "description": stored_db_article.get("description") if stored_db_article else "",
                        "summary": stored_db_article.get("summary") if stored_db_article else None,
                        "key_points": stored_db_article.get("key_points") if stored_db_article else None,
                        "category": stored_db_article.get("ai_category") or stored_db_article.get("category") if stored_db_article else None,
                        "keywords": stored_db_article.get("keywords") if stored_db_article else None,
                        "sentiment": stored_db_article.get("sentiment") if stored_db_article else None,
                        "published_at": extracted_data.get("published_at"),
                    }
                ]

                for lbl in selected_candidate_labels:
                    cand_obj = candidate_options_map[lbl]
                    selected_articles_for_comp.append({
                        "url": cand_obj.get("url"),
                        "title": cand_obj.get("title"),
                        "source": cand_obj.get("source") or "News Source",
                        "article_text": cand_obj.get("article_text") or cand_obj.get("description", ""),
                        "description": cand_obj.get("description", ""),
                        "summary": cand_obj.get("summary"),
                        "key_points": cand_obj.get("key_points"),
                        "category": cand_obj.get("ai_category") or cand_obj.get("category"),
                        "keywords": cand_obj.get("keywords"),
                        "sentiment": cand_obj.get("sentiment"),
                        "published_at": cand_obj.get("published_at"),
                    })

                comp_cache_key = f"{active_url}_" + "_".join(sorted([a.get('url', '') for a in selected_articles_for_comp]))

                if compare_clicked:
                    if len(selected_articles_for_comp) < 2:
                        st.warning("Please select at least 1 additional article to compare (minimum 2 total sources).")
                    else:
                        with st.spinner(f"⚖️ Comparing {len(selected_articles_for_comp)} news sources with AI..."):
                            comp_result = source_comparison.compare_articles(selected_articles_for_comp)
                            st.session_state["comparison_results"][comp_cache_key] = comp_result

                # Render Comparison Results if present
                active_comp_res = st.session_state["comparison_results"].get(comp_cache_key)

                if active_comp_res:
                    if active_comp_res.get("success"):
                        with st.container(border=True):
                            st.markdown(f"### ⚖️ Multi-Source Comparison ({len(selected_articles_for_comp)} Publishers)")
                            method_badge = "Gemini AI" if active_comp_res.get("method") == "gemini" else "Local Fallback"
                            st.caption(f"**Analysis Method:** {method_badge} • *Coverage comparison only; does not evaluate absolute truth or bias.*")

                            # 1. Comparison Table
                            st.markdown("#### 📰 Source Overview & Headline Comparison")
                            comp_table_rows = []
                            for s_art in selected_articles_for_comp:
                                s_name = s_art.get("source") or "Source"
                                s_title = s_art.get("title") or "Untitled"
                                s_cat = s_art.get("category") or "—"
                                s_sent = s_art.get("sentiment") or "Not analyzed"
                                s_date = utils.format_date_human(s_art.get("published_at"))
                                kw_count = len(source_comparison._normalize_keyword_set(s_art.get("keywords")))

                                comp_table_rows.append({
                                    "Publisher": s_name,
                                    "Headline": s_title,
                                    "Category": s_cat,
                                    "Sentiment": s_sent,
                                    "Keywords": f"{kw_count} tags",
                                    "Published Date": s_date,
                                })
                            st.dataframe(comp_table_rows, use_container_width=True, hide_index=True)

                            if active_comp_res.get("headline_comparison"):
                                st.info(f"**Headline Focus:** {active_comp_res['headline_comparison']}")

                            st.divider()

                            # 2. Common Information
                            st.markdown("#### 🟢 Common Information Across Sources")
                            common_pts = active_comp_res.get("common_points", [])
                            if common_pts:
                                for pt in common_pts:
                                    st.markdown(f"- {pt}")
                            else:
                                st.write("*No overarching common points identified.*")

                            st.divider()

                            # 3. Differences & Framing Angles
                            st.markdown("#### 🔵 Differences & Reporting Angles")
                            diffs = active_comp_res.get("differences", [])
                            if diffs:
                                d_cols = st.columns(min(len(diffs), 3))
                                for idx, diff_item in enumerate(diffs):
                                    col = d_cols[idx % len(d_cols)]
                                    with col:
                                        st.markdown(
                                            f"""
                                            <div class="diff-card">
                                                <span class="diff-card-source">🏢 {diff_item.get('source', 'Source')}</span>
                                                <div class="diff-card-text">
                                                    {diff_item.get('focus', 'No specific focal angle reported.')}
                                                </div>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                            else:
                                st.write("*No major divergent framing angles detected.*")

                            st.divider()

                            # 4. Unique Information by Source
                            st.markdown("#### 🟡 Unique Information by Source")
                            unique_infos = active_comp_res.get("unique_information", [])
                            if unique_infos:
                                for u_item in unique_infos:
                                    src_name = u_item.get("source", "Source")
                                    pts = u_item.get("points", [])
                                    pts_str = " ".join(pts) if pts else "Shares core facts with other sources."
                                    st.markdown(f"**{src_name}:** {pts_str}")
                            else:
                                st.write("*No distinct unique details reported exclusively by one source.*")

                            st.divider()

                            # 5. Shared & Unique Keywords
                            st.markdown("#### 🔖 Shared & Unique Keywords")
                            sh_kws = active_comp_res.get("shared_keywords", [])
                            un_kws = active_comp_res.get("unique_keywords", {})

                            if sh_kws:
                                sh_chips = " ".join([f'<span class="meta-pill meta-pill-provider">🤝 {kw}</span>' for kw in sh_kws])
                                st.markdown(f"**Shared Keywords (in $\\ge 2$ sources):**<br><div style='line-height:2.2; margin-top:4px;'>{sh_chips}</div>", unsafe_allow_html=True)

                            if un_kws:
                                for u_src, u_tags in un_kws.items():
                                    if u_tags:
                                        u_chips = " ".join([f'<span class="meta-pill">🏷️ {kw}</span>' for kw in u_tags[:5]])
                                        st.markdown(f"**{u_src} Unique Keywords:**<br><div style='line-height:2.2; margin-top:4px;'>{u_chips}</div>", unsafe_allow_html=True)

                            st.divider()

                            # 6. Overall Comparison Summary
                            st.markdown("#### 🤖 AI Comparison Summary")
                            st.markdown(
                                f"""
                                <div class="ai-result-box" style="font-size:0.96rem; line-height:1.7;">
                                    {active_comp_res.get('overall_summary', 'Multi-source summary completed.')}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.error(f"Unable to complete comparison. {active_comp_res.get('error', '')}")

            st.markdown("---")

            # ---------------------------------------------------------------
            # Feature 15: AI Article Chat (Grounded Q&A)
            # ---------------------------------------------------------------
            st.markdown("## 💬 Ask AI About This Article")
            st.caption("Ask questions strictly grounded in the content of this news story. The AI will not hallucinate facts outside the article.")

            # Initialize chat history for the active article URL
            if active_url not in st.session_state["chat_messages_by_url"]:
                st.session_state["chat_messages_by_url"][active_url] = []

            article_chat_history = st.session_state["chat_messages_by_url"][active_url]

            # Chat Banner & Controls
            chat_head_c1, chat_head_c2 = st.columns([3.5, 1])
            with chat_head_c1:
                st.markdown(
                    f"""
                    <div class="chat-banner">
                        💬 <strong>Discussing:</strong> {art_title[:75]}{'...' if len(art_title) > 75 else ''} &nbsp;|&nbsp; 
                        <strong>Source:</strong> {extracted_data.get('source') or 'News Publisher'}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with chat_head_c2:
                if st.button("🗑️ Clear Chat", key=f"btn_clear_chat_{hash(active_url)}", use_container_width=True):
                    st.session_state["chat_messages_by_url"][active_url] = []
                    st.rerun()

            # Suggested Quick Questions
            st.markdown("<div style='font-size:0.85rem; font-weight:600; color:var(--text-muted); margin-top:10px; margin-bottom:4px;'>Suggested Questions:</div>", unsafe_allow_html=True)
            q_c1, q_c2, q_c3, q_c4 = st.columns(4)
            clicked_quick_q = None
            with q_c1:
                if st.button("📌 What is this about?", key=f"qq1_{hash(active_url)}", use_container_width=True):
                    clicked_quick_q = "What is this article about?"
            with q_c2:
                if st.button("👥 Who is involved?", key=f"qq2_{hash(active_url)}", use_container_width=True):
                    clicked_quick_q = "Who are the key people and organizations involved in this article?"
            with q_c3:
                if st.button("⚡ What happened?", key=f"qq3_{hash(active_url)}", use_container_width=True):
                    clicked_quick_q = "What major event or decision happened according to this article?"
            with q_c4:
                if st.button("🎯 Key takeaways?", key=f"qq4_{hash(active_url)}", use_container_width=True):
                    clicked_quick_q = "What are the main takeaways and implications stated in the article?"

            # Display prior conversation messages
            if article_chat_history:
                for msg in article_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            else:
                st.caption("No messages yet. Ask any question below or click a suggested question above.")

            # Handle chat input (either from chat_input or quick questions)
            user_chat_input = st.chat_input("Ask something about this article...", key=f"chat_input_{hash(active_url)}")
            prompt_to_process = clicked_quick_q if clicked_quick_q else user_chat_input

            if prompt_to_process:
                # Append user message
                st.session_state["chat_messages_by_url"][active_url].append({"role": "user", "content": prompt_to_process})

                # Extract supplementary context if available
                supp_ctx = {}
                if stored_db_article:
                    supp_ctx["summary"] = stored_db_article.get("summary")
                    supp_ctx["category"] = stored_db_article.get("ai_category") or stored_db_article.get("category")
                    supp_ctx["sentiment"] = stored_db_article.get("sentiment")
                    if stored_db_article.get("key_points"):
                        try:
                            supp_ctx["key_points"] = json.loads(stored_db_article["key_points"])
                        except Exception:
                            pass

                with st.spinner("🤖 Thinking..."):
                    response = chat.ask_article_question(
                        article_title=art_title,
                        article_text=extracted_data.get("text"),
                        question=prompt_to_process,
                        chat_history=article_chat_history[:-1],
                        supplementary_context=supp_ctx,
                    )
                    answer_text = response.get("answer", "Unable to generate an answer right now.")
                    st.session_state["chat_messages_by_url"][active_url].append({"role": "assistant", "content": answer_text})
                st.rerun()

            st.markdown("---")
            # Action button to read full article online
            st.link_button("🔗 Read Original Article on Publisher Website", url=active_url)

        else:
            st.warning("⚠ Full article content could not be extracted from this source.")
            err_msg = extracted_data.get("error", "Website blocked access or article contains no readable text.")
            st.caption(f"**Details:** {err_msg}")
            st.link_button("🔗 Open Original Article", url=active_url)


# ---------------------------------------------------------------------------
# TAB 4: Stored Articles History (Database View)
# ---------------------------------------------------------------------------

with tab_history:
    st.markdown("## 💾 Stored Database Articles")
    st.caption("Browse previously fetched and processed articles persisted in the local SQLite database.")

    try:
        recent_records = database.get_recent_articles(limit=20)
    except Exception as e:
        st.error(f"Unable to load database records: {e}")
        recent_records = []

    if not recent_records:
        st.info("No articles stored in the database yet. Fetch news from the sidebar to populate the database.")
    else:
        for rec in recent_records:
            with st.container(border=True):
                r_col1, r_col2 = st.columns([3.5, 1.2])
                with r_col1:
                    st.markdown(f"**{rec.get('title')}**")
                    r_src = rec.get("source") or "Unknown"
                    r_date = utils.format_date_human(rec.get("published_at"))
                    r_cat = rec.get("category") or "General"
                    st.caption(f"🏢 {r_src} &nbsp;|&nbsp; 📅 {r_date} &nbsp;|&nbsp; 🏷️ Category: `{r_cat}`")

                    # Indicator flags for stored AI outputs
                    ai_badges = []
                    if rec.get("summary"):
                        ai_badges.append("Summary ✓")
                    if rec.get("key_points"):
                        ai_badges.append("Key Points ✓")
                    if rec.get("ai_category"):
                        ai_badges.append(f"AI Category: {rec.get('ai_category')} ✓")
                    if rec.get("keywords"):
                        ai_badges.append("Keywords ✓")

                    if ai_badges:
                        st.markdown(
                            " ".join([f"<span class='meta-pill'>{b}</span>" for b in ai_badges]),
                            unsafe_allow_html=True,
                        )

                with r_col2:
                    if st.button("📄 Analyze Article", key=f"db_analyze_{rec.get('id')}_{hash(rec.get('url', ''))}"):
                        st.session_state["active_article_url"] = rec.get("url")
                        st.session_state["active_article_source"] = rec.get("source")
                        st.rerun()



# ---------------------------------------------------------------------------
# TAB 5: News Analytics Dashboard (Feature 12)
# ---------------------------------------------------------------------------

with tab_dashboard:
    st.markdown("## 📊 News Analytics Dashboard")
    st.caption("Your AI-powered news intelligence overview — all data read from SQLite. No Gemini or API calls.")

    # ── Dashboard Filters ──────────────────────────────────────────────────
    dash_col_f1, dash_col_f2, dash_col_f3, dash_col_f4 = st.columns([2, 2, 2, 1])
    with dash_col_f1:
        dash_date_filter = st.selectbox(
            "Date Range",
            options=["All Time", "Today", "Last 7 Days", "Last 30 Days", "Custom Range"],
            index=0,
            key="dash_date_filter",
        )
    with dash_col_f2:
        dash_category_filter = st.selectbox(
            "Category",
            options=["All"] + news_fetcher.SUPPORTED_CATEGORIES,
            index=0,
            key="dash_category_filter",
        )
    with dash_col_f3:
        dash_custom_start = None
        dash_custom_end = None
        if dash_date_filter == "Custom Range":
            dash_custom_start = st.date_input("From", key="dash_from")
            dash_custom_end = st.date_input("To", key="dash_to")
        else:
            st.write("")
    with dash_col_f4:
        st.write("")
        st.write("")
        refresh_dash = st.button("🔄 Refresh", use_container_width=True, key="dash_refresh_btn")

    dash_cat_arg = None if dash_category_filter == "All" else dash_category_filter
    dash_start_str = str(dash_custom_start) if dash_custom_start else None
    dash_end_str = str(dash_custom_end) if dash_custom_end else None

    st.markdown("---")

    # ── Load all dashboard data from SQLite ─────────────────────────────────
    try:
        dash_metrics = database.get_dashboard_summary_metrics(
            category=dash_cat_arg,
            date_filter=dash_date_filter,
            start_date=dash_start_str,
            end_date=dash_end_str,
        )
    except Exception:
        dash_metrics = {
            "total_articles": 0,
            "total_categories": 0,
            "total_sources": 0,
            "ai_processed_count": 0,
            "today_articles": 0,
            "week_articles": 0,
            "avg_word_count": 0,
        }

    total_articles = dash_metrics.get("total_articles", 0)

    # ── Empty State ────────────────────────────────────────────────────────
    if total_articles == 0:
        st.info("### 📊 No Stored News Available Yet\n\nNo stored articles match the selected filters.\n\nFetch some news first from the **📰 Latest News** tab to populate the dashboard.")
    else:
        # ── TOP METRICS BAR ────────────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("📰 Total Articles", total_articles)
        with mc2:
            st.metric("🤖 AI Processed", dash_metrics.get("ai_processed_count", 0))
        with mc3:
            st.metric("🗂 Categories", dash_metrics.get("total_categories", 0))
        with mc4:
            st.metric("🌐 Sources", dash_metrics.get("total_sources", 0))

        mc5, mc6, mc7, _ = st.columns(4)
        with mc5:
            st.metric("📅 Today", dash_metrics.get("today_articles", 0))
        with mc6:
            st.metric("📆 This Week", dash_metrics.get("week_articles", 0))
        with mc7:
            avg_wc = dash_metrics.get("avg_word_count", 0)
            st.metric("📝 Avg. Article Length", f"{avg_wc:,} words" if avg_wc else "N/A")

        st.markdown("---")

        # ── NEWS VOLUME OVER TIME ──────────────────────────────────────────
        st.markdown("### 📈 News Volume Over Time")
        try:
            time_data = database.get_articles_over_time(
                category=dash_cat_arg,
                date_filter=dash_date_filter,
                start_date=dash_start_str,
                end_date=dash_end_str,
            )
            if time_data:
                dates = [d["date"] for d in time_data]
                counts = [d["count"] for d in time_data]
                fig_time = go.Figure()
                fig_time.add_trace(go.Bar(
                    x=dates,
                    y=counts,
                    marker_color="#3b82f6",
                    name="Articles",
                    hovertemplate="<b>%{x}</b><br>Articles: %{y}<extra></extra>",
                ))
                if len(dates) > 2:
                    fig_time.add_trace(go.Scatter(
                        x=dates,
                        y=counts,
                        mode="lines",
                        line=dict(color="#f59e0b", width=2, dash="dot"),
                        name="Trend",
                        hoverinfo="skip",
                    ))
                fig_time.update_layout(
                    template=theme.get_plotly_template(),
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=280,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(showgrid=False, tickangle=-30),
                    yaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"], title="Articles"),
                    font=dict(color=theme.get_theme_tokens()["text_primary"]),
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No date data available for the selected filters.")
        except Exception as e:
            st.warning(f"Could not load news trend chart: {e}")

        st.markdown("---")

        # ── CATEGORY + SOURCE (side by side) ──────────────────────────────
        chart_col_left, chart_col_right = st.columns(2)

        with chart_col_left:
            st.markdown("#### 🗂 Articles by Category")
            try:
                cat_data = database.get_category_distribution(
                    date_filter=dash_date_filter,
                    start_date=dash_start_str,
                    end_date=dash_end_str,
                )
                if cat_data:
                    cat_names = [d["category"] for d in cat_data]
                    cat_counts = [d["count"] for d in cat_data]
                    fig_cat = go.Figure(go.Bar(
                        x=cat_counts,
                        y=cat_names,
                        orientation="h",
                        marker_color="#8b5cf6",
                        hovertemplate="<b>%{y}</b><br>Articles: %{x}<extra></extra>",
                    ))
                    fig_cat.update_layout(
                        template=theme.get_plotly_template(),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=300,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"]),
                        yaxis=dict(autorange="reversed"),
                        font=dict(color=theme.get_theme_tokens()["text_primary"]),
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.info("No category data available.")
            except Exception as e:
                st.warning(f"Could not load category chart: {e}")

        with chart_col_right:
            st.markdown("#### 🌐 Articles by Source (Top 10)")
            try:
                src_data = database.get_source_distribution(
                    category=dash_cat_arg,
                    date_filter=dash_date_filter,
                    start_date=dash_start_str,
                    end_date=dash_end_str,
                    limit=10,
                )
                if src_data:
                    src_names = [d["source"] for d in src_data]
                    src_counts = [d["count"] for d in src_data]
                    fig_src = go.Figure(go.Bar(
                        x=src_counts,
                        y=src_names,
                        orientation="h",
                        marker_color="#10b981",
                        hovertemplate="<b>%{y}</b><br>Articles: %{x}<extra></extra>",
                    ))
                    fig_src.update_layout(
                        template=theme.get_plotly_template(),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=300,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"]),
                        yaxis=dict(autorange="reversed"),
                        font=dict(color=theme.get_theme_tokens()["text_primary"]),
                    )
                    st.plotly_chart(fig_src, use_container_width=True)
                else:
                    st.info("No source data available.")
            except Exception as e:
                st.warning(f"Could not load source chart: {e}")

        st.markdown("---")

        # ── TRENDING TOPICS IN STORED NEWS (Feature 14) ────────────────────
        st.markdown("### 🔥 Trending Topics in Stored News")
        st.caption("Identifies the most frequently and recently discussed topics across stored articles with recency-weighted trending scores.")

        tr_c1, tr_c2, tr_c3 = st.columns([1.5, 1.2, 1])
        with tr_c1:
            trend_days_preset = st.selectbox(
                "Time Window",
                options=["Today (24h)", "Last 3 Days", "Last 7 Days", "Last 30 Days"],
                index=2,  # Default: Last 7 Days
                key="trend_window_select",
                help="Select the recency evaluation window."
            )
        with tr_c2:
            trend_limit_sel = st.selectbox(
                "Top Topics Limit",
                options=[5, 10, 15],
                index=1,  # Default: 10
                key="trend_limit_select",
                help="Maximum number of trending topics to display."
            )
        with tr_c3:
            st.write("")
            st.write("")
            refresh_trends = st.button("🔄 Refresh Trends", use_container_width=True)

        days_map = {
            "Today (24h)": 1,
            "Last 3 Days": 3,
            "Last 7 Days": 7,
            "Last 30 Days": 30,
        }
        eval_days = days_map.get(trend_days_preset, 7)

        try:
            trending_data = trending_topics.get_trending_topics(
                days=eval_days,
                limit=trend_limit_sel,
                category=dash_cat_arg,
            )

            if trending_data:
                t_col_chart, t_col_cards = st.columns([1.3, 1.7])

                with t_col_chart:
                    topic_names = [t["topic"] for t in trending_data]
                    topic_scores = [t["score"] for t in trending_data]
                    fig_trend = go.Figure(go.Bar(
                        x=topic_scores,
                        y=topic_names,
                        orientation="h",
                        marker=dict(
                            color=topic_scores,
                            colorscale="YlOrRd",
                            showscale=False,
                        ),
                        hovertemplate="<b>%{y}</b><br>Trend Score: %{x}<extra></extra>",
                    ))
                    fig_trend.update_layout(
                        template=theme.get_plotly_template(),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=max(260, len(trending_data) * 28),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"], title="Trending Score (Frequency × Recency)"),
                        yaxis=dict(autorange="reversed"),
                        font=dict(color=theme.get_theme_tokens()["text_primary"]),
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

                with t_col_cards:
                    st.markdown("**🔥 Top Trending Topics:**")
                    for rank, t_item in enumerate(trending_data, start=1):
                        t_name = t_item["topic"]
                        t_cnt = t_item["article_count"]
                        t_score = t_item["score"]
                        t_dir = t_item["trend"]
                        t_sent = t_item.get("dominant_sentiment")

                        if t_dir == "Rising":
                            dir_badge = '<span style="color:#ef4444; font-weight:700;">🔥 Rising</span>'
                        elif t_dir == "Declining":
                            dir_badge = '<span style="color:#64748b; font-weight:600;">↓ Declining</span>'
                        elif t_dir == "Stable":
                            dir_badge = '<span style="color:#3b82f6; font-weight:600;">→ Stable</span>'
                        else:
                            dir_badge = '<span style="color:#8b5cf6; font-weight:600;">⚡ Current</span>'

                        sent_badge = ""
                        if t_sent == "Positive":
                            sent_badge = '&nbsp;•&nbsp;<span style="color:#10b981;">🟢 Pos</span>'
                        elif t_sent == "Negative":
                            sent_badge = '&nbsp;•&nbsp;<span style="color:#ef4444;">🔴 Neg</span>'
                        elif t_sent == "Neutral":
                            sent_badge = '&nbsp;•&nbsp;<span style="color:#64748b;">⚪ Neu</span>'

                        st.markdown(
                            f"""
                            <div class="trend-card">
                                <div>
                                    <div class="trend-card-title">#{rank} {t_name}</div>
                                    <div class="trend-card-meta">
                                        {t_cnt} article(s) • Score: <strong>{t_score}</strong> {sent_badge}
                                    </div>
                                </div>
                                <div>{dir_badge}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with st.expander("📋 View Complete Trending Topics Table", expanded=False):
                    table_rows = []
                    for rank, t_item in enumerate(trending_data, start=1):
                        table_rows.append({
                            "Rank": f"#{rank}",
                            "Topic": t_item["topic"],
                            "Articles": t_item["article_count"],
                            "Trend Score": t_item["score"],
                            "Trend Movement": t_item["trend"],
                            "Dominant Sentiment": t_item.get("dominant_sentiment") or "—",
                        })
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No trending topics found in the selected time window. Fetch more news from the Latest News tab to generate trends.")
        except Exception as e:
            st.warning(f"Could not calculate trending topics: {e}")

        st.markdown("---")

        # ── SENTIMENT DISTRIBUTION (Feature 13) ────────────────────────────
        st.markdown("### 💭 Sentiment Distribution")
        try:
            sent_data = database.get_sentiment_distribution(
                category=dash_cat_arg,
                date_filter=dash_date_filter,
                start_date=dash_start_str,
                end_date=dash_end_str,
            )
            if sent_data:
                sent_names = [d["sentiment"] for d in sent_data]
                sent_counts = [d["count"] for d in sent_data]
                color_map = {"Positive": "#10b981", "Neutral": "#64748b", "Negative": "#ef4444"}
                bar_colors = [color_map.get(name, "#8b5cf6") for name in sent_names]

                fig_sent = go.Figure(go.Bar(
                    x=sent_names,
                    y=sent_counts,
                    marker_color=bar_colors,
                    hovertemplate="<b>%{x}</b><br>Articles: %{y}<extra></extra>",
                ))
                fig_sent.update_layout(
                    template=theme.get_plotly_template(),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=260,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"], title="Articles"),
                    xaxis=dict(showgrid=False),
                    font=dict(color=theme.get_theme_tokens()["text_primary"]),
                )
                st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.info("No sentiment data available yet. Analyze sentiment from the Article Details tab first.")
        except Exception as e:
            st.warning(f"Could not load sentiment chart: {e}")

        st.markdown("---")

        # ── TOP KEYWORDS ───────────────────────────────────────────────────
        st.markdown("### 🔥 Top Keywords")
        try:
            kw_data = database.get_top_keywords(
                category=dash_cat_arg,
                date_filter=dash_date_filter,
                start_date=dash_start_str,
                end_date=dash_end_str,
                top_n=15,
            )
            if kw_data:
                kw_col_chart, kw_col_tags = st.columns([2, 1])
                with kw_col_chart:
                    kw_labels = [d["keyword"] for d in kw_data]
                    kw_counts = [d["count"] for d in kw_data]
                    fig_kw = go.Figure(go.Bar(
                        x=kw_counts,
                        y=kw_labels,
                        orientation="h",
                        marker=dict(
                            color=kw_counts,
                            colorscale="Oranges",
                            showscale=False,
                        ),
                        hovertemplate="<b>%{y}</b><br>Appearances: %{x}<extra></extra>",
                    ))
                    fig_kw.update_layout(
                        template=theme.get_plotly_template(),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=380,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"], title="Appearances"),
                        yaxis=dict(autorange="reversed"),
                        font=dict(color=theme.get_theme_tokens()["text_primary"]),
                    )
                    st.plotly_chart(fig_kw, use_container_width=True)

                with kw_col_tags:
                    st.markdown("**Keyword Tags:**")
                    tags_html = " ".join([
                        f'<span class="keyword-tag-pill">'
                        f'🏷️ {kw["keyword"]} <strong style="color:var(--accent);">×{kw["count"]}</strong></span>'
                        for kw in kw_data[:10]
                    ])
                    st.markdown(f'<div style="line-height:2.2;">{tags_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No AI keywords have been generated yet.")
        except Exception as e:
            st.warning(f"Could not load keyword data: {e}")

        st.markdown("---")

        # ── AI PROCESSING STATUS ───────────────────────────────────────────
        st.markdown("### 🤖 AI Processing Status")
        try:
            ai_stats = database.get_ai_processing_breakdown(
                category=dash_cat_arg,
                date_filter=dash_date_filter,
                start_date=dash_start_str,
                end_date=dash_end_str,
            )
            ai_total = ai_stats.get("total", 0)

            if ai_total > 0:
                ai_c1, ai_c2 = st.columns(2)
                ai_features = [
                    ("🤖 Summary", "summary_count", "summary_pct", "#3b82f6"),
                    ("🔑 Key Points", "key_points_count", "key_points_pct", "#8b5cf6"),
                    ("🏷️ AI Category", "category_count", "category_pct", "#10b981"),
                    ("🔖 Keywords", "keywords_count", "keywords_pct", "#f59e0b"),
                    ("💭 Sentiment", "sentiment_count", "sentiment_pct", "#ec4899"),
                ]
                for i, (label, cnt_key, pct_key, color) in enumerate(ai_features):
                    col = ai_c1 if i % 2 == 0 else ai_c2
                    with col:
                        cnt = ai_stats.get(cnt_key, 0)
                        pct = ai_stats.get(pct_key, 0.0)
                        st.markdown(
                            f"""
                            <div class="ai-progress-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                                    <span class="ai-progress-label">{label}</span>
                                    <span class="ai-progress-meta">{cnt} / {ai_total} &nbsp; <strong style="color:{color};">{pct}%</strong></span>
                                </div>
                                <div class="ai-progress-track">
                                    <div style="width:{pct}%; background:{color}; height:10px; border-radius:8px; transition:width 0.5s;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Summary type breakdown
                try:
                    sum_types = database.get_summary_type_distribution(
                        category=dash_cat_arg,
                        date_filter=dash_date_filter,
                        start_date=dash_start_str,
                        end_date=dash_end_str,
                    )
                    if sum_types:
                        with st.expander("📊 Summary Type Breakdown", expanded=False):
                            sum_labels = [d["type"] for d in sum_types]
                            sum_counts = [d["count"] for d in sum_types]
                            fig_sum = go.Figure(go.Bar(
                                x=sum_labels,
                                y=sum_counts,
                                marker_color=["#3b82f6", "#8b5cf6", "#10b981"][:len(sum_labels)],
                                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                            ))
                            fig_sum.update_layout(
                                template=theme.get_plotly_template(),
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=200,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                yaxis=dict(showgrid=True, gridcolor=theme.get_theme_tokens()["border_color"]),
                                font=dict(color=theme.get_theme_tokens()["text_primary"]),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_sum, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("No articles match the selected filters.")
        except Exception as e:
            st.warning(f"Could not load AI processing stats: {e}")

        st.markdown("---")

        # ── DUPLICATE DETECTION NOTE ────────────────────────────────────────
        with st.expander("🔗 Duplicate & Related News", expanded=False):
            st.info(
                "Duplicate and related article detection is performed dynamically using TF-IDF + Cosine Similarity. "
                "To find similar articles for a specific article, open it in the **📄 Article Details & AI Insights** tab "
                "and click **🔍 Find Related News**."
            )

        st.markdown("---")

        # ── RECENT ARTICLES TABLE ───────────────────────────────────────────
        st.markdown("### 📰 Recent Articles")
        try:
            recent_arts = database.get_dashboard_recent_articles(
                category=dash_cat_arg,
                date_filter=dash_date_filter,
                start_date=dash_start_str,
                end_date=dash_end_str,
                limit=10,
            )
            if recent_arts:
                for rec_art in recent_arts:
                    rec_title = (rec_art.get("title") or "Untitled")[:65]
                    rec_source = rec_art.get("source") or "—"
                    rec_cat = rec_art.get("category") or "—"
                    rec_date = utils.format_date_human(rec_art.get("published_at") or rec_art.get("created_at"))
                    ai_status_parts = []
                    if rec_art.get("has_summary"):
                        ai_status_parts.append("🤖")
                    if rec_art.get("has_key_points"):
                        ai_status_parts.append("🔑")
                    if rec_art.get("has_ai_category"):
                        ai_status_parts.append("🏷️")
                    if rec_art.get("has_keywords"):
                        ai_status_parts.append("🔖")
                    if rec_art.get("has_sentiment"):
                        ai_status_parts.append("💭")
                    ai_status = " ".join(ai_status_parts) if ai_status_parts else "—"

                    with st.container():
                        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns([3, 1.2, 1.2, 1.2, 0.8, 1.2])
                        with rc1:
                            st.markdown(f"**{rec_title}**{'…' if len(rec_art.get('title','')) > 65 else ''}")
                        with rc2:
                            st.caption(rec_source)
                        with rc3:
                            st.caption(rec_cat)
                        with rc4:
                            st.caption(rec_date)
                        with rc5:
                            st.caption(ai_status)
                        with rc6:
                            if st.button("📄 Open", key=f"dash_open_{rec_art.get('id')}_{hash(rec_art.get('url',''))}"):
                                st.session_state["active_article_url"] = rec_art.get("url")
                                st.session_state["active_article_source"] = rec_art.get("source")
                                st.rerun()
                    st.divider()
            else:
                st.info("No recent articles found for the selected filters.")
        except Exception as e:
            st.warning(f"Could not load recent articles: {e}")


# ---------------------------------------------------------------------------
# 8. Professional Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-footer">
        AI News Summarizer • Powered by Gemini AI • NewsAPI / RSS • SQLite Database
    </div>
    """,
    unsafe_allow_html=True,
)
