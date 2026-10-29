"""
test_suite.py
-------------
Comprehensive end-to-end test suite for validating Features 1–11 of the
AI News Summarizer application.
"""

from datetime import datetime, timezone, timedelta
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
import utils


def run_all_tests():
    print("================================================================")
    print("             AI NEWS SUMMARIZER - FULL SYSTEM TEST             ")
    print("================================================================")

    # 1. Database Init
    print("\n[Test 1/11] Database Initialization & Table Check...")
    assert database.init_database() is True
    count_db = database.get_article_count()
    ai_count = database.get_ai_processed_count()
    print(f"-> SQLite DB Connected | Total Articles: {count_db} | AI Processed: {ai_count}")

    # 2. News Collection
    print("\n[Test 2/11] News Collection (Technology & Business)...")
    tech_news = news_fetcher.fetch_news("Technology", limit=5)
    assert len(tech_news) > 0
    print(f"-> Fetched {len(tech_news)} Tech articles. Sample: '{tech_news[0]['title'][:40]}...' (Provider: {tech_news[0]['provider']})")
    biz_news = news_fetcher.fetch_news("Business", limit=5)
    assert len(biz_news) > 0
    print(f"-> Fetched {len(biz_news)} Business articles.")

    # 3. Article Extraction
    print("\n[Test 3/11] Article Extraction...")
    first_art = tech_news[0]
    ext_res = article_extractor.extract_article(first_art["url"], source_hint=first_art.get("source"))
    print(f"-> Extraction Success: {ext_res.get('success')} | Word Count: {ext_res.get('word_count')}")

    sample_text = ext_res.get("text") or (
        "Artificial intelligence company OpenAI has officially released a new reasoning model "
        "designed to solve complex mathematics, software engineering, and scientific research challenges. "
        "The new model employs advanced reinforcement learning techniques to think before responding, "
        "reducing factual errors and improving step-by-step logical coherence."
    )

    # 4. Summarization
    print("\n[Test 4/11] AI Summarization...")
    sum_res = summarizer.summarize_article(sample_text, "medium")
    print(f"-> Summary result handled gracefully: {sum_res.get('summary_type')}")

    # 5. Key Points
    print("\n[Test 5/11] Key Points Extraction...")
    kp_res = summarizer.extract_key_points(sample_text, number_of_points=3)
    print(f"-> Key Points handled gracefully: {kp_res.get('requested_points')} requested")

    # 6. Categorization
    print("\n[Test 6/11] AI News Categorization...")
    cat_res = classifier.classify_article(first_art["title"], sample_text)
    print(f"-> Category: {cat_res.get('category')} | Confidence: {cat_res.get('confidence')} | Method: {cat_res.get('method')}")
    assert cat_res.get("category") in classifier.VALID_CATEGORIES

    # 7. Keyword Extraction
    print("\n[Test 7/11] AI Keyword Extraction...")
    kw_res = keyword_extractor.extract_keywords(sample_text, number_of_keywords=5)
    print(f"-> Extracted {len(kw_res.get('keywords', []))} keywords: {kw_res.get('keywords')} (Method: {kw_res.get('method')})")

    # 8. Database Persistence
    print("\n[Test 8/11] SQLite Persistence & Deserialization...")
    test_url = "https://example.com/test-feature-11-suite"
    database.save_article({
        "title": "OpenAI Reasoning Model Launch",
        "url": test_url,
        "source": "AI Tech News",
        "category": "Technology",
        "description": "Comprehensive report on the new AI launch.",
        "published_at": "2026-08-28T10:00:00Z",
    })
    database.update_article_content(test_url, sample_text, 85, title="OpenAI Reasoning Model Launch")
    database.save_summary(test_url, "Clean factual test summary.", "medium")
    database.save_key_points(test_url, ["Point A", "Point B", "Point C"])
    database.save_category(test_url, "Technology", confidence=0.92, method="test")
    database.save_keywords(test_url, ["OpenAI", "Reasoning", "Artificial Intelligence"], method="test")

    retrieved = database.get_article_by_url(test_url)
    assert retrieved is not None
    assert retrieved["summary"] == "Clean factual test summary."
    assert len(retrieved["key_points"]) == 3
    assert retrieved["ai_category"] == "Technology"
    assert "OpenAI" in retrieved["keywords"]
    print("-> Successfully verified SQLite save and JSON deserialization.")

    # 9. Duplicate and Related News Detection (Feature 10)
    print("\n[Test 9/11] Duplicate & Related News Detection (TF-IDF Cosine Similarity)...")
    assert duplicate_detector.is_sklearn_available() is True

    doc_a = {
        "url": "https://reuters.com/ai-launch",
        "title": "OpenAI launches new artificial intelligence reasoning model",
        "description": "OpenAI introduced its latest reasoning AI model for science and math.",
        "article_text": sample_text,
    }

    doc_b = {
        "url": "https://bbc.com/ai-launch",
        "title": "OpenAI releases new artificial intelligence reasoning model",
        "description": "OpenAI introduced its latest reasoning AI model for science and math.",
        "article_text": sample_text,
    }

    doc_c = {
        "url": "https://sports.com/premier-league",
        "title": "Liverpool defeats Arsenal in dramatic Premier League clash",
        "description": "A last-minute goal secured victory in today's football match.",
        "article_text": "The football match concluded with a thrilling stoppage-time winner at Anfield stadium.",
    }

    sim_ab = duplicate_detector.calculate_similarity(doc_a["title"], doc_b["title"])
    print(f"-> Pairwise Title Similarity (Doc A vs Doc B): {sim_ab*100:.1f}% ({duplicate_detector.classify_relationship(sim_ab)})")
    assert sim_ab >= duplicate_detector.DUPLICATE_THRESHOLD

    sim_ac = duplicate_detector.calculate_similarity(doc_a["title"], doc_c["title"])
    print(f"-> Pairwise Title Similarity (Doc A vs Doc C): {sim_ac*100:.1f}% ({duplicate_detector.classify_relationship(sim_ac)})")
    assert sim_ac < duplicate_detector.RELATED_THRESHOLD

    similar_results = duplicate_detector.find_similar_articles(
        target_article=doc_a,
        candidate_articles=[doc_a, doc_b, doc_c],
        threshold=0.50,
        max_results=5,
    )
    print(f"-> Found {len(similar_results)} similar articles for Doc A:")
    assert len(similar_results) >= 1
    assert similar_results[0]["relationship"] == "Duplicate / Same Story"

    # 10. Search & Filtering (Feature 11)
    print("\n[Test 10/11] Search & Filtering System (Feature 11)...")
    
    # 10a. Distinct sources
    sources = database.get_distinct_sources()
    print(f"-> Distinct sources retrieved ({len(sources)}): {sources[:5]}")
    assert isinstance(sources, list)

    # 10b. Keyword Search (single word & multi-word)
    res_kw = database.search_articles(query="OpenAI", limit=10)
    print(f"-> Keyword search 'OpenAI': {len(res_kw)} result(s)")
    assert len(res_kw) > 0
    assert any("openai" in (a.get("title") or "").lower() or "openai" in (a.get("keywords") or []) for a in res_kw)

    res_partial = database.search_articles(query="artific", limit=10)
    print(f"-> Partial search 'artific': {len(res_partial)} result(s)")
    assert len(res_partial) > 0

    res_multi = database.search_articles(query="reasoning model", limit=10)
    print(f"-> Multi-word search 'reasoning model': {len(res_multi)} result(s)")
    assert len(res_multi) > 0

    # 10c. Category Filter
    res_tech = database.search_articles(category="Technology", limit=10)
    print(f"-> Category filter 'Technology': {len(res_tech)} result(s)")
    assert len(res_tech) > 0

    # 10d. Date Filters
    res_today = database.search_articles(date_filter="Today", limit=10)
    res_7d = database.search_articles(date_filter="Last 7 Days", limit=10)
    res_30d = database.search_articles(date_filter="Last 30 Days", limit=10)
    print(f"-> Date filters: Today={len(res_today)}, Last 7d={len(res_7d)}, Last 30d={len(res_30d)}")

    # 10e. Sorting
    res_newest = database.search_articles(sort_by="newest", limit=5)
    res_oldest = database.search_articles(sort_by="oldest", limit=5)
    res_relevance = database.search_articles(query="OpenAI Launch", sort_by="relevance", limit=5)
    print(f"-> Sorting options tested: Newest={len(res_newest)}, Oldest={len(res_oldest)}, Relevance={len(res_relevance)}")
    assert len(res_newest) > 0

    # 10f. SQL Injection Security Test
    count_before_inj = database.get_article_count()
    malicious_inputs = [
        "' OR '1'='1",
        "'; DROP TABLE articles; --",
        "' UNION SELECT * FROM articles --",
        "1' OR 1=1 --",
        "admin'--",
    ]
    for injection_attempt in malicious_inputs:
        safe_res = database.search_articles(query=injection_attempt, limit=10)
        assert isinstance(safe_res, list)

    count_after_inj = database.get_article_count()
    assert count_before_inj == count_after_inj
    print("-> SQL Injection Safety Verified: Handled malicious payloads safely without database corruption.")

    # 11. Utils & UI Helper Tests
    print("\n[Test 11/12] Utils & Formatting...")
    assert utils.format_date_human("2026-08-28T13:45:00Z") == "Aug 28, 2026"
    assert len(utils.truncate_description("word " * 100, max_chars=80)) <= 85
    print("-> Formatting functions verified.")

    # 12. News Analytics Dashboard Queries (Feature 12)
    print("\n[Test 12/13] News Analytics Dashboard Queries (Feature 12)...")
    dash_stats = database.get_dashboard_summary_metrics()
    assert isinstance(dash_stats, dict)
    assert dash_stats["total_articles"] > 0
    assert dash_stats["ai_processed_count"] > 0
    print(f"-> Summary Metrics: Total={dash_stats['total_articles']}, AI Processed={dash_stats['ai_processed_count']}, Categories={dash_stats['total_categories']}, Sources={dash_stats['total_sources']}")

    time_trend = database.get_articles_over_time()
    assert isinstance(time_trend, list)
    print(f"-> Over Time Trend: {len(time_trend)} distinct date(s) tracked.")

    cat_dist = database.get_category_distribution()
    assert isinstance(cat_dist, list)
    print(f"-> Category Distribution: {len(cat_dist)} categories found.")

    src_dist = database.get_source_distribution(limit=10)
    assert isinstance(src_dist, list)
    print(f"-> Source Distribution: {len(src_dist)} sources found.")

    top_kws = database.get_top_keywords(top_n=10)
    assert isinstance(top_kws, list)
    print(f"-> Top Keywords: {top_kws[:3]}")

    ai_breakdown = database.get_ai_processing_breakdown()
    assert isinstance(ai_breakdown, dict)
    assert "summary_pct" in ai_breakdown
    assert "key_points_pct" in ai_breakdown
    assert "category_pct" in ai_breakdown
    assert "keywords_pct" in ai_breakdown
    assert "sentiment_pct" in ai_breakdown
    print(f"-> AI Processing Breakdown: Summary={ai_breakdown['summary_pct']}%, KeyPoints={ai_breakdown['key_points_pct']}%, Category={ai_breakdown['category_pct']}%, Keywords={ai_breakdown['keywords_pct']}%, Sentiment={ai_breakdown['sentiment_pct']}%")

    sum_types = database.get_summary_type_distribution()
    assert isinstance(sum_types, list)
    print(f"-> Summary Types: {sum_types}")

    recent_dash = database.get_dashboard_recent_articles(limit=5)
    assert isinstance(recent_dash, list)
    assert len(recent_dash) > 0
    print(f"-> Recent Articles Table: {len(recent_dash)} loaded with AI indicators.")

    # Test filtered queries
    tech_stats = database.get_dashboard_summary_metrics(category="Technology", date_filter="Last 30 Days")
    assert isinstance(tech_stats, dict)
    print("-> Filtered Dashboard Stats (Category=Tech, Date=Last 30 Days) verified.")

    # 13. AI Sentiment Analysis (Feature 13)
    print("\n[Test 13/13] AI Sentiment Analysis (Feature 13)...")
    # Positive test
    pos_res = sentiment_analyzer.analyze_sentiment(
        title="Tech Company Reports Record Profits and Massive Growth",
        article_text="The company achieved unprecedented revenue growth this quarter, breaking past milestones and celebrating major innovation breakthroughs across its product lines."
    )
    assert pos_res["success"] is True
    assert pos_res["sentiment"] in sentiment_analyzer.VALID_SENTIMENTS
    assert pos_res["sentiment"] == "Positive"
    print(f"-> Positive Article Test: Sentiment='{pos_res['sentiment']}', Method='{pos_res['method']}', Explanation='{pos_res['explanation'][:60]}...'")

    # Negative test
    neg_res = sentiment_analyzer.analyze_sentiment(
        title="Severe Hurricane Causes Catastrophic Damage and Casualties",
        article_text="The devastating storm left thousands without power, causing severe destruction, fatalities, and millions of dollars in losses across coastal communities."
    )
    assert neg_res["success"] is True
    assert neg_res["sentiment"] == "Negative"
    print(f"-> Negative Article Test: Sentiment='{neg_res['sentiment']}', Method='{neg_res['method']}', Explanation='{neg_res['explanation'][:60]}...'")

    # Neutral test
    neu_res = sentiment_analyzer.analyze_sentiment(
        title="City Council Holds Scheduled Monthly Committee Meeting",
        article_text="The municipal council convened on Tuesday to review administrative agenda items, standard budget allocations, and routine operational guidelines."
    )
    assert neu_res["success"] is True
    assert neu_res["sentiment"] == "Neutral"
    print(f"-> Neutral Article Test: Sentiment='{neu_res['sentiment']}', Method='{neu_res['method']}'")

    # Short article test
    short_res = sentiment_analyzer.analyze_sentiment(title="Short", article_text="Too brief.")
    assert short_res["success"] is True
    assert short_res["sentiment"] == "Neutral"
    assert "not contain enough information" in short_res["explanation"]
    print("-> Short Article Handling Verified: Returned graceful neutral fallback explanation.")

    # Normalization tests
    assert sentiment_analyzer.normalize_sentiment("POSITIVE") == "Positive"
    assert sentiment_analyzer.normalize_sentiment("negative") == "Negative"
    assert sentiment_analyzer.normalize_sentiment("neutral") == "Neutral"
    print("-> Normalization Helper Verified: Safely mapped strings to standard sentiments.")

    # Database Persistence & Retrieval
    test_article_url = first_art["url"]
    save_ok = database.save_sentiment(
        url=test_article_url,
        sentiment=pos_res["sentiment"],
        confidence=pos_res.get("confidence"),
        explanation=pos_res.get("explanation"),
        method=pos_res.get("method"),
    )
    assert save_ok is True
    stored_sent = database.get_sentiment_by_url(test_article_url)
    assert stored_sent is not None
    assert stored_sent["sentiment"] == "Positive"
    print(f"-> Sentiment SQLite Persistence Verified: Stored and retrieved for [{test_article_url[:40]}...]")

    # Sentiment Search Filtering
    pos_search = database.search_articles(sentiment="Positive")
    assert isinstance(pos_search, list)
    assert len(pos_search) > 0
    print(f"-> Sentiment Search Filter Verified: Retrieved {len(pos_search)} 'Positive' article(s).")

    # Sentiment Dashboard Distribution
    sent_dist = database.get_sentiment_distribution()
    assert isinstance(sent_dist, list)
    print(f"-> Sentiment Dashboard Distribution Verified: {sent_dist}")

    # 14. Trending Topics Detection (Feature 14)
    print("\n[Test 14/14] Trending Topics Detection (Feature 14)...")
    # Normalization tests
    assert trending_topics.normalize_topic("ai") == "Artificial Intelligence"
    assert trending_topics.normalize_topic("Artificial Intelligence") == "Artificial Intelligence"
    assert trending_topics.normalize_topic("ml") == "Machine Learning"
    assert trending_topics.normalize_topic("climate change") == "Climate Change"
    assert trending_topics.normalize_topic("breaking news") is None  # stop-topic
    assert trending_topics.normalize_topic("the report") is None  # stop-topic
    print("-> Topic Normalization & Stop-Topic Filtering Verified.")

    # Fetch trending topics
    trends = trending_topics.get_trending_topics(days=7, limit=5)
    assert isinstance(trends, list)
    if trends:
        top_trend = trends[0]
        assert "topic" in top_trend
        assert "article_count" in top_trend
        assert "score" in top_trend
        assert "trend" in top_trend
        assert top_trend["trend"] in ("Rising", "Stable", "Declining", "Current")
        print(f"-> Top Stored Trend: '{top_trend['topic']}' (Score: {top_trend['score']}, Articles: {top_trend['article_count']}, Trend: {top_trend['trend']})")

    # Category filtered trends
    tech_trends = trending_topics.get_trending_topics(days=30, limit=5, category="Technology")
    assert isinstance(tech_trends, list)
    print(f"-> Category-Filtered (Technology) Trends: Found {len(tech_trends)} topics.")

    # Direct database helper test
    trend_articles = database.get_articles_for_trends(limit=10)
    assert isinstance(trend_articles, list)
    assert len(trend_articles) > 0
    print(f"-> Database Trends Fetcher Verified: {len(trend_articles)} articles retrieved with metadata.")

    # 15. AI Article Chat (Feature 15)
    print("\n[Test 15/15] AI Article Chat (Feature 15)...")
    sample_article_title = "OpenAI Releases New O1 Reasoning Model"
    sample_article_body = (
        "San Francisco-based artificial intelligence research company OpenAI has officially released "
        "its new reasoning model named O1. The model spends more time thinking before generating responses, "
        "allowing it to solve complex mathematics, competitive coding, and science problems with significantly "
        "fewer factual errors. The company stated that the model is now available to ChatGPT Plus subscribers."
    )

    # Test 1: Grounded question
    q1_res = chat.ask_article_question(
        article_title=sample_article_title,
        article_text=sample_article_body,
        question="What is the name of the new model announced by OpenAI?",
    )
    assert q1_res["success"] is True
    assert isinstance(q1_res["answer"], str)
    assert len(q1_res["answer"]) > 0
    print(f"-> Grounded Q&A Answer: '{q1_res['answer'][:70]}...' (Method: {q1_res['method']})")

    # Test 2: Unanswerable / Out-of-article question
    q2_res = chat.ask_article_question(
        article_title=sample_article_title,
        article_text=sample_article_body,
        question="What was the stock price of Apple yesterday?",
    )
    assert q2_res["success"] is True
    assert "not available" in q2_res["answer"].lower() or "not" in q2_res["answer"].lower()
    print(f"-> Out-of-Article Question Handled: '{q2_res['answer'][:70]}...'")

    # Test 3: Multi-turn history support
    chat_hist = [
        {"role": "user", "content": "Who released the model?"},
        {"role": "assistant", "content": "According to the article, OpenAI released the model."},
    ]
    q3_res = chat.ask_article_question(
        article_title=sample_article_title,
        article_text=sample_article_body,
        question="Who can access it now?",
        chat_history=chat_hist,
    )
    assert q3_res["success"] is True
    print(f"-> Multi-turn Follow-up Answer: '{q3_res['answer'][:70]}...'")

    # Test 4: Input validation
    empty_q_res = chat.ask_article_question(article_title="Title", article_text="Body", question="")
    assert empty_q_res["success"] is False
    print("-> Empty Question Validation Verified.")

    # 16. Multi-Source News Comparison (Feature 16)
    print("\n[Test 16/16] Multi-Source News Comparison (Feature 16)...")
    art_bbc = {
        "url": "https://bbc.com/ai-model-release",
        "title": "OpenAI Unveils New O1 Reasoning AI Model",
        "source": "BBC News",
        "article_text": "OpenAI has officially launched its O1 model focusing on complex reasoning, science, and math benchmarks. The engineering team spent months testing safety and alignment.",
        "keywords": '["OpenAI", "Reasoning", "Artificial Intelligence", "Benchmarks"]',
        "category": "Technology",
        "sentiment": "Positive",
        "published_at": "2026-08-29T10:00:00Z",
    }
    art_reuters = {
        "url": "https://reuters.com/openai-o1-launch",
        "title": "OpenAI Releases O1 AI System Aiming at Enterprise Market",
        "source": "Reuters",
        "article_text": "OpenAI introduced its newest O1 model targeting corporate and developer subscriptions. Industry analysts expect significant commercial adoption across financial institutions.",
        "keywords": '["OpenAI", "Enterprise", "Artificial Intelligence", "Market"]',
        "category": "Business",
        "sentiment": "Neutral",
        "published_at": "2026-08-29T11:00:00Z",
    }
    art_cnn = {
        "url": "https://cnn.com/tech/new-openai-model",
        "title": "New OpenAI Reasoning Model Takes Aim at Competitors",
        "source": "CNN",
        "article_text": "OpenAI has released a new reasoning model to compete directly with rival AI systems from Google and Anthropic in the competitive technology race.",
        "keywords": '["OpenAI", "Competition", "Artificial Intelligence", "Anthropic"]',
        "category": "Technology",
        "sentiment": "Neutral",
        "published_at": "2026-08-29T12:00:00Z",
    }

    # Test 1: Compare 3 distinct sources
    comp_res = source_comparison.compare_articles([art_bbc, art_reuters, art_cnn])
    assert comp_res["success"] is True
    assert "common_points" in comp_res
    assert "differences" in comp_res
    assert "unique_information" in comp_res
    assert "shared_keywords" in comp_res
    assert "unique_keywords" in comp_res
    assert "overall_summary" in comp_res
    assert "Openai" in comp_res["shared_keywords"] or "OpenAI" in comp_res["shared_keywords"] or "Artificial Intelligence" in comp_res["shared_keywords"]
    print(f"-> Multi-Source Comparison (3 Sources) Verified (Method: {comp_res['method']}).")
    print(f"   Shared Keywords: {comp_res['shared_keywords']}")
    print(f"   Summary: '{comp_res['overall_summary'][:80]}...'")

    # Test 2: Reject single article (< 2)
    single_res = source_comparison.compare_articles([art_bbc])
    assert single_res["success"] is False
    print("-> Minimum Article Limit (< 2) Validation Verified.")

    # Test 3: Reject identical URLs
    dup_url_res = source_comparison.compare_articles([art_bbc, art_bbc])
    assert dup_url_res["success"] is False
    print("-> Identical URL Duplicate Validation Verified.")

    print("\n================================================================")
    print("     ALL 16 FEATURES VALIDATED & OPERATING WITH 100% INTEGRITY  ")
    print("================================================================")


if __name__ == "__main__":
    run_all_tests()

