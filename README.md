# AI News Summarizer

An enterprise-grade, AI-powered news analysis and intelligence platform. It collects news articles across diverse publishers via NewsAPI and live RSS feeds, extracts clean article bodies, generates executive summaries and key bullet points using Google Gemini AI, classifies news categories, extracts contextual keywords, stores full metadata in a local SQLite database, identifies duplicate and syndicated coverage via TF-IDF cosine similarity, computes recency-weighted trending topics, analyzes journalistic sentiment, facilitates grounded AI article Q&A, and compares multi-source coverage across publishers.

---

## 🌟 Key Features

| # | Feature | Description |
| :--- | :--- | :--- |
| **1** | **Project Setup & Architecture** | Multi-tiered modular architecture with strict decoupling between data collection, AI inference, and UI rendering. |
| **2** | **News Collection (NewsAPI & RSS)** | Automated fetching from NewsAPI with an integrated live RSS fallback mechanism across 7 major news categories. |
| **3** | **Article Content Extraction** | High-precision article scraping with `newspaper4k` and `BeautifulSoup`, extracting titles, authors, body texts, and timestamps. |
| **4** | **AI News Summarization** | Executive summary generation powered by Google Gemini (with configurable Short, Medium, and Detailed modes) and local fallback. |
| **5** | **AI Key Point Extraction** | Contextual extraction of 3, 5, or 7 structured bullet points capturing core takeaways. |
| **6** | **AI News Categorization** | Multi-class classification mapping news into *Technology*, *Business*, *Sports*, *Science*, *Health*, *World*, or *Entertainment*. |
| **7** | **AI Keyword Extraction** | Extraction of 5 to 10 prominent topical tags and named entities stored with JSON serialization. |
| **8** | **SQLite Database Persistence** | Thread-safe, non-destructive SQLite storage (`data/news.db`) with parameterized queries and schema migrations. |
| **9** | **Professional Streamlit UI/UX** | Responsive interface with dark/light harmony, news cards, status indicators, and collapsible insights. |
| **10** | **Duplicate & Related News Detection** | 100% local, scikit-learn TF-IDF vectorization and cosine similarity to detect syndicated stories and related developments. |
| **11** | **Search & Advanced Filtering** | Full-text search across headlines, keywords, and text with category, source, date, sentiment filters, and sorting options. |
| **12** | **News Analytics Dashboard** | Real-time analytics dashboard with Plotly charts for publication trends, categories, sources, keywords, and AI metrics. |
| **13** | **AI Sentiment Analysis** | Journalistic tone analysis classifying articles into *Positive*, *Neutral*, or *Negative* with confidence scores and explanations. |
| **14** | **Trending Topics Detection** | Dynamic recency-weighted frequency scoring identifying trending topics in stored news with trend direction badges (🔥 Rising, → Stable, ↓ Declining, ⚡ Current). |
| **15** | **AI Article Chat (Grounded Q&A)** | Conversational Q&A grounded strictly in the active article text with prompt injection defense and out-of-scope rejection. |
| **16** | **Multi-Source News Comparison** | Cross-publisher comparison of 2 to 5 articles covering the same event, analyzing common facts, distinct framing angles, unique details, and sentiment. |

---

## 🛠️ Technology Stack

* **Frontend & Web Framework**: [Streamlit](https://streamlit.io/)
* **Generative AI & LLMs**: [Google GenAI SDK (`google-genai`)](https://pypi.org/project/google-genai/) (Gemini 2.5 Flash)
* **News Feeds & Web Ingestion**: [NewsAPI](https://newsapi.org/), [feedparser](https://pypi.org/project/feedparser/), [newspaper4k](https://pypi.org/project/newspaper4k/), [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/)
* **Machine Learning & NLP**: [scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorization, Cosine Similarity)
* **Data Visualization**: [Plotly](https://plotly.com/)
* **Database & Storage**: SQLite 3 (`sqlite3`) with parameterized queries
* **Environment Configuration**: [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 📂 Project Structure

```text
AI-News-Summarizer/
│
├── app.py                  → Main Streamlit multi-tab user interface
├── news_fetcher.py         → NewsAPI client & live RSS fallback collectors
├── article_extractor.py    → HTML article scraper & metadata extractor
├── summarizer.py           → Google Gemini executive summarization & key points
├── classifier.py           → AI news categorization & keyword fallback engine
├── keyword_extractor.py    → AI named-entity & keyword tag extraction
├── duplicate_detector.py   → TF-IDF cosine similarity for duplicate/related detection
├── sentiment_analyzer.py   → Journalistic sentiment & tone analysis engine
├── trending_topics.py      → Recency-weighted trending topic detection engine
├── chat.py                 → Article-grounded conversational Q&A assistant
├── source_comparison.py    → Multi-source cross-publisher comparison engine
├── database.py             → SQLite connection manager, schema migrations, and queries
├── theme.py                → Centralized Light/Dark theme configuration & Plotly styling
├── utils.py                → Date formatters, text helpers, and CSS styling definitions
├── test_suite.py           → Comprehensive 16-feature automated validation suite
│
├── data/
│   └── news.db             → Local SQLite database (created automatically)
│
├── .streamlit/
│   └── config.toml         → Streamlit server and theme configuration
│
├── requirements.txt        → Pinned external Python dependencies
├── .env.example            → Template for required environment variables
├── .gitignore              → Secret and build file exclusions
├── LICENSE                 → MIT License
├── TEST_REPORT.md          → Comprehensive test results and validation log
├── DEPLOYMENT.md           → Step-by-step production deployment guide
└── README.md               → Project overview and documentation
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "AI News Summarizer"
```

### 2. Create and Activate Virtual Environment
* **Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and configure your API keys:
```ini
# News API Key (https://newsapi.org/) - optional, auto-falls back to RSS feeds
NEWS_API_KEY=your_news_api_key_here

# Google Gemini API Key (https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: customize model (defaults to gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-flash
```

---

## 🎨 Theme Support (Light / Dark Mode)

The application features a built-in, dynamic **Theme Switcher** located in the top-right header:
* 🌞 **Light Mode** (Default): Crisp white surfaces, deep slate text, subtle light borders, and vibrant accents.
* 🌙 **Dark Mode**: High-contrast slate-900 background, slate-800 cards, and luminescent data tags.

> **Note**: Theme selection is a purely client-side session preference. Switching themes dynamically updates all Plotly charts (`plotly_white` vs `plotly_dark`) and custom CSS cards **without triggering any AI/API calls or resetting active article/search filters**.

---

## 🖥️ Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 Running the Automated Test Suite

Execute the full 16-feature end-to-end verification suite:
```bash
python test_suite.py
```

---

## 🔧 Troubleshooting

### 1. Gemini API Key Not Configured / Quota Exceeded
* **Behavior**: The application displays informational fallback badges (`Local Fallback` / `Keyword Fallback`) and continues to operate without crashing.
* **Resolution**: Ensure a valid `GEMINI_API_KEY` is placed in `.env` or Streamlit Secrets.

### 2. NewsAPI Rate Limits
* **Behavior**: If NewsAPI returns HTTP 429 (rate limit) or fails, the application automatically switches to live RSS feeds from BBC, Reuters, TechCrunch, Wired, and ESPN.

### 3. Article Extraction Blocked by Website
* **Behavior**: Some publishers block automated scrapers with Cloudflare protection.
* **Resolution**: The system displays a warning message with direct links to the publisher's site and leverages the article's available RSS description for analysis.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
