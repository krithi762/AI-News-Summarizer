# Deployment Guide — AI News Summarizer

This guide describes how to deploy the **AI News Summarizer** application to local environments, cloud containers, and hosted platforms such as **Streamlit Community Cloud** or **Hugging Face Spaces**.

---

## 📋 Pre-Deployment Checklist

Before deploying, ensure you have:
1. A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
2. An optional **NewsAPI Key** from [NewsAPI.org](https://newsapi.org/) (the app automatically falls back to live RSS feeds if omitted).
3. Verified the codebase by running `python test_suite.py`.

---

## 1. Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the fastest and most direct way to host this application.

### Steps:
1. **Push your repository to GitHub**:
   Ensure all project files are committed to your GitHub repository:
   ```bash
   git add .
   git commit -m "Complete AI News Summarizer deployment ready"
   git push origin main
   ```
2. **Log into Streamlit Community Cloud**:
   Visit [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. **Create a New App**:
   * **Repository**: Select your GitHub repository.
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. **Configure Secrets (Environment Variables)**:
   In the Streamlit Cloud deployment dashboard, navigate to **Settings** $\rightarrow$ **Secrets**, and paste your keys in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   NEWS_API_KEY = "your_actual_news_api_key"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
   *(Note: The `python-dotenv` and `os.getenv` system in `summarizer.py` will read these secrets seamlessly).*
5. **Deploy**:
   Click **Deploy!**. Streamlit will install dependencies from `requirements.txt` and launch the app.

---

## 2. Deploying with Docker

You can containerize the application for deployment to AWS ECS, Google Cloud Run, Azure Container Apps, or any Docker host.

### Sample `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/cells/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:
```bash
docker build -t ai-news-summarizer .
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key" ai-news-summarizer
```

---

## 3. Deploying to Linux Virtual Machine (Ubuntu / Debian / systemd)

For running on an AWS EC2, DigitalOcean Droplet, or Linode instance:

### 1. Setup Environment:
```bash
sudo apt update && sudo apt install python3-pip python3-venv git -y
git clone <your-repo-url> /opt/ai-news-summarizer
cd /opt/ai-news-summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # Enter your API keys
```

### 2. Configure systemd Service (`/etc/systemd/system/ainews.service`):
```ini
[Unit]
Description=AI News Summarizer Streamlit Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/ai-news-summarizer
ExecStart=/opt/ai-news-summarizer/.venv/bin/streamlit run app.py --server.port=8501 --server.headless=true
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3. Start Service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ainews
sudo systemctl start ainews
```

---

## ⚠️ Important Note on SQLite Persistence in Hosted Cloud Platforms

* The application stores articles and AI-generated insights in a local SQLite file located at `data/news.db`.
* **Ephemeral Storage Warning**: Serverless platforms (such as basic Heroku or standard Cloud Run without mounted volumes) reset ephemeral disk storage upon instance restarts.
* **Persistent Volumes**: If deploying on Kubernetes or Docker, mount a persistent volume to `/app/data` to retain news articles and historical trends permanently across container updates.

---

## 🛠️ Post-Deployment Verification

Once deployed:
1. Verify the homepage loads without runtime errors.
2. Navigate to **📰 Latest News** and click **Fetch News Articles**.
3. Select an article, verify text extraction, and generate an AI summary.
4. Verify that the **📊 Dashboard** and **🔥 Trending Topics** charts render properly.
