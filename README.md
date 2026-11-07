# AI-Powered YouTube Video Summarizer & Q&A Bot

An interactive **YouTube Video Summarizer and RAG Question-Answering Bot** (`ytbot.py`) built with **LangChain**, **FAISS**, **Local Sentence Embeddings**, **Google Gemini (`gemini-3.5-flash-lite`)**, and **Gradio**.

Designed from the `Youtube_transcripts.pdf` specification and optimized to run smoothly on local machines with **16GB RAM and 2GB GPU VRAM** without requiring an IBM Cloud / Watsonx account.

---

## Key Features

- **Video ID & Multi-Language Transcript Extraction**: Extracts the 11-character YouTube video ID from standard (`youtube.com/watch?v=...`) and shortened (`youtu.be/...`) URLs, prioritizing manual captions over auto-generated captions for **English (`en`)** and **Vietnamese (`vi`)** (with automatic fallback to any available subtitle language).
- **Single-Paragraph Video Summarization**: Summarizes dense video transcripts into a single concise paragraph focused on spoken content while omitting raw timestamp markers.
- **RAG Question Answering (FAISS + Local CPU Embeddings)**: Splits transcripts into overlapping chunks (`chunk_size=200`, `chunk_overlap=20`), indexes them in-memory using `faiss-cpu` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on CPU (`~250MB RAM`, `0MB GPU VRAM`), retrieves the top `k=7` most relevant segments, and answers questions in the same language as the user's prompt (English or Vietnamese).
- **Smart Session Caching & Status Feedback**: Reuses the fetched transcript and FAISS index across follow-up questions for the same video and automatically refreshes when you paste a new YouTube URL.

---

## Quickstart (Windows PowerShell)

### 1. Create and Activate Virtual Environment

```powershell
python -m venv my_env
.\my_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and set your free Google Gemini API key (from [Google AI Studio](https://aistudio.google.com/)):

```powershell
Copy-Item .env.example .env
```

```ini
GOOGLE_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
PORT=7860
```

### 3. Run Unit & Contract Tests

```powershell
pytest -v
```

### 4. Launch the Gradio Web App

```powershell
python ytbot.py
```

Open **`http://localhost:7860`** in your browser.

---

## Run with Docker

Ensure you have configured your `.env` file with a valid `GOOGLE_API_KEY`.

### Option 1: Using Docker Compose (Recommended)

```powershell
# Build and start the container in the background
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Stop the container
docker compose down
```

### Option 2: Using Docker CLI

```powershell
# Build the Docker image
docker build -t youtube-transcripts-bot .

# Run the container using your .env file
docker run -d --name youtube-transcripts-bot -p 7860:7860 --env-file .env youtube-transcripts-bot
```

> **Tip**: If you want to run the FastAPI interface (`app.py`) instead of Gradio (`ytbot.py`), override the startup command:
> ```powershell
> docker run -d --name youtube-transcripts-api -p 7860:7860 --env-file .env youtube-transcripts-bot uvicorn app:app --host 0.0.0.0 --port 7860
> ```

---

## Example Test Flow

1. **YouTube Video URL**: `https://www.youtube.com/watch?v=T-D1OfcDW1M`
2. Click **Summarize Video** to view a single-paragraph summary of the video.
3. In **Ask a Question About the Video**, try:
   - `How does one reduce hallucinations?`
   - `Which problems does RAG solve, according to the video?`
   - `RAG giải quyết những vấn đề gì theo nội dung video?`
4. Click **Ask a Question** to receive a context-grounded answer.
