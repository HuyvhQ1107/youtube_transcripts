"""
Unit and contract test suite for ytbot.py covering:
- Phase 2 Foundational: get_video_id, get_transcript (en/vi/fallback, manual vs auto), process, setup_credentials, define_parameters
- Phase 3 User Story 1 (P1): create_summary_prompt, create_summary_chain, summarize_video
- Phase 4 User Story 2 (P2): chunk_transcript, create_faiss_index, perform_similarity_search, retrieve, create_qa_prompt_template, create_qa_chain, generate_answer, answer_question
- Phase 5 User Story 3 (P3): session cache reuse across follow-up questions, cache invalidation on URL change, and get_transcript_status feedback
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ytbot


class DummyEmbeddings:
    """Deterministic lightweight in-memory embedding model for fast unit testing with FAISS."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            lower = t.lower()
            vectors.append(
                [
                    float(len(t) % 100) / 100.0 + 0.01,
                    1.0 if "hallucination" in lower or "rag" in lower else 0.1,
                    1.0 if "quantum" in lower else 0.1,
                    1.0 if "strangers" in lower or "love" in lower else 0.1,
                ]
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)


class DummyLLM:
    """Fake LLM that returns deterministic responses and records formatted prompts."""

    def __init__(self, response_text: str = "Mock LLM output"):
        self.response_text = response_text
        self.last_prompt: str | None = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return SimpleNamespace(content=self.response_text)


@pytest.fixture(autouse=True)
def reset_ytbot_session_state(monkeypatch):
    """Reset module-level session cache before each test."""
    ytbot.current_video_id = None
    ytbot.fetched_transcript = None
    ytbot.processed_transcript = ""
    ytbot.faiss_index = None
    ytbot.transcript_status_msg = "Ready."
    ytbot._cached_embedding_model = None
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-api-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# =====================================================================
# Phase 2: Foundational Tests (T010)
# =====================================================================
def test_get_video_id_standard_and_short_urls():
    assert ytbot.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ytbot.get_video_id("https://www.youtube.com/watch?v=T-D1OfcDW1M&t=15s") == "T-D1OfcDW1M"
    assert ytbot.get_video_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert ytbot.get_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_get_video_id_invalid_urls():
    assert ytbot.get_video_id("") is None
    assert ytbot.get_video_id(None) is None  # type: ignore[arg-type]
    assert ytbot.get_video_id("https://www.google.com") is None
    assert ytbot.get_video_id("https://www.youtube.com/watch?v=short") is None


def test_get_transcript_prioritizes_manual_over_generated(monkeypatch):
    auto_en = SimpleNamespace(
        language_code="en",
        is_generated=True,
        fetch=lambda: [{"text": "Auto EN", "start": 0.0, "duration": 2.0}],
    )
    manual_en = SimpleNamespace(
        language_code="en",
        is_generated=False,
        fetch=lambda: [{"text": "Manual EN", "start": 0.0, "duration": 2.0}],
    )

    mock_api_instance = MagicMock()
    mock_api_instance.list.return_value = [auto_en, manual_en]
    monkeypatch.setattr(ytbot, "YouTubeTranscriptApi", lambda: mock_api_instance)

    result = ytbot.get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result == [{"text": "Manual EN", "start": 0.0, "duration": 2.0}]
    assert "type=manual" in ytbot.get_transcript_status()


def test_get_transcript_supports_vietnamese_and_fallback(monkeypatch):
    vi_track = SimpleNamespace(
        language_code="vi",
        is_generated=True,
        fetch=lambda: [{"text": "Xin chào các bạn", "start": 1.0, "duration": 3.0}],
    )

    mock_api_instance = MagicMock()
    mock_api_instance.list.return_value = [vi_track]
    monkeypatch.setattr(ytbot, "YouTubeTranscriptApi", lambda: mock_api_instance)

    result = ytbot.get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result == [{"text": "Xin chào các bạn", "start": 1.0, "duration": 3.0}]
    assert "language=vi" in ytbot.get_transcript_status()


def test_process_formats_dict_and_object_entries_and_skips_malformed():
    sample_transcript = [
        {"text": "We're no strangers to love.", "start": 0.0, "duration": 3.5},
        SimpleNamespace(text="You know the rules and so do I.", start=3.5, duration=4.0),
        {"invalid_entry": "missing keys"},
        SimpleNamespace(text="A full commitment's what I'm thinking of.", start=7.5, duration=4.0),
    ]
    formatted = ytbot.process(sample_transcript)
    expected = (
        "Text: We're no strangers to love. Start: 0.0\n"
        "Text: You know the rules and so do I. Start: 3.5\n"
        "Text: A full commitment's what I'm thinking of. Start: 7.5\n"
    )
    assert formatted == expected
    assert ytbot.process(None) == ""
    assert ytbot.process([]) == ""


def test_setup_credentials_and_define_parameters():
    model_id, creds, client, project_id = ytbot.setup_credentials()
    assert model_id == "gemini-3.5-flash-lite"
    assert creds["api_key"] == "test-google-api-key-123"
    params = ytbot.define_parameters()
    assert params["temperature"] == 0.0
    assert params["max_output_tokens"] == 900


# =====================================================================
# Phase 3: User Story 1 Tests - Summarize a YouTube Video (T011)
# =====================================================================
def test_create_summary_prompt_and_chain():
    prompt = ytbot.create_summary_prompt()
    assert "transcript" in prompt.input_variables
    formatted = prompt.format(transcript="Text: Hello world Start: 0.0")
    assert "Summarize the transcript in a single concise paragraph" in formatted
    assert "Ignore any timestamps" in formatted

    dummy_llm = DummyLLM("This video introduces greeting conventions in a single concise paragraph.")
    chain = ytbot.create_summary_chain(dummy_llm, prompt)
    summary = chain.run({"transcript": "Text: Hello world Start: 0.0"})
    assert summary == "This video introduces greeting conventions in a single concise paragraph."
    assert "Start:" not in summary


def test_summarize_video_validation_and_success(monkeypatch):
    # 1. Empty or invalid URL
    assert ytbot.summarize_video("") == "Please provide a valid YouTube URL."
    assert ytbot.summarize_video("https://example.com") == "Please provide a valid YouTube URL."

    # 2. Valid URL but no transcript available
    monkeypatch.setattr(ytbot, "get_transcript", lambda url: None)
    assert (
        ytbot.summarize_video("https://www.youtube.com/watch?v=T-D1OfcDW1M")
        == "No transcript available. Please fetch the transcript first."
    )

    # 3. Valid URL with transcript -> generates single-paragraph summary
    monkeypatch.setattr(
        ytbot,
        "get_transcript",
        lambda url: [{"text": "RAG reduces hallucinations in LLMs.", "start": 0.0, "duration": 5.0}],
    )
    monkeypatch.setattr(
        ytbot,
        "initialize_watsonx_llm",
        lambda *args, **kwargs: DummyLLM(
            "Retrieval-Augmented Generation (RAG) grounds large language models on reliable external sources to reduce hallucinations."
        ),
    )

    summary = ytbot.summarize_video("https://www.youtube.com/watch?v=T-D1OfcDW1M")
    assert "Retrieval-Augmented Generation" in summary
    assert "Start:" not in summary
    assert ytbot.current_video_id == "T-D1OfcDW1M"


# =====================================================================
# Phase 4: User Story 2 Tests - Ask Questions About Video Content (T016)
# =====================================================================
def test_chunk_transcript_splits_with_overlap():
    long_text = (
        "Text: We're no strangers to love. Start: 0.0\n"
        "Text: You know the rules and so do I. Start: 3.5\n"
        "Text: A full commitment's what I'm thinking of. Start: 7.5\n"
        "Text: You wouldn't get this from any other guy. Start: 11.5\n"
        "Text: I just wanna tell you how I'm feeling. Start: 15.0\n"
    )
    chunks = ytbot.chunk_transcript(long_text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) <= 120 for c in chunks)


def test_faiss_index_and_retrieve_and_generate_answer():
    chunks = [
        "Text: RAG reduces hallucinations by grounding LLMs in external knowledge. Start: 10.0",
        "Text: Quantum physics studies subatomic particles and wave-particle duality. Start: 50.0",
    ]
    emb = DummyEmbeddings()
    index = ytbot.create_faiss_index(chunks, emb)

    sim_results = ytbot.perform_similarity_search(index, "How to reduce hallucinations?", k=1)
    assert len(sim_results) == 1
    assert "hallucinations" in sim_results[0].page_content

    retrieved = ytbot.retrieve("hallucinations", index, k=2)
    assert len(retrieved) == 2

    qa_prompt = ytbot.create_qa_prompt_template()
    assert set(qa_prompt.input_variables) == {"context", "question"}

    dummy_llm = DummyLLM("One reduces hallucinations by using RAG to ground responses in retrieved context.")
    qa_chain = ytbot.create_qa_chain(dummy_llm, qa_prompt)
    answer = ytbot.generate_answer("How does one reduce hallucinations?", index, qa_chain, k=1)
    assert "RAG" in answer
    assert "How does one reduce hallucinations?" in (dummy_llm.last_prompt or "")


def test_answer_question_auto_fetches_and_validates(monkeypatch):
    # Missing URL when no transcript cached
    assert ytbot.answer_question("", "What is RAG?") == "Please provide a valid YouTube URL."

    # Valid URL, auto-fetch transcript, empty question
    monkeypatch.setattr(
        ytbot,
        "get_transcript",
        lambda url: [{"text": "RAG solves hallucination issues.", "start": 0.0, "duration": 4.0}],
    )
    assert (
        ytbot.answer_question("https://www.youtube.com/watch?v=T-D1OfcDW1M", "")
        == "Please provide a valid question and ensure the transcript has been fetched."
    )

    # Full valid question answering flow
    monkeypatch.setattr(ytbot, "setup_embedding_model", lambda *a, **kw: DummyEmbeddings())
    monkeypatch.setattr(
        ytbot,
        "initialize_watsonx_llm",
        lambda *a, **kw: DummyLLM("Theo video, RAG giúp giảm ảo giác (hallucinations) cho LLM."),
    )
    ans = ytbot.answer_question(
        "https://www.youtube.com/watch?v=T-D1OfcDW1M",
        "RAG giải quyết vấn đề gì theo video?",
    )
    assert "RAG" in ans


# =====================================================================
# Phase 5: User Story 3 Tests - Cache Reuse & URL Switch (T023)
# =====================================================================
def test_session_cache_reused_for_same_video_and_invalidated_on_url_change(monkeypatch):
    fetch_calls = []

    def fake_get_transcript(url):
        vid = ytbot.get_video_id(url)
        fetch_calls.append(vid)
        return [{"text": f"Content for video {vid}", "start": 0.0, "duration": 5.0}]

    monkeypatch.setattr(ytbot, "get_transcript", fake_get_transcript)
    monkeypatch.setattr(ytbot, "setup_embedding_model", lambda *a, **kw: DummyEmbeddings())
    monkeypatch.setattr(
        ytbot,
        "initialize_watsonx_llm",
        lambda *a, **kw: DummyLLM("Grounded answer from cached or fresh index."),
    )

    url_a = "https://www.youtube.com/watch?v=T-D1OfcDW1M"
    url_b = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Question 1 on Video A -> fetches Video A once
    ytbot.answer_question(url_a, "First question?")
    assert fetch_calls == ["T-D1OfcDW1M"]
    assert ytbot.current_video_id == "T-D1OfcDW1M"

    # Question 2 on Video A -> reuses cached transcript & FAISS index without re-fetching
    ytbot.answer_question(url_a, "Second follow-up question?")
    assert fetch_calls == ["T-D1OfcDW1M"]
    assert "Reusing cached transcript" in ytbot.get_transcript_status()

    # Question 3 on Video B -> detects URL change and fetches Video B automatically
    ytbot.answer_question(url_b, "Question about second video?")
    assert fetch_calls == ["T-D1OfcDW1M", "dQw4w9WgXcQ"]
    assert ytbot.current_video_id == "dQw4w9WgXcQ"
    assert "dQw4w9WgXcQ" in ytbot.processed_transcript


def test_youtube_proxy_configuration_from_env(monkeypatch):
    # 1. Default (no proxy env vars) -> None
    monkeypatch.delenv("WEBSHARE_PROXY_USERNAME", raising=False)
    monkeypatch.delenv("WEBSHARE_PROXY_PASSWORD", raising=False)
    monkeypatch.delenv("YOUTUBE_PROXY_URL", raising=False)
    monkeypatch.delenv("YOUTUBE_HTTP_PROXY", raising=False)
    monkeypatch.delenv("YOUTUBE_HTTPS_PROXY", raising=False)
    assert ytbot._build_youtube_proxy_config() is None

    # 2. Generic Proxy Config
    monkeypatch.setenv("YOUTUBE_PROXY_URL", "socks5h://127.0.0.1:1080")
    generic_cfg = ytbot._build_youtube_proxy_config()
    assert generic_cfg is not None
    assert generic_cfg.to_requests_dict() == {
        "http": "socks5h://127.0.0.1:1080",
        "https": "socks5h://127.0.0.1:1080",
    }

    # 3. Webshare Proxy Config takes priority when credentials are set
    monkeypatch.setenv("WEBSHARE_PROXY_USERNAME", "myuser")
    monkeypatch.setenv("WEBSHARE_PROXY_PASSWORD", "mypass")
    monkeypatch.setenv("WEBSHARE_PROXY_LOCATIONS", "vn, sg")
    ws_cfg = ytbot._build_youtube_proxy_config()
    assert ws_cfg is not None
    reqs_dict = ws_cfg.to_requests_dict()
    assert "myuser-VN-SG-rotate:mypass@p.webshare.io:80" in reqs_dict["http"]

