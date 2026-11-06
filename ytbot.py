"""
AI-Powered YouTube Video Summarizer & Q&A Bot (ytbot.py)

Implements the complete pipeline from Youtube_transcripts.pdf:
- YouTube video ID extraction (standard watch URLs & shortened youtu.be URLs)
- Multi-language transcript retrieval (prioritizing 'en' and 'vi', manual over auto-generated, with fallback)
- Transcript formatting ("Text: ... Start: ...")
- RecursiveCharacterTextSplitter chunking (chunk_size=200, chunk_overlap=20)
- Local CPU/GPU sentence embeddings with FAISS similarity search (top-k retrieval)
- Google Gemini LLM (defaulting to gemini-3.5-flash-lite) for summarization and Q&A
- Session caching per video ID & Gradio Blocks web UI on port 7860
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:  # pragma: no cover
    from langchain.prompts import PromptTemplate

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS

# Load environment variables from .env if present
load_dotenv()

# =====================================================================
# Session Cache State (Module-level state matching PDF globals + cache)
# =====================================================================
current_video_id: Optional[str] = None
fetched_transcript: Optional[List[Any]] = None
processed_transcript: str = ""
faiss_index: Optional[Any] = None
transcript_status_msg: str = "Ready. Enter a YouTube Video URL to begin."
_cached_embedding_model: Optional[Any] = None


def get_transcript_status() -> str:
    """Return the latest human-readable transcript status message for the Gradio UI."""
    return transcript_status_msg


# =====================================================================
# 1. Extracting YouTube Transcripts
# =====================================================================
def get_video_id(url: str) -> Optional[str]:
    """
    Extract the unique 11-character YouTube video ID from a provided URL.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    patterns = [
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/watch\?(?:.*&)?v=([a-zA-Z0-9_-]{11})(?:&|$|#|\?)",
        r"(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})(?:\?|&|$|#)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/(?:embed|shorts)\/([a-zA-Z0-9_-]{11})(?:\?|&|$|#)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript(url: str, preferred_languages: Optional[List[str]] = None) -> Optional[List[Any]]:
    """
    Fetch the transcript for a given YouTube URL.

    Prioritizes preferred languages ('en', 'vi' by default), selecting manually
    created transcripts over auto-generated ones when both exist, and falling back
    to the first available transcript track if neither 'en' nor 'vi' is present.
    """
    global transcript_status_msg

    video_id = get_video_id(url)
    if not video_id:
        transcript_status_msg = "Invalid YouTube URL. Could not extract 11-character video ID."
        return None

    if preferred_languages is None:
        preferred_languages = ["en", "vi"]

    try:
        ytt_api = YouTubeTranscriptApi()
        if hasattr(ytt_api, "list"):
            transcript_list = list(ytt_api.list(video_id))
        else:  # Fallback for older youtube-transcript-api versions
            transcript_list = list(YouTubeTranscriptApi.list_transcripts(video_id))
    except Exception as exc:
        transcript_status_msg = f"Failed to fetch transcripts for video '{video_id}': {exc}"
        return None

    if not transcript_list:
        transcript_status_msg = f"No transcripts available for video '{video_id}'."
        return None

    selected_track = None

    # Search preferred languages in order ('en', then 'vi')
    for lang in preferred_languages:
        manual_track = None
        generated_track = None
        for t in transcript_list:
            lang_code = getattr(t, "language_code", "")
            # Match exact code or regional variant (e.g., 'en-US', 'vi-VN')
            if lang_code == lang or lang_code.startswith(f"{lang}-"):
                if getattr(t, "is_generated", False):
                    if generated_track is None:
                        generated_track = t
                else:
                    manual_track = t
                    break  # Prioritize manually created transcript immediately
        if manual_track is not None:
            selected_track = manual_track
            break
        if generated_track is not None:
            selected_track = generated_track
            break

    # Fallback to first available transcript track if neither 'en' nor 'vi' is present
    if selected_track is None:
        manual_fallback = next(
            (t for t in transcript_list if not getattr(t, "is_generated", False)),
            None,
        )
        selected_track = manual_fallback if manual_fallback is not None else transcript_list[0]

    try:
        fetched = selected_track.fetch()
        track_lang = getattr(selected_track, "language_code", "unknown")
        track_type = "auto-generated" if getattr(selected_track, "is_generated", False) else "manual"
        transcript_status_msg = (
            f"Transcript fetched successfully (video_id={video_id}, language={track_lang}, type={track_type})."
        )
        return fetched if fetched else None
    except Exception as exc:
        transcript_status_msg = f"Error fetching transcript content: {exc}"
        return None


def process(transcript: Optional[List[Any]]) -> str:
    """
    Transform the fetched transcript into a readable multi-line string format:
    'Text: {text} Start: {start}\\n'

    Supports both object attribute access (i.text, i.start) and dictionary access
    (i['text'], i['start']), gracefully skipping malformed entries.
    """
    if not transcript:
        return ""

    txt = ""
    for entry in transcript:
        try:
            if isinstance(entry, dict):
                text_val = entry["text"]
                start_val = entry["start"]
            else:
                text_val = getattr(entry, "text") if hasattr(entry, "text") else entry["text"]
                start_val = getattr(entry, "start") if hasattr(entry, "start") else entry["start"]

            if text_val is None or start_val is None:
                continue
            txt += f"Text: {text_val} Start: {start_val}\n"
        except (KeyError, AttributeError, TypeError):
            pass

    return txt


def chunk_transcript(
    processed_transcript_str: str,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> List[str]:
    """
    Split a processed transcript string into overlapping chunks using
    RecursiveCharacterTextSplitter (default chunk_size=200, chunk_overlap=20).
    """
    if not processed_transcript_str:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(processed_transcript_str)
    return chunks


# =====================================================================
# 2. Setting Up Google Gemini LLM & Local CPU Embeddings
# =====================================================================
def setup_credentials() -> Tuple[str, dict, Any, str]:
    """
    Set up credentials and model configuration for Google Gemini API.

    Maintains the 4-tuple return signature `(model_id, credentials, client, project_id)`
    from Youtube_transcripts.pdf so all caller code works seamlessly without IBM Watsonx.
    """
    model_id = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    credentials = {
        "api_key": api_key,
        "url": "https://generativelanguage.googleapis.com",
    }
    client = None
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "local-gemini-project")
    return model_id, credentials, client, project_id


def define_parameters() -> dict:
    """
    Define generation parameters for the language model (greedy/deterministic decoding,
    max output tokens = 900).
    """
    return {
        "temperature": 0.0,
        "max_output_tokens": 900,
    }


def initialize_watsonx_llm(
    model_id: str,
    credentials: dict,
    project_id: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> Any:
    """
    Instantiate the Google Gemini LLM (replacing WatsonxLLM from the PDF while preserving
    the exact function signature).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    params = parameters or define_parameters()
    api_key = ""
    if isinstance(credentials, dict):
        api_key = credentials.get("api_key", "") or os.getenv("GOOGLE_API_KEY", "")
    else:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    if not api_key or api_key == "your_google_gemini_api_key_here":
        raise ValueError(
            "Missing GOOGLE_API_KEY. Please set GOOGLE_API_KEY in your .env file or environment variables."
        )

    return ChatGoogleGenerativeAI(
        model=model_id,
        google_api_key=api_key,
        temperature=params.get("temperature", 0.0),
        max_output_tokens=params.get("max_output_tokens", 900),
    )


# Alias for clarity
initialize_llm = initialize_watsonx_llm


def setup_embedding_model(
    credentials: Optional[dict] = None,
    project_id: Optional[str] = None,
) -> Any:
    """
    Initialize and cache the embedding model.
    - On local machines (when sentence-transformers & langchain-huggingface are installed
      and VERCEL is not set): defaults to 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
      on CPU (~250-400MB RAM, 0MB GPU VRAM).
    - On Vercel Serverless (or when EMBEDDING_PROVIDER=gemini / local packages are absent):
      automatically uses GoogleGenerativeAIEmbeddings ('models/gemini-embedding-001') with
      the same GOOGLE_API_KEY so the Serverless bundle stays well under Vercel's 250MB limit.
    """
    global _cached_embedding_model
    if _cached_embedding_model is not None:
        return _cached_embedding_model

    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    is_vercel = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENV"))

    if provider != "gemini" and not is_vercel:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            embedding_model_name = os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip()
            device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"

            _cached_embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
            return _cached_embedding_model
        except ImportError:
            pass

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    api_key = ""
    if isinstance(credentials, dict):
        api_key = credentials.get("api_key", "") or os.getenv("GOOGLE_API_KEY", "")
    else:
        api_key = os.getenv("GOOGLE_API_KEY", "")

    gemini_emb_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
    _cached_embedding_model = GoogleGenerativeAIEmbeddings(
        model=gemini_emb_model,
        google_api_key=api_key,
    )
    return _cached_embedding_model


def create_faiss_index(chunks: List[str], embedding_model: Any) -> FAISS:
    """
    Create a FAISS index from text chunks using the specified embedding model.
    """
    return FAISS.from_texts(chunks, embedding_model)


def perform_similarity_search(faiss_index_obj: FAISS, query: str, k: int = 3) -> List[Any]:
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    """
    results = faiss_index_obj.similarity_search(query, k=k)
    return results


# =====================================================================
# 3. Summarization & Q&A Prompt Templates and Chains
# =====================================================================
class RunnableLLMChainAdapter:
    """
    Lightweight adapter around `prompt | llm` that provides `.run()`, `.predict()`,
    and `.invoke()` compatibility matching LangChain's classic `LLMChain` interface
    used in Youtube_transcripts.pdf without deprecation warnings.
    """

    def __init__(self, llm: Any, prompt: PromptTemplate, verbose: bool = True) -> None:
        self.llm = llm
        self.prompt = prompt
        self.verbose = verbose

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return "".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            return str(content).strip()
        if isinstance(response, dict) and "text" in response:
            return str(response["text"]).strip()
        return str(response).strip()

    def invoke(self, inputs: dict) -> str:
        formatted_prompt = self.prompt.format(**inputs)
        if hasattr(self.llm, "invoke"):
            raw = self.llm.invoke(formatted_prompt)
        elif callable(self.llm):
            raw = self.llm(formatted_prompt)
        else:  # pragma: no cover
            raw = str(formatted_prompt)
        return self._extract_text(raw)

    def run(self, inputs: Any, **kwargs: Any) -> str:
        if isinstance(inputs, dict):
            merged = {**inputs, **kwargs}
        else:
            input_var = self.prompt.input_variables[0]
            merged = {input_var: inputs, **kwargs}
        return self.invoke(merged)

    def predict(self, **kwargs: Any) -> str:
        return self.invoke(kwargs)


def create_summary_prompt() -> PromptTemplate:
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    """
    template = """You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph.
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.

Note: In the transcript, "Text" refers to the spoken words in the video, and "Start" indicates the timestamp when that part begins in the video.

Please summarize the following YouTube video transcript:

{transcript}
"""
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template,
    )
    return prompt


def create_summary_chain(
    llm: Any,
    prompt: PromptTemplate,
    verbose: bool = True,
) -> RunnableLLMChainAdapter:
    """
    Create an LLMChain-compatible runner for generating summaries.
    """
    return RunnableLLMChainAdapter(llm=llm, prompt=prompt, verbose=verbose)


def retrieve(query: str, faiss_index_obj: FAISS, k: int = 7) -> List[Any]:
    """
    Retrieve relevant context from the FAISS index based on the user's query (default k=7).
    """
    relevant_context = faiss_index_obj.similarity_search(query, k=k)
    return relevant_context


def create_qa_prompt_template() -> PromptTemplate:
    """
    Create a PromptTemplate for question answering based on video content.
    Responds in the same language as the user's question (e.g., English or Vietnamese).
    """
    qa_template = """You are an expert assistant providing detailed and accurate answers based on the following video content. Your responses should be:
1. Precise and free from repetition
2. Consistent with the information provided in the video
3. Well-organized and easy to understand
4. Focused on addressing the user's question directly
5. Written in the same language as the user's question (for example, if the question is in Vietnamese, answer in Vietnamese; if in English, answer in English).

If you encounter conflicting information in the video content, use your best judgment to provide the most likely correct answer based on context.
Note: In the transcript, "Text" refers to the spoken words in the video, and "Start" indicates the timestamp when that part begins in the video.

Relevant Video Context: {context}

Based on the above context, please answer the following question:
Question: {question}
"""
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template,
    )
    return prompt_template


def create_qa_chain(
    llm: Any,
    prompt_template: PromptTemplate,
    verbose: bool = True,
) -> RunnableLLMChainAdapter:
    """
    Create an LLMChain-compatible runner for question answering.
    """
    return RunnableLLMChainAdapter(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(
    question: str,
    faiss_index_obj: FAISS,
    qa_chain: Any,
    k: int = 7,
) -> str:
    """
    Retrieve relevant context and generate an answer based on user input.
    """
    relevant_context = retrieve(question, faiss_index_obj, k=k)
    answer = qa_chain.predict(context=relevant_context, question=question)
    return answer


# =====================================================================
# 4. High-Level Action Handlers (Summarize & Q&A with Smart Caching)
# =====================================================================
def _ensure_transcript_loaded(video_url: str, force_refresh: bool = False) -> Optional[str]:
    """
    Ensure `processed_transcript` is loaded for `video_url`.
    Invalidates cached `faiss_index` whenever the video changes or is refreshed.
    Returns an error string if validation/fetching fails, or `None` on success.
    """
    global current_video_id, fetched_transcript, processed_transcript, faiss_index, transcript_status_msg

    if not video_url or not str(video_url).strip():
        transcript_status_msg = "Please provide a valid YouTube URL."
        return "Please provide a valid YouTube URL."

    extracted_id = get_video_id(video_url)
    if not extracted_id:
        transcript_status_msg = "Please provide a valid YouTube URL."
        return "Please provide a valid YouTube URL."

    needs_fetch = (
        force_refresh
        or not processed_transcript
        or (current_video_id is not None and current_video_id != extracted_id)
    )

    if needs_fetch:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
        if processed_transcript:
            current_video_id = extracted_id
            faiss_index = None  # Reset FAISS index for the newly fetched transcript
        else:
            current_video_id = None
            faiss_index = None
    else:
        transcript_status_msg = f"Using cached transcript for video '{current_video_id}'."

    if not processed_transcript:
        if not transcript_status_msg.startswith("Failed") and not transcript_status_msg.startswith("No transcripts"):
            transcript_status_msg = "No transcript available. Please fetch the transcript first."
        return "No transcript available. Please fetch the transcript first."

    return None


def summarize_video(video_url: str) -> str:
    """
    Generate a single-paragraph summary of a YouTube video using the preprocessed transcript.
    """
    global transcript_status_msg

    if not video_url or not str(video_url).strip() or not get_video_id(video_url):
        transcript_status_msg = "Please provide a valid YouTube URL."
        return "Please provide a valid YouTube URL."

    err = _ensure_transcript_loaded(video_url, force_refresh=True)
    if err:
        return err

    try:
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)
        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    except Exception as exc:
        error_msg = f"Error generating summary: {exc}"
        transcript_status_msg = error_msg
        return error_msg


def answer_question(video_url: str, user_question: str) -> str:
    """
    Retrieve relevant context from the FAISS index based on the user's query
    and generate an answer using the preprocessed transcript.
    Automatically fetches the transcript if not yet fetched or if the video URL changed,
    and reuses the cached FAISS index across follow-up questions on the same video.
    """
    global current_video_id, fetched_transcript, processed_transcript, faiss_index, transcript_status_msg

    extracted_id = get_video_id(video_url) if video_url else None
    url_changed = (
        extracted_id is not None
        and current_video_id is not None
        and extracted_id != current_video_id
    )

    if not processed_transcript or url_changed:
        if video_url and extracted_id:
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
            current_video_id = extracted_id if processed_transcript else None
            faiss_index = None
        else:
            transcript_status_msg = "Please provide a valid YouTube URL."
            return "Please provide a valid YouTube URL."
    else:
        if current_video_id:
            transcript_status_msg = f"Reusing cached transcript and FAISS index for video '{current_video_id}'."

    if processed_transcript and user_question and str(user_question).strip():
        try:
            model_id, credentials, client, project_id = setup_credentials()
            llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

            if faiss_index is None:
                chunks = chunk_transcript(processed_transcript)
                embedding_model = setup_embedding_model(credentials, project_id)
                faiss_index = create_faiss_index(chunks, embedding_model)

            qa_prompt = create_qa_prompt_template()
            qa_chain = create_qa_chain(llm, qa_prompt)
            answer = generate_answer(user_question, faiss_index, qa_chain)
            return answer
        except Exception as exc:
            error_msg = f"Error generating answer: {exc}"
            transcript_status_msg = error_msg
            return error_msg
    else:
        transcript_status_msg = "Please provide a valid question and ensure the transcript has been fetched."
        return "Please provide a valid question and ensure the transcript has been fetched."


# =====================================================================
# 5. Setting Up the Gradio Interface
# =====================================================================
def build_gradio_interface():
    """Construct and return the Gradio Blocks interface."""
    import gradio as gr

    with gr.Blocks(title="YouTube Video Summarizer and Q&A") as app_interface:
        gr.Markdown(
            "<h2 style='text-align: center;'>YouTube Video Summarizer and Q&A</h2>"
        )

        # Input field for YouTube URL
        video_url = gr.Textbox(
            label="YouTube Video URL",
            placeholder="Enter the YouTube Video URL (e.g., https://www.youtube.com/watch?v=T-D1OfcDW1M)",
        )

        # Outputs for summary and answer
        summary_output = gr.Textbox(label="Video Summary", lines=5)
        question_input = gr.Textbox(
            label="Ask a Question About the Video",
            placeholder="Ask your question (English or Tiếng Việt)",
        )
        answer_output = gr.Textbox(label="Answer to Your Question", lines=5)

        # Buttons for selecting functionalities
        with gr.Row():
            summarize_btn = gr.Button("Summarize Video", variant="primary")
            question_btn = gr.Button("Ask a Question", variant="secondary")

        # Display status message for transcript fetch
        transcript_status = gr.Textbox(
            label="Transcript Status",
            value=transcript_status_msg,
            interactive=False,
        )

        # Set up button actions (preserving single-output function signatures + status update)
        summarize_btn.click(
            fn=summarize_video,
            inputs=video_url,
            outputs=summary_output,
        ).then(
            fn=get_transcript_status,
            inputs=None,
            outputs=transcript_status,
        )

        question_btn.click(
            fn=answer_question,
            inputs=[video_url, question_input],
            outputs=answer_output,
        ).then(
            fn=get_transcript_status,
            inputs=None,
            outputs=transcript_status,
        )

    return app_interface


interface = None


if __name__ == "__main__":
    interface = build_gradio_interface()
    port = int(os.getenv("PORT", "7860"))
    interface.launch(server_name="0.0.0.0", server_port=port)
