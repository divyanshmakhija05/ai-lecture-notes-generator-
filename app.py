import streamlit as st
import whisper
import re
import tempfile
from transformers import pipeline
import yt_dlp
import os
import uuid

st.title("🎓 Lecture Voice-to-Notes Generator")

# Loading Models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("tiny")

    summarizer = pipeline(
        "summarization",
        model="t5-small"
    )

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return whisper_model, summarizer, generator


whisper_model, summarizer, generator = load_models()


def clean_text(text):
    text = re.sub(r"\b(um|uh|you know)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def cleanup_old_audio():
    for file in os.listdir():
        if file.endswith((".mp3", ".webm", ".m4a", ".wav")):
            try:
                os.remove(file)
            except:
                pass


# Download YouTube audio


def download_audio(url):
    unique_id = uuid.uuid4().hex

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{unique_id}.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for file in os.listdir():
        if file.startswith(unique_id):
            return file

# Split text into chunks

def split_into_chunks(text, max_words=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


if "last_url" not in st.session_state:
    st.session_state.last_url = ""


#Inputs

uploaded_file = st.file_uploader("Upload Lecture Audio", type=["mp3", "wav", "m4a"])
youtube_url = st.text_input("Or Paste YouTube Video Link")

audio_path = None

if uploaded_file is not None:
    cleanup_old_audio()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

elif youtube_url:
    if youtube_url != st.session_state.last_url:
        cleanup_old_audio()
        st.session_state.last_url = youtube_url
    audio_path = download_audio(youtube_url)




from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generate_pdf(topic_notes, overall_summary, questions):
    file_path = "Lecture_Notes.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Lecture Notes</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>Topic-wise Summary</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for i, note in enumerate(topic_notes):
        content.append(Paragraph(f"<b>Topic {i+1}:</b> {note}", styles["Normal"]))
        content.append(Spacer(1, 8))

    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Overall Summary</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))
    content.append(Paragraph(overall_summary, styles["Normal"]))

    content.append(Spacer(1, 12))
    content.append(Paragraph("<b>Quiz Questions</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))

    for q in questions:
        content.append(Paragraph(q, styles["Normal"]))
        content.append(Spacer(1, 6))

    doc.build(content)
    return file_path

#Variables
topic_notes = []
combined_summary = ""
all_questions = []


if audio_path:
    st.info("Processing audio... Please wait ⏳")

    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]

    cleaned_text = clean_text(transcript)

    st.subheader("📝 Full Transcript")
    st.write(cleaned_text)

    
    
    
    st.subheader("📘 Topic-wise Notes")

    chunks = split_into_chunks(cleaned_text)
    topic_notes = []

    for i, chunk in enumerate(chunks):
        summary = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False
        )
        topic_notes.append(summary[0]["summary_text"])
        st.markdown(f"**Topic {i+1}:** {summary[0]['summary_text']}")

    
    
    
    combined_summary = " ".join(topic_notes)

    st.subheader("📕 Overall Summary")
    st.write(combined_summary)

    
    st.subheader("❓ Quiz Questions (Topic-wise)")

    all_questions = []
    question_number = 1

    for i, chunk in enumerate(chunks):
        quiz_prompt = f"""
        Generate exactly TWO meaningful short-answer questions from the following lecture content.
        Do NOT ask about names, people, or titles.
        Focus on concepts, explanations, or applications.
        Output only numbered questions.

        Lecture Content:
        {chunk}
        """

        quiz = generator(
        quiz_prompt,
        max_length=180,
        do_sample=False
        )

        raw_output = quiz[0]["generated_text"]

        # Extract proper que
        lines = raw_output.split("\n")
        for line in lines:
            line = line.strip()
            if line.endswith("?") and len(line) > 20:
                all_questions.append(f"{question_number}. {line}")
                question_number += 1

    # Display que
    if all_questions:
        for q in all_questions[:6]:  # limit to 6 
            st.write(q)
    else:
        st.warning("Could not generate quality questions. Try a longer lecture.")
        

# PDF Download

pdf_path = generate_pdf(
    topic_notes=topic_notes,
    overall_summary=combined_summary,
    questions=all_questions
)

with open(pdf_path, "rb") as pdf_file:
    st.download_button(
        label="📥 Download Notes as PDF",
        data=pdf_file,
        file_name="Lecture_Notes.pdf",
        mime="application/pdf"
    )