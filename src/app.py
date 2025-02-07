import gradio as gr
from transformers import pipeline
import io
import ffmpeg
import soundfile as sf

from process_audio import *

# Pipeline pour la transcription audio
# Ce modèle est multilingue, il peut donc gérer aussi bien le français que l'anglais.
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Pipeline pour la génération de texte (LLM)
llm_pipeline = pipeline("text-generation", model="gpt2")

# Prompt interne servant à orienter le modèle LLM pour structurer la transcription.
prompt_template = (
    "Tu es un assistant expert chargé de structurer et organiser la retranscription d'une réunion. "
    "Génère un rapport clair, concis et bien structuré en te basant sur le contenu suivant. N'invente rien.\n\n"
    "Transcription de la réunion :\n{transcription}\n\n"
    "Rapport :"
)

# Création de l'interface Gradio
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Importer un fichier MP3"),
    outputs=gr.Markdown(label="Rapport généré"),
    title="Application de Transcription et Rapport de Réunion",
    description=(
        "Importez un fichier MP3 (en français ou en anglais) contenant l'enregistrement d'une réunion. "
        "L'application transcrit l'audio puis, utilise un LLM open-source pour générer un rapport"
        "structuré à partir de la transcription."
    )
)

# Lancement de l'application
interface.launch()
