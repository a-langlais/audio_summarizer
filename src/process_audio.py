import io
import ffmpeg
import soundfile as sf

def load_audio_ffmpeg(file_path):
    """
    Charge le fichier audio en utilisant ffmpeg pour le convertir en WAV,
    puis lit le flux audio en mémoire avec soundfile.
    
    :param file_path: Chemin vers le fichier audio (MP3).
    :return: Tuple (audio_array, sampling_rate)
    """
    try:
        # On demande à ffmpeg de lire le fichier et de le convertir en WAV envoyé sur stdout.
        out, err = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format = 'wav', acodec='pcm_s16le', ac = 1, ar = '16000')  # Mono, 16kHz
            .run(capture_stdout = True, capture_stderr = True)
        )
        # On lit le flux WAV en mémoire grâce à soundfile
        audio_array, sampling_rate = sf.read(io.BytesIO(out))
        return audio_array, sampling_rate
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement audio avec ffmpeg : {e}")

def process_audio(audio_file_path):
    """
    Traite le fichier audio :
      - Chargement du fichier audio via ffmpeg.
      - Transcription de l'audio avec la pipeline ASR.
      - Construction du prompt et génération du rapport via le LLM.
    
    :param audio_file_path: Chemin vers le fichier MP3 importé.
    :return: Rapport généré sous forme de texte.
    """
    # 1. Charger l'audio avec ffmpeg
    try:
        audio_data, sr = load_audio_ffmpeg(audio_file_path)
    except Exception as e:
        return f"Erreur lors du chargement audio : {e}"
    
    # 2. Transcription audio en passant le tableau numpy à la pipeline ASR
    try:
        # On précise la fréquence d'échantillonnage obtenue
        transcription_result = asr_pipeline(audio_data, sampling_rate = sr)
        transcription_text = transcription_result["text"]
    except Exception as e:
        return f"Erreur lors de la transcription : {e}"
    
    # 3. Construction du prompt pour le LLM
    prompt = prompt_template.format(transcription = transcription_text)
    
    # 4. Génération du rapport par le LLM
    try:
        generation = llm_pipeline(prompt, max_length = 500, do_sample = True, temperature = 0.7)
        report_text = generation[0]['generated_text']
    except Exception as e:
        return f"Erreur lors de la génération du rapport : {e}"
    
    return report_text