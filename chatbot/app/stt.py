"""Speech-to-Text (optionnel) avec faster-whisper.

Installation : pip install faster-whisper
Modele       : variable d'environnement WHISPER_MODEL (defaut "small" ; "medium"/"large-v3" = plus precis, plus lent)
Le premier appel telecharge le modele (connexion internet requise, une seule fois).

Whisper comprend bien l'arabe et le francais ; la darija est moins bien reconnue
(a mesurer avec le WER sur les phrases de audio_to_record.csv).
"""
import os
import tempfile
from pathlib import Path

_model = None

# amorce de vocabulaire : aide Whisper a reconnaitre les mots du domaine
PROMPT = ("pharmacie, médicament, prix, remboursement, composition, disponible, Doliprane, Augmentin, Efferalgan, "
          "صيدلية، دواء، ثمن، شحال، واش كاين، ch7al taman, wach kayn, fin kayna pharmacie de garde")


class STTUnavailable(RuntimeError):
    pass


def _load():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTUnavailable("faster-whisper n'est pas installé (pip install faster-whisper)") from e
        _model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="auto", compute_type="auto")
    return _model


def transcribe(audio: bytes, suffix: str = ".webm", language: str = None):
    """retourne (texte, langue_detectee)"""
    model = _load()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        segments, info = model.transcribe(path, language=language, initial_prompt=PROMPT, vad_filter=True,
                                          beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language
    finally:
        Path(path).unlink(missing_ok=True)
