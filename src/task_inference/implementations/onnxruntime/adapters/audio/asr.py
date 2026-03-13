# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime ASR dialect adapters.

Supported dialects
------------------
wav2vec2-ctc
    CTC-based sequence-to-sequence ASR (wav2vec2, HuBERT,
    Wav2Vec2Conformer).  Single ONNX file ``model.onnx``.
    Input  ``input_values [1, seq_len]`` (z-normalised float32 PCM).
    Output ``logits [1, T, vocab_size]``: per-frame log-probabilities.
    Decoding: greedy argmax + CTC blank-collapse + vocab token lookup.

whisper-encoder-decoder
    Whisper encoder-decoder ASR.  Two ONNX files in the same directory:
    ``encoder_model.onnx`` and ``decoder_model.onnx``.
    Encoder input  ``input_features [1, n_mels, T]`` (log-mel spectrogram).
    Encoder output ``last_hidden_state [1, T', hidden]``.
    Decoder inputs ``input_ids [1, seq_len]``, ``encoder_hidden_states``.
    Decoder output ``logits [1, seq_len, vocab_size]``.
    Decoding: greedy autoregressive loop with optional timestamp tokens.

Dialect selection
-----------------
``resolve_asr_adapter`` first inspects ``config.json`` → ``model_type``
and the filesystem (presence of ``encoder_model.onnx``) to detect Whisper.
For all other models it creates the main session and delegates to
``resolve_adapter``.
"""

from __future__ import annotations

import json
import pathlib
from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from ..base import OnnxDialectAdapter, resolve_adapter, io_names_from_session
from ...base import OnnxRuntimeTaskMixin as _M
from .....tasks.audio.speech_recognition import ASRChunk


# ---------------------------------------------------------------------------
# Per-task abstract base
# ---------------------------------------------------------------------------

class ASRAdapter(OnnxDialectAdapter):
    """Abstract adapter for automatic speech recognition tasks."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        return_timestamps: bool,
    ) -> tuple[str, list[ASRChunk] | None]:
        """Transcribe a 1-D float32 PCM array.

        Returns
        -------
        text : str
        chunks : list[ASRChunk] | None
            Populated only when ``return_timestamps=True`` and the model
            supports timestamps.
        """


# ---------------------------------------------------------------------------
# Dialect: wav2vec2-ctc
# ---------------------------------------------------------------------------

class Wav2Vec2CTCAdapter(ASRAdapter):
    """CTC ASR dialect adapter (wav2vec2, HuBERT, …).

    Expects:
    * Input  ``input_values [1, seq_len]`` (z-normalised float32 PCM)
    * Output ``logits [1, T, vocab_size]``
    """

    DIALECT: ClassVar[str] = "wav2vec2-ctc"

    def __init__(
        self,
        session: Any,
        do_normalize: bool,
        id2token: dict[int, str],
        blank_id: int,
    ) -> None:
        self._session = session
        self._do_normalize = do_normalize
        self._id2token = id2token
        self._blank_id = blank_id

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return "input_values" in input_names and "logits" in output_names

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        return_timestamps: bool,
    ) -> tuple[str, list[ASRChunk] | None]:
        if self._do_normalize:
            mean = float(audio.mean())
            std = float(audio.std()) + 1e-9
            audio = (audio - mean) / std

        input_values = audio[np.newaxis, :]  # [1, seq_len]
        out = self._session.run(["logits"], {"input_values": input_values})
        logits = out[0][0]  # [T, vocab_size]

        ids = np.argmax(logits, axis=-1).tolist()
        # CTC collapse: remove repeated tokens then blanks
        collapsed: list[int] = []
        prev = None
        for i in ids:
            if i != prev:
                collapsed.append(i)
            prev = i
        token_ids = [i for i in collapsed if i != self._blank_id]
        tokens = [self._id2token.get(i, "") for i in token_ids]
        text = "".join(tokens).replace("|", " ").strip()
        return text, None  # CTC does not produce timestamp chunks


# ---------------------------------------------------------------------------
# Dialect: whisper-encoder-decoder
# ---------------------------------------------------------------------------

class WhisperAdapter(ASRAdapter):
    """Whisper encoder-decoder dialect adapter.

    Loads ``encoder_model.onnx`` and ``decoder_model.onnx`` from the same
    directory.  Preprocessing uses ``librosa`` to compute log-mel
    spectrograms (must be installed separately).
    """

    DIALECT: ClassVar[str] = "whisper-encoder-decoder"

    def __init__(
        self,
        model_dir: str,
        providers: list[str],
        cfg: dict[str, Any],
    ) -> None:
        p = pathlib.Path(model_dir)
        pp_cfg = _M._load_preprocessor_config(model_dir)

        self._n_mels: int = int(pp_cfg.get("feature_size", 80))
        self._n_fft: int = int(pp_cfg.get("n_fft", 400))
        self._hop_length: int = int(pp_cfg.get("hop_length", 160))
        self._n_samples: int = int(pp_cfg.get("n_samples", 480000))
        self._nb_max_frames: int = int(pp_cfg.get("nb_max_frames", 3000))

        encoder_path = str(p / "encoder_model.onnx")
        decoder_path = str(p / "decoder_model.onnx")
        if not (p / "decoder_model.onnx").exists():
            decoder_merged = p / "decoder_model_merged.onnx"
            if decoder_merged.exists():
                decoder_path = str(decoder_merged)
        self._encoder_session = _M._create_session(encoder_path, providers)
        self._decoder_session = _M._create_session(decoder_path, providers)

        tokenizer_path = p / "tokenizer.json"
        tokenizer_data: dict[str, Any] = json.loads(tokenizer_path.read_text())
        self._added_tokens: dict[str, int] = {
            t["content"]: t["id"] for t in tokenizer_data.get("added_tokens", [])
        }
        self._eos_token_id: int = self._added_tokens.get(
            "<|endoftext|>", cfg.get("eos_token_id", 50256)
        )
        self._bos_token_id: int = cfg.get(
            "decoder_start_token_id",
            self._added_tokens.get("<|startoftranscript|>", 50258),
        )

        from tokenizers import Tokenizer  # noqa: PLC0415
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # ---- classmethod not used via the generic registry for Whisper ----

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        # Detection is done statically in resolve_asr_adapter before any
        # session is created; this method is kept for API completeness.
        return config.get("model_type") == "whisper"

    # ------------------------------------------------------------------
    # Mel spectrogram
    # ------------------------------------------------------------------

    def _compute_features(self, audio: np.ndarray) -> np.ndarray:
        try:
            import librosa  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Whisper ASR requires 'librosa'. Install it with: pip install librosa"
            ) from exc

        if len(audio) < self._n_samples:
            audio = np.pad(audio, (0, self._n_samples - len(audio)))
        else:
            audio = audio[: self._n_samples]

        mel = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=self._n_fft, hop_length=self._hop_length,
            n_mels=self._n_mels, window="hann", center=True, norm=None, power=2.0,
        )
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0

        frames = log_mel.shape[1]
        if frames > self._nb_max_frames:
            log_mel = log_mel[:, : self._nb_max_frames]
        elif frames < self._nb_max_frames:
            log_mel = np.pad(log_mel, ((0, 0), (0, self._nb_max_frames - frames)))

        return log_mel[np.newaxis, :, :].astype(np.float32)  # [1, n_mels, T]

    # ------------------------------------------------------------------
    # Greedy decoder
    # ------------------------------------------------------------------

    def _greedy_decode(
        self,
        input_features: np.ndarray,
        language: str | None,
        return_timestamps: bool,
        max_new_tokens: int = 448,
    ) -> tuple[str, list[ASRChunk] | None]:
        enc_out = self._encoder_session.run(
            ["last_hidden_state"], {"input_features": input_features}
        )
        encoder_hidden_states: np.ndarray = enc_out[0]  # [1, T', hidden]

        prefix: list[int] = [self._bos_token_id]
        lang_token = f"<|{language or 'en'}|>"
        if lang_token in self._added_tokens:
            prefix.append(self._added_tokens[lang_token])
        if "<|transcribe|>" in self._added_tokens:
            prefix.append(self._added_tokens["<|transcribe|>"])
        if not return_timestamps and "<|notimestamps|>" in self._added_tokens:
            prefix.append(self._added_tokens["<|notimestamps|>"])

        generated: list[int] = list(prefix)
        for _ in range(max_new_tokens):
            dec_out = self._decoder_session.run(
                ["logits"],
                {
                    "input_ids": np.array([generated], dtype=np.int64),
                    "encoder_hidden_states": encoder_hidden_states,
                },
            )
            next_token = int(np.argmax(dec_out[0][0, -1, :]))
            if next_token == self._eos_token_id:
                break
            generated.append(next_token)

        text_ids = [i for i in generated[len(prefix):] if i < 50257]
        text: str = self._tokenizer.decode(text_ids)

        chunks: list[ASRChunk] | None = None
        if return_timestamps:
            chunks = self._parse_timestamps(generated[len(prefix):])
        return text, chunks

    def _parse_timestamps(self, token_ids: list[int]) -> list[ASRChunk]:
        timestamp_begin = self._added_tokens.get("<|0.00|>", 50364)
        time_per_token = self._hop_length * 2 / 16000
        chunks: list[ASRChunk] = []
        text_ids: list[int] = []
        start: float | None = None
        for tok in token_ids:
            if tok >= timestamp_begin:
                t = (tok - timestamp_begin) * time_per_token
                if start is None:
                    start = t
                else:
                    text = self._tokenizer.decode([i for i in text_ids if i < 50257])
                    if text.strip():
                        chunks.append(ASRChunk(text=text, timestamp_start=start, timestamp_end=t))
                    text_ids = []
                    start = t
            elif tok < 50257:
                text_ids.append(tok)
        if text_ids and start is not None:
            text = self._tokenizer.decode([i for i in text_ids if i < 50257])
            if text.strip():
                chunks.append(ASRChunk(text=text, timestamp_start=start, timestamp_end=None))
        return chunks

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        return_timestamps: bool,
    ) -> tuple[str, list[ASRChunk] | None]:
        features = self._compute_features(audio)
        return self._greedy_decode(features, language, return_timestamps)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_CTC_ADAPTERS: list[type[ASRAdapter]] = [
    Wav2Vec2CTCAdapter,
]


def _load_ctc_vocab(model_dir: str) -> tuple[dict[int, str], int]:
    """Load CTC vocabulary from ``tokenizer.json`` or ``vocab.json``."""
    p = pathlib.Path(model_dir)
    tokenizer_path = p / "tokenizer.json"
    vocab_path = p / "vocab.json"
    if tokenizer_path.exists():
        tok_data: dict[str, Any] = json.loads(tokenizer_path.read_text())
        vocab: dict[str, int] = tok_data.get("model", {}).get("vocab", {})
        if not vocab:
            vocab = tok_data.get("vocab", {})
    elif vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
    else:
        vocab = {}
    id2token = {v: k for k, v in vocab.items()}
    blank_id = 0  # <pad> is id 0 for wav2vec2
    return id2token, blank_id


def resolve_asr_adapter(
    model_dir: str,
    providers: list[str],
    config: dict[str, Any],
) -> ASRAdapter:
    """Detect the ASR dialect and return an instantiated adapter.

    Whisper detection is performed *before* creating a session (by
    inspecting ``config.json`` → ``model_type`` and the presence of
    ``encoder_model.onnx``) to avoid loading the wrong file.  All other
    models fall through to session-based dialect auto-detection.
    """
    p = pathlib.Path(model_dir)
    is_whisper = (
        config.get("model_type") == "whisper"
        or (p / "encoder_model.onnx").exists()
    )
    if is_whisper:
        return WhisperAdapter(model_dir, providers, config)

    pp_cfg = _M._load_preprocessor_config(model_dir)
    do_normalize: bool = bool(pp_cfg.get("do_normalize", True))
    session = _M._create_session(_M._find_onnx_file(model_dir), providers)
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_CTC_ADAPTERS, input_names, output_names, config)
    id2token, blank_id = _load_ctc_vocab(model_dir)
    return adapter_cls(session, do_normalize, id2token, blank_id)
