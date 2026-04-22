"""LLM-based editing, evaluation, and reporting helpers for the poetry project."""

import csv
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - optional dependency in non-Colab runs.
    genai = None
    types = None


VOWELS = "аеёиоуыэюя"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
SCORE_FIELDS = [
    "semantic_coherence",
    "grammar",
    "theme_consistency",
    "poetic_quality",
    "overall",
]
REPORT_NUMERIC_FIELDS = [
    "requested_lines",
    "actual_lines",
    "word_count_total",
    "char_count_total",
    "unique_line_count",
    "rhyme_success",
    "rhyme_quality",
    "unique_rate",
    "unique_words",
    "avg_words_per_line",
    "avg_vowels_per_line",
    *SCORE_FIELDS,
]


@dataclass
class LLMConfig:
    """Configuration for the optional LLM editor/evaluator."""

    provider: str = "gemini"
    model_name: str = DEFAULT_GEMINI_MODEL
    editor_temperature: float = 0.35
    evaluator_temperature: float = 0.0
    max_retries: int = 2
    retry_delay_seconds: float = 2.0
    request_timeout_seconds: float = 90.0
    ollama_base_url: str = "http://127.0.0.1:11434"


def _get_colab_secret(name: str) -> Optional[str]:
    """Read a secret from Google Colab if the notebook is running there."""

    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        return value or None
    except Exception:
        return None


def get_gemini_api_key() -> Optional[str]:
    """Find a Gemini API key in environment variables or Colab Secrets."""

    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _get_colab_secret("GEMINI_API_KEY")
        or _get_colab_secret("GOOGLE_API_KEY")
    )


def get_llm_provider() -> Optional[str]:
    """Find the selected LLM provider in environment variables or Colab Secrets."""

    return os.getenv("LLM_PROVIDER") or _get_colab_secret("LLM_PROVIDER")


def get_ollama_base_url() -> Optional[str]:
    """Find the Ollama base URL in environment variables or Colab Secrets."""

    return (
        os.getenv("OLLAMA_BASE_URL")
        or _get_colab_secret("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or _get_colab_secret("OLLAMA_HOST")
    )


def tokenize_russian(text: str) -> List[str]:
    """Extract Russian words from text."""

    return re.findall(r"[а-яё]+", text.lower())


def normalize_poem(poem: Sequence[str] | str | None) -> List[str]:
    """Convert a poem represented as text or list of lines to clean lines."""

    if poem is None:
        return []

    if isinstance(poem, str):
        raw_lines = poem.splitlines()
    else:
        raw_lines = list(poem)

    lines = []
    for line in raw_lines:
        clean = str(line).strip()
        if not clean:
            continue
        clean = re.sub(r"^\s*[\-\*\d]+[\.\)\:\-]?\s*", "", clean).strip()
        if clean:
            lines.append(clean)
    return lines


def poem_to_text(poem: Sequence[str] | str | None) -> str:
    """Return a poem as newline-separated text."""

    return "\n".join(normalize_poem(poem))


def print_poem(title: str, poem: Sequence[str] | str | None) -> None:
    """Print a poem with a compact title."""

    print(f"\n{title}\n")
    lines = normalize_poem(poem)
    if not lines:
        print("Стихотворение отсутствует")
        return
    for line in lines:
        print(line)


def simplify_rhyme_scheme(scheme: str, length: int) -> List[str]:
    """Repeat or trim a rhyme scheme to match the poem length."""

    clean = "".join(ch.upper() for ch in str(scheme) if ch.isalpha()) or "AABB"
    if len(clean) < length:
        clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
    return list(clean[:length])


def extract_last_word(line: str) -> Optional[str]:
    """Return the last Russian word from a line."""

    words = tokenize_russian(line)
    return words[-1] if words else None


def calculate_rhyme_quality(line1: str, line2: str) -> float:
    """Score a rhyme by matching suffixes of the last words."""

    last1 = extract_last_word(line1)
    last2 = extract_last_word(line2)
    if not last1 or not last2:
        return 0.0
    if len(last1) < 3 or len(last2) < 3:
        return 0.5

    for suffix_len in range(3, 0, -1):
        if last1[-suffix_len:] == last2[-suffix_len:]:
            return {3: 1.0, 2: 0.8, 1: 0.5}[suffix_len]
    return 0.2


def calculate_formal_metrics(
    poem: Sequence[str] | str | None,
    rhyme_scheme: str = "AABB",
) -> Dict[str, Any]:
    """Calculate simple comparable metrics for raw or LLM-edited poems."""

    lines = normalize_poem(poem)
    if not lines:
        return {
            "lines": 0,
            "rhyme_success": 0.0,
            "rhyme_quality": 0.0,
            "unique_rate": 0.0,
            "unique_words": 0,
            "avg_words_per_line": 0.0,
            "avg_vowels_per_line": 0.0,
        }

    scheme = simplify_rhyme_scheme(rhyme_scheme, len(lines))
    last_line_by_letter: Dict[str, str] = {}
    rhyme_scores = []

    for line, letter in zip(lines, scheme):
        previous_line = last_line_by_letter.get(letter)
        if previous_line:
            rhyme_scores.append(calculate_rhyme_quality(previous_line, line))
        last_line_by_letter[letter] = line

    tokenized = [tokenize_russian(line) for line in lines]
    all_words = [word for words in tokenized for word in words]
    vowels_per_line = [
        sum(1 for char in line.lower() if char in VOWELS) for line in lines
    ]
    words_per_line = [len(words) for words in tokenized]

    unique_lines = len({line.lower() for line in lines})
    successful_rhymes = sum(1 for score in rhyme_scores if score >= 0.5)

    return {
        "lines": len(lines),
        "rhyme_success": round(successful_rhymes / max(1, len(rhyme_scores)) * 100, 1),
        "rhyme_quality": round(
            sum(rhyme_scores) / max(1, len(rhyme_scores)) * 100, 1
        ),
        "unique_rate": round(unique_lines / len(lines) * 100, 1),
        "unique_words": len(set(all_words)),
        "avg_words_per_line": round(sum(words_per_line) / len(words_per_line), 1),
        "avg_vowels_per_line": round(sum(vowels_per_line) / len(vowels_per_line), 1),
    }


def build_formal_metrics_table(
    poem_versions: Dict[str, Sequence[str] | str | None],
    rhyme_scheme: str = "AABB",
) -> List[Dict[str, Any]]:
    """Build rows for a formal metrics comparison table."""

    rows = []
    for name, poem in poem_versions.items():
        row = {"name": name}
        row.update(calculate_formal_metrics(poem, rhyme_scheme=rhyme_scheme))
        rows.append(row)
    return rows


def poem_lines_to_columns(
    poem: Sequence[str] | str | None,
    max_lines: int = 16,
) -> Dict[str, str]:
    """Expand poem lines into fixed CSV-friendly columns."""

    lines = normalize_poem(poem)
    columns = {}
    for index in range(max_lines):
        key = f"line_{index + 1}"
        columns[key] = lines[index] if index < len(lines) else ""
    return columns


def detect_source_model(version_name: str) -> str:
    """Infer the base model from a version label."""

    label = str(version_name)
    if "Markov" in label:
        return "Markov"
    if "LSTM" in label:
        return "LSTM"
    return ""


def build_poem_report_row(
    version: str,
    poem: Sequence[str] | str | None,
    report_scope: str = "batch",
    run: Optional[int] = None,
    source_model: Optional[str] = None,
    stage: Optional[str] = None,
    start_line: str = "",
    rhyme_scheme: str = "AABB",
    requested_lines: Optional[int] = None,
    llm_model_name: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    evaluation: Optional[Dict[str, Any]] = None,
    max_lines: int = 16,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one detailed CSV row for a poem version."""

    lines = normalize_poem(poem)
    text = poem_to_text(lines)
    metrics = metrics or calculate_formal_metrics(lines, rhyme_scheme=rhyme_scheme)
    evaluation = evaluation or {}
    source_model = source_model or detect_source_model(version)
    stage = stage or ("llm_edited" if "+ LLM" in str(version) else "raw")
    words = tokenize_russian(text)

    row: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "report_scope": report_scope,
        "run": run if run is not None else "",
        "source_model": source_model,
        "version": version,
        "stage": stage,
        "llm_used": "yes" if stage == "llm_edited" else "no",
        "llm_model_name": llm_model_name if llm_model_name else "",
        "start_line": start_line.strip() if start_line else "",
        "rhyme_scheme": rhyme_scheme,
        "requested_lines": requested_lines if requested_lines is not None else len(lines),
        "actual_lines": len(lines),
        "unique_line_count": len({line.lower() for line in lines}),
        "word_count_total": len(words),
        "char_count_total": len(text),
        "poem_text": text,
    }
    row.update(poem_lines_to_columns(lines, max_lines=max_lines))
    row.update(metrics)

    for field in SCORE_FIELDS:
        row[field] = evaluation.get(field, "")
    row["llm_comment"] = evaluation.get("comment", "")
    row["llm_raw_response"] = evaluation.get("raw_response", "")

    if extra:
        row.update(extra)

    return row


def summarize_report_rows(
    rows: Sequence[Dict[str, Any]],
    group_fields: Sequence[str] = ("report_scope", "version", "stage"),
) -> List[Dict[str, Any]]:
    """Aggregate detailed report rows into a compact summary table."""

    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group in grouped.items():
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["row_count"] = len(group)
        summary["source_model"] = group[0].get("source_model", "")
        summary["llm_used"] = group[0].get("llm_used", "")
        summary["llm_model_name"] = group[0].get("llm_model_name", "")

        for field in REPORT_NUMERIC_FIELDS:
            values = []
            for row in group:
                value = row.get(field, "")
                if value in ("", None):
                    continue
                try:
                    values.append(float(value))
                except Exception:
                    continue
            summary[f"mean_{field}"] = (
                round(statistics.mean(values), 3) if values else ""
            )

        comments = [str(row.get("llm_comment", "")).strip() for row in group]
        comments = [comment for comment in comments if comment]
        summary["sample_comment"] = comments[0] if comments else ""
        summary_rows.append(summary)

    return summary_rows


def write_csv_report(
    rows: Sequence[Dict[str, Any]],
    output_path: str | Path,
    preferred_fields: Optional[Sequence[str]] = None,
) -> str:
    """Write rows to CSV and return the resulting path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preferred_fields = list(preferred_fields or [])
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    extra_fields = [key for key in keys if key not in preferred_fields]
    fieldnames = preferred_fields + extra_fields

    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return str(path)


def export_report_bundle(
    rows: Sequence[Dict[str, Any]],
    output_dir: str | Path,
    prefix: str = "poetry_report",
    summary_group_fields: Sequence[str] = ("report_scope", "version", "stage"),
) -> Dict[str, Any]:
    """Export detailed and summary CSV reports."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    preferred_detail_fields = [
        "created_at",
        "report_scope",
        "run",
        "source_model",
        "version",
        "stage",
        "llm_used",
        "llm_model_name",
        "start_line",
        "rhyme_scheme",
        "requested_lines",
        "actual_lines",
        "unique_line_count",
        "word_count_total",
        "char_count_total",
        "rhyme_success",
        "rhyme_quality",
        "unique_rate",
        "unique_words",
        "avg_words_per_line",
        "avg_vowels_per_line",
        *SCORE_FIELDS,
        "llm_comment",
        "llm_raw_response",
        "poem_text",
    ] + [f"line_{index}" for index in range(1, 17)]

    detailed_rows = list(rows)
    summary_rows = summarize_report_rows(
        detailed_rows, group_fields=summary_group_fields
    )
    detailed_path = output_dir / f"{prefix}_detailed_{timestamp}.csv"
    summary_path = output_dir / f"{prefix}_summary_{timestamp}.csv"

    write_csv_report(
        detailed_rows,
        detailed_path,
        preferred_fields=preferred_detail_fields,
    )
    write_csv_report(summary_rows, summary_path)

    return {
        "detailed_path": str(detailed_path),
        "summary_path": str(summary_path),
        "detailed_rows": detailed_rows,
        "summary_rows": summary_rows,
    }


class LLMPoetryAssistant:
    """Optional LLM-based editor and evaluator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.config = config or LLMConfig()
        provider_value = provider or get_llm_provider() or self.config.provider
        self.provider = str(provider_value).strip().lower() or "gemini"
        if model_name:
            self.config.model_name = model_name

        self.api_key = api_key or get_gemini_api_key()
        self.client = None
        self.base_url: Optional[str] = None
        self.last_error: Optional[str] = None

        if self.provider not in {"gemini", "ollama"}:
            self.last_error = (
                f"Unsupported LLM provider: {self.provider}. "
                "Use 'gemini' or 'ollama'."
            )
            return

        if self.provider == "ollama":
            self._init_ollama(base_url=base_url)
            return

        if genai is None:
            self.last_error = "Не установлена библиотека google-genai."
            return

        if not self.api_key:
            self.last_error = (
                "Не найден GEMINI_API_KEY. Добавьте ключ в Colab Secrets "
                "или в переменную окружения."
            )
            return

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            self.last_error = f"Не удалось создать Gemini client: {exc}"

    def _init_ollama(self, base_url: Optional[str] = None) -> None:
        if not self.config.model_name or self.config.model_name == DEFAULT_GEMINI_MODEL:
            self.config.model_name = DEFAULT_OLLAMA_MODEL

        ollama_base_url = base_url or get_ollama_base_url() or self.config.ollama_base_url
        self.base_url = str(ollama_base_url).rstrip("/") if ollama_base_url else None

        if not self.base_url:
            self.last_error = (
                "OLLAMA_BASE_URL is not set. In Colab, point it to a public tunnel "
                "that forwards requests to your local Ollama server."
            )
            return

        self.client = "ollama"

    def is_available(self) -> bool:
        """Return True if the selected LLM backend is ready."""

        if self.provider == "ollama":
            return bool(self.base_url and self.client)
        return self.client is not None

    def status_message(self) -> str:
        """Return a user-facing status line."""

        if self.is_available() and self.provider == "ollama":
            return (
                "LLM-Ð¼Ð¾Ð´ÑƒÐ»ÑŒ Ð³Ð¾Ñ‚Ð¾Ð²: "
                f"Ollama / {self.config.model_name} / {self.base_url}"
            )

        if self.is_available():
            return f"LLM-модуль готов: {self.config.model_name}"
        return self.last_error or "LLM-модуль недоступен."

    def _generate(
        self,
        prompt: str,
        temperature: float,
        response_mime_type: Optional[str] = None,
    ) -> str:
        if self.provider == "ollama":
            return self._generate_ollama(
                prompt,
                temperature=temperature,
                response_mime_type=response_mime_type,
            )

        if not self.client:
            raise RuntimeError(self.status_message())

        config_kwargs: Dict[str, Any] = {"temperature": temperature}
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        generation_config = (
            types.GenerateContentConfig(**config_kwargs) if types else None
        )

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=generation_config,
                )
                return (response.text or "").strip()
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(f"Ошибка Gemini API: {last_exc}")

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float,
        response_mime_type: Optional[str] = None,
    ) -> str:
        if not self.base_url:
            raise RuntimeError(self.status_message())

        endpoint = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if response_mime_type == "application/json":
            payload["format"] = "json"

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.request_timeout_seconds,
                ) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                return str(response_data.get("response", "")).strip()
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(
                    f"Ollama HTTP {exc.code}: {error_body or exc.reason}"
                )
            except Exception as exc:
                last_exc = exc

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise RuntimeError(f"Ollama API error: {last_exc}")

    def edit_poem(
        self,
        draft_poem: Sequence[str] | str | None,
        start_line: str,
        rhyme_scheme: str = "AABB",
        preserve_first_line: bool = True,
    ) -> List[str]:
        """Edit a generated draft while preserving the generator's role."""

        draft_lines = normalize_poem(draft_poem)
        if not draft_lines:
            return []

        expected_lines = len(draft_lines)
        draft_text = "\n".join(draft_lines)
        first_line_rule = (
            f'- Первую строку сохрани без изменений: "{start_line}".'
            if preserve_first_line and start_line
            else "- Первую строку можно слегка отредактировать, если это необходимо."
        )

        prompt = f"""
Ты литературный редактор русскоязычных стихотворений.

Ниже дан черновик, созданный алгоритмом генерации текста.
Твоя задача - отредактировать черновик, а не написать стихотворение с нуля.

Требования:
- Сохрани количество строк: {expected_lines}.
- Сохрани схему рифмовки: {rhyme_scheme}.
{first_line_rule}
- Сохрани тему и основные образы черновика, если они не ломают смысл.
- Исправь грамматику и сделай текст более осмысленным.
- Не добавляй заголовок, пояснения, нумерацию или комментарии.
- Верни только итоговое стихотворение.

Черновик:
{draft_text}
""".strip()

        edited_text = self._generate(
            prompt,
            temperature=self.config.editor_temperature,
            response_mime_type="text/plain",
        )
        edited_lines = normalize_poem(edited_text)

        if not edited_lines:
            return draft_lines

        if len(edited_lines) > expected_lines:
            edited_lines = edited_lines[:expected_lines]
        elif len(edited_lines) < expected_lines:
            edited_lines.extend(draft_lines[len(edited_lines) : expected_lines])

        if preserve_first_line and edited_lines and start_line:
            edited_lines[0] = start_line.strip()

        return edited_lines

    def evaluate_poem(
        self,
        poem: Sequence[str] | str | None,
        start_line: str,
    ) -> Dict[str, Any]:
        """Evaluate semantic quality with a fixed blind rubric."""

        poem_lines = normalize_poem(poem)
        if not poem_lines:
            return self._empty_evaluation("Стихотворение отсутствует.")

        poem_text = "\n".join(poem_lines)
        prompt = f"""
Ты независимый эксперт по оценке русскоязычных стихотворений.
Ты не знаешь, каким алгоритмом создан текст, и не должен это угадывать.

Оцени стихотворение по шкале от 1 до 5:
1 - очень плохо, 2 - плохо, 3 - средне, 4 - хорошо, 5 - отлично.

Критерии:
- semantic_coherence: смысловая связность текста;
- grammar: грамматическая правильность;
- theme_consistency: сохранение темы первой строки;
- poetic_quality: художественность и выразительность;
- overall: общее качество.

Не оценивай строго классический стихотворный размер.
Не переписывай стихотворение.
Верни только JSON без markdown-разметки.

Первая строка:
"{start_line}"

Стихотворение:
{poem_text}

Формат ответа:
{{
  "semantic_coherence": 1,
  "grammar": 1,
  "theme_consistency": 1,
  "poetic_quality": 1,
  "overall": 1,
  "comment": "краткий комментарий"
}}
""".strip()

        raw_response = self._generate(
            prompt,
            temperature=self.config.evaluator_temperature,
            response_mime_type="application/json",
        )
        parsed = self._parse_evaluation(raw_response)
        parsed["raw_response"] = raw_response
        return parsed

    def evaluate_versions(
        self,
        poem_versions: Dict[str, Sequence[str] | str | None],
        start_line: str,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple poems without exposing generator names to the LLM."""

        rows = []
        for name, poem in poem_versions.items():
            evaluation = self.evaluate_poem(poem, start_line=start_line)
            row = {"name": name}
            for field in SCORE_FIELDS:
                row[field] = evaluation.get(field, 0)
            row["comment"] = evaluation.get("comment", "")
            rows.append(row)
        return rows

    def _parse_evaluation(self, raw_response: str) -> Dict[str, Any]:
        text = raw_response.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)

        try:
            data = json.loads(text)
        except Exception:
            return self._empty_evaluation("Не удалось разобрать JSON-оценку.")

        normalized: Dict[str, Any] = {}
        for field in SCORE_FIELDS:
            normalized[field] = self._normalize_score(data.get(field))
        normalized["comment"] = str(data.get("comment", "")).strip()
        return normalized

    @staticmethod
    def _normalize_score(value: Any) -> int:
        try:
            score = int(round(float(value)))
        except Exception:
            score = 0
        return max(1, min(5, score)) if score else 0

    @staticmethod
    def _empty_evaluation(comment: str) -> Dict[str, Any]:
        data = {field: 0 for field in SCORE_FIELDS}
        data["comment"] = comment
        return data
