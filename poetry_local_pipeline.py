#!/usr/bin/env python3
"""Local end-to-end runner for the poetry generation project."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import py_compile
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DATASET_JSON_URL = (
    "https://raw.githubusercontent.com/Koziev/Rifma/main/rifma_dataset.json"
)
DEFAULT_START_LINE = "В тиши ночной дрожит звезда"
DEFAULT_RHYME_SCHEME = "AABB"
DEFAULT_POEM_LINES = 8
DEFAULT_REPORT_PREFIX = "poetry_report"


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_windows_tensorflow_gpu_note() -> None:
    if platform.system() != "Windows":
        return

    print(
        "Note: на native Windows TensorFlow 2.11+ не использует NVIDIA GPU. "
        "Для GPU-запуска LSTM нужен WSL2/Linux-окружение."
    )


def detect_nvidia_gpus() -> List[str]:
    """Return GPU names reported by nvidia-smi, if available."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines


def detect_tensorflow_gpu_devices() -> Dict[str, Any]:
    """Return TensorFlow GPU visibility for the current interpreter."""

    try:
        import tensorflow as tf
    except Exception as exc:
        return {
            "ok": False,
            "message": f"TensorFlow import failed: {exc}",
            "devices": [],
        }

    try:
        devices = tf.config.list_physical_devices("GPU")
        device_names = [getattr(device, "name", str(device)) for device in devices]
        return {
            "ok": True,
            "message": "TensorFlow GPU devices detected." if device_names else "TensorFlow sees no GPU devices.",
            "devices": device_names,
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": f"TensorFlow GPU probe failed: {exc}",
            "devices": [],
        }


def print_local_hardware_summary() -> None:
    """Print a compact hardware summary before LSTM-related work."""

    print_section("Аппаратная диагностика")

    nvidia_gpus = detect_nvidia_gpus()
    if nvidia_gpus:
        print("NVIDIA GPU detected:")
        for gpu in nvidia_gpus:
            print(f"  - {gpu}")
    else:
        print("NVIDIA GPU not detected via nvidia-smi.")

    tf_gpu_info = detect_tensorflow_gpu_devices()
    print(tf_gpu_info["message"])
    if tf_gpu_info["devices"]:
        for device in tf_gpu_info["devices"]:
            print(f"  - {device}")


def ensure_supported_python() -> None:
    py_version = sys.version_info[:2]
    if not ((3, 10) <= py_version <= (3, 13)):
        raise RuntimeError(
            "TensorFlow currently supports Python 3.10-3.13. "
            f"Current interpreter: {sys.version.split()[0]}. "
            "Create or select a Python 3.11/3.12 environment and rerun the script."
        )


def lstm_requested(args: argparse.Namespace) -> bool:
    return not args.prepare_data_only and (
        not args.skip_quick_lstm or not args.skip_full_lstm
    )


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def ensure_on_syspath(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def import_project_module(module_dir: Path, module_name: str):
    ensure_on_syspath(module_dir)
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "dependency"
        raise RuntimeError(
            f"Cannot import '{module_name}' because '{missing_name}' is missing. "
            "Install the local dependencies first, for example: "
            f"`{sys.executable} -m pip install -r requirements.txt`."
        ) from exc


def validate_project_layout(project_dir: Path) -> None:
    required = [
        project_dir / "Markov_chain_2" / "Markov_chain.py",
        project_dir / "LSTM_generation_2" / "lstm_generation.py",
        project_dir / "llm_poetry_tools.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "Project files are missing. Run the script from the repository root.\n"
            f"{missing_text}"
        )


def validate_python_files(project_dir: Path) -> None:
    print_section("Проверка Python-файлов")
    for source_path in [
        project_dir / "Markov_chain_2" / "Markov_chain.py",
        project_dir / "LSTM_generation_2" / "lstm_generation.py",
        project_dir / "llm_poetry_tools.py",
        project_dir / "poetry_local_pipeline.py",
    ]:
        py_compile.compile(str(source_path), doraise=True)
        print(f"OK: {source_path.relative_to(project_dir)}")


def count_good_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        words = re.findall(r"[А-Яа-яЁё]+", line)
        if len(words) >= 2:
            count += 1
    return count


def extract_text_from_item(item: Any) -> str:
    candidates: List[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            candidates.append(obj)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(item)
    if not candidates:
        return ""

    candidates.sort(
        key=lambda value: (
            count_good_lines(value),
            len(re.findall(r"[А-Яа-яЁё]", value)),
        ),
        reverse=True,
    )
    return candidates[0]


def remove_russian_accent_marks(text: str) -> str:
    return text.replace("\u0301", "").replace("\u0300", "")


def normalize_json_root(data: Any) -> Iterable[Any]:
    if isinstance(data, dict):
        for key in ["data", "dataset", "items", "poems", "samples"]:
            value = data.get(key)
            if isinstance(value, list):
                return value
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset structure: expected a list or a dict with a list.")


def download_dataset_json(dataset_url: str, dataset_json_path: Path, force: bool) -> Path:
    dataset_json_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_json_path.exists() and not force:
        print(f"JSON-датасет уже есть: {dataset_json_path}")
        return dataset_json_path

    print(f"Скачиваем датасет: {dataset_url}")
    urllib.request.urlretrieve(dataset_url, dataset_json_path)
    print(f"JSON-датасет сохранён: {dataset_json_path}")
    return dataset_json_path


def build_poems_clean_file(dataset_json_path: Path, dataset_txt_path: Path) -> Path:
    print_section("Подготовка poems_clean.txt")
    with dataset_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    items = normalize_json_root(data)
    poems: List[str] = []

    for item in items:
        text = extract_text_from_item(item)
        if not text:
            continue

        text = remove_russian_accent_marks(text)
        clean_lines = []

        for line in text.splitlines():
            line = line.strip()
            line = re.sub(r"[^А-Яа-яЁё\-\s]", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            words = re.findall(r"[А-Яа-яЁё]+", line)
            if len(words) >= 2:
                clean_lines.append(line)

        if len(clean_lines) >= 2:
            poems.append("\n".join(clean_lines))

    if not poems:
        raise ValueError(
            "Не удалось извлечь стихи из JSON. Проверьте структуру датасета или укажите свой poems_clean.txt."
        )

    dataset_txt_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_txt_path.write_text("\n\n".join(poems), encoding="utf-8")
    print(f"Готово. Сохранено стихотворных фрагментов: {len(poems)}")
    print(f"Файл: {dataset_txt_path}")
    return dataset_txt_path


def prepare_dataset_from_custom_text(source_path: Path, dataset_txt_path: Path) -> Path:
    print_section("Подготовка пользовательского poems_clean.txt")
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")
    dataset_txt_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dataset_txt_path)
    print(f"Используем пользовательский датасет: {source_path}")
    print(f"Локальная копия: {dataset_txt_path}")
    return dataset_txt_path


def copy_dataset_to_model_dirs(project_dir: Path, dataset_txt_path: Path) -> Dict[str, Path]:
    print_section("Копирование датасета в проект")
    targets = {
        "markov": project_dir / "Markov_chain_2" / "poems_clean.txt",
        "lstm": project_dir / "LSTM_generation_2" / "poems_clean.txt",
    }
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_txt_path, target)
        print(target)
    return targets


def clean_poem_lines(poem: Sequence[str] | None) -> List[str]:
    if not poem:
        return []
    return [str(line).strip() for line in poem if str(line).strip()]


def generate_poem_with_retries(
    generator: Any,
    lines: int,
    rhyme_scheme: str,
    start_line: Optional[str],
    attempts: int,
    progress_label: str = "",
    show_progress: bool = False,
    **kwargs: Any,
) -> List[str]:
    best_poem: List[str] = []
    total_start = time.perf_counter()
    label = progress_label or "generation"

    for attempt in range(1, attempts + 1):
        if show_progress:
            print(f"[{label}] Попытка {attempt}/{attempts}...")
            attempt_start = time.perf_counter()

        poem = generator.generate_poem(
            lines=lines,
            rhyme_scheme=rhyme_scheme,
            start_line=start_line,
            **kwargs,
        )
        poem_lines = clean_poem_lines(poem)

        if show_progress:
            elapsed = time.perf_counter() - attempt_start
            print(
                f"[{label}] Получено строк: {len(poem_lines)}/{lines} "
                f"за {elapsed:.1f} c"
            )

        if len(poem_lines) > len(best_poem):
            best_poem = poem_lines
        if len(poem_lines) >= lines:
            if show_progress:
                total_elapsed = time.perf_counter() - total_start
                print(f"[{label}] Готово за {total_elapsed:.1f} c")
            return poem_lines[:lines]

    if show_progress:
        total_elapsed = time.perf_counter() - total_start
        print(
            f"[{label}] Завершено без полного результата. "
            f"Лучший вариант: {len(best_poem)}/{lines} строк за {total_elapsed:.1f} c"
        )
    return best_poem


def get_lstm_cache_files(model_name: str = "poet") -> List[str]:
    return [
        f"{model_name}_model.keras",
        f"{model_name}_metadata.json",
        f"{model_name}_rhymes.json",
    ]


def restore_lstm_cache(cache_dir: Path, local_dir: Path, model_name: str = "poet") -> bool:
    cache_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    for filename in get_lstm_cache_files(model_name):
        src = cache_dir / filename
        dst = local_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            restored.append(filename)

    if restored:
        print(f"Восстановлен LSTM-кэш: {', '.join(restored)}")
        return True

    print("Сохранённая LSTM-модель в кэше пока не найдена.")
    return False


def save_lstm_cache(cache_dir: Path, local_dir: Path, model_name: str = "poet") -> bool:
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for filename in get_lstm_cache_files(model_name):
        src = local_dir / filename
        dst = cache_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            saved.append(filename)

    if saved:
        print(f"LSTM-файлы сохранены в кэш: {', '.join(saved)}")
        return True

    print("LSTM-файлы для сохранения не найдены.")
    return False


def run_markov_demo(
    project_dir: Path,
    poem_lines: int,
    rhyme_scheme: str,
    start_line: Optional[str],
    retries: int,
) -> Dict[str, Any]:
    print_section("Markov")
    markov_dir = project_dir / "Markov_chain_2"
    markov_module = import_project_module(markov_dir, "Markov_chain")
    generator_class = getattr(markov_module, "RhymingPoetryGenerator")

    with working_directory(markov_dir):
        generator = generator_class(model_name="poet_markov")
        trained = generator.load_and_train("poems_clean.txt", state_size=2)
        if not trained:
            return {"ok": False, "poem": [], "generator": generator, "metrics": {}}

        poem = generate_poem_with_retries(
            generator,
            lines=poem_lines,
            rhyme_scheme=rhyme_scheme,
            start_line=start_line,
            attempts=retries,
        )
        generator.display_poem(poem)
        generator.display_metrics()
        return {
            "ok": True,
            "poem": poem,
            "generator": generator,
            "metrics": generator.metrics.get_statistics(),
        }


def run_quick_lstm_check(
    project_dir: Path,
    rhyme_scheme: str,
    start_line: Optional[str],
    epochs: int,
) -> Dict[str, Any]:
    print_section("LSTM: быстрая проверка")
    lstm_dir = project_dir / "LSTM_generation_2"
    lstm_module = import_project_module(lstm_dir, "lstm_generation")
    generator_class = getattr(lstm_module, "LSTMRhymingPoetryGenerator")

    with working_directory(lstm_dir):
        quick_lstm = generator_class(model_name="poet_quick")
        quick_ok = quick_lstm.load_and_train(
            "poems_clean.txt",
            epochs=epochs,
            batch_size=16,
            max_vocab_size=3000,
            max_training_lines=3000,
        )
        if not quick_ok:
            print("LSTM быстрая проверка не прошла.")
            return {"ok": False}

        poem = quick_lstm.generate_poem(
            lines=4,
            rhyme_scheme=rhyme_scheme,
            start_line=start_line,
            temperature=0.65,
        )
        quick_lstm.display_poem(poem)
        quick_lstm.display_metrics()
        return {"ok": True, "poem": clean_poem_lines(poem)}


def run_full_lstm_demo(
    project_dir: Path,
    cache_dir: Path,
    use_cache: bool,
    poem_lines: int,
    rhyme_scheme: str,
    start_line: Optional[str],
    full_epochs: int,
    demo_retries: int,
    free_line_candidates: int,
    rhyme_candidates: int,
    top_k: int,
    max_context_tokens: int,
) -> Dict[str, Any]:
    print_section("LSTM: основной запуск")
    lstm_dir = project_dir / "LSTM_generation_2"
    lstm_module = import_project_module(lstm_dir, "lstm_generation")
    generator_class = getattr(lstm_module, "LSTMRhymingPoetryGenerator")

    with working_directory(lstm_dir):
        if use_cache:
            restore_lstm_cache(cache_dir, lstm_dir, model_name="poet")

        generator = generator_class(model_name="poet")
        full_ok = generator.load_and_train(
            "poems_clean.txt",
            epochs=full_epochs,
            batch_size=32,
            max_vocab_size=10000,
            max_training_lines=20000,
        )
        if not full_ok:
            print("Полная LSTM-модель не обучилась.")
            return {"ok": False, "poem": [], "generator": generator, "metrics": {}}

        if use_cache:
            save_lstm_cache(cache_dir, lstm_dir, model_name="poet")

        print("Начинаем итоговую генерацию стихотворения через LSTM...")
        poem = generate_poem_with_retries(
            generator,
            lines=poem_lines,
            rhyme_scheme=rhyme_scheme,
            start_line=start_line,
            attempts=demo_retries,
            progress_label="LSTM demo",
            show_progress=True,
            temperature=0.65,
            free_line_candidates=free_line_candidates,
            rhyme_candidates=rhyme_candidates,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
        )
        generator.display_poem(poem)
        generator.display_metrics()
        return {
            "ok": True,
            "poem": poem,
            "generator": generator,
            "metrics": generator.metrics.get_statistics(),
        }


def build_llm_assistant(
    project_dir: Path,
    provider: str,
    model_name: Optional[str],
) -> Dict[str, Any]:
    print_section("LLM-модуль")
    ensure_on_syspath(project_dir)
    llm_tools = import_project_module(project_dir, "llm_poetry_tools")

    normalized_provider = (provider or "disabled").strip().lower()
    os.environ["LLM_PROVIDER"] = normalized_provider

    assistant = llm_tools.LLMPoetryAssistant(
        provider=normalized_provider,
        model_name=model_name,
    )
    print(assistant.status_message())
    return {"assistant": assistant, "tools": llm_tools}


def run_batch_experiment(
    markov_generator: Any,
    lstm_generator: Optional[Any],
    runs: int,
    poem_lines: int,
    rhyme_scheme: str,
    start_line: Optional[str],
    markov_retries: int,
    lstm_retries: int,
    free_line_candidates: int,
    rhyme_candidates: int,
    top_k: int,
    max_context_tokens: int,
    llm_tools: Any,
) -> Dict[str, Any]:
    print_section("Batch experiment")
    batch_poem_records: List[Dict[str, Any]] = []
    batch_formal_rows: List[Dict[str, Any]] = []

    for run_index in range(1, runs + 1):
        print(f"Генерация {run_index}/{runs}")
        generated = {
            "Markov": generate_poem_with_retries(
                markov_generator,
                lines=poem_lines,
                rhyme_scheme=rhyme_scheme,
                start_line=start_line,
                attempts=markov_retries,
            ),
        }
        if lstm_generator is not None:
            print(f"Генерируем LSTM-стихотворение для batch {run_index}/{runs}...")
            generated["LSTM"] = generate_poem_with_retries(
                lstm_generator,
                lines=poem_lines,
                rhyme_scheme=rhyme_scheme,
                start_line=start_line,
                attempts=lstm_retries,
                progress_label=f"LSTM batch {run_index}",
                show_progress=True,
                temperature=0.65,
                free_line_candidates=free_line_candidates,
                rhyme_candidates=rhyme_candidates,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
            )

        for model_name, poem in generated.items():
            if not poem:
                continue
            batch_poem_records.append(
                {
                    "run": run_index,
                    "model": model_name,
                    "poem": poem,
                }
            )
            row = {"run": run_index, "version": model_name}
            row.update(
                llm_tools.calculate_formal_metrics(poem, rhyme_scheme=rhyme_scheme)
            )
            batch_formal_rows.append(row)

    return {
        "poem_records": batch_poem_records,
        "formal_rows": batch_formal_rows,
    }


def run_batch_llm(
    llm: Any,
    llm_tools: Any,
    batch_poem_records: Sequence[Dict[str, Any]],
    experiment_llm_runs: int,
    poem_lines: int,
    rhyme_scheme: str,
    start_line: Optional[str],
    enable_llm_evaluation: bool,
) -> Dict[str, Any]:
    print_section("Batch LLM")
    batch_llm_rows: List[Dict[str, Any]] = []
    batch_llm_records: List[Dict[str, Any]] = []

    if not llm.is_available():
        print("LLM недоступен, batch-редактирование пропущено.")
        return {"rows": batch_llm_rows, "records": batch_llm_records}

    records_for_llm = [
        record
        for record in batch_poem_records
        if int(record.get("run", 0)) <= experiment_llm_runs
    ]

    for index, record in enumerate(records_for_llm, start=1):
        print(
            f"LLM обработка {index}/{len(records_for_llm)}: "
            f"{record['model']} run {record['run']}"
        )
        edited = llm.edit_poem(
            record["poem"],
            start_line=start_line,
            rhyme_scheme=rhyme_scheme,
            target_lines=poem_lines,
        )

        versions = {
            record["model"]: record["poem"],
            f"{record['model']} + LLM": edited,
        }

        for version_name, poem in versions.items():
            metrics = llm_tools.calculate_formal_metrics(poem, rhyme_scheme=rhyme_scheme)
            evaluation = (
                llm.evaluate_poem(poem, start_line=start_line)
                if enable_llm_evaluation
                else {}
            )
            row = {"run": record["run"], "version": version_name}
            row.update(metrics)
            for field in [
                "semantic_coherence",
                "grammar",
                "theme_consistency",
                "poetic_quality",
                "overall",
            ]:
                row[field] = evaluation.get(field, 0)
            row["comment"] = evaluation.get("comment", "")
            batch_llm_rows.append(row)
            batch_llm_records.append(
                {
                    "run": record["run"],
                    "source_model": record["model"],
                    "version": version_name,
                    "stage": "llm_edited" if "+ LLM" in version_name else "raw",
                    "poem": poem,
                    "metrics": metrics,
                    "evaluation": evaluation,
                }
            )

    return {"rows": batch_llm_rows, "records": batch_llm_records}


def export_reports(
    llm_tools: Any,
    llm: Any,
    llm_model_name: str,
    reports_dir: Path,
    poem_versions: Dict[str, Sequence[str]],
    formal_rows: Sequence[Dict[str, Any]],
    llm_evaluation_rows: Sequence[Dict[str, Any]],
    batch_poem_records: Sequence[Dict[str, Any]],
    batch_llm_records: Sequence[Dict[str, Any]],
    start_line: Optional[str],
    rhyme_scheme: str,
    poem_lines: int,
    experiment_runs: int,
    experiment_llm_runs: int,
    markov_generation_retries: int,
    lstm_demo_retries: int,
    lstm_batch_retries: int,
) -> Dict[str, Any]:
    print_section("Отчёты")
    report_rows: List[Dict[str, Any]] = []
    demo_formal_map = {row["name"]: row for row in formal_rows}
    demo_eval_map = {row["name"]: row for row in llm_evaluation_rows}

    extra_settings = {
        "experiment_runs": experiment_runs,
        "experiment_llm_runs": experiment_llm_runs,
        "markov_generation_retries": markov_generation_retries,
        "lstm_demo_retries": lstm_demo_retries,
        "lstm_batch_retries": lstm_batch_retries,
    }

    for version_name, poem in poem_versions.items():
        report_rows.append(
            llm_tools.build_poem_report_row(
                version=version_name,
                poem=poem,
                report_scope="demo",
                run=1,
                start_line=start_line or "",
                rhyme_scheme=rhyme_scheme,
                requested_lines=poem_lines,
                llm_model_name=llm_model_name if llm.is_available() else "",
                metrics=demo_formal_map.get(version_name),
                evaluation=demo_eval_map.get(version_name),
                extra=extra_settings,
            )
        )

    batch_eval_keys = {
        (record.get("run"), record.get("version")) for record in batch_llm_records
    }
    for record in batch_poem_records:
        key = (record.get("run"), record.get("model"))
        if key in batch_eval_keys:
            continue
        report_rows.append(
            llm_tools.build_poem_report_row(
                version=record["model"],
                poem=record["poem"],
                report_scope="batch",
                run=record["run"],
                source_model=record["model"],
                start_line=start_line or "",
                rhyme_scheme=rhyme_scheme,
                requested_lines=poem_lines,
                metrics=llm_tools.calculate_formal_metrics(
                    record["poem"], rhyme_scheme=rhyme_scheme
                ),
                extra=extra_settings,
            )
        )

    for record in batch_llm_records:
        report_rows.append(
            llm_tools.build_poem_report_row(
                version=record["version"],
                poem=record["poem"],
                report_scope="batch",
                run=record["run"],
                source_model=record.get("source_model"),
                stage=record.get("stage"),
                start_line=start_line or "",
                rhyme_scheme=rhyme_scheme,
                requested_lines=poem_lines,
                llm_model_name=llm_model_name if llm.is_available() else "",
                metrics=record.get("metrics"),
                evaluation=record.get("evaluation"),
                extra=extra_settings,
            )
        )

    report_bundle = llm_tools.export_report_bundle(
        report_rows,
        output_dir=reports_dir,
        prefix=DEFAULT_REPORT_PREFIX,
    )
    print(f"Detailed report: {report_bundle['detailed_path']}")
    print(f"Summary report: {report_bundle['summary_path']}")
    if report_bundle.get("detailed_csv_path") and report_bundle["detailed_csv_path"] != report_bundle["detailed_path"]:
        print(f"Detailed CSV fallback: {report_bundle['detailed_csv_path']}")
    if report_bundle.get("summary_csv_path") and report_bundle["summary_csv_path"] != report_bundle["summary_path"]:
        print(f"Summary CSV fallback: {report_bundle['summary_csv_path']}")
    return report_bundle


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the poetry project locally without Google Colab."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="Repository root. Defaults to the directory of this script.",
    )
    parser.add_argument(
        "--dataset-txt",
        default="",
        help="Path to a ready local poems_clean.txt. If omitted, the script downloads and prepares RIFMA.",
    )
    parser.add_argument(
        "--dataset-json-url",
        default=DATASET_JSON_URL,
        help="URL of the source JSON dataset used when --dataset-txt is not provided.",
    )
    parser.add_argument(
        "--start-line",
        default=DEFAULT_START_LINE,
        help="Fixed first line for all generators.",
    )
    parser.add_argument(
        "--rhyme-scheme",
        default=DEFAULT_RHYME_SCHEME,
        help="Rhyme scheme like AABB, ABAB, ABBA, AAAA.",
    )
    parser.add_argument(
        "--poem-lines",
        type=int,
        default=DEFAULT_POEM_LINES,
        help="Requested number of lines in each poem.",
    )
    parser.add_argument(
        "--quick-epochs",
        type=int,
        default=1,
        help="Epochs for the quick LSTM smoke test.",
    )
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=20,
        help="Epochs for the main LSTM model.",
    )
    parser.add_argument(
        "--experiment-runs",
        type=int,
        default=2,
        help="Number of batch comparison generations.",
    )
    parser.add_argument(
        "--experiment-llm-runs",
        type=int,
        default=2,
        help="How many batch runs should also be processed by the LLM.",
    )
    parser.add_argument(
        "--llm-provider",
        default=os.getenv("LLM_PROVIDER", "disabled"),
        choices=["disabled", "groq", "gemini", "ollama", "transformers"],
        help="Optional local LLM backend.",
    )
    parser.add_argument(
        "--llm-model-name",
        default="",
        help="Override the default model name for the selected LLM backend.",
    )
    parser.add_argument(
        "--enable-llm-evaluation",
        action="store_true",
        help="Run the additional LLM scoring step.",
    )
    parser.add_argument(
        "--skip-llm-editing",
        action="store_true",
        help="Skip LLM editing even if the provider is configured.",
    )
    parser.add_argument(
        "--skip-quick-lstm",
        action="store_true",
        help="Skip the quick LSTM smoke test.",
    )
    parser.add_argument(
        "--skip-full-lstm",
        action="store_true",
        help="Skip the main LSTM training and generation.",
    )
    parser.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip the multi-run experiment and only generate the demo poems.",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Only download/prepare the dataset and copy poems_clean.txt into the project folders.",
    )
    parser.add_argument(
        "--no-lstm-cache",
        action="store_true",
        help="Do not restore or save the local LSTM cache.",
    )
    parser.add_argument(
        "--force-dataset-refresh",
        action="store_true",
        help="Force re-download of the JSON dataset and rebuild poems_clean.txt.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use lighter defaults for quicker local debugging.",
    )
    return parser


def apply_fast_mode(args: argparse.Namespace) -> None:
    if not args.fast:
        return
    args.quick_epochs = 1
    args.full_epochs = min(args.full_epochs, 5)
    args.experiment_runs = min(args.experiment_runs, 1)
    args.experiment_llm_runs = min(args.experiment_llm_runs, 1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    apply_fast_mode(args)
    configure_console_output()

    project_dir = Path(args.project_dir).resolve()
    data_dir = project_dir / "data"
    cache_dir = project_dir / "cache" / "poetry_lstm_cache"
    reports_dir = project_dir / "reports"
    dataset_json_path = data_dir / "rifma_dataset.json"
    dataset_txt_path = data_dir / "poems_clean.txt"

    validate_project_layout(project_dir)

    print_section("Локальный запуск проекта")
    print(f"Project dir: {project_dir}")
    print(f"Dataset dir: {data_dir}")
    print(f"Reports dir: {reports_dir}")
    print(f"LLM provider: {args.llm_provider}")

    if lstm_requested(args):
        ensure_supported_python()
        print_local_hardware_summary()
        print_windows_tensorflow_gpu_note()

    if args.dataset_txt:
        prepare_dataset_from_custom_text(Path(args.dataset_txt).resolve(), dataset_txt_path)
    else:
        download_dataset_json(
            dataset_url=args.dataset_json_url,
            dataset_json_path=dataset_json_path,
            force=args.force_dataset_refresh,
        )
        if args.force_dataset_refresh or not dataset_txt_path.exists():
            build_poems_clean_file(dataset_json_path, dataset_txt_path)
        else:
            print(f"poems_clean.txt уже готов: {dataset_txt_path}")

    copy_dataset_to_model_dirs(project_dir, dataset_txt_path)
    validate_python_files(project_dir)

    if args.prepare_data_only:
        print("\nПодготовка данных завершена. Модели не запускались.")
        return 0

    poem_versions: Dict[str, Sequence[str]] = {}
    formal_rows: List[Dict[str, Any]] = []
    llm_evaluation_rows: List[Dict[str, Any]] = []
    batch_poem_records: List[Dict[str, Any]] = []
    batch_llm_records: List[Dict[str, Any]] = []

    markov_result = run_markov_demo(
        project_dir=project_dir,
        poem_lines=args.poem_lines,
        rhyme_scheme=args.rhyme_scheme,
        start_line=args.start_line,
        retries=30,
    )
    if markov_result["ok"] and markov_result["poem"]:
        poem_versions["Markov"] = markov_result["poem"]

    if not args.skip_quick_lstm:
        run_quick_lstm_check(
            project_dir=project_dir,
            rhyme_scheme=args.rhyme_scheme,
            start_line=args.start_line,
            epochs=args.quick_epochs,
        )

    full_lstm_result = {"ok": False, "poem": [], "generator": None, "metrics": {}}
    if not args.skip_full_lstm:
        full_lstm_result = run_full_lstm_demo(
            project_dir=project_dir,
            cache_dir=cache_dir,
            use_cache=not args.no_lstm_cache,
            poem_lines=args.poem_lines,
            rhyme_scheme=args.rhyme_scheme,
            start_line=args.start_line,
            full_epochs=args.full_epochs,
            demo_retries=5,
            free_line_candidates=12,
            rhyme_candidates=20,
            top_k=40,
            max_context_tokens=64,
        )
        if full_lstm_result["ok"] and full_lstm_result["poem"]:
            poem_versions["LSTM"] = full_lstm_result["poem"]

    llm_bundle = build_llm_assistant(
        project_dir=project_dir,
        provider=args.llm_provider,
        model_name=args.llm_model_name or None,
    )
    llm = llm_bundle["assistant"]
    llm_tools = llm_bundle["tools"]
    effective_llm_model_name = getattr(llm.config, "model_name", "") if llm else ""

    if llm.is_available() and not args.skip_llm_editing:
        if "Markov" in poem_versions:
            edited_markov = llm.edit_poem(
                poem_versions["Markov"],
                start_line=args.start_line,
                rhyme_scheme=args.rhyme_scheme,
                target_lines=args.poem_lines,
            )
            poem_versions["Markov + LLM"] = edited_markov
            llm_tools.print_poem("Markov + LLM", edited_markov)
        if "LSTM" in poem_versions:
            edited_lstm = llm.edit_poem(
                poem_versions["LSTM"],
                start_line=args.start_line,
                rhyme_scheme=args.rhyme_scheme,
                target_lines=args.poem_lines,
            )
            poem_versions["LSTM + LLM"] = edited_lstm
            llm_tools.print_poem("LSTM + LLM", edited_lstm)
    elif args.skip_llm_editing:
        print("\nLLM-редактирование отключено флагом --skip-llm-editing.")
    else:
        print("\nLLM-редактирование пропущено: модуль недоступен.")

    formal_rows = llm_tools.build_formal_metrics_table(
        poem_versions,
        rhyme_scheme=args.rhyme_scheme,
    )
    print_section("Формальные метрики")
    for row in formal_rows:
        print(row)

    if llm.is_available() and poem_versions and args.enable_llm_evaluation:
        print_section("LLM-оценка")
        llm_evaluation_rows = llm.evaluate_versions(
            poem_versions,
            start_line=args.start_line,
        )
        for row in llm_evaluation_rows:
            print(row)
    elif args.enable_llm_evaluation:
        print("\nLLM-оценка пропущена: модуль недоступен.")

    if not args.skip_batch and markov_result["ok"]:
        batch_result = run_batch_experiment(
            markov_generator=markov_result["generator"],
            lstm_generator=full_lstm_result["generator"] if full_lstm_result["ok"] else None,
            runs=args.experiment_runs,
            poem_lines=args.poem_lines,
            rhyme_scheme=args.rhyme_scheme,
            start_line=args.start_line,
            markov_retries=30,
            lstm_retries=8,
            free_line_candidates=12,
            rhyme_candidates=20,
            top_k=40,
            max_context_tokens=64,
            llm_tools=llm_tools,
        )
        batch_poem_records = batch_result["poem_records"]

        if llm.is_available() and not args.skip_llm_editing:
            batch_llm_result = run_batch_llm(
                llm=llm,
                llm_tools=llm_tools,
                batch_poem_records=batch_poem_records,
                experiment_llm_runs=args.experiment_llm_runs,
                poem_lines=args.poem_lines,
                rhyme_scheme=args.rhyme_scheme,
                start_line=args.start_line,
                enable_llm_evaluation=args.enable_llm_evaluation,
            )
            batch_llm_records = batch_llm_result["records"]
    elif args.skip_batch:
        print("\nBatch experiment пропущен флагом --skip-batch.")

    report_bundle = export_reports(
        llm_tools=llm_tools,
        llm=llm,
        llm_model_name=effective_llm_model_name,
        reports_dir=reports_dir,
        poem_versions=poem_versions,
        formal_rows=formal_rows,
        llm_evaluation_rows=llm_evaluation_rows,
        batch_poem_records=batch_poem_records,
        batch_llm_records=batch_llm_records,
        start_line=args.start_line,
        rhyme_scheme=args.rhyme_scheme,
        poem_lines=args.poem_lines,
        experiment_runs=args.experiment_runs,
        experiment_llm_runs=args.experiment_llm_runs,
        markov_generation_retries=30,
        lstm_demo_retries=5,
        lstm_batch_retries=8,
    )

    print_section("Готово")
    print(f"Подробный отчёт: {report_bundle['detailed_path']}")
    print(f"Сводный отчёт: {report_bundle['summary_path']}")
    if report_bundle.get("detailed_csv_path") and report_bundle["detailed_csv_path"] != report_bundle["detailed_path"]:
        print(f"CSV-копия подробного отчёта: {report_bundle['detailed_csv_path']}")
    if report_bundle.get("summary_csv_path") and report_bundle["summary_csv_path"] != report_bundle["summary_path"]:
        print(f"CSV-копия сводного отчёта: {report_bundle['summary_csv_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
