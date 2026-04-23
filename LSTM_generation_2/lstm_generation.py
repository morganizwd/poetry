
#-------ДОБАВИЛА МЕТРИКИ И МЕХАНИЗМ ВНИМАНИЯ-------------

import re, json, random, numpy as np, tensorflow as tf, os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Input,
    Dropout,
    Layer,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from collections import defaultdict, Counter
import statistics

# МЕХАНИЗМ ВНИМАНИЯ БАХДАНАУ (Additive Attention)

@tf.keras.utils.register_keras_serializable(package="poetry")
class BahdanauAttention(Layer):
    """Простой causal additive attention для языковой модели"""

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.supports_masking = True
        self.W_query = None
        self.W_value = None
        self.V = None
        self._build_input_shape = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self._build_input_shape = input_shape

        self.W_query = Dense(self.units, use_bias=False, name="query_projection")
        self.W_value = Dense(self.units, use_bias=False, name="value_projection")
        self.V = Dense(1, use_bias=False, name="score_projection")

        self.W_query.build(input_shape)
        self.W_value.build(input_shape)
        self.V.build(tf.TensorShape([None, None, None, self.units]))
        super(BahdanauAttention, self).build(input_shape)

    def call(self, values, mask=None):
        query = self.W_query(values)
        value = self.W_value(values)

        score = self.V(
            tf.nn.tanh(tf.expand_dims(query, axis=2) + tf.expand_dims(value, axis=1))
        )
        score = tf.squeeze(score, axis=-1)

        seq_len = tf.shape(values)[1]
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0
        )
        large_negative = tf.cast(-1e9, score.dtype)
        score += (
            1.0 - tf.cast(causal_mask[tf.newaxis, :, :], score.dtype)
        ) * large_negative

        if mask is not None:
            value_mask = tf.cast(mask[:, tf.newaxis, :], score.dtype)
            score += (1.0 - value_mask) * large_negative

        attention_weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(attention_weights, values)

        if mask is not None:
            context *= tf.cast(mask[:, :, tf.newaxis], context.dtype)

        return context

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

    def get_build_config(self):
        if self._build_input_shape is None:
            return {}
        return {"input_shape": self._build_input_shape.as_list()}

    def build_from_config(self, config):
        input_shape = config.get("input_shape")
        if input_shape:
            self.build(input_shape)


# КЛАСС ДЛЯ ПОИСКА РИФМ

class RhymeSearch:
    """Поиск рифм по окончаниям"""

    def __init__(self):
        self.rhymes_dict = defaultdict(list)

    def train(self, words):
        """обучение словаря"""
        for word in words:
            clean = self._clean_word(word)
            if len(clean) < 2:
                continue
            if len(clean) >= 3:
                suffix3 = clean[-3:]
                if word not in self.rhymes_dict[suffix3]:
                    self.rhymes_dict[suffix3].append(word)
            suffix2 = clean[-2:]
            if word not in self.rhymes_dict[suffix2]:
                self.rhymes_dict[suffix2].append(word)

    def _clean_word(self, word):
        """очистка слова"""
        return re.sub(r"[^\w\-]", "", word.lower()).strip()

    def give_rhyme(self, word, exclude_words=None):
        """поиск рифмы"""
        if exclude_words is None:
            exclude_words = set()
        clean = self._clean_word(word)
        if len(clean) < 2:
            return None
        candidates = []
        if len(clean) >= 3:
            suffix3 = clean[-3:]
            candidates = [
                w
                for w in self.rhymes_dict.get(suffix3, [])
                if w not in exclude_words and self._clean_word(w) != clean
            ]
        if len(candidates) < 3:
            suffix2 = clean[-2:]
            candidates.extend(
                [
                    w
                    for w in self.rhymes_dict.get(suffix2, [])
                    if w not in exclude_words and self._clean_word(w) != clean
                ]
            )
        candidates = [w for w in set(candidates) if self._clean_word(w) != clean]
        return random.choice(candidates) if candidates else None

    def save_json(self, path):
        """сохранение словаря"""
        data = {k: v for k, v in self.rhymes_dict.items() if len(v) > 1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# КЛАСС ДЛЯ СБОРА МЕТРИК

class MetricsCollector:
    """Сбор и расчет метрик качества генерации"""

    def __init__(self):
        self.data = {
            'rhyme_success': [],
            'unique_rate': [],
            'rhyme_quality': [],
            'vowels': [],
            'lengths': [],
            'unique_words': []
        }

    def _safe_stat(self, values, func, default=0):
        """Безопасное вычисление статистики"""
        return func(values) if values else default

    def add_generation(self, rhyme_success, unique_rate, rhyme_quality=None):
        """Добавление метрик одного стихотворения"""
        self.data['rhyme_success'].append(rhyme_success)
        self.data['unique_rate'].append(unique_rate)
        if rhyme_quality is not None:
            self.data['rhyme_quality'].append(rhyme_quality)

    def add_line_metrics(self, line):
        """Добавление метрик одной строки"""
        vowel_count = sum(1 for c in line.lower() if c in "аеёиоуыэюя")
        self.data['vowels'].append(vowel_count)
        self.data['lengths'].append(len(line.split()))

    def add_poem_metrics(self, poem):
        """Добавление метрик стихотворения"""
        words = {w for line in poem for w in re.findall(r'[а-яё]+', line.lower())}
        self.data['unique_words'].append(len(words))

    def calculate_rhyme_quality(self, line1, line2):
        """Оценка качества рифмы (0-1)"""
        words1 = re.findall(r'[а-яё]+', line1.lower())
        words2 = re.findall(r'[а-яё]+', line2.lower())
        if not words1 or not words2:
            return 0

        last1, last2 = words1[-1], words2[-1]
        if len(last1) < 3 or len(last2) < 3:
            return 0.5

        for i in range(3, 0, -1):
            if last1[-i:] == last2[-i:]:
                return {3: 1.0, 2: 0.8, 1: 0.5}[i]
        return 0.2

    def get_statistics(self):
        """Вычисление статистических показателей"""
        d = self.data
        stats = {
            'avg_rhyme_success': self._safe_stat(d['rhyme_success'], lambda x: round(statistics.mean(x) * 100, 1)),
            'avg_rhyme_quality': self._safe_stat(d['rhyme_quality'], lambda x: round(statistics.mean(x) * 100, 1)),
            'avg_unique_rate': self._safe_stat(d['unique_rate'], lambda x: round(statistics.mean(x) * 100, 1)),
            'avg_unique_words_per_poem': self._safe_stat(d['unique_words'], lambda x: round(statistics.mean(x), 1)),
            'avg_vowels_per_line': self._safe_stat(d['vowels'], lambda x: round(statistics.mean(x), 1)),
            'min_vowels': self._safe_stat(d['vowels'], min, 0),
            'max_vowels': self._safe_stat(d['vowels'], max, 0),
            'avg_words_per_line': self._safe_stat(d['lengths'], lambda x: round(statistics.mean(x), 1)),
            'min_words': self._safe_stat(d['lengths'], min, 0),
            'max_words': self._safe_stat(d['lengths'], max, 0),
        }
        return stats

    def print_summary(self):
        """Вывод сводной статистики"""
        stats = self.get_statistics()
        print("\nМетрики качества:")
        print(f"   • Успешность рифмовки: {stats.get('avg_rhyme_success', 0):.1f}%")
        print(f"   • Среднее качество рифмы: {stats.get('avg_rhyme_quality', 0):.1f}%")
        print(f"   • Уникальность строк: {stats.get('avg_unique_rate', 0):.1f}%")
        print(f"   • Уникальных слов на стих: {stats.get('avg_unique_words_per_poem', 0):.1f}")

        print("\nМетрики строк:")
        print(f"   • Средняя длина строки: {stats.get('avg_words_per_line', 0):.1f} слов")
        print(f"   • Среднее количество гласных: {stats.get('avg_vowels_per_line', 0):.1f}")
        print(f"   • Диапазон слов: {stats.get('min_words', 0)}-{stats.get('max_words', 0)}")
        print(f"   • Диапазон гласных: {stats.get('min_vowels', 0)}-{stats.get('max_vowels', 0)}")


# ГЕНЕРАТОР СТИХОВ С LSTM

class LSTMRhymingPoetryGenerator:
    """Генератор стихов с контролем рифмы и механизмом внимания Бахданау"""

    MODEL_VERSION = 6

    def __init__(self, model_name="poet"):
        self.model_name = model_name
        self.vowels = "аеёиоуыэюя"
        self.model = None
        self.rhyme_search = None
        self.vocab = self._create_base_vocab()
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.VOCAB_SIZE = len(self.vocab)
        self.original_lines = set()
        self.poems = []
        self.metrics = MetricsCollector()

    def _create_base_vocab(self):
        return {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<line>": 4,
        }

    def _reset_vocab(self):
        self.vocab = self._create_base_vocab()
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.VOCAB_SIZE = len(self.vocab)

    def _tokenize_russian(self, text):
        """Токенизация"""
        return re.findall(r"[а-яё]+", text.lower())

    def _extract_last_word(self, line):
        """Извлекает последнее слово из строки"""
        words = self._tokenize_russian(line)
        return words[-1] if words else None

    def _count_vowels(self, text):
        """Подсчёт гласных"""
        return sum(1 for c in text.lower() if c in self.vowels)

    def _is_unique_line(self, line, threshold=0.85):
        """Проверка уникальности"""
        if not line:
            return False
        words_list = self._tokenize_russian(line)
        words = set(words_list)
        if len(words) < 3:
            return True
        for orig in self.original_lines:
            orig_words_list = self._tokenize_russian(orig)
            orig_words = set(orig_words_list)
            if words_list == orig_words_list:
                return False
            if words and len(words & orig_words) / len(words) >= threshold:
                return False
        return True

    def _is_good_generated_line(self, words, min_words):
        """Простой фильтр строк, которые выглядят явно неудачно."""
        if len(words) < min_words:
            return False

        bad_end_words = {
            "и", "а", "но", "или", "что", "как", "не", "ни", "же", "ли", "бы",
            "в", "во", "на", "к", "ко", "с", "со", "у", "о", "об", "от", "до",
            "по", "за", "из", "над", "под", "для", "без", "при", "меж",
        }
        stop_words = bad_end_words | {
            "я", "ты", "он", "она", "мы", "вы", "они", "это", "этот", "тот",
            "все", "всё", "мне", "тебе", "его", "ее", "её", "их", "мой",
            "твой", "свой", "был", "была", "были", "будет", "есть",
        }

        if words[-1] in bad_end_words:
            return False

        for prev_word, word in zip(words, words[1:]):
            if prev_word == word:
                return False

        if max(words.count(word) for word in set(words)) > 2:
            return False

        content_words = [word for word in words if word not in stop_words]
        if len(content_words) < 2:
            return False

        stop_ratio = 1 - len(content_words) / len(words)
        if stop_ratio > 0.75:
            return False

        if not any(len(word) >= 4 for word in words):
            return False

        return True

    def _postprocess_line(self, words, min_words):
        """Постобработка сгенерированной строки"""
        if len(words) < min_words:
            return None
        normal_line = " ".join(words)
        if len(normal_line) < 10 or self._count_vowels(normal_line) < 2:
            return None
        if not self._is_good_generated_line(words, min_words):
            return None
        if not self._is_unique_line(normal_line, threshold=0.85):
            return None
        if normal_line and normal_line[0].isalpha():
            normal_line = normal_line[0].upper() + normal_line[1:]
        return normal_line

    def load_and_train(
        self,
        filepath="poems_clean.txt",
        epochs=20,
        batch_size=32,
        max_vocab_size=12000,
        max_line_words=12,
        max_training_lines=20000,
        max_sequence_len=80,
    ):
        """обучение моделей"""

        model_path = f"{self.model_name}_model.keras"
        metadata_path = f"{self.model_name}_metadata.json"
        rhymes_path = f"{self.model_name}_rhymes.json"

        if (
            os.path.exists(model_path)
            and os.path.exists(metadata_path)
            and os.path.exists(rhymes_path)
        ):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if metadata.get("model_version") == self.MODEL_VERSION:
                print(f"Найдена сохранённая модель, загружаем...")
                self.vocab = metadata["vocab"]
                self.id2word = {int(k): v for k, v in metadata["id2word"].items()}
                self.VOCAB_SIZE = metadata["vocab_size"]

                try:
                    self.model = load_model(
                        model_path,
                        custom_objects={'BahdanauAttention': BahdanauAttention},
                        compile=False,
                    )
                    self._compile_model(self.model)
                except Exception as exc:
                    print("Не удалось загрузить сохранённую модель.")
                    print(f"Причина: {exc}")
                    print(
                        "Кэш модели несовместим с текущей версией Keras/кода. "
                        "Обучаем заново и затем перезапишем кэш."
                    )
                    self.model = None
                    self._reset_vocab()
                else:
                    self.rhyme_search = RhymeSearch()
                    with open(rhymes_path, "r", encoding="utf-8") as f:
                        rhymes_data = json.load(f)
                        self.rhyme_search.rhymes_dict = defaultdict(list, rhymes_data)

                    if os.path.exists(filepath):
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        self.original_lines = {
                            line.strip()
                            for line in content.split("\n")
                            if line.strip() and len(line.strip()) > 10
                        }

                        self.poems = [
                            p.strip() for p in content.split("\n\n") if p.strip()
                        ]
                    else:
                        print(
                            "Файл датасета не найден, "
                            "проверка копирования строк пропущена."
                        )

                    print("Модель загружена")
                    return True

            print("Сохранённая модель создана старой версией кода. Обучаем заново.")

        if not os.path.exists(filepath):
            print(f"Файл '{filepath}' не найден!")
            return False

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        self.original_lines = {
            line.strip()
            for line in content.split("\n")
            if line.strip() and len(line.strip()) > 10
        }

        poem_token_lines, word_counts = [], Counter()
        poems = [p.strip() for p in content.split("\n\n") if p.strip()]
        self.poems = poems

        for poem in poems:
            current_poem = []
            lines = [l.strip() for l in poem.split("\n") if l.strip()]
            for line in lines:
                tokens = self._tokenize_russian(line)
                vowel_count = self._count_vowels(line)
                if 3 <= len(tokens) <= max_line_words and vowel_count >= 2:
                    current_poem.append(tokens)
                    word_counts.update(w for w in tokens if len(w) >= 2)
            if len(current_poem) >= 2:
                poem_token_lines.append(current_poem)

        total_lines = sum(len(poem) for poem in poem_token_lines)
        if not poem_token_lines:
            print("Не найдено подходящих строк для обучения.")
            return False

        if total_lines > max_training_lines:
            random.shuffle(poem_token_lines)
            selected_poems, selected_lines = [], 0
            for poem in poem_token_lines:
                if selected_lines >= max_training_lines:
                    break
                selected_poems.append(poem)
                selected_lines += len(poem)
            poem_token_lines = selected_poems
            total_lines = selected_lines

        print(f"Статистика датасета:")
        print(f"   Стихотворений для обучения: {len(poem_token_lines)}")
        print(f"   Строк для обучения: {total_lines}")
        print(f"   Уникальных слов до ограничения словаря: {len(word_counts)}")

        available_vocab_slots = max_vocab_size - len(self.vocab)
        for word, _ in word_counts.most_common(available_vocab_slots):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.id2word = {v: k for k, v in self.vocab.items()}
        self.VOCAB_SIZE = len(self.vocab)
        print(f"   Размер словаря модели: {self.VOCAB_SIZE}")

        self.rhyme_search = RhymeSearch()
        model_words = [
            w for w in self.vocab
            if w not in ["<pad>", "<unk>", "<bos>", "<eos>", "<line>"]
        ]
        self.rhyme_search.train(model_words)
        self.rhyme_search.save_json(f"{self.model_name}_rhymes.json")

        encoded_sequences = []
        for poem in poem_token_lines:
            ids = [self.vocab["<bos>"]]
            for line_index, tokens in enumerate(poem):
                ids.extend(self.vocab.get(w, self.vocab["<unk>"]) for w in tokens)
                if line_index < len(poem) - 1:
                    ids.append(self.vocab["<line>"])
            ids.append(self.vocab["<eos>"])

            for start in range(0, len(ids) - 1, max_sequence_len):
                chunk = ids[start : start + max_sequence_len + 1]
                if len(chunk) >= 4:
                    encoded_sequences.append(chunk)

        if not encoded_sequences:
            print("Не удалось подготовить последовательности для LSTM.")
            return False

        random.shuffle(encoded_sequences)
        batches = []
        for i in range(0, len(encoded_sequences), batch_size):
            batch = encoded_sequences[i : i + batch_size]
            maxlen = max(len(x) for x in batch)
            x, y = [], []
            for seq in batch:
                pad = [0] * (maxlen - len(seq))
                x.append(seq[:-1] + pad)
                y.append(seq[1:] + pad)
            batches.append(
                (np.array(x, dtype=np.int32), np.array(y, dtype=np.int32))
            )

        print(f"\nОбучение модели ({epochs} эпох, batch_size={batch_size})...")
        print(f"   Батчей в одной эпохе: {len(batches)}")
        self.model = self._build_model()

        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            print(f"\nEpoch {epoch + 1}/{epochs}")
            for batch_index, (x_batch, y_batch) in enumerate(batches, start=1):
                if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(batches):
                    print(f"   batch {batch_index}/{len(batches)}...", flush=True)
                sample_weight = (y_batch != self.vocab["<pad>"]).astype("float32")
                result = self.model.train_on_batch(
                    x_batch, y_batch, sample_weight=sample_weight
                )
                loss_value = result[0] if isinstance(result, (list, tuple)) else result
                total_loss += float(loss_value)
                total_batches += 1
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model.save(f"{self.model_name}_model.keras")
        metadata = {
            "model_version": self.MODEL_VERSION,
            "vocab": self.vocab,
            "id2word": self.id2word,
            "vocab_size": self.VOCAB_SIZE,
        }
        with open(f"{self.model_name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return True

    def _compile_model(self, model):
        model.compile(
            optimizer=Adam(learning_rate=3e-4),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy'],
        )

    def _build_model(
        self, embedding_dim=128, hidden_size=128, num_layers=1,
        dropout=0.2, attention_units=64
    ):
        """
        Создание архитектуры модели с механизмом внимания Бахданау
        """

        inputs = Input(shape=(None,), name='input_sequence', dtype='int32')
        x = Embedding(
            self.VOCAB_SIZE, embedding_dim, mask_zero=True, name='embedding'
        )(inputs)

        prev_output = x

        for i in range(num_layers):
            layer_dropout = dropout if i < num_layers - 1 else 0.0

            prev_output = LSTM(
                hidden_size,
                return_sequences=True,
                dropout=layer_dropout,
                recurrent_dropout=0.0,
                name=f'lstm_{i+1}'
            )(prev_output)

        attention = BahdanauAttention(attention_units, name='bahdanau_attention')
        attention_output = attention(prev_output)

        combined = Concatenate(name='attention_concat')(
            [prev_output, attention_output]
        )
        combined = Dense(hidden_size, activation='tanh', name='attention_combine')(
            combined
        )
        combined = Dropout(dropout)(combined)

        outputs = Dense(self.VOCAB_SIZE, activation='linear', name='logits')(combined)

        model = Model(inputs=inputs, outputs=outputs, name='lstm_attention_poet')
        self._compile_model(model)
        return model

    def _sample_next_token(
        self,
        logits,
        temperature=0.65,
        allow_eos=True,
        top_k=40,
        extra_blocked_tokens=None,
    ):
        """Выбор следующего токена без служебных токенов в тексте строки."""
        temperature = max(float(temperature), 1e-6)
        probs = tf.nn.softmax(logits / temperature).numpy().astype(np.float64)

        blocked_tokens = [
            self.vocab["<pad>"],
            self.vocab["<unk>"],
            self.vocab["<bos>"],
        ]
        if not allow_eos:
            blocked_tokens.append(self.vocab["<eos>"])
            blocked_tokens.append(self.vocab["<line>"])
        if extra_blocked_tokens:
            blocked_tokens.extend(extra_blocked_tokens)

        for token_id in blocked_tokens:
            if token_id is not None and 0 <= token_id < len(probs):
                probs[token_id] = 0.0

        positive_ids = np.flatnonzero(probs > 0)
        if top_k and len(positive_ids) > top_k:
            positive_probs = probs[positive_ids]
            keep_ids = positive_ids[np.argpartition(positive_probs, -top_k)[-top_k:]]
            filtered = np.zeros_like(probs)
            filtered[keep_ids] = probs[keep_ids]
            probs = filtered

        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return None

        probs /= total
        return int(np.random.choice(len(probs), p=probs))

    def _words_to_ids(self, words):
        """Перевод слов в id с учётом неизвестных слов."""
        return [self.vocab.get(word, self.vocab["<unk>"]) for word in words]

    def _trim_context(self, token_ids, max_context_tokens=80):
        """Ограничение контекста, чтобы attention не становился слишком тяжёлым."""
        return token_ids[-max_context_tokens:]

    def _prepare_generation_input(self, token_ids, max_context_tokens=60):
        """Подготовка контекста фиксированной длины для более быстрого инференса."""
        trimmed = self._trim_context(token_ids, max_context_tokens)
        pad_id = self.vocab["<pad>"]
        if len(trimmed) < max_context_tokens:
            trimmed = [pad_id] * (max_context_tokens - len(trimmed)) + trimmed
        return tf.convert_to_tensor([trimmed], dtype=tf.int32)

    def _generate_candidate_words(
        self,
        context_ids,
        min_words=4,
        max_words=9,
        temperature=0.65,
        top_k=30,
        max_context_tokens=60,
    ):
        """Генерация одной строки слева направо."""
        tokens = list(context_ids)
        words = []

        for _ in range(max_words * 2):
            input_seq = self._prepare_generation_input(tokens, max_context_tokens)
            logits = self.model(input_seq, training=False)[0, -1, :].numpy()
            next_token = self._sample_next_token(
                logits,
                temperature=temperature,
                allow_eos=len(words) >= min_words,
                top_k=top_k,
                extra_blocked_tokens=[tokens[-1] if tokens else None],
            )
            if next_token is None:
                break

            next_word = self.id2word.get(next_token, "<unk>")
            if next_word in ["<line>", "<eos>"]:
                break

            tokens.append(next_token)
            words.append(next_word)

            if len(words) >= max_words:
                break

        return words

    def _line_score(self, words, min_words):
        """Простая оценка качества строки без сложной лингвистики."""
        if not self._is_good_generated_line(words, min_words):
            return None

        line = " ".join(words)
        if not self._is_unique_line(line, threshold=0.85):
            return None

        unique_ratio = len(set(words)) / len(words)
        length_bonus = 1.0 - min(abs(len(words) - 7) * 0.08, 0.5)
        return unique_ratio + length_bonus

    def _generate_best_line(
        self,
        context_ids,
        target_word=None,
        candidates=16,
        min_words=4,
        max_words=9,
        temperature=0.65,
        top_k=30,
        max_context_tokens=60,
    ):
        """Генерация нескольких вариантов и выбор лучшего."""
        best_line, best_last_word, best_quality, best_score = None, None, 0, -1

        for _ in range(candidates):
            words = self._generate_candidate_words(
                context_ids,
                min_words=min_words,
                max_words=max_words,
                temperature=temperature,
                top_k=top_k,
                max_context_tokens=max_context_tokens,
            )
            score = self._line_score(words, min_words)
            if score is None:
                continue

            line = self._postprocess_line(words, min_words)
            if not line:
                continue

            last_word = self._extract_last_word(line)
            rhyme_quality = 0
            if target_word:
                rhyme_quality = self.metrics.calculate_rhyme_quality(
                    target_word, line
                )
                if rhyme_quality < 0.2:
                    continue
                score += rhyme_quality * 3

            if score > best_score:
                best_line = line
                best_last_word = last_word
                best_quality = rhyme_quality
                best_score = score

        if best_line:
            return best_line, best_last_word, best_quality
        return None, None, 0

    def _generate_line_with_rhyme(
        self,
        target_word,
        context_ids,
        max_attempts=20,
        min_words=4,
        max_words=9,
        temperature=0.65,
        top_k=30,
        max_context_tokens=60,
    ):
        """Генерация строки слева направо с последующим выбором рифмы."""
        if not self.model or not target_word:
            return None, None

        line, last_word, quality = self._generate_best_line(
            context_ids,
            target_word=target_word,
            candidates=max_attempts,
            min_words=min_words,
            max_words=max_words,
            temperature=temperature,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
        )
        if line and quality >= 0.5:
            return line, last_word

        fallback_line = self._generate_free_line(
            context_ids,
            max_attempts=max(6, max_attempts // 2),
            min_words=min_words,
            max_words=max_words,
            temperature=temperature,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
        )
        return fallback_line, None

    def _generate_free_line(
        self,
        context_ids,
        max_attempts=12,
        min_words=4,
        max_words=9,
        temperature=0.65,
        top_k=30,
        max_context_tokens=60,
    ):
        """Генерация свободной строки слева направо."""
        if not self.model:
            return None

        line, _, _ = self._generate_best_line(
            context_ids,
            target_word=None,
            candidates=max_attempts,
            min_words=min_words,
            max_words=max_words,
            temperature=temperature,
            top_k=top_k,
            max_context_tokens=max_context_tokens,
        )
        return line

    def _simplify_scheme(self, scheme, length):
        """Приведение схемы рифмовки к нужной длине"""
        clean = "".join(c.upper() for c in scheme if c.isalpha()) or "AABB"
        if len(clean) < length:
            clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
        return list(clean[:length])

    def generate_poem(
        self,
        lines=8,
        rhyme_scheme="AABB",
        start_line=None,
        temperature=0.65,
        free_line_candidates=12,
        rhyme_candidates=20,
        top_k=30,
        max_context_tokens=60,
    ):
        """Генерация стихотворения с метриками"""
        if not self.model or not self.rhyme_search:
            print("Модели не обучены. Вызовите load_and_train() сначала.")
            return []

        rhyme_success_count, rhyme_pairs_count, rhyme_quality_sum = 0, 0, 0
        scheme = self._simplify_scheme(rhyme_scheme, lines)
        poem, line_scheme, rhyme_map = [], [], {}
        context_ids = [self.vocab["<bos>"]]
        start_idx = 0

        if start_line and len(start_line.strip()) >= 10:
            start_words = self._tokenize_russian(start_line.strip())
            clean_start = " ".join(start_words)
            if clean_start and (last_word := self._extract_last_word(clean_start)):
                clean_start = clean_start[0].upper() + clean_start[1:]
                poem.append(clean_start)
                line_scheme.append(scheme[0])
                rhyme_map[scheme[0]] = last_word
                context_ids.extend(self._words_to_ids(start_words))
                context_ids.append(self.vocab["<line>"])
                start_idx = 1

        for i in range(start_idx, lines):
            rhyme_letter = scheme[i]
            if rhyme_letter not in rhyme_map:
                line = self._generate_free_line(
                    context_ids,
                    max_attempts=free_line_candidates,
                    min_words=4,
                    max_words=10,
                    temperature=temperature,
                    top_k=top_k,
                    max_context_tokens=max_context_tokens,
                )
                if not line:
                    continue
                poem.append(line)
                line_scheme.append(rhyme_letter)
                if last_word := self._extract_last_word(line):
                    rhyme_map[rhyme_letter] = last_word
                context_ids.extend(self._words_to_ids(self._tokenize_russian(line)))
                context_ids.append(self.vocab["<line>"])
            else:
                target_word = rhyme_map[rhyme_letter]
                line, rhyme_word = self._generate_line_with_rhyme(
                    target_word,
                    context_ids,
                    max_attempts=rhyme_candidates,
                    min_words=4,
                    max_words=10,
                    temperature=temperature,
                    top_k=top_k,
                    max_context_tokens=max_context_tokens,
                )
                if not line:
                    continue
                previous_line = None
                for j in range(len(poem) - 1, -1, -1):
                    if line_scheme[j] == rhyme_letter:
                        previous_line = poem[j]
                        break

                poem.append(line)
                line_scheme.append(rhyme_letter)
                context_ids.extend(self._words_to_ids(self._tokenize_russian(line)))
                context_ids.append(self.vocab["<line>"])
                rhyme_pairs_count += 1
                if rhyme_word:
                    rhyme_success_count += 1
                    rhyme_map[rhyme_letter] = rhyme_word
                    if previous_line:
                        quality = self.metrics.calculate_rhyme_quality(
                            previous_line, line
                        )
                        rhyme_quality_sum += quality
                else:
                    rhyme_map[rhyme_letter] = target_word

        if poem:
            final_poem = poem[:lines]
            if len(final_poem) < 2:
                return None

            unique_lines = sum(1 for line in final_poem if self._is_unique_line(line))
            unique_rate = unique_lines / len(final_poem)
            rhyme_success_rate = rhyme_success_count / max(1, rhyme_pairs_count)
            avg_rhyme_quality = rhyme_quality_sum / max(1, rhyme_success_count)

            for line in final_poem:
                self.metrics.add_line_metrics(line)
            self.metrics.add_generation(
                rhyme_success_rate, unique_rate, avg_rhyme_quality
            )
            self.metrics.add_poem_metrics(final_poem)
            return final_poem
        return None

    def display_poem(self, poem):
        """Вывод стихотворения"""
        print("\nСгенерированное стихотворение\n")
        if not poem:
            print("Не удалось сгенерировать стихотворение")
            return
        for line in poem:
            print(line)

    def display_metrics(self):
        """Вывод собранных метрик"""
        self.metrics.print_summary()


# ЗАПУСК

def main():
    print("ГЕНЕРАЦИЯ СТИХОВ С ПОМОЩЬЮ LSTM\n")
    generator = LSTMRhymingPoetryGenerator(model_name="poet")

    if not generator.load_and_train("poems_clean.txt", epochs=20):
        print("\nОшибка обучения. Проверьте наличие файла poems_clean.txt")
        return

    start_line = input(
        "\nВведите начальную строку (Enter для случайной генерации):\n> "
    ).strip()

    schemes = {
        "1": ("AABB", "Парная рифма"),
        "2": ("ABAB", "Перекрестная рифма"),
        "3": ("ABBA", "Опоясывающая рифма"),
        "4": ("AAAA", "Монорифма"),
    }

    print("\nВыберите схему рифмовки:")
    for key, (scheme, desc) in schemes.items():
        print(f"   {key}. {desc} ({scheme})")

    choice = input("Ваш выбор [1-4, по умолчанию 1]: ").strip() or "1"
    rhyme_scheme, _ = schemes.get(choice, schemes["1"])

    lines_input = input("Количество строк [4-16, по умолчанию 8]: ").strip() or "8"
    lines = max(4, min(16, int(lines_input))) if lines_input.isdigit() else 8

    poem = None
    for attempt in range(50):
        if attempt % 10 == 0 and attempt > 0:
            print(f"   Попытка {attempt}...")
        poem = generator.generate_poem(
            lines=lines, rhyme_scheme=rhyme_scheme, start_line=start_line or None
        )
        if poem and len(poem) >= 2:
            break

    if poem and len(poem) >= 2:
        generator.display_poem(poem)
        generator.display_metrics()
    else:
        print("\nНе удалось сгенерировать стихотворение")


if __name__ == "__main__":
    main()
