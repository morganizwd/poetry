
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

class BahdanauAttention(Layer):
    """Простой causal additive attention для языковой модели"""

    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.supports_masking = True

    def build(self, input_shape):
        self.W_query = Dense(self.units, use_bias=False)
        self.W_value = Dense(self.units, use_bias=False)
        self.V = Dense(1, use_bias=False)
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

    MODEL_VERSION = 3

    def __init__(self, model_name="poet"):
        self.model_name = model_name
        self.vowels = "аеёиоуыэюя"
        self.model = None
        self.rhyme_search = None
        self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.id2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.VOCAB_SIZE = 4
        self.original_lines = set()
        self.poems = []
        self.metrics = MetricsCollector()

    def _tokenize_russian(self, text):
        """Токенизация"""
        return re.findall(r"[а-яё]+", text.lower())

    def _reverse_words(self, line):
        """Реверсирование строки"""
        return " ".join(reversed(self._tokenize_russian(line)))

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

    def _postprocess_line(self, words, min_words):
        """Постобработка сгенерированной строки"""
        if len(words) < min_words:
            return None
        normal_line = " ".join(reversed(words))
        if len(normal_line) < 10 or self._count_vowels(normal_line) < 2:
            return None
        if not self._is_unique_line(normal_line, threshold=0.85):
            return None
        if normal_line and normal_line[0].isalpha():
            normal_line = normal_line[0].upper() + normal_line[1:]
        return normal_line

    def load_and_train(
        self,
        filepath="poems_clean.txt",
        epochs=10,
        batch_size=32,
        max_vocab_size=8000,
        max_line_words=12,
        max_training_lines=20000,
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

                self.model = load_model(
                    model_path,
                    custom_objects={'BahdanauAttention': BahdanauAttention},
                    compile=False,
                )
                self._compile_model(self.model)

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

        all_lines, word_counts = [], Counter()
        poems = [p.strip() for p in content.split("\n\n") if p.strip()]
        self.poems = poems

        for poem in poems:
            lines = [l.strip() for l in poem.split("\n") if l.strip()]
            for line in lines:
                tokens = self._tokenize_russian(line)
                vowel_count = self._count_vowels(line)
                if 3 <= len(tokens) <= max_line_words and vowel_count >= 2:
                    all_lines.append(line)
                    word_counts.update(w for w in tokens if len(w) >= 2)

        if not all_lines:
            print("Не найдено подходящих строк для обучения.")
            return False

        if len(all_lines) > max_training_lines:
            random.shuffle(all_lines)
            all_lines = all_lines[:max_training_lines]

        print(f"Статистика датасета:")
        print(f"   Строк для обучения: {len(all_lines)}")
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
            if w not in ["<pad>", "<unk>", "<bos>", "<eos>"]
        ]
        self.rhyme_search.train(model_words)
        self.rhyme_search.save_json(f"{self.model_name}_rhymes.json")

        reversed_encoded = []
        for line in all_lines:
            tokens = self._tokenize_russian(line)
            ids = (
                [self.vocab["<bos>"]]
                + [self.vocab.get(w, self.vocab["<unk>"]) for w in reversed(tokens)]
                + [self.vocab["<eos>"]]
            )
            reversed_encoded.append(ids)

        random.shuffle(reversed_encoded)
        batches = []
        for i in range(0, len(reversed_encoded), batch_size):
            batch = reversed_encoded[i : i + batch_size]
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
        self.model = self._build_model()

        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            for x_batch, y_batch in batches:
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

    def _sample_next_token(self, logits, temperature=0.85, allow_eos=True):
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

        for token_id in blocked_tokens:
            probs[token_id] = 0.0

        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return None

        probs /= total
        return int(np.random.choice(len(probs), p=probs))

    def _generate_line_with_rhyme(
        self, target_word, max_attempts=50, min_words=4, max_words=9, temperature=0.85
    ):
        """Генерация строки с рифмой (с учётом attention)"""
        if not self.model or not target_word:
            return None, None

        exclude_words = {target_word}
        for _ in range(max_attempts):
            rhyme_word = self.rhyme_search.give_rhyme(target_word, exclude_words)
            if not rhyme_word or rhyme_word not in self.vocab:
                continue

            exclude_words.add(rhyme_word)
            tokens = [self.vocab["<bos>"], self.vocab[rhyme_word]]
            words = [rhyme_word]

            for _ in range(max_words * 2):
                if len(words) >= max_words:
                    break

                input_seq = np.array([tokens], dtype=np.int32)

                logits = self.model.predict(input_seq, verbose=0)[0, -1, :]
                next_token = self._sample_next_token(
                    logits,
                    temperature=temperature,
                    allow_eos=len(words) >= min_words,
                )
                if next_token is None:
                    break

                next_word = self.id2word.get(next_token, "<unk>")

                if next_word == "<eos>":
                    break

                tokens.append(next_token)
                words.append(next_word)

            if len(words) < min_words:
                continue

            line = self._postprocess_line(words, min_words)
            if line:
                return line, rhyme_word

        return None, None

    def _generate_free_line(
        self, max_attempts=40, min_words=4, max_words=9, temperature=0.85
    ):
        """Генерация свободной строки (с учётом attention)"""
        if not self.model:
            return None

        for _ in range(max_attempts):
            valid_words = [
                w for w in self.vocab
                if w not in ["<pad>", "<unk>", "<bos>", "<eos>"]
            ]
            if not valid_words:
                return None

            start_word = random.choice(valid_words)
            if start_word not in self.vocab:
                continue

            tokens = [self.vocab["<bos>"], self.vocab[start_word]]
            words = [start_word]

            for _ in range(max_words * 2):
                if len(words) >= max_words:
                    break

                input_seq = np.array([tokens], dtype=np.int32)

                logits = self.model.predict(input_seq, verbose=0)[0, -1, :]
                next_token = self._sample_next_token(
                    logits,
                    temperature=temperature,
                    allow_eos=len(words) >= min_words,
                )
                if next_token is None:
                    break

                next_word = self.id2word.get(next_token, "<unk>")

                if next_word == "<eos>":
                    break

                tokens.append(next_token)
                words.append(next_word)

            if len(words) < min_words:
                continue

            line = self._postprocess_line(words, min_words)
            if line:
                return line

        return None

    def _simplify_scheme(self, scheme, length):
        """Приведение схемы рифмовки к нужной длине"""
        clean = "".join(c.upper() for c in scheme if c.isalpha()) or "AABB"
        if len(clean) < length:
            clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
        return list(clean[:length])

    def generate_poem(
        self, lines=8, rhyme_scheme="AABB", start_line=None, temperature=0.85
    ):
        """Генерация стихотворения с метриками"""
        if not self.model or not self.rhyme_search:
            print("Модели не обучены. Вызовите load_and_train() сначала.")
            return []

        rhyme_success_count, rhyme_pairs_count, rhyme_quality_sum = 0, 0, 0
        scheme = self._simplify_scheme(rhyme_scheme, lines)
        poem, line_scheme, rhyme_map = [], [], {}
        start_idx = 0

        if start_line and len(start_line.strip()) >= 10:
            clean_start = " ".join(self._tokenize_russian(start_line.strip()))
            if clean_start and (last_word := self._extract_last_word(clean_start)):
                clean_start = clean_start[0].upper() + clean_start[1:]
                poem.append(clean_start)
                line_scheme.append(scheme[0])
                rhyme_map[scheme[0]] = last_word
                start_idx = 1

        for i in range(start_idx, lines):
            rhyme_letter = scheme[i]
            if rhyme_letter not in rhyme_map:
                line = self._generate_free_line(
                    min_words=4, max_words=9, temperature=temperature
                )
                if not line:
                    continue
                poem.append(line)
                line_scheme.append(rhyme_letter)
                if last_word := self._extract_last_word(line):
                    rhyme_map[rhyme_letter] = last_word
            else:
                target_word = rhyme_map[rhyme_letter]
                line, rhyme_word = self._generate_line_with_rhyme(
                    target_word,
                    min_words=4,
                    max_words=9,
                    temperature=temperature,
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

    if not generator.load_and_train("poems_clean.txt", epochs=10):
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
