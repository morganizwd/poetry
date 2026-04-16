# ----------------ДОБАВИЛА МЕТРИКИ------------------------------

import re, random, json, os
from collections import defaultdict
import markovify
import statistics


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
        """сохранение словаря рифм"""
        data = {k: v for k, v in self.rhymes_dict.items() if len(v) > 1}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# КЛАСС ДЛЯ СБОРА МЕТРИК
class MetricsCollector:
    """Сбор и расчет метрик качества генерации"""

    def __init__(self):
        self.data = {
            "rhyme_success": [],
            "unique_rate": [],
            "rhyme_quality": [],
            "vowels": [],
            "lengths": [],
            "unique_words": [],
        }

    def _safe_stat(self, values, func, default=0):
        """Безопасное вычисление статистики: возвращает default если нет данных"""
        return func(values) if values else default

    def add_generation(self, rhyme_success, unique_rate, rhyme_quality=None):
        """Добавление метрик одного стихотворения"""
        self.data["rhyme_success"].append(rhyme_success)
        self.data["unique_rate"].append(unique_rate)
        if rhyme_quality is not None:
            self.data["rhyme_quality"].append(rhyme_quality)

    def add_line_metrics(self, line):
        """Добавление метрик одной строки: количество гласных, длина"""
        vowel_count = sum(1 for c in line.lower() if c in "аеёиоуыэюя")
        self.data["vowels"].append(vowel_count)
        self.data["lengths"].append(len(line.split()))

    def add_poem_metrics(self, poem):
        """Добавление метрик стихотворения: количество уникальных слов"""
        words = {w for line in poem for w in re.findall(r"[а-яё]+", line.lower())}
        self.data["unique_words"].append(len(words))

    def calculate_rhyme_quality(self, line1, line2):
        """Оценка качества рифмы (0-1): 1.0=3 буквы, 0.8=2 буквы, 0.5=1 буква, 0.2=нет"""
        words1 = re.findall(r"[а-яё]+", line1.lower())
        words2 = re.findall(r"[а-яё]+", line2.lower())
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
        """Вычисление статистических показателей по всем метрикам"""
        d = self.data
        stats = {
            "avg_rhyme_success": self._safe_stat(
                d["rhyme_success"], lambda x: round(statistics.mean(x) * 100, 1)
            ),
            "avg_rhyme_quality": self._safe_stat(
                d["rhyme_quality"], lambda x: round(statistics.mean(x) * 100, 1)
            ),
            "avg_unique_rate": self._safe_stat(
                d["unique_rate"], lambda x: round(statistics.mean(x) * 100, 1)
            ),
            "avg_unique_words_per_poem": self._safe_stat(
                d["unique_words"], lambda x: round(statistics.mean(x), 1)
            ),
            "avg_vowels_per_line": self._safe_stat(
                d["vowels"], lambda x: round(statistics.mean(x), 1)
            ),
            "min_vowels": self._safe_stat(d["vowels"], min, 0),
            "max_vowels": self._safe_stat(d["vowels"], max, 0),
            "avg_words_per_line": self._safe_stat(
                d["lengths"], lambda x: round(statistics.mean(x), 1)
            ),
            "min_words": self._safe_stat(d["lengths"], min, 0),
            "max_words": self._safe_stat(d["lengths"], max, 0),
        }
        return stats

    def print_summary(self):
        """Вывод сводной статистики в консоль"""
        stats = self.get_statistics()
        print("\nМетрики качества:")
        print(f"   • Успешность рифмовки: {stats.get('avg_rhyme_success', 0):.1f}%")
        print(f"   • Среднее качество рифмы: {stats.get('avg_rhyme_quality', 0):.1f}%")
        print(f"   • Уникальность строк: {stats.get('avg_unique_rate', 0):.1f}%")
        print(
            f"   • Уникальных слов на стих: {stats.get('avg_unique_words_per_poem', 0):.1f}"
        )

        print("\nМетрики строк:")
        print(
            f"   • Средняя длина строки: {stats.get('avg_words_per_line', 0):.1f} слов"
        )
        print(
            f"   • Среднее количество гласных: {stats.get('avg_vowels_per_line', 0):.1f}"
        )
        print(
            f"   • Диапазон слов: {stats.get('min_words', 0)}-{stats.get('max_words', 0)}"
        )
        print(
            f"   • Диапазон гласных: {stats.get('min_vowels', 0)}-{stats.get('max_vowels', 0)}"
        )


# ГЕНЕРАТОР СТИХОВ
class RhymingPoetryGenerator:
    def __init__(self, model_name="poet"):
        self.model_name = model_name
        self.vowels = "аеёиоуыэюя"
        self.markov_model_reversed = None
        self.rhyme_search = None
        self.original_lines = set()
        self.metrics = MetricsCollector()

    def _tokenize_russian(self, text):
        """Токенизация"""
        return re.findall(r"[а-яё]+", text.lower())

    def _reverse_words(self, line):
        """реверсирование"""
        return " ".join(reversed(self._tokenize_russian(line)))

    def _extract_last_word(self, line):
        """Извлечение последнего слова из строки"""
        words = self._tokenize_russian(line)
        return words[-1] if words else None

    def _count_vowels(self, text):
        """Подсчет количества гласных букв в тексте"""
        return sum(1 for c in text.lower() if c in self.vowels)

    def load_and_train(self, filepath="poems_clean.txt", state_size=2):
        """Обучение модели"""
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

        all_lines, all_words = [], set()
        for poem in re.split(r"\n\s*\n", content):
            for line in [l.strip() for l in poem.split("\n") if l.strip()]:
                if len(line) < 8 or self._count_vowels(line) < 2:
                    continue
                all_lines.append(line)
                all_words.update(w for w in self._tokenize_russian(line) if len(w) >= 2)

        print(f"Статистика датасета:")
        print(f"   Строк: {len(all_lines)}")
        print(f"   Уникальных слов: {len(all_words)}")

        if not all_lines:
            print("Не найдено подходящих строк для обучения.")
            return False

        self.rhyme_search = RhymeSearch()
        self.rhyme_search.train(list(all_words))
        self.rhyme_search.save_json(f"{self.model_name}_rhymes.json")

        reversed_text = "\n".join(self._reverse_words(line) for line in all_lines)
        self.markov_model_reversed = markovify.NewlineText(
            reversed_text,
            state_size=state_size,
            retain_original=False,
        )
        return True

    def _is_unique_line(self, line, threshold=0.85):
        """Проверка уникальности строки: не является ли она копией из датасета"""
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

    def _generate_line_with_rhyme(
        self, target_word, max_attempts=50, min_words=3, max_words=10
    ):
        """Генерация строки, заканчивающейся словом, рифмующимся с target_word"""
        if not self.markov_model_reversed or not target_word:
            return None, None
        exclude_words = {target_word}
        for attempt in range(max_attempts):
            rhyme_word = self.rhyme_search.give_rhyme(target_word, exclude_words)
            if not rhyme_word:
                continue
            exclude_words.add(rhyme_word)
            try:
                rev_line = self.markov_model_reversed.make_sentence_with_start(
                    rhyme_word,
                    strict=False,
                    min_words=min_words,
                    max_words=max_words,
                    test_output=False,
                )
                if rev_line:
                    words = self._tokenize_russian(rev_line)
                    line = self._postprocess_line(words, min_words)
                    if line:
                        return line, rhyme_word
            except Exception:
                continue
        return None, None

    def _generate_free_line(self, max_attempts=40, min_words=4, max_words=10):
        """Генерация свободной строки без фиксированной рифмы"""
        if not self.markov_model_reversed:
            return None
        for attempt in range(max_attempts):
            try:
                rev_line = self.markov_model_reversed.make_sentence(
                    min_words=min_words,
                    max_words=max_words,
                    tries=20,
                    test_output=False,
                )
                if rev_line:
                    words = self._tokenize_russian(rev_line)
                    line = self._postprocess_line(words, min_words)
                    if line:
                        return line
            except Exception:
                continue
        return None

    def _simplify_scheme(self, scheme, length):
        """Приведение схемы рифмовки к нужной длине"""
        clean = "".join(c.upper() for c in scheme if c.isalpha()) or "AABB"
        if len(clean) < length:
            clean = (clean * ((length + len(clean) - 1) // len(clean)))[:length]
        return list(clean[:length])

    def generate_poem(self, lines=8, rhyme_scheme="AABB", start_line=None):
        """Генерация стихотворения с заданной схемой рифмовки и стартовой строкой"""
        if not self.markov_model_reversed or not self.rhyme_search:
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
                line = self._generate_free_line(min_words=4, max_words=9)
                if not line:
                    continue
                poem.append(line)
                line_scheme.append(rhyme_letter)
                if last_word := self._extract_last_word(line):
                    rhyme_map[rhyme_letter] = last_word
            else:
                target_word = rhyme_map[rhyme_letter]
                line, rhyme_word = self._generate_line_with_rhyme(
                    target_word, min_words=4, max_words=9
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

            # Вычисление метрик для стихотворения
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
        """Вывод сгенерированного стихотворения"""
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
    """Основная функция: обучение модели, настройка параметров, генерация стиха, вывод метрик"""
    print("ГЕНЕРАЦИЯ СТИХОВ С ПОМОЩЬЮ ЦЕПЕЙ МАРКОВА\n")
    generator = RhymingPoetryGenerator(model_name="poet")

    if not generator.load_and_train("poems_clean.txt", state_size=2):
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
