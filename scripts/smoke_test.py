"""Смоук-тест клиента: один запрос с вопросами всех трёх типов, ответ печатается в консоль.

    venv\\Scripts\\python scripts\\smoke_test.py
    venv\\Scripts\\python scripts\\smoke_test.py --model jev-preview
    venv\\Scripts\\python scripts\\smoke_test.py --base-url http://localhost:8888 --model laya
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client import API_KEY_ENV, DEFAULT_MODEL, POLZA_BASE_URL, JevClient, JevError, choice, noul, score  # noqa: E402

STATE = (
    "Вал-шестерня 7.40.112, сталь 40Х. Операция 015 — токарная с ЧПУ: точить наружные поверхности "
    "Ø45h7 и Ø52 с припуском под шлифование, подрезать торцы в размер 186±0,2, точить канавку 3×0,5. "
    "После операции — закалка ТВЧ шеек Ø45 до HRC 48…52."
)

QUESTIONS = {
    "work_type": choice(
        "К какому виду работ относится операция, описанная в тексте?",
        {
            "литьё по выплавляемым моделям": "Изготовление отливок: модели, оболочковые формы, заливка металла",
            "механообработка": "Обработка резанием: точение, фрезерование, сверление, шлифование",
            "сварка": "Соединение деталей сваркой, наплавка",
            "сборка": "Соединение готовых деталей в узел или изделие",
            "прочее": "Ничего из перечисленного",
        },
    ),
    "complexity": score(
        "Насколько сложна операция для рабочего?",
        [
            "Простая: грубые размеры без допусков, универсальное оборудование, обычная разметка",
            "Средняя: размеры с допусками, стандартный инструмент и мерительный инструмент",
            "Сложная: точные посадки, высокая чистота поверхности, станок с ЧПУ или сложная наладка",
        ],
    ),
    "has_heat_treatment": noul(
        "Упоминается ли в тексте термообработка детали: закалка, отпуск, нормализация, цементация и т. п.?"
    ),
}


def bar(probability: float, width: int = 30) -> str:
    return "█" * round(probability * width)


def print_answer(name: str, answer) -> None:
    if answer.type == "noul":
        print(f"\n{name} [noul]  P(да) = {answer.noul:.3f}  {bar(answer.noul)}")
    elif answer.type == "choice":
        print(f"\n{name} [choice]  → {answer.choice}   confidence {answer.confidence:.2f}")
        for key, probability in sorted(answer.probabilities.items(), key=lambda item: -item[1]):
            print(f"    {key:<32} {probability:.3f}  {bar(probability)}")
    else:
        top_level = len(answer.probabilities) - 1
        print(f"\n{name} [score]  = {answer.score:.2f} из {top_level}   confidence {answer.confidence:.2f}")
        for level, probability in answer.probabilities.items():
            legend = str(answer.legend.get(level, ""))[:60]
            print(f"    {level}  {probability:.3f}  {bar(probability):<30}  {legend}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Один запрос к System One с вопросами noul, choice и score")
    parser.add_argument("--base-url", default=POLZA_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    jev = JevClient(base_url=args.base_url, model=args.model)
    if not jev.api_key and args.base_url == POLZA_BASE_URL:
        print(f"{API_KEY_ENV} не задан: скопируйте .env.example в .env и впишите ключ")
        return 2

    print(f"POST {jev.url}")
    print(f"Модель: {jev.model}, ключ: {'задан' if jev.api_key else 'не задан'}")
    try:
        result = jev.evaluate(STATE, QUESTIONS)
    except JevError as exc:
        print(f"\nОШИБКА\n{exc}")
        return 1

    response = result.response
    print(f"Ответила модель: {response.model}")
    print(f"Латентность: {result.latency_ms:.0f} мс, попыток: {result.attempts}")
    if response.usage:
        cost = f"{response.usage.cost_rub} ₽" if response.usage.cost_rub is not None else "не указана"
        print(f"Токены: вход {response.usage.input_tokens}, выход {response.usage.output_tokens}; стоимость: {cost}")

    for name, answer in response.answers.items():
        print_answer(name, answer)

    print("\nСырой ответ:")
    print(json.dumps(result.raw, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
