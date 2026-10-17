"""Клиент решающих моделей System One (Jev от TypeSafe) через Polza.AI
или любой сервер с тем же контрактом. От UI не зависит.

    from client import JevClient, choice, noul, score

    jev = JevClient()  # ключ берётся из переменной окружения POLZA_API_KEY
    result = jev.evaluate(
        state="Кроссовки пришли не того размера, нужен обмен до субботы",
        questions={
            "is_urgent": noul("Клиент сообщает о срочной проблеме?"),
            "department": choice("Какая команда должна заняться обращением?",
                                 {"returns": "Обмен и возврат", "other": None}),
            "frustration": score("Насколько раздражён клиент?",
                                 ["Спокоен", "Раздражён, но вежлив", "Очень зол"]),
        },
    )
    print(result.response.answers["department"].choice, result.latency_ms)

Контракт: https://polza.ai/docs/api-reference/systemone/create.md
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Annotated, Any, Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

POLZA_BASE_URL = "https://polza.ai/api"
DEFAULT_MODEL = "typesafe/jev"
JEV_MODELS = ["typesafe/jev", "jev-latest", "jev-preview", "jev-1.13.0"]
API_KEY_ENV = "POLZA_API_KEY"

# Лимиты из документации Polza.AI (systemone/create, jev-recipes)
MAX_QUESTIONS = 256
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_STATE_CHARS = 400_000
MAX_QUESTIONS_CHARS = 400_000
QUESTION_TYPES = ("noul", "choice", "score")

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 60  # секунд; если сервер просит ждать дольше (дневной лимит), не ждём

ERROR_HINTS = {
    400: "Запрос не прошёл валидацию: схема вопросов, размер state или модель не поддерживает этот эндпоинт",
    401: "Неверный или отозванный API-ключ",
    402: "Недостаточно средств на балансе",
    403: "Доступ запрещён",
    404: "Не найдено: проверьте base_url и название модели",
    408: "Сервер не дождался ответа модели",
    409: "Конфликт запроса",
    413: "Запрос слишком большой",
    422: "Запрос не прошёл валидацию",
    429: "Превышен лимит запросов",
    500: "Внутренняя ошибка сервера",
    502: "Провайдер модели вернул некорректный ответ",
    503: "Провайдер модели недоступен или перегружен",
    504: "Шлюз не дождался ответа провайдера",
}


# --- Вопросы ---------------------------------------------------------------

def noul(instructions: Any, criteria: dict[str, Any] | None = None) -> dict:
    """Да/нет. criteria необязательны: {"true": "когда да", "false": "когда нет"}."""
    question = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    """Один вариант из списка. criteria: {"ключ": "описание" | None | {...}}, до 255 вариантов."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: Any, criteria: list[Any]) -> dict:
    """Положение на шкале. criteria: описания уровней по возрастанию, от 2 до 10."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def check_request(payload: dict) -> list[str]:
    """Проверяет тело запроса по лимитам из документации. Пустой список — всё в порядке."""
    problems = []
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        problems.append("не указана модель")

    state = payload.get("state")
    if not isinstance(state, (str, dict, list)):
        problems.append("state должен быть строкой, JSON-объектом или массивом")
    elif not (state.strip() if isinstance(state, str) else state):
        problems.append("state пустой")
    elif _chars(state) > MAX_STATE_CHARS:
        problems.append(f"state длиннее {MAX_STATE_CHARS:,} символов ({_chars(state):,})")

    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        problems.append("нужен хотя бы один вопрос: questions — объект {имя: вопрос}")
        return problems
    if len(questions) > MAX_QUESTIONS:
        problems.append(f"вопросов {len(questions)}, максимум {MAX_QUESTIONS}")
    if _chars(questions) > MAX_QUESTIONS_CHARS:
        problems.append(f"questions длиннее {MAX_QUESTIONS_CHARS:,} символов")
    for name, question in questions.items():
        if not str(name).strip():
            problems.append("у вопроса пустое имя")
        problems += [f"«{name}»: {problem}" for problem in _check_question(question)]
    return problems


def _check_question(question: Any) -> list[str]:
    if not isinstance(question, dict):
        return ["вопрос должен быть объектом {type, instructions, criteria}"]
    problems = []
    kind = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")

    if not instructions or (isinstance(instructions, str) and not instructions.strip()):
        problems.append("пустые instructions")

    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
            problems.append('criteria для noul — объект с ключами "true" и/или "false"')
    elif kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            problems.append("для choice нужны criteria: {вариант: описание или null}")
        elif len(criteria) > MAX_CHOICE_OPTIONS:
            problems.append(f"вариантов {len(criteria)}, максимум {MAX_CHOICE_OPTIONS}")
        elif any(not key.strip() for key in criteria):
            problems.append("у варианта пустой ключ")
    elif kind == "score":
        if not isinstance(criteria, list) or not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            problems.append(f"для score нужны criteria: список из {MIN_SCORE_LEVELS}–{MAX_SCORE_LEVELS} уровней по возрастанию")
        elif any(level is None or (isinstance(level, str) and not level.strip()) for level in criteria):
            problems.append("у уровня шкалы пустое описание")
    else:
        problems.append(f"неизвестный тип {kind!r}, допустимы: {', '.join(QUESTION_TYPES)}")
    return problems


def _chars(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(json.dumps(value, ensure_ascii=False))


# --- Ответ -----------------------------------------------------------------

class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["noul"]
    noul: float  # вероятность «да», 0..1


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["choice"]
    choice: str  # ключ победившего варианта
    confidence: float
    probabilities: dict[str, float]  # ключ варианта → вероятность


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["score"]
    score: float  # позиция на шкале от 0 до (число уровней − 1), может быть между уровнями
    confidence: float
    probabilities: dict[str, float]  # номер уровня строкой ("0", "1", …) → вероятность
    legend: dict[str, Any] = Field(default_factory=dict)  # номер уровня → описание


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0  # тарифицируются только входные
    output_tokens: int = 0
    cost_rub: float | None = None


class JevResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str  # версия, которая ответила, например jev-1.13.0
    answers: dict[str, Answer]
    usage: Usage | None = None


@dataclass
class JevResult:
    response: JevResponse
    raw: dict  # ответ сервера как есть
    latency_ms: float  # время успешной попытки: от отправки до ответа
    attempts: int  # 1 — без повторов


# --- Ошибки ----------------------------------------------------------------

class JevError(Exception):
    """Базовая ошибка клиента."""


class JevValidationError(JevError):
    """Запрос не прошёл проверку и не был отправлен."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Запрос не отправлен:\n" + "\n".join(f"• {problem}" for problem in problems))


class JevAPIError(JevError):
    """Ошибка при отправке: HTTP-статус, сеть или ответ не по контракту."""

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None,
                 trace_id: str | None = None, retry_after: float | None = None,
                 retryable: bool = False, body: Any = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.trace_id = trace_id
        self.retry_after = retry_after
        self.retryable = retryable
        self.body = body  # тело ответа сервера, если было


def _api_error(response: httpx.Response) -> JevAPIError:
    """Ошибка Polza: {"error": {code, message, trace_id, details, metadata}, "trace_id", "detail"?}."""
    try:
        body = response.json()
    except ValueError:
        body = response.text
    data = body if isinstance(body, dict) else {}
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    status = response.status_code
    code = error.get("code")
    trace_id = error.get("trace_id") or data.get("trace_id")
    retry_after = _retry_after(response)

    title = ERROR_HINTS.get(status, f"Ошибка HTTP {status}")
    lines = [f"{title} (HTTP {status}{', ' + code if code else ''})"]
    server_message = error.get("message") or _detail_text(data.get("detail")) or (body if isinstance(body, str) else "")
    if server_message:
        lines.append(f"Сервер: {server_message.strip()[:1000]}")
    if error.get("details"):
        lines.append(f"Подробности: {json.dumps(error['details'], ensure_ascii=False)[:1000]}")
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    if metadata.get("provider_name"):
        lines.append(f"Провайдер: {metadata['provider_name']} {metadata.get('raw') or ''}".strip())
    if retry_after is not None:
        lines.append(f"Повторить можно через {retry_after:.0f} с")
    if trace_id:
        lines.append(f"trace_id: {trace_id}")
    return JevAPIError("\n".join(lines), status_code=status, code=code, trace_id=trace_id,
                       retry_after=retry_after, retryable=status in RETRY_STATUSES, body=body)


def _detail_text(detail: Any) -> str:
    """detail бывает в стиле TypeSafe {error_type, message} или FastAPI [{loc, msg}]."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        return detail.get("message") or detail.get("error_type") or ""
    if isinstance(detail, list):
        return "; ".join(
            f"{'.'.join(map(str, item.get('loc', [])))}: {item.get('msg')}" if isinstance(item, dict) else str(item)
            for item in detail
        )
    return ""


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            return max(parsedate_to_datetime(value).timestamp() - time.time(), 0.0)
        except (TypeError, ValueError):
            return None


def _network_error(exc: httpx.TransportError, url: str) -> JevAPIError:
    if isinstance(exc, httpx.TimeoutException):
        text = f"Нет ответа от {url} за отведённое время"
    elif isinstance(exc, httpx.ConnectError):
        text = f"Не удалось подключиться к {url}: сервер не запущен или адрес неверный"
    else:
        text = f"Сетевая ошибка при обращении к {url}"
    # Отказ в соединении сам не пройдёт, остальное (таймауты, обрывы) пробуем повторить
    return JevAPIError(f"{text} ({type(exc).__name__}: {exc})", retryable=not isinstance(exc, httpx.ConnectError))


def _parse_result(response: httpx.Response, latency_ms: float, attempts: int) -> JevResult:
    try:
        raw = response.json()
    except ValueError:
        raise JevAPIError(f"Сервер вернул не JSON (HTTP {response.status_code}): {response.text[:500]}",
                          status_code=response.status_code, body=response.text) from None
    try:
        parsed = JevResponse.model_validate(raw)
    except ValidationError as exc:
        raise JevAPIError(f"Ответ не соответствует контракту System One:\n{exc}",
                          status_code=response.status_code, body=raw) from exc
    return JevResult(response=parsed, raw=raw, latency_ms=latency_ms, attempts=attempts)


# --- Клиент ----------------------------------------------------------------

def endpoint_url(base_url: str) -> str:
    """https://polza.ai/api → https://polza.ai/api/v1/systemone. Полный адрес эндпоинта не меняется."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/systemone"):
        return url
    if url.endswith("/v1"):
        return f"{url}/systemone"
    return f"{url}/v1/systemone"


def to_curl(payload: dict, url: str, api_key_env: str | None = API_KEY_ENV) -> str:
    """curl для bash. Вместо ключа подставляется переменная окружения, сам ключ в команду не попадает.
    api_key_env=None — без заголовка авторизации (локальный сервер)."""
    body = json.dumps(payload, ensure_ascii=False, indent=2).replace("'", "'\\''")
    auth = f"  -H \"Authorization: Bearer ${api_key_env}\" \\\n" if api_key_env else ""
    return (
        f"curl -X POST '{url}' \\\n"
        f"{auth}"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{body}'"
    )


class JevClient:
    """POST {base_url}/v1/systemone: синхронно, асинхронно и пачкой с ограничением параллельности."""

    def __init__(self, api_key: str | None = None, base_url: str = POLZA_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: float = 60.0, max_retries: int = 3, validate: bool = True):
        self.api_key = api_key if api_key is not None else os.getenv(API_KEY_ENV)
        self.url = endpoint_url(base_url)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries  # повторы на 408, 429, 5xx и сетевые сбои
        self.validate = validate  # проверять лимиты до отправки

    def __repr__(self) -> str:
        return f"JevClient(url={self.url!r}, model={self.model!r}, api_key={'***' if self.api_key else None})"

    def payload(self, state: str | dict | list, questions: dict[str, dict], model: str | None = None) -> dict:
        return {"model": model or self.model, "state": state, "questions": questions}

    def evaluate(self, state: str | dict | list, questions: dict[str, dict], model: str | None = None) -> JevResult:
        return self.send(self.payload(state, questions, model))

    async def aevaluate(self, state: str | dict | list, questions: dict[str, dict],
                        model: str | None = None) -> JevResult:
        return await self.asend(self.payload(state, questions, model))

    def send(self, payload: dict) -> JevResult:
        """Отправляет готовое тело запроса, например из редактора сырого JSON."""
        self._check(payload)
        with httpx.Client(timeout=self.timeout) as http:
            attempt = 0
            while True:
                attempt += 1
                started = time.perf_counter()
                try:
                    response = http.post(self.url, json=payload, headers=self._headers())
                except httpx.TransportError as exc:
                    error = _network_error(exc, self.url)
                else:
                    if response.is_success:
                        return _parse_result(response, _ms_since(started), attempt)
                    error = _api_error(response)
                delay = self._retry_delay(attempt, error)
                if delay is None:
                    raise error
                time.sleep(delay)

    async def asend(self, payload: dict, http: httpx.AsyncClient | None = None) -> JevResult:
        if http is None:
            async with httpx.AsyncClient(timeout=self.timeout) as http:
                return await self.asend(payload, http)

        self._check(payload)
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                response = await http.post(self.url, json=payload, headers=self._headers())
            except httpx.TransportError as exc:
                error = _network_error(exc, self.url)
            else:
                if response.is_success:
                    return _parse_result(response, _ms_since(started), attempt)
                error = _api_error(response)
            delay = self._retry_delay(attempt, error)
            if delay is None:
                raise error
            await asyncio.sleep(delay)

    async def asend_many(self, payloads: list[dict], concurrency: int = 4,
                         on_progress: Callable[[int, int], None] | None = None) -> list[JevResult | JevError]:
        """Отправляет запросы параллельно, не больше concurrency одновременно.

        Результаты в том же порядке, что и payloads; на месте неудачного запроса — объект ошибки.
        on_progress(готово, всего) вызывается после каждого запроса.
        """
        semaphore = asyncio.Semaphore(concurrency)
        finished = 0

        async with httpx.AsyncClient(timeout=self.timeout) as http:
            async def run(payload: dict) -> JevResult | JevError:
                nonlocal finished
                async with semaphore:
                    try:
                        result = await self.asend(payload, http)
                    except JevError as exc:
                        result = exc
                finished += 1
                if on_progress:
                    on_progress(finished, len(payloads))
                return result

            return await asyncio.gather(*(run(payload) for payload in payloads))

    def _check(self, payload: dict) -> None:
        if self.validate:
            problems = check_request(payload)
            if problems:
                raise JevValidationError(problems)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _retry_delay(self, attempt: int, error: JevAPIError) -> float | None:
        """Сколько ждать перед повтором; None — больше не повторять."""
        if attempt > self.max_retries or not error.retryable:
            return None
        if error.retry_after is not None:
            return error.retry_after if error.retry_after <= MAX_RETRY_AFTER else None
        # 1, 2, 4… с и небольшой разброс, чтобы параллельные запросы не повторялись разом
        return 2 ** (attempt - 1) + random.uniform(0, 0.5)


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
