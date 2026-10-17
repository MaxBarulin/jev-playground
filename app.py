"""Песочница для решающих моделей System One (Jev) через Polza.AI или свой сервер с тем же контрактом.

Запуск: venv/Scripts/streamlit run app.py
"""

import asyncio
import io
import json
import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import history
import render
from client import (API_KEY_ENV, JEV_MODELS, POLZA_BASE_URL, QUESTION_TYPES, JevClient, JevError, JevResponse,
                    JevResult, JevValidationError, check_request, endpoint_url)
from presets import PRESETS

load_dotenv(Path(__file__).resolve().parent / ".env")
st.set_page_config(page_title="Jev playground", page_icon=":material/rule:", layout="wide")

TABS = ["Песочница", "Эксперименты", "Пакетный режим", "История"]
FLAG_SCREEN = "rgba(227, 73, 72, 0.18)"  # строки «на проверку» в таблице
FLAG_XLSX = "#F8D7DA"  # они же в выгрузке


# --- Отправка --------------------------------------------------------------

def send(jev: JevClient, payload: dict, source: str) -> JevResult | JevError:
    try:
        outcome = jev.send(payload)
    except JevError as exc:
        outcome = exc
    if not isinstance(outcome, JevValidationError):  # неотправленное в историю не пишем
        history.append([history.entry(source, jev.url, payload, outcome)])
    return outcome


def run_many(jev: JevClient, payloads: list[dict], concurrency: int, source: str) -> list[JevResult | JevError]:
    progress = st.progress(0.0, text=f"0 из {len(payloads)}")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"{done} из {total}")

    outcomes = asyncio.run(jev.asend_many(payloads, concurrency=concurrency, on_progress=on_progress))
    history.append([history.entry(source, jev.url, payload, outcome)
                    for payload, outcome in zip(payloads, outcomes) if not isinstance(outcome, JevValidationError)])
    return outcomes


# --- Сайдбар ---------------------------------------------------------------

def sidebar() -> tuple[JevClient, str | None]:
    """Возвращает клиент и имя переменной окружения с ключом для curl (None — без ключа)."""
    with st.sidebar:
        st.header("Провайдер")
        provider = st.radio("Провайдер", ["Polza.AI", "Свой сервер"], key="provider", label_visibility="collapsed")
        if provider == "Polza.AI":
            base_url = POLZA_BASE_URL
            model = st.selectbox("Модель", JEV_MODELS, key="polza-model", accept_new_options=True)
            api_key = os.getenv(API_KEY_ENV) or ""
            curl_env = API_KEY_ENV
            if api_key:
                st.caption(f"Ключ: {API_KEY_ENV} из .env")
            else:
                st.error(f"{API_KEY_ENV} не найден. Скопируйте .env.example в .env и впишите ключ.")
        else:
            base_url = st.text_input("base_url", "http://127.0.0.1:8888", key="custom-url",
                                     help="/v1/systemone допишется сам, если его нет в адресе. "
                                          "127.0.0.1 вместо localhost: на Windows localhost сначала пробует IPv6 "
                                          "и добавляет к латентности около 2 с")
            model = st.text_input("Модель", "laya", key="custom-model")
            # Пустая строка, а не None: иначе клиент возьмёт ключ Polza из окружения и отправит его чужому серверу
            api_key = st.text_input("API-ключ, если нужен", type="password", key="custom-key")
            curl_env = "API_KEY" if api_key else None
        st.caption(f"POST {endpoint_url(base_url)}")

        with st.expander("Параметры запросов"):
            timeout = st.number_input("Таймаут, с", 5, 600, 60, key="timeout")
            retries = st.number_input("Повторов на 408, 429, 5xx", 0, 10, 3, key="retries")
            validate = st.toggle("Проверять лимиты до отправки", True, key="validate",
                                 help="Выключите, чтобы увидеть, как ошибку вернёт сам API")

    jev = JevClient(api_key=api_key, base_url=base_url, model=model or "", timeout=timeout,
                    max_retries=retries, validate=validate)
    return jev, curl_env


# --- Черновик песочницы ----------------------------------------------------
# draft — снимок, из которого виджеты конструктора берут начальные значения. Правки в виджетах
# в draft не пишутся: запрос собирается из виджетов на каждом прогоне. Пресет, JSON или запись
# истории заменяют draft и увеличивают ver; ver входит в ключи виджетов, и они создаются заново.

def to_text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def from_text(text: str):
    """Пусто → None, текст вида {...} или [...] → JSON-объект, иначе строка."""
    text = text.strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def cell(value) -> str:
    """Ячейка data_editor → строка; у новых строк там None или NaN."""
    return "" if value is None or pd.isna(value) else str(value)


def draft_question(name: str, question: dict) -> dict:
    kind = question.get("type") if question.get("type") in QUESTION_TYPES else "noul"
    criteria = question.get("criteria")
    return {
        "uid": uuid4().hex[:8],
        "name": name,
        "type": kind,
        "instructions": to_text(question.get("instructions")),
        "options": [{"Ключ": key, "Описание": to_text(value)} for key, value in criteria.items()]
        if kind == "choice" and isinstance(criteria, dict) else [],
        "levels": [{"Описание уровня": to_text(value)} for value in criteria]
        if kind == "score" and isinstance(criteria, list) else [],
        "if_true": to_text(criteria.get("true")) if kind == "noul" and isinstance(criteria, dict) else "",
        "if_false": to_text(criteria.get("false")) if kind == "noul" and isinstance(criteria, dict) else "",
    }


def load_draft(state, questions: dict) -> None:
    is_text = isinstance(state, str)
    st.session_state.draft = {
        "state_mode": "Текст" if is_text else "JSON",
        "state_text": state if is_text else json.dumps(state, ensure_ascii=False, indent=2),
        "questions": [draft_question(name, question) for name, question in questions.items()],
    }
    st.session_state.ver = st.session_state.get("ver", 0) + 1
    st.session_state.json_error = None


def load_preset(title: str) -> None:
    load_draft(PRESETS[title]["state"], PRESETS[title]["questions"])


def open_in_sandbox(request: dict) -> None:
    questions = request.get("questions")
    load_draft(request.get("state", ""), questions if isinstance(questions, dict) else {})
    st.session_state.tab = "Песочница"


def add_question(kind: str) -> None:
    questions = st.session_state.draft["questions"]
    names = {question["name"] for question in questions}
    number = len(questions) + 1
    while f"{kind}_{number}" in names:
        number += 1
    question = draft_question(f"{kind}_{number}", {"type": kind})
    question["options"] = [{"Ключ": "", "Описание": ""} for _ in range(2)]
    question["levels"] = [{"Описание уровня": ""} for _ in range(3)]
    questions.append(question)


def remove_question(uid: str) -> None:
    draft = st.session_state.draft
    draft["questions"] = [question for question in draft["questions"] if question["uid"] != uid]


def apply_raw_json() -> None:
    try:
        data = json.loads(st.session_state.raw_json)
    except json.JSONDecodeError as exc:
        st.session_state.json_error = f"Невалидный JSON: {exc.msg} (строка {exc.lineno}, столбец {exc.colno})"
        return
    problem = raw_json_problem(data)
    if problem:
        st.session_state.json_error = problem
        return
    load_draft(data["state"], data["questions"])
    if "model" in data:
        st.toast("Поле model из JSON не используется: модель выбирается в сайдбаре")


def raw_json_problem(data) -> str | None:
    """Можно ли разложить JSON по конструктору. Лимиты потом проверит check_request."""
    if not isinstance(data, dict) or "state" not in data or not isinstance(data.get("questions"), dict):
        return 'Нужен объект {"state": ..., "questions": {имя: вопрос}}'
    if not isinstance(data["state"], (str, dict, list)):
        return "state должен быть строкой, объектом или массивом"
    expected = {
        "noul": ((dict, type(None)), 'объект {"true": ..., "false": ...} или отсутствовать'),
        "choice": (dict, "объект {вариант: описание}"),
        "score": (list, "список уровней"),
    }
    for name, question in data["questions"].items():
        if not isinstance(question, dict) or question.get("type") not in QUESTION_TYPES:
            return f"«{name}»: нужен объект с type из {', '.join(QUESTION_TYPES)}"
        types, description = expected[question["type"]]
        if not isinstance(question.get("criteria"), types):
            return f"«{name}»: criteria у {question['type']} — {description}"
    return None


def reset_json_error() -> None:
    st.session_state.json_error = None


# --- Песочница -------------------------------------------------------------

def sandbox_tab(jev: JevClient, curl_env: str | None):
    """Возвращает текущие state (None, если JSON в state невалиден) и questions."""
    draft, ver = st.session_state.draft, st.session_state.ver

    with st.container(horizontal=True):
        for title, preset in PRESETS.items():
            st.button(title, key=f"preset-{title}", help=preset["description"], on_click=load_preset, args=(title,))
        st.button("Очистить", key="preset-clear", icon=":material/close:", on_click=load_draft, args=("", {}))

    left, right = st.columns([3, 2], gap="large")
    with left:
        state = state_editor(draft, ver)
        st.markdown("**Вопросы**")
        questions = {}
        for question in draft["questions"]:
            name, built = question_editor(question, ver)
            if name in questions:
                st.warning(f"Имя «{name}» повторяется: в запрос попадёт только последний вопрос")
            questions[name] = built
        with st.container(horizontal=True):
            for kind in QUESTION_TYPES:
                st.button(kind, key=f"add-{kind}", icon=":material/add:", on_click=add_question, args=(kind,),
                          help=f"Добавить вопрос «{render.TYPE_LABELS[kind]}»")
    with right:
        raw_json_editor(state, questions)

    payload = jev.payload(state, questions) if state is not None else None
    problems = check_request(payload) if payload else []
    if problems:
        st.warning("Проверка до отправки:\n" + "\n".join(f"- {problem}" for problem in problems))
    blocked = payload is None or (bool(problems) and jev.validate)
    if st.button("Отправить", type="primary", key="send", icon=":material/send:", disabled=blocked):
        with st.spinner("Жду ответ…"):
            outcome = send(jev, payload, "sandbox")
        st.session_state.sandbox_outcome = {"payload": payload, "outcome": outcome, "url": jev.url,
                                            "api_key_env": curl_env}

    if "sandbox_outcome" in st.session_state:
        st.divider()
        render.show_outcome(**st.session_state.sandbox_outcome, key="sandbox")
    return state, questions


def state_editor(draft: dict, ver: int):
    mode = st.segmented_control("state", ["Текст", "JSON"], default=draft["state_mode"], required=True,
                                key=f"{ver}-state-mode")
    text = st.text_area("state", draft["state_text"], height=160, key=f"{ver}-state", label_visibility="collapsed")
    if mode == "Текст":
        return text
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"state: невалидный JSON — {exc.msg} (строка {exc.lineno}, столбец {exc.colno})")
        return None
    if not isinstance(value, (dict, list, str)):
        st.error("state в режиме JSON должен быть объектом или массивом")
        return None
    st.caption("JSON валиден. На поля можно ссылаться из вопросов в обратных кавычках: `поле`, `список[0]`.")
    return value


def question_editor(question: dict, ver: int) -> tuple[str, dict]:
    key = f"{ver}-{question['uid']}"
    with st.container(border=True):
        name_col, type_col, delete_col = st.columns([6, 4, 1], vertical_alignment="bottom")
        name = name_col.text_input("Имя", question["name"], key=f"{key}-name")
        kind = type_col.selectbox("Тип", QUESTION_TYPES, index=QUESTION_TYPES.index(question["type"]),
                                  format_func=lambda value: f"{value} · {render.TYPE_LABELS[value]}", key=f"{key}-type")
        delete_col.button(":material/delete:", key=f"{key}-delete", help="Удалить вопрос",
                          on_click=remove_question, args=(question["uid"],))
        instructions = st.text_area("instructions — сам вопрос", question["instructions"], height=68,
                                    key=f"{key}-instructions")
        built = {"type": kind, "instructions": from_text(instructions) or ""}
        criteria = criteria_editor(question, kind, key)
        if criteria is not None:
            built["criteria"] = criteria
    return name.strip(), built


def criteria_editor(question: dict, kind: str, key: str):
    if kind == "choice":
        table = st.data_editor(
            pd.DataFrame(question["options"], columns=["Ключ", "Описание"]),
            num_rows="dynamic", hide_index=True, key=f"{key}-options",
            column_config={"Ключ": st.column_config.TextColumn(width="medium"),
                           "Описание": st.column_config.TextColumn(width="large")},
        )
        criteria, repeated = {}, []
        for row in table.to_dict("records"):
            option, description = cell(row["Ключ"]).strip(), cell(row["Описание"])
            if not option and not description.strip():
                continue
            if option in criteria:
                repeated.append(option)
            criteria[option] = from_text(description)
        if repeated:
            st.warning(f"Повторяются ключи: {', '.join(repeated)}")
        st.caption(f"Вариантов: {len(criteria)}. Пустое описание уйдёт как null, {{...}} — как JSON-объект.")
        return criteria

    if kind == "score":
        table = st.data_editor(
            pd.DataFrame(question["levels"], columns=["Описание уровня"]),
            num_rows="dynamic", hide_index=True, key=f"{key}-levels",
        )
        levels = [from_text(cell(row["Описание уровня"])) for row in table.to_dict("records")]
        levels = [level for level in levels if level is not None]
        st.caption(f"Уровней: {len(levels)}. Сверху вниз по возрастанию, верхний — уровень 0.")
        return levels

    true_col, false_col = st.columns(2)
    if_true = true_col.text_input("criteria.true", question["if_true"], key=f"{key}-true",
                                  placeholder="когда «да», необязательно")
    if_false = false_col.text_input("criteria.false", question["if_false"], key=f"{key}-false",
                                    placeholder="когда «нет», необязательно")
    criteria = {flag: from_text(text) for flag, text in (("true", if_true), ("false", if_false)) if text.strip()}
    return criteria or None


def raw_json_editor(state, questions: dict) -> None:
    st.markdown("**JSON запроса**")
    if st.session_state.json_error:
        st.error(st.session_state.json_error)
        st.button("Вернуть JSON из конструктора", key="json-reset", on_click=reset_json_error)
    elif state is not None:
        st.session_state.raw_json = json.dumps({"state": state, "questions": questions}, ensure_ascii=False, indent=2)
    st.text_area("JSON запроса", key="raw_json", height=640, on_change=apply_raw_json, label_visibility="collapsed",
                 disabled=state is None and not st.session_state.json_error)
    if state is None:
        st.caption("Исправьте JSON в state, и редактор обновится.")
    else:
        st.caption("Синхронизирован с конструктором. Правка применяется по Ctrl+Enter или при выходе из поля. "
                   "model задаётся в сайдбаре.")


# --- Эксперименты ----------------------------------------------------------

def experiments_tab(jev: JevClient, state, questions: dict) -> None:
    payload = jev.payload(state, questions) if state is not None else None
    repeat_tab, ab_tab = st.tabs(["Повтор N раз", "A/B формулировок"])
    with repeat_tab:
        repeat_experiment(jev, payload)
    with ab_tab:
        ab_experiment(jev, state, questions)


def repeat_experiment(jev: JevClient, payload: dict | None) -> None:
    st.caption("Текущий запрос из песочницы уходит N раз: видно, насколько стабильны вероятности и латентность.")
    if payload is None:
        st.info("Сначала соберите корректный запрос в песочнице.")
        return
    n_col, concurrency_col, run_col = st.columns([1, 1, 2], vertical_alignment="bottom")
    n = n_col.number_input("Сколько раз", 2, 100, 10, key="repeat-n")
    concurrency = concurrency_col.number_input("Параллельно", 1, 16, 1, key="repeat-concurrency",
                                               help="1 — по очереди: латентность не искажают соседние запросы")
    blocked = bool(check_request(payload)) and jev.validate
    if run_col.button(f"Отправить {n} раз", type="primary", key="repeat-run", disabled=blocked):
        st.session_state.repeat = run_many(jev, [payload] * n, concurrency, "repeat")
    if "repeat" in st.session_state:
        show_repeat(st.session_state.repeat)


def show_repeat(outcomes: list) -> None:
    results = [outcome for outcome in outcomes if isinstance(outcome, JevResult)]
    errors = Counter(str(outcome) for outcome in outcomes if not isinstance(outcome, JevResult))
    st.markdown(f"Успешно **{len(results)}** из {len(outcomes)}")
    for message, count in errors.items():
        st.error(f"×{count}: " + message.replace("\n", "  \n"))
    if not results:
        return

    st.markdown("**Латентность, мс**")
    latency = pd.Series([result.latency_ms for result in results])
    stats = {"мин": latency.min(), "медиана": latency.median(), "среднее": latency.mean(),
             "p90": latency.quantile(0.9), "макс": latency.max(), "std": latency.std()}
    for column, (label, value) in zip(st.columns(len(stats)), stats.items()):
        column.metric(label, "—" if pd.isna(value) else f"{value:.0f}")
    runs = pd.DataFrame({"запуск": range(1, len(results) + 1), "мс": latency})
    st.plotly_chart(render.spread_figure(runs, x="мс", y=None), key="repeat-latency", config=render.CHART_CONFIG)

    st.markdown("**Ответы по запускам**")
    winners: dict[str, Counter] = {}
    rows = []
    for run, result in enumerate(results, start=1):
        for name, answer in result.response.answers.items():
            winners.setdefault(name, Counter())[top_answer(answer)] += 1
            rows += [{"запуск": run, "вопрос": name, "метрика": metric, "значение": value}
                     for metric, value in render.metrics(answer).items()]
    for name, counts in winners.items():
        st.markdown(f"`{name}`: " + ", ".join(f"**{answer}** ×{count}" for answer, count in counts.most_common()))

    long = pd.DataFrame(rows)
    summary = long.groupby(["вопрос", "метрика"], sort=False)["значение"].agg(["mean", "std", "min", "max"])
    summary["размах"] = summary["max"] - summary["min"]
    summary = summary.reset_index().rename(columns={"mean": "среднее", "min": "мин", "max": "макс"})
    number = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(summary, hide_index=True, key="repeat-summary",
                 column_config={column: number for column in ["среднее", "std", "мин", "макс", "размах"]})

    st.caption("Разброс вероятностей и confidence по запускам; score — в таблице выше, у него своя шкала.")
    probabilities = long[long["метрика"] != "score"]
    names = list(winners)
    for start in range(0, len(names), 2):
        for column, name in zip(st.columns(2), names[start:start + 2]):
            with column:
                st.markdown(f"`{name}`")
                frame = probabilities[probabilities["вопрос"] == name]
                st.plotly_chart(render.spread_figure(frame, x="значение", y="метрика", x_range=[-0.02, 1.02]),
                                key=f"repeat-{name}", config=render.CHART_CONFIG)


def top_answer(answer) -> str:
    if answer.type == "noul":
        return "да" if answer.noul >= 0.5 else "нет"
    if answer.type == "choice":
        return answer.choice
    return f"уровень {max(answer.probabilities, key=answer.probabilities.get)}"


def fill_ab(question: dict) -> None:
    text = json.dumps(question, ensure_ascii=False, indent=2)
    st.session_state["ab-a"] = text
    st.session_state["ab-b"] = text


def ab_experiment(jev: JevClient, state, questions: dict) -> None:
    st.caption("Две формулировки одного вопроса на state из песочницы. Оба варианта уходят одним запросом: "
               "Jev отвечает на вопросы независимо, а state и служебные токены оплачиваются один раз.")
    if state is None or not questions:
        st.info("Сначала соберите в песочнице state и хотя бы один вопрос.")
        return
    base_col, fill_col = st.columns([3, 1], vertical_alignment="bottom")
    base = base_col.selectbox("Вопрос из песочницы за основу", list(questions), key="ab-base")
    fill_col.button("Скопировать в A и B", key="ab-fill", on_click=fill_ab, args=(questions[base],))
    if "ab-a" not in st.session_state:
        fill_ab(questions[base])

    variants = {}
    for column, variant in zip(st.columns(2), ["A", "B"]):
        with column:
            text = st.text_area(f"Вариант {variant}: JSON вопроса", key=f"ab-{variant.lower()}", height=300)
            try:
                variants[variant] = json.loads(text)
            except json.JSONDecodeError as exc:
                st.error(f"Невалидный JSON: {exc.msg} (строка {exc.lineno}, столбец {exc.colno})")
    if len(variants) < 2:
        return

    payload = jev.payload(state, variants)
    problems = check_request(payload)
    if problems:
        st.warning("Проверка до отправки:\n" + "\n".join(f"- {problem}" for problem in problems))
    if st.button("Сравнить", type="primary", key="ab-run", disabled=bool(problems) and jev.validate):
        with st.spinner("Жду ответ…"):
            st.session_state.ab = send(jev, payload, "ab")
    if "ab" in st.session_state:
        show_ab(st.session_state.ab)


def show_ab(outcome: JevResult | JevError) -> None:
    if not isinstance(outcome, JevResult):
        render.show_error(outcome)
        return
    render.result_meta(outcome)
    answers = outcome.response.answers
    a, b = answers.get("A"), answers.get("B")
    if a is None or b is None:
        st.error("В ответе нет вариантов A и B")
        return
    for column, (variant, answer) in zip(st.columns(2), [("A", a), ("B", b)]):
        with column:
            render.answer_card(variant, answer, key=f"ab-card-{variant}")
    if a.type != b.type:
        st.info("У вариантов разные типы вопросов, общий график не строю.")
        return

    dist_a, dist_b = render.distribution(a), render.distribution(b)
    if a.type == "score":
        dist_a = {f"уровень {level}": p for level, p in dist_a.items()}
        dist_b = {f"уровень {level}": p for level, p in dist_b.items()}
    st.plotly_chart(render.compare_figure(dist_a, dist_b), key="ab-compare", config=render.CHART_CONFIG)
    table = pd.DataFrame({"A": dist_a, "B": dist_b}).fillna(0.0)
    table["B − A"] = table["B"] - table["A"]
    number = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(table, key="ab-table", column_config={"A": number, "B": number, "B − A": number})


# --- Пакетный режим --------------------------------------------------------

@st.cache_data(show_spinner=False)
def excel_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def read_table(data: bytes, name: str, sheet: str | None) -> pd.DataFrame:
    if name.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet)
    for encoding in ("utf-8-sig", "cp1251"):
        try:  # sep=None — разделитель (запятая, точка с запятой, табуляция) определяется сам
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("кодировка не UTF-8 и не Windows-1251")


def row_states(rows: pd.DataFrame, columns: list) -> list:
    """Одна колонка → state строкой, несколько → объект {колонка: значение}."""
    records = json.loads(rows[columns].to_json(orient="records", force_ascii=False, date_format="iso"))
    if len(columns) == 1:
        only = str(columns[0])
        return ["" if record[only] is None else str(record[only]) for record in records]
    return records


def batch_tab(jev: JevClient, questions: dict) -> None:
    st.caption("Вопросы берутся из песочницы, state — из колонок файла. "
               "Строки с уверенностью ниже порога помечаются на ручную проверку.")
    upload = st.file_uploader("CSV или XLSX", type=["csv", "xlsx"], key="batch-file")
    if upload is not None:
        batch_setup(jev, questions, upload)
    if "batch" in st.session_state:
        st.divider()
        show_batch(st.session_state.batch)


def batch_setup(jev: JevClient, questions: dict, upload) -> None:
    data, file_key = upload.getvalue(), upload.file_id
    try:
        sheet = None
        if upload.name.lower().endswith(".xlsx"):
            sheets = excel_sheets(data)
            sheet = st.selectbox("Лист", sheets, key=f"batch-sheet-{file_key}") if len(sheets) > 1 else sheets[0]
        table = read_table(data, upload.name, sheet)
    except Exception as exc:  # битый файл, чужая кодировка, пустой лист
        st.error(f"Не удалось прочитать файл: {exc}")
        return
    if table.empty:
        st.warning("В файле нет строк")
        return

    st.caption(f"Строк: {len(table)}. Первые пять:")
    st.dataframe(table.head(5), hide_index=True, key="batch-preview")
    text_columns = [column for column in table.columns if pd.api.types.is_string_dtype(table[column])]
    columns = st.multiselect("Колонки для state", list(table.columns), default=(text_columns or list(table.columns))[:1],
                             key=f"batch-columns-{file_key}",
                             help="Одна колонка — state строкой. Несколько — объектом {колонка: значение}, "
                                  "на поля можно ссылаться из вопросов: `колонка`.")
    if questions:
        st.caption("Вопросы из песочницы: " + ", ".join(f"{name} ({q['type']})" for name, q in questions.items()))
    problems = check_request(jev.payload("проверка схемы", questions))
    if problems:
        st.warning("Схема вопросов не проходит проверку:\n" + "\n".join(f"- {problem}" for problem in problems))

    limit_col, concurrency_col, run_col = st.columns(3, vertical_alignment="bottom")
    limit = limit_col.number_input("Сколько строк обработать", 1, len(table), len(table), key=f"batch-limit-{file_key}")
    concurrency = concurrency_col.number_input("Параллельно", 1, 16, 4, key="batch-concurrency")
    blocked = not columns or (bool(problems) and jev.validate)
    if run_col.button(f"Запустить на {limit} строках", type="primary", key="batch-run", disabled=blocked):
        rows = table.head(limit)
        payloads = [jev.payload(state, questions) for state in row_states(rows, columns)]
        outcomes = run_many(jev, payloads, concurrency, "batch")
        st.session_state.batch = {"source": rows[columns].reset_index(drop=True), "outcomes": outcomes,
                                  "name": Path(upload.name).stem}


def batch_table(source: pd.DataFrame, outcomes: list, threshold: float) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        if isinstance(outcome, JevResult):
            answers = outcome.response.answers
            low = {name: render.certainty(answer) for name, answer in answers.items()
                   if render.certainty(answer) < threshold}
            reason = ("низкая уверенность: " + ", ".join(f"{name} {value:.2f}" for name, value in low.items())
                      if low else "")
            rows.append({"на проверку": bool(low), "причина": reason, "мс": round(outcome.latency_ms),
                         **render.flatten(answers)})
        elif isinstance(outcome, JevValidationError):
            rows.append({"на проверку": True, "причина": "не отправлен: " + "; ".join(outcome.problems)})
        else:
            rows.append({"на проверку": True, "причина": "ошибка: " + str(outcome).splitlines()[0]})
    answers_frame = pd.DataFrame(rows)
    answers_frame.columns = [f"{column} (ответ)" if column in source.columns else column
                             for column in answers_frame.columns]
    return pd.concat([source, answers_frame], axis=1)


def highlight(table: pd.DataFrame, color: str):
    return table.style.apply(
        lambda row: [f"background-color: {color}" if row["на проверку"] else ""] * len(row), axis=1)


def to_xlsx(table: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        highlight(table, FLAG_XLSX).to_excel(writer, sheet_name="Jev", index=False)
        sheet = writer.sheets["Jev"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    return buffer.getvalue()


def show_batch(batch: dict) -> None:
    threshold = st.slider("Порог уверенности", 0.0, 1.0, 0.6, 0.05, key="batch-threshold",
                          help="Для choice и score — confidence, для noul — |2·P(да) − 1|. "
                               "Строка уходит на проверку, если хоть один вопрос ниже порога или запрос упал.")
    table = batch_table(batch["source"], batch["outcomes"], threshold)
    answered = sum(isinstance(outcome, JevResult) for outcome in batch["outcomes"])
    st.markdown(f"Ответов **{answered}** из {len(table)}, на ручную проверку **{int(table['на проверку'].sum())}**")
    only_flagged = st.toggle("Только строки на проверку", key="batch-only-flagged")
    view = table[table["на проверку"]] if only_flagged else table
    st.dataframe(highlight(view, FLAG_SCREEN).format(precision=3), key="batch-table")
    st.download_button("Скачать XLSX", to_xlsx(table), file_name=f"{batch['name']}_jev.xlsx", key="batch-download",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       icon=":material/download:")


# --- История ---------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=2)
def load_history(mtime: int) -> list[dict]:
    return history.read()


def preview(state, limit: int = 90) -> str:
    text = " ".join(to_text(state).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def history_row(record: dict) -> dict:
    request = record.get("request") or {}
    response = record.get("response") if isinstance(record.get("response"), dict) else {}
    usage = response.get("usage") or {}
    questions = request.get("questions")
    return {
        "#": record["id"],
        "время": record.get("ts", "").replace("T", " "),
        "источник": record.get("source"),
        "модель": response.get("model") or request.get("model"),
        "вопросы": ", ".join(map(str, questions)) if isinstance(questions, dict) else "",
        "state": preview(request.get("state")),
        "мс": record.get("latency_ms"),
        "₽": usage.get("cost_rub"),
        "итог": "ошибка" if record.get("error") else "ок",
    }


def history_tab(jev: JevClient) -> None:
    records = load_history(history.mtime())[::-1]
    if not records:
        st.info("История пуста: здесь появятся отправленные запросы.")
        return
    sources = sorted({record.get("source", "?") for record in records})
    chosen = st.pills("Источник (ничего не выбрано — все)", sources, selection_mode="multi",
                      key="history-sources") or sources
    shown = [record for record in records if record.get("source") in chosen][:500]
    st.caption(f"Показано {len(shown)} из {len(records)}, новые сверху. Файл: {history.HISTORY_PATH.name}")
    if not shown:
        return
    st.dataframe(pd.DataFrame([history_row(record) for record in shown]), hide_index=True, height=280,
                 key="history-table")

    labels = {record["id"]: f"#{record['id']} · {history_row(record)['время']} · {record.get('source')} · "
                            f"{history_row(record)['вопросы']}" for record in shown}
    # Ключ зависит от числа записей: после нового запроса выбор встаёт на самую свежую
    selected = st.selectbox("Запись", list(labels), format_func=labels.get, key=f"history-selected-{len(records)}")
    show_history_record(jev, next(record for record in shown if record["id"] == selected))


def show_history_record(jev: JevClient, record: dict) -> None:
    request = record.get("request") or {}
    request_col, response_col = st.columns(2)
    with request_col:
        st.markdown("**Запрос**")
        st.json(request, expanded=1)
    with response_col:
        st.markdown("**Ответ**")
        if record.get("error"):
            st.error(record["error"].replace("\n", "  \n"))
        if record.get("response") is not None:
            st.json(record["response"], expanded=1)

    if not record.get("error") and isinstance(record.get("response"), dict):
        try:
            response = JevResponse.model_validate(record["response"])
        except ValueError:
            response = None
        if response:
            render.answers_grid(response.answers, key=f"history-{record['id']}")

    with st.container(horizontal=True):
        resend = st.button("Отправить повторно", type="primary", key="history-resend", icon=":material/replay:")
        st.button("Открыть в песочнице", key="history-open", icon=":material/edit:",
                  on_click=open_in_sandbox, args=(request,))
    st.caption(f"Повтор уйдёт на {jev.url} с моделью {jev.model or '—'} из сайдбара "
               "и откроется здесь новой записью.")
    if resend:
        with st.spinner("Жду ответ…"):
            outcome = send(jev, {**request, "model": jev.model}, "history")
        if isinstance(outcome, JevValidationError):
            render.show_error(outcome)
        else:
            st.rerun()


# --- Страница --------------------------------------------------------------

jev, curl_env = sidebar()
if "draft" not in st.session_state:
    load_preset(next(iter(PRESETS)))

st.title("Jev · песочница System One")
sandbox, experiments, batch, journal = st.tabs(TABS, key="tab", on_change="rerun")
with sandbox:
    current_state, current_questions = sandbox_tab(jev, curl_env)
with experiments:
    experiments_tab(jev, current_state, current_questions)
with batch:
    batch_tab(jev, current_questions)
with journal:
    history_tab(jev)
