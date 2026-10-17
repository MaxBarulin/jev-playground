"""Отображение ответов Jev в Streamlit: карточки с графиками, метрики запроса, ошибки,
плоские строки для таблиц."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from client import JevError, JevResult, JevValidationError, to_curl

ACCENT = "#2a78d6"  # самый вероятный вариант
REST = "#a3a29b"  # остальные варианты
TRACK = "rgba(137, 135, 129, 0.22)"  # фон шкалы noul
GRID = "rgba(137, 135, 129, 0.25)"
AB_COLORS = {"A": "#2a78d6", "B": "#eb6834"}
MAX_BARS = 12  # варианты choice сверх этого сворачиваются в «прочие»
CHART_CONFIG = {"displayModeBar": False}
TYPE_LABELS = {"noul": "да/нет", "choice": "выбор", "score": "шкала"}


# --- Числа из ответа -------------------------------------------------------

def certainty(answer) -> float:
    """Уверенность для порога. У noul confidence нет, берём |2P − 1|: 0 при 0.5, 1 при 0 или 1."""
    if answer.type == "noul":
        return abs(2 * answer.noul - 1)
    return answer.confidence


def distribution(answer) -> dict[str, float]:
    """Вероятности по вариантам: для noul — да/нет, для score — номера уровней."""
    if answer.type == "noul":
        return {"да": answer.noul, "нет": 1 - answer.noul}
    return dict(answer.probabilities)


def metrics(answer) -> dict[str, float]:
    """Все числа ответа для экспериментов с повторами."""
    if answer.type == "noul":
        return {"P(да)": answer.noul}
    values = {"confidence": answer.confidence}
    if answer.type == "score":
        values["score"] = answer.score
    values.update({f"P({key})": p for key, p in answer.probabilities.items()})
    return values


def flatten(answers: dict) -> dict:
    """Ответы одного запроса одной строкой таблицы: ответ, confidence и вероятности по вариантам."""
    row = {}
    for name, answer in answers.items():
        if answer.type == "noul":
            row[f"{name}: P(да)"] = answer.noul
            continue
        row[name] = answer.choice if answer.type == "choice" else answer.score
        row[f"{name}: confidence"] = answer.confidence
        row.update({f"{name}: P({key})": p for key, p in answer.probabilities.items()})
    return row


# --- Графики ---------------------------------------------------------------

def bars_figure(labels: list[str], values: list[float], highlight: int | None, hover: list[str]) -> go.Figure:
    colors = [ACCENT if i == highlight else REST for i in range(len(values))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{value:.2f}" for value in values], textposition="outside", cliponaxis=False,
        customdata=hover, hovertemplate="%{customdata}<br>P = %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(height=46 + 30 * len(labels), margin=dict(l=4, r=44, t=6, b=6), bargap=0.35, showlegend=False)
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(type="category", autorange="reversed", automargin=True)
    return fig


def noul_figure(p: float) -> go.Figure:
    """Шкала 0–1 с отметкой 0.5 («не знает»)."""
    fig = go.Figure()
    fig.add_bar(x=[1], y=[""], orientation="h", width=0.5, marker=dict(color=TRACK, cornerradius=4), hoverinfo="skip")
    fig.add_bar(x=[p], y=[""], orientation="h", width=0.5, marker=dict(color=ACCENT, cornerradius=4),
                hovertemplate=f"P(да) = {p:.3f}<extra></extra>")
    fig.add_vline(x=0.5, line=dict(color=REST, width=1, dash="dot"))
    fig.update_layout(barmode="overlay", height=90, margin=dict(l=36, r=28, t=6, b=30), showlegend=False)
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], ticktext=["0 · нет", "0.5", "1 · да"],
                     showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(visible=False)
    return fig


def compare_figure(dist_a: dict[str, float], dist_b: dict[str, float]) -> go.Figure:
    """A и B рядом на одной оси вероятностей."""
    labels = list(dict.fromkeys([*dist_a, *dist_b]))
    fig = go.Figure()
    for variant, dist in (("A", dist_a), ("B", dist_b)):
        values = [dist.get(label, 0.0) for label in labels]
        fig.add_bar(
            name=variant, x=values, y=labels, orientation="h",
            marker=dict(color=AB_COLORS[variant], cornerradius=4),
            text=[f"{value:.2f}" for value in values], textposition="outside", cliponaxis=False,
            hovertemplate=f"{variant}: %{{x:.3f}}<extra>%{{y}}</extra>",
        )
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, height=70 + 46 * len(labels),
                      margin=dict(l=0, r=44, t=30, b=6),
                      legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"))
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.5, 1], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(type="category", autorange="reversed", automargin=True)
    return fig


def spread_figure(frame: pd.DataFrame, x: str, y: str | None, x_range: list[float] | None = None) -> go.Figure:
    """Точки по запускам: как разбросаны значения от запроса к запросу."""
    fig = px.strip(frame, x=x, y=y, hover_data=["запуск"], color_discrete_sequence=[ACCENT])
    rows = frame[y].nunique() if y else 1
    fig.update_traces(marker=dict(size=9, opacity=0.6))
    fig.update_layout(height=70 + 34 * rows, margin=dict(l=0, r=8, t=6, b=6))
    fig.update_xaxes(range=x_range, gridcolor=GRID, zeroline=False, title=None if y else x)
    fig.update_yaxes(title=None, automargin=True)
    return fig


def _choice_bars(answer) -> tuple[list[str], list[float], list[str]]:
    ranked = sorted(answer.probabilities.items(), key=lambda item: -item[1])
    if len(ranked) > MAX_BARS:
        rest = ranked[MAX_BARS - 1:]
        ranked = ranked[:MAX_BARS - 1] + [(f"прочие ({len(rest)})", sum(p for _, p in rest))]
    labels = [_short(key, 40) for key, _ in ranked]
    return labels, [p for _, p in ranked], [key for key, _ in ranked]


def _score_bars(answer) -> tuple[list[str], list[float], list[str]]:
    levels = list(answer.probabilities)
    full = [f"{level} · {_text(answer.legend.get(level, ''))}" for level in levels]
    return [_short(label, 40) for label in full], [answer.probabilities[level] for level in levels], full


def _short(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# --- Карточки и блоки ------------------------------------------------------

def answer_card(name: str, answer, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{name}** · {TYPE_LABELS.get(answer.type, answer.type)}")
        if answer.type == "noul":
            verdict = "да" if answer.noul >= 0.5 else "нет"
            st.markdown(f"Ответ: **{verdict}** · P(да) = **{answer.noul:.2f}** · уверенность {certainty(answer):.2f}")
            st.plotly_chart(noul_figure(answer.noul), key=key, config=CHART_CONFIG)
            return

        if answer.type == "choice":
            labels, values, hover = _choice_bars(answer)
            highlight = hover.index(answer.choice) if answer.choice in hover else None
            st.markdown(f"Ответ: **{answer.choice}**")
        else:
            labels, values, hover = _score_bars(answer)
            highlight = values.index(max(values))
            top_level = len(values) - 1
            st.markdown(f"Ответ: **{answer.score:.2f}** из {top_level} · самый вероятный уровень {highlight}")
        st.progress(min(max(answer.confidence, 0.0), 1.0), text=f"confidence {answer.confidence:.2f}")
        st.plotly_chart(bars_figure(labels, values, highlight, hover), key=key, config=CHART_CONFIG)


def answers_grid(answers: dict, key: str, columns: int = 2) -> None:
    items = list(answers.items())
    for start in range(0, len(items), columns):
        for column, (name, answer) in zip(st.columns(columns), items[start:start + columns]):
            with column:
                answer_card(name, answer, key=f"{key}-{name}")


def result_meta(result: JevResult) -> None:
    usage = result.response.usage
    cols = st.columns(5)
    cols[0].metric("Латентность", f"{result.latency_ms:.0f} мс")
    cols[1].metric("Ответила модель", result.response.model)
    cols[2].metric("Токены: вход / выход", f"{usage.input_tokens} / {usage.output_tokens}" if usage else "—")
    cols[3].metric("Стоимость", f"{usage.cost_rub:.4f} ₽" if usage and usage.cost_rub is not None else "—")
    cols[4].metric("Попыток", result.attempts)


def show_error(error: JevError) -> None:
    if isinstance(error, JevValidationError):
        st.warning("Запрос не отправлен:\n" + "\n".join(f"- {problem}" for problem in error.problems))
        return
    st.error(str(error).replace("\n", "  \n"))
    body = getattr(error, "body", None)
    if body:
        with st.expander("Тело ответа сервера"):
            if isinstance(body, (dict, list)):
                st.json(body)
            else:
                st.code(str(body), language=None)


def show_outcome(payload: dict, outcome: JevResult | JevError, url: str, key: str,
                 api_key_env: str | None) -> None:
    """Итог запроса: ответы или ошибка, сырой ответ, запрос и curl."""
    is_ok = isinstance(outcome, JevResult)
    if is_ok:
        result_meta(outcome)
    tabs = st.tabs(["Ответы" if is_ok else "Ошибка", "Сырой ответ", "Запрос", "curl"])
    with tabs[0]:
        if is_ok:
            answers_grid(outcome.response.answers, key=key)
        else:
            show_error(outcome)
    with tabs[1]:
        raw = outcome.raw if is_ok else getattr(outcome, "body", None)
        if isinstance(raw, (dict, list)):
            st.json(raw)
        else:
            st.caption(str(raw) if raw else "Сервер не вернул тело ответа")
    with tabs[2]:
        st.json(payload)
    with tabs[3]:
        st.code(to_curl(payload, url, api_key_env), language="bash")
        hint = f"Ключ не подставлен: перед запуском задайте переменную `{api_key_env}`. " if api_key_env else ""
        st.caption(hint + "Команда для bash (Git Bash, WSL); копирование — кнопкой в углу блока.")
