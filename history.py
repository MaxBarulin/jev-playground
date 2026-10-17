"""Журнал запросов в history.jsonl: одна строка — один запрос и его итог. Ключ API сюда не попадает."""

import json
from datetime import datetime
from pathlib import Path

from client import JevError, JevResult

HISTORY_PATH = Path(__file__).resolve().parent / "history.jsonl"


def entry(source: str, url: str, payload: dict, outcome: JevResult | JevError) -> dict:
    """source — откуда запрос: sandbox, repeat, ab, batch, history."""
    record = {"ts": datetime.now().isoformat(timespec="seconds"), "source": source, "url": url, "request": payload}
    if isinstance(outcome, JevResult):
        record.update(response=outcome.raw, latency_ms=round(outcome.latency_ms, 1), attempts=outcome.attempts)
    else:
        record.update(error=str(outcome), status=getattr(outcome, "status_code", None),
                      response=getattr(outcome, "body", None))
    return record


def append(entries: list[dict], path: Path = HISTORY_PATH) -> None:
    with path.open("a", encoding="utf-8") as file:
        for record in entries:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read(path: Path = HISTORY_PATH) -> list[dict]:
    """Все записи, от старых к новым. id — номер строки в файле. Битые строки пропускаются."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["id"] = number
            records.append(record)
    return records


def mtime(path: Path = HISTORY_PATH) -> int:
    return path.stat().st_mtime_ns if path.exists() else 0
