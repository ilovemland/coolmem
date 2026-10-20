from __future__ import annotations

from datetime import UTC, datetime

_WEEKDAYS_ZH = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
_WEEKDAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def format_time(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    moment = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    numeric = moment.strftime("%Y-%m-%d %H:%M")
    chinese = f"{moment.year}年{moment.month}月{moment.day}日 {_WEEKDAYS_ZH[moment.weekday()]}"
    english = (
        f"{moment.strftime('%B')} {moment.year} {_WEEKDAYS_EN[moment.weekday()]} "
        f"{moment.strftime('%I:%M %p')}"
    )
    return f"{numeric} {chinese} {english}"
