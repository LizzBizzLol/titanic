from __future__ import annotations

import pandas as pd


EMBARKED_MAP = {
    "S": "Southampton",
    "C": "Cherbourg",
    "Q": "Queenstown",
}


def compute_dead_children_by_embarked(df: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """
    Вариант №13:
    Подсчитать количество погибших детей (Survived==0) по каждому пункту посадки (Embarked),
    указав максимальный возраст ребенка (max_age от 1 до 18).

    Возвращает DataFrame с колонками:
      - Embarked
      - EmbarkedName
      - DeadChildrenCount
      - MaxAgeInGroup
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    if not isinstance(max_age, int):
        raise TypeError("max_age must be int")

    if max_age < 1 or max_age > 18:
        raise ValueError("max_age must be in range [1, 18]")

    required_cols = {"Survived", "Age", "Embarked"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Копия, чтобы не трогать исходный df
    data = df[["Survived", "Age", "Embarked"]].copy()

    # Age может быть NaN, приводим к числу корректно
    data["Age"] = pd.to_numeric(data["Age"], errors="coerce")

    # Фильтр: дети до max_age включительно + погибшие
    dead_children = data[(data["Survived"] == 0) & (data["Age"].notna()) & (data["Age"] <= max_age)]

    if dead_children.empty:
        # Возвращаем пустую таблицу правильной формы
        return pd.DataFrame(
            columns=["Embarked", "EmbarkedName", "DeadChildrenCount", "MaxAgeInGroup"]
        )

    # Группируем по Embarked
    grouped = (
        dead_children.groupby("Embarked", dropna=False)
        .agg(
            DeadChildrenCount=("Embarked", "size"),
            MaxAgeInGroup=("Age", "max"),
        )
        .reset_index()
    )

    # Человеческое имя пункта посадки
    grouped["Embarked"] = grouped["Embarked"].fillna("Unknown")
    grouped["EmbarkedName"] = grouped["Embarked"].map(EMBARKED_MAP).fillna("Unknown")

    # Сортировка: сначала больше погибших детей
    grouped = grouped.sort_values(by=["DeadChildrenCount", "Embarked"], ascending=[False, True])

    # Красивые типы
    grouped["DeadChildrenCount"] = grouped["DeadChildrenCount"].astype(int)

    return grouped[["Embarked", "EmbarkedName", "DeadChildrenCount", "MaxAgeInGroup"]]
