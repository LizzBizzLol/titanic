import pandas as pd
import pytest

from core import compute_dead_children_by_embarked


def make_df():
    """
    Делаем маленький "титаник"-датасет прямо в тесте (автономность).
    Колонки: Survived, Age, Embarked.
    """
    return pd.DataFrame(
        {
            "Survived": [0, 0, 1, 0, 0, 0, 0],
            "Age":      [5, 17, 10, 19, None, 12, 18],
            "Embarked": ["S", "C", "S", "S", "Q", "Q", None],
        }
    )
    # Пояснение по строкам:
    # idx0: погиб, 5 лет, S -> ребенок
    # idx1: погиб, 17, C -> ребенок
    # idx2: выжил, 10, S -> не считаем
    # idx3: погиб, 19, S -> старше max_age (если max_age<=18)
    # idx4: погиб, Age=None, Q -> Age NaN, не считаем
    # idx5: погиб, 12, Q -> ребенок
    # idx6: погиб, 18, Embarked=None -> ребенок, неизвестный пункт


def test_counts_by_embarked_max_age_18():
    df = make_df()
    res = compute_dead_children_by_embarked(df, max_age=18)

    # Проверяем, что есть ожидаемые колонки
    assert list(res.columns) == ["Embarked", "EmbarkedName", "DeadChildrenCount", "MaxAgeInGroup"]

    # Переводим в словарь для удобства проверок
    counts = dict(zip(res["Embarked"], res["DeadChildrenCount"]))

    # Ожидаем:
    # S: idx0 (5) -> 1
    # C: idx1 (17) -> 1
    # Q: idx5 (12) -> 1
    # Unknown: idx6 (18) -> 1
    assert counts["S"] == 1
    assert counts["C"] == 1
    assert counts["Q"] == 1
    assert counts["Unknown"] == 1


def test_max_age_filter_strictness():
    df = make_df()

    # max_age=12 -> должны попасть: idx0 (5, S) и idx5 (12, Q). idx1 (17) уже не должен.
    res = compute_dead_children_by_embarked(df, max_age=12)
    counts = dict(zip(res["Embarked"], res["DeadChildrenCount"]))

    assert counts.get("S", 0) == 1
    assert counts.get("Q", 0) == 1
    assert counts.get("C", 0) == 0  # не должно быть погибших детей в C при max_age=12


def test_max_age_in_group_is_correct():
    df = make_df()
    res = compute_dead_children_by_embarked(df, max_age=18)

    # Для Q у нас только один ребенок 12 -> max должен быть 12
    q_row = res[res["Embarked"] == "Q"].iloc[0]
    assert float(q_row["MaxAgeInGroup"]) == 12.0

    # Для Unknown у нас один ребенок 18 -> max должен быть 18
    u_row = res[res["Embarked"] == "Unknown"].iloc[0]
    assert float(u_row["MaxAgeInGroup"]) == 18.0


def test_invalid_max_age_raises():
    df = make_df()
    with pytest.raises(ValueError):
        compute_dead_children_by_embarked(df, max_age=0)
    with pytest.raises(ValueError):
        compute_dead_children_by_embarked(df, max_age=19)


def test_missing_required_columns_raises():
    df = pd.DataFrame({"Survived": [0], "Age": [5]})  # нет Embarked
    with pytest.raises(ValueError):
        compute_dead_children_by_embarked(df, max_age=10)
