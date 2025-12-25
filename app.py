import streamlit as st
import pandas as pd

from core import compute_dead_children_by_embarked


st.set_page_config(page_title="Titanic — Streamlit Lab #3", layout="centered")

st.title("Лабораторная №3: Streamlit + Titanic")
st.caption(
    "13: Количество погибших детей по пунктам посадки "
    "при заданном максимальном возрасте."
)

# --- Загрузка данных ---
with st.sidebar:
    st.header("Настройки")
    csv_path = st.text_input("Путь к файлу CSV", value="titanic_train.csv")
    max_age = st.slider("Максимальный возраст ребёнка", min_value=1, max_value=18, value=12)
    show_raw = st.checkbox("Показать исходные данные (первые 20 строк)", value=False)

st.divider()


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Не удалось прочитать файл `{csv_path}`.\n\nОшибка: {e}")
    st.stop()

# --- Показ исходника (опционально) ---
if show_raw:
    st.subheader("Исходные данные (head)")
    st.dataframe(df.head(20), use_container_width=True)

# --- Основной расчёт ---
st.subheader("Результат (таблица)")

try:
    result = compute_dead_children_by_embarked(df, max_age=max_age)
except Exception as e:
    st.error(f"Ошибка при вычислении результата: {e}")
    st.stop()

if result.empty:
    st.info("По заданному возрасту не найдено погибших детей (или нет подходящих данных).")
else:
    st.dataframe(result, use_container_width=True)

    total_dead_children = int(result["DeadChildrenCount"].sum())
    st.success(f"Всего погибших детей (Age ≤ {max_age}): {total_dead_children}")
