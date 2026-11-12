import os
import urllib.request

import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="산업용 기계 고장 예측 및 예방 정비 시스템",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ 산업용 기계 고장 예측 및 예방 정비 시스템")
st.caption("UCI AI4I 2020 데이터로 산업용 기계의 운전 조건과 고장 여부를 분석합니다.")

DATA_FILE = "ai4i2020.csv"

# UCI 공식 저장소에 있는 원본 CSV 주소
UCI_CSV_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

TARGET = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_LABELS = {
    "Type": "제품 유형 (Type)",
    "Air temperature [K]": "공기 온도 (K)",
    "Process temperature [K]": "공정 온도 (K)",
    "Rotational speed [rpm]": "회전 속도 (rpm)",
    "Torque [Nm]": "토크 (Nm)",
    "Tool wear [min]": "공구 마모 시간 (min)",
}

REQUIRED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]


# ============================================================
# 2. CSV 파일 준비
#    GitHub에 CSV가 있으면 그것을 사용하고,
#    없으면 UCI 공식 주소에서 같은 원본 CSV를 내려받습니다.
# ============================================================
def prepare_csv():
    if os.path.exists(DATA_FILE):
        return True

    try:
        st.info("저장소에서 CSV를 찾지 못해 UCI 공식 사이트에서 원본 데이터를 가져옵니다.")
        urllib.request.urlretrieve(UCI_CSV_URL, DATA_FILE)
        return True
    except Exception as error:
        st.error("UCI 공식 데이터 파일을 자동으로 내려받지 못했습니다.")
        st.write(f"오류 내용: {error}")
        st.markdown(
            "직접 [UCI AI4I 2020 데이터셋 페이지]"
            "(https://archive.ics.uci.edu/dataset/601/ai4i)에서 "
            "`ai4i2020.csv`를 내려받아 `main.py`와 같은 폴더에 올려 주세요."
        )
        return False


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding="utf-8-sig")


if not prepare_csv():
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as error:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    st.error("CSV 파일의 열 이름이 예상한 형식과 다릅니다.")
    st.write("누락된 열:", missing_columns)
    st.write("현재 파일의 열:", list(df.columns))
    st.stop()

for col in NUMERIC_FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)
df["Type"] = df["Type"].fillna("미상").astype(str)

st.success("UCI AI4I 2020 데이터 파일을 불러왔습니다.")

# ============================================================
# 3. 데이터 소개
# ============================================================
st.header("1. 데이터 살펴보기")

st.write(
    "이 데이터는 산업용 기계의 운전 조건과 고장 여부를 담은 "
    "UCI AI4I 2020 Predictive Maintenance Dataset입니다."
)
st.warning(
    "이 데이터는 실제 공장에서 직접 측정한 자료가 아니라 "
    "산업 환경을 모사해 만든 합성 데이터입니다."
)

c1, c2, c3 = st.columns(3)
c1.metric("데이터 행 수", f"{len(df):,}개")
c2.metric("변수 수", f"{len(df.columns):,}개")
c3.metric("고장 데이터 수", f"{int((df[TARGET] == 1).sum()):,}개")

with st.expander("데이터 미리보기"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("변수 설명"):
    st.markdown(
        """
        - **Type**: 제품 유형(L, M, H)
        - **Air temperature [K]**: 공기 온도(K)
        - **Process temperature [K]**: 공정 온도(K)
        - **Rotational speed [rpm]**: 회전 속도(rpm)
        - **Torque [Nm]**: 토크(Nm)
        - **Tool wear [min]**: 공구 마모 시간(min)
        - **Machine failure**: 고장 여부(0=정상, 1=고장)
        - **UDI, Product ID**: 식별자이므로 예측 변수에서 제외
        - **TWF, HDF, PWF, OSF, RNF**: 고장 유형 열이므로 데이터 누수 방지를 위해 제외
        """
    )

st.markdown(
    "데이터 출처: [UCI AI4I 2020 공식 페이지]"
    "(https://archive.ics.uci.edu/dataset/601/ai4i)"
)

# ============================================================
# 4. 정상/고장 데이터 분포
# ============================================================
st.header("2. 고장 데이터 분포")

counts = (
    df[TARGET]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(index={0: "정상", 1: "고장"})
    .rename_axis("기계 상태")
    .reset_index(name="데이터 수")
)

fig_counts = px.bar(
    counts,
    x="기계 상태",
    y="데이터 수",
    text="데이터 수",
    title="정상 및 고장 데이터 수",
)
st.plotly_chart(fig_counts, use_container_width=True)
st.info(
    "고장 데이터가 정상 데이터보다 적은 불균형 데이터입니다. "
    "따라서 정확도뿐 아니라 정밀도, 재현율, F1-score도 함께 확인합니다."
)

# ============================================================
# 5. 운전 조건과 고장 여부 그래프
# ============================================================
st.header("3. 운전 조건과 고장 여부의 관계")

chart_titles = {
    "Air temperature [K]": "공기 온도와 고장 여부",
    "Process temperature [K]": "공정 온도와 고장 여부",
    "Rotational speed [rpm]": "회전 속도와 고장 여부",
    "Torque [Nm]": "토크와 고장 여부",
    "Tool wear [min]": "공구 마모 시간과 고장 여부",
}

for col, title in chart_titles.items():
    temp = df[[col, TARGET]].dropna().copy()
    temp["기계 상태"] = temp[TARGET].map({0: "정상", 1: "고장"})
    fig = px.box(
        temp,
        x="기계 상태",
        y=col,
        color="기계 상태",
        points="outliers",
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"정상과 고장 데이터에서 {col}의 분포를 비교합니다. "
        "분포 차이만으로 직접적인 원인과 결과를 단정할 수는 없습니다."
    )

# ============================================================
# 6. 예측 변수 선택 및 모델 학습
# ============================================================
st.header("4. 머신러닝 모델 학습")

st.write("학습에 사용할 변수를 선택해 주세요.")
selected_features = []
feature_cols = st.columns(3)

for index, (col, label) in enumerate(FEATURE_LABELS.items()):
    with feature_cols[index % 3]:
        default_value = True
        if st.checkbox(label, value=default_value, key=f"select_{col}"):
            selected_features.append(col)

if len(selected_features) == 0:
    st.warning("예측 변수를 한 개 이상 선택해 주세요.")
    st.stop()

test_percent = st.slider(
    "테스트 데이터 비율(%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)
seed = st.number_input(
    "무작위 분할 기준값",
    min_value=0,
    max_value=9999,
    value=42,
    step=1,
)

if df[TARGET].nunique() != 2:
    st.error("Machine failure 열에 정상(0)과 고장(1)이 모두 있어야 합니다.")
    st.stop()

X = df[selected_features].copy()
y = df[TARGET].copy()

numeric_selected = [c for c in selected_features if c in NUMERIC_FEATURES]
categorical_selected = [c for c in selected_features if c == "Type"]

transformers = []

if numeric_selected:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers.append(("numeric", numeric_pipeline, numeric_selected))

if categorical_selected:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers.append(("categorical", categorical_pipeline, categorical_selected))

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=int(seed),
            ),
        ),
    ]
)

try:
    class_counts = y.value_counts()
    stratify_value = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_percent / 100,
        random_state=int(seed),
        stratify=stratify_value,
    )

    if y_train.nunique() < 2:
        st.error("학습 데이터에 정상과 고장 두 종류가 모두 포함되지 않았습니다.")
        st.stop()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

except Exception as error:
    st.error(f"모델 학습 중 오류가 발생했습니다: {error}")
    st.stop()

# ============================================================
# 7. 모델 평가
# ============================================================
st.subheader("모델 평가 결과")

m1, m2, m3, m4 = st.columns(4)
m1.metric("정확도", f"{accuracy_score(y_test, y_pred):.3f}")
m2.metric("정밀도", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
m3.metric("재현율", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
m4.metric("F1-score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

st.write("**혼동 행렬**")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["실제 정상", "실제 고장"],
    columns=["예측 정상", "예측 고장"],
)
st.dataframe(cm_df, use_container_width=True)

st.write("**분류 보고서**")
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["정상", "고장"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.caption(
    "재현율은 실제 고장 데이터를 고장이라고 찾아낸 비율입니다. "
    "고장을 놓치는 상황도 중요하므로 여러 평가 지표를 함께 살펴야 합니다."
)

# ============================================================
# 8. 사용자 입력을 통한 예측
# ============================================================
st.header("5. 운전 조건을 입력해 고장 가능성 예측하기")
st.write("제품 유형과 운전 조건을 입력한 뒤 예측 버튼을 눌러 주세요.")

with st.form("prediction_form"):
    input_values = {}

    if "Type" in selected_features:
        input_values["Type"] = st.selectbox(
            "제품 유형",
            options=["L", "M", "H"],
        )

    input_columns = st.columns(2)
    defaults = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
    }

    for index, col in enumerate(numeric_selected):
        with input_columns[index % 2]:
            input_values[col] = st.number_input(
                FEATURE_LABELS[col],
                value=defaults[col],
                key=f"input_{col}",
            )

    submitted = st.form_submit_button("고장 가능성 예측")

if submitted:
    input_df = pd.DataFrame([input_values], columns=selected_features)

    try:
        predicted_class = int(model.predict(input_df)[0])
        failure_probability = float(model.predict_proba(input_df)[0][1])

        if predicted_class == 1:
            st.error("예측 결과: 고장 가능성이 있는 상태입니다.")
        else:
            st.success("예측 결과: 정상으로 분류되었습니다.")

        st.metric("예측된 고장 확률", f"{failure_probability * 100:.1f}%")
        st.progress(min(max(failure_probability, 0.0), 1.0))

    except Exception as error:
        st.error(f"예측 중 오류가 발생했습니다: {error}")

st.warning(
    "예측 결과는 참고용이며 실제 정비 판단이나 안전 점검을 대신하지 않습니다."
)

# ============================================================
# 9. 기계공학 관점 해석
# ============================================================
st.header("6. 기계공학 관점에서 생각하기")

st.markdown(
    """
    - **온도 변화**: 공기 온도와 공정 온도는 기계의 열적 작동 환경을 나타냅니다.
      온도 변화는 부품의 열팽창과 냉각 상태를 살펴보는 단서가 될 수 있습니다.
    - **회전 속도**: 회전 속도는 회전체의 작동 조건입니다.
      속도 변화에 따라 진동이나 동적 하중이 달라질 수 있습니다.
    - **토크**: 토크는 회전시키는 힘과 관련된 값으로 기계에 걸리는 회전 부하를
      이해하는 데 도움이 됩니다.
    - **공구 마모**: 공구 사용 시간이 누적되면 마모 상태를 점검하고 교체 시기를
      판단할 필요가 있습니다.
    - **예방 정비**: 온도, 회전 속도, 토크, 공구 마모 등을 주기적으로 확인하면
      이상 징후를 조기에 발견하는 데 도움이 될 수 있습니다.
    """
)

st.info(
    "위 내용은 기계공학 관점의 해석입니다. 데이터의 관계만으로 특정 조건이 "
    "고장의 직접 원인이라고 단정할 수는 없습니다."
)

st.divider()
st.caption(
    "데이터 출처: UCI Machine Learning Repository, "
    "AI4I 2020 Predictive Maintenance Dataset"
)
st.caption("사용 기술: Python, Streamlit, pandas, Plotly, scikit-learn")

