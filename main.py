 적용할게.

요청한 조건에 맞춰 준비된 파일은 다음과 같아.

main.py 다운로드

requirements.txt 다운로드

이제 초보자 기준으로 한 단계씩 진행하자.

1단계. 데이터 파일 준비하기
먼저 UCI 공식 사이트에서 ai4i2020.csv 파일을 준비해야 해.

UCI AI4I 2020 데이터셋

UCI 공식 데이터셋 페이지 열기

위 링크를 클릭해.

페이지에서 Download를 눌러 데이터 파일을 받아.

압축 파일이라면 압축을 풀어.

안에 있는 CSV 파일을 확인해.

주의 파일 이름을 반드시 ai4i2020.csv로 맞춰야 해. 데이터 내용은 수정하거나 다른 파일로 대체하지 마.

먼저 한 가지만 확인할게.

ai4i2020.csv 파일을 다운로드했어?


main.py
코드

requirements.txt
문서




# ============================================================
# 산업용 기계 고장 예측 및 예방 정비 시스템
# 사용 데이터: UCI AI4I 2020 Predictive Maintenance Dataset
# 필수 데이터 파일: 같은 폴더의 ai4i2020.csv
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ------------------------------------------------------------
# 1. 페이지 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="산업용 기계 고장 예측 및 예방 정비 시스템",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ 산업용 기계 고장 예측 및 예방 정비 시스템")
st.caption("UCI AI4I 2020 데이터로 산업용 기계의 운전 조건과 고장 여부를 분석합니다.")

DATA_FILE = "ai4i2020.csv"
TARGET = "Machine failure"

# 예측에 사용할 수 있는 변수
FEATURE_LABELS = {
    "Type": "제품 유형 (Type)",
    "Air temperature [K]": "공기 온도 (K)",
    "Process temperature [K]": "공정 온도 (K)",
    "Rotational speed [rpm]": "회전 속도 (rpm)",
    "Torque [Nm]": "토크 (Nm)",
    "Tool wear [min]": "공구 마모 시간 (min)",
}

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]

# 고장 원인을 나타내는 세부 열은 목표값을 미리 알려주는 데이터 누수 가능성이 있어 제외
EXCLUDED_COLUMNS = [
    "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"
]

REQUIRED_COLUMNS = [
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"
]


# ------------------------------------------------------------
# 2. CSV 파일 불러오기 및 데이터 확인
# ------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    # 반드시 사용자가 제공한 ai4i2020.csv만 읽습니다.
    data = pd.read_csv(file_path, encoding="utf-8")
    return data


if not os.path.exists(DATA_FILE):
    st.error(
        f"'{DATA_FILE}' 파일을 찾을 수 없습니다. "
        "GitHub 저장소에서 main.py와 같은 위치(저장소 최상위 폴더)에 "
        "ai4i2020.csv를 업로드해 주세요."
    )
    st.info(
        "이 앱은 임의 데이터나 다른 데이터셋을 대신 사용하지 않습니다. "
        "UCI 공식 AI4I 2020 데이터 파일이 준비된 뒤 다시 실행해 주세요."
    )
    st.markdown(
