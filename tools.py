from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict, Counter
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional, Any, Literal
from holidayskr import year_holidays
from datetime import date
from dateutil.relativedelta import relativedelta
from functools import lru_cache
import datetime
import requests
import json
import os
import pandas as pd


load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

BASE_DIR = Path(__file__).resolve().parent

with open(f"{BASE_DIR}/settings/payload.json", "r", encoding="utf-8") as fr:
    payload = json.load(fr)["func_param"]

with open(f"{BASE_DIR}/data/merchant_cls.json", "r", encoding="utf-8") as fr:
    merchant_cls = json.load(fr)

time_payload = payload["time_tool"]
weather_payload = payload["weather_tool"]
url_payload = payload["url_tool"]


class ToolOutput(BaseModel):
    """모든 도구가 공통으로 반환하는 표준 출력 타입 클래스"""

    status: Literal["success", "error"] = Field(
        description="도구 실행의 성공 여부(success or error)"
    )

    message: str = Field("", description="실행 결과에 대한 메시지")

    data: Any | None = Field(
        None, description="도구의 실행 결과로 담기는 데이터, 도구마다 형식이 다름"
    )

    error_status: str | None = Field(
        None, description="프로그래밍 에러에 대한 상세 정보"
    )

def load_df() -> pd.DataFrame:
    return pd.read_csv(f"{BASE_DIR}/data/data_final.csv", encoding="utf-8")

def _normalize_name(s: str) -> str:
    return s.strip().replace("*", "")

def filter_data_with_df(df: pd.DataFrame, name: str) -> ToolOutput:
    """가맹점명에 맞는 데이터를 필터링하는 함수입니다. asterisk를 제외한 이름이 같은 것만 필터링합니다.

    Args:
        name (str): 입력된 가맹점명

    Returns:
        ToolOutput: tool 실행 결과.
    """
    try:
        left = df["가맹점명"].astype(str).str.replace("*", "", regex=False).str.strip()
        right = _normalize_name(name)
        result = df[left == right]
        if result.empty:
            return ToolOutput(
                status="error",
                message=f"'{name}'과(와) 일치하는 가맹점명을 찾을 수 없습니다.",
            )
        elif len(result) > 24:
            # 필터링된 가맹점이 하나 이상인 경우
            return ToolOutput(
                status="error",
                message=f"'{name}'과(와) 일치하는 가맹점이 너무 많습니다. 다른 정보도 같이 입력해주세요."
            )
        else:
            return ToolOutput(
                status="success",
                message=f"'{name}'에 대한 가맹점 정보를 성공적으로 조회했습니다. 총 {len(result)}건의 데이터가 있습니다.",
                data=result.to_dict("records"),
            )
    except Exception as e:
        return ToolOutput(
            status="error",
            error_status=str(e)
        )

def _get_events(
    year_str: str, extra_events: Optional[Dict[str, Tuple[int, int]]] = None
) -> List[Tuple[date, str]]:
    """필요한 기념일들의 날짜와 이름을 반환하는 함수

    Args:
        year_str (str): 탐색 연도
        extra_events (Dict[str, Tuple[int, int]]): 추가할 기념일들

    Returns:
        List[Tuple[date, str]]: 통합된 기념일
    """
    holidays = year_holidays(year_str)
    exist_events = {"신정", "설날", "어린이날", "추석", "크리스마스"}

    year_events = [
        (d, name) for d, name in holidays if any(ev in name for ev in exist_events)
    ]

    if extra_events:
        extra_events = [
            (datetime.datetime.strptime(f"{year_str}/{d}", "%Y/%m/%d").date(), name)
            for name, d in extra_events.items()
        ]
        return year_events + extra_events
    else:
        return year_events

def get_holidays_in_months() -> ToolOutput:
    """n개월 이내의 기념일들을 반환합니다.

    Returns:
        ToolOutput: tool 실행 결과
    """
    try:
        year_str: str = time_payload["year_str"]
        n: int = time_payload["n"]
        extra_events: Optional[Dict[str, str]] = time_payload.get("extra_events", None)
        undated_events: Optional[List[str]] = time_payload.get("undated_events", None)

        events = _get_events(year_str, extra_events)
        today = date.today()
        n_later = today + relativedelta(months=n + 1, day=1) - relativedelta(days=1)
        # n개월 후 달 전체릂 포함하기 위해 n + 1개월에서 하루를 뺌

        if n_later.year != today.year:
            events += _get_events(str(n_later.year), extra_events)
        
        result = []
        # 오늘부터 n개월 내에 있는 모든 holiday 반환
        for d, name in events:
            if today <= d <= n_later:
                result.append(name)

        if undated_events:
            result += undated_events

        return ToolOutput(
            status="success",
            message=f"{n}개월 이내의 가까운 기념일들을 {len(result)}개 찾았습니다.",
            data=result,
        )
    except Exception as e:
        return ToolOutput(
            status="error",
            message="툴 함수 실행 중 오류가 발생하였습니다.",
            error_status=str(e),
        )


def get_weather() -> ToolOutput:
    """오늘 날짜로부터 5일 간의 전반적인 날씨와 기온을 반환합니다.

    Returns:
        ToolOutput: tool 실행 결과
    """
    try:
        lat: str = weather_payload["lat"]
        lon: str = weather_payload["lon"]
        lang: str = weather_payload["lang"]

        api = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={WEATHER_API_KEY}&lang={lang}"
        result = requests.get(api, timeout=8)
        if result.status_code != 200:
            return ToolOutput(status="error", message="날씨 응답 형식 오류", error_status=data)
        data = result.json()
        weather_data = data["list"]
        weather_data = [
            (data["weather"][0]["description"], data["main"]["temp"], data["dt_txt"])
            for data in weather_data
        ]

        result = {}
        by_day = defaultdict(list)
        for weather, temp, dt in weather_data:
            day = dt.split(" ")[0]
            by_day[day].append((weather, temp))

        for day, records in by_day.items():
            avg_temp = sum(t for _, t in records) / len(records)

            weather_count = Counter(w for w, _ in records)
            most_common_weather = ", ".join(
                w for w, c in weather_count.items() if c == max(weather_count.values())
            )

            result[day] = {
                "weather": most_common_weather,
                "avg_temp": round(avg_temp, 2),
            }

        return ToolOutput(
            status="success",
            message="오늘 날짜로부터 5일간의 날씨, 평균 기온입니다.",
            data=result
        )
    
    except Exception as e:
        return ToolOutput(
            status="error",
            message="툴 함수 실행 중 오류가 발생하였습니다.",
            error_status=str(e),
        )

def get_sns_url() -> ToolOutput:
    """get right SNS URL

    Returns:
        ToolOutput: tool 실행 결과
    """
    return ToolOutput(
        status="success",
        message="참조할 SNS URL을 반환해드립니다.",
        data = url_payload
    )

def get_merchant_cls(industry: str) -> str:
    """업종을 받아 대분류를 반환하는 함수

    Args:
        industry (str): 입력된 업종
        
    Returns:
        str: 대분류
    """
    for d in merchant_cls:
        if industry == d["업종"]:
            return d["분류"]
    else:
        return "분류 없음"