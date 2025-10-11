import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data1 = pd.read_csv('./data/raw_data/big_data_set1_f.csv' , encoding='cp949')

col_mapping = {
    "ENCODED_MCT": "가맹점구분번호",
    "MCT_BSE_AR": "가맹점주소",
    "MCT_NM": "가맹점명",
    "MCT_BRD_NUM": "브랜드구분코드",
    "MCT_SIGUNGU_NM": "가맹점지역",
    "HPSN_MCT_ZCD_NM": "업종",
    "HPSN_MCT_BZN_CD_NM": "상권",
    "ARE_D": "개설일",
    "MCT_ME_D": "폐업일"
}

data1 = data1.rename(columns=col_mapping)
data1 = data1.drop(columns=['가맹점지역', '브랜드구분코드'])
data1["상권"].fillna("성동구", inplace=True)
data1 = data1.drop(columns=['상권포함여부'])

data2 = pd.read_csv('./data/raw_data/big_data_set2_f.csv' , encoding='cp949')

col_mapping = {
    "ENCODED_MCT": "가맹점구분번호",
    "TA_YM": "기준년월",
    "MCT_OPE_MS_CN": "가맹점 운영개월수 구간",
    "RC_M1_SAA": "매출금액 구간",
    "RC_M1_TO_UE_CT": "매출건수 구간",
    "RC_M1_UE_CUS_CN": "유니크 고객 수 구간",
    "RC_M1_AV_NP_AT": "객단가 구간",
    "APV_CE_RAT": "취소율 구간",
    "DLV_SAA_RAT": "배달매출금액 비율",
    "M1_SME_RY_SAA_RAT": "동일 업종 매출금액 비율",
    "M1_SME_RY_CNT_RAT": "동일 업종 매출건수 비율",
    "M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "M12_SME_RY_ME_MCT_RAT": "동일 업종 내 해지 가맹점 비중",
    "M12_SME_BZN_ME_MCT_RAT": "동일 상권 내 해지 가맹점 비중"
}

data2 = data2.rename(columns=col_mapping)
data2_sorted = data2.sort_values(by=["가맹점구분번호", "기준년월"]).reset_index(drop=True)
data2_sorted = data2_sorted.replace(-999999.9, np.nan)

data3 = pd.read_csv('./data/raw_data/big_data_set3_f.csv' , encoding='cp949')

col_mapping = {
    "ENCODED_MCT": "가맹점구분번호",
    "TA_YM": "기준년월",
    "M12_MAL_1020_RAT": "남성 20대이하 고객 비중",
    "M12_MAL_30_RAT": "남성 30대 고객 비중",
    "M12_MAL_40_RAT": "남성 40대 고객 비중",
    "M12_MAL_50_RAT": "남성 50대 고객 비중",
    "M12_MAL_60_RAT": "남성 60대이상 고객 비중",
    "M12_FME_1020_RAT": "여성 20대이하 고객 비중",
    "M12_FME_30_RAT": "여성 30대 고객 비중",
    "M12_FME_40_RAT": "여성 40대 고객 비중",
    "M12_FME_50_RAT": "여성 50대 고객 비중",
    "M12_FME_60_RAT": "여성 60대이상 고객 비중",
    "MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "RC_M1_SHC_RSD_UE_CLN_RAT": "거주 이용 고객 비율",
    "RC_M1_SHC_WP_UE_CLN_RAT": "직장 이용 고객 비율",
    "RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 이용 고객 비율"
}

data3 = data3.rename(columns=col_mapping)
data3_sorted = data3.sort_values(by=["가맹점구분번호", "기준년월"]).reset_index(drop=True)
data3_sorted = data3_sorted.replace(-999999.9, np.nan)

data23 = pd.merge(data2_sorted, data3_sorted, on=["가맹점구분번호", "기준년월"], how="left")
data23 = data23.sort_values(by=["가맹점구분번호", "기준년월"]).reset_index(drop=True)
data123 = pd.merge(data23, data1, on="가맹점구분번호", how="left")
data123 = data123.sort_values(by=["가맹점구분번호", "기준년월"]).reset_index(drop=True)

cols = [
    "남성 20대이하 고객 비중", "남성 30대 고객 비중", "남성 40대 고객 비중",
    "남성 50대 고객 비중", "남성 60대이상 고객 비중",
    "여성 20대이하 고객 비중", "여성 30대 고객 비중",
    "여성 40대 고객 비중", "여성 50대 고객 비중", "여성 60대이상 고객 비중"
]

data123_filled = data123[cols].fillna(0)
data123["고객비중합"] = data123_filled.sum(axis=1)

data123["고객비중합"] = data123[cols].sum(axis=1)
mask = data123["고객비중합"] > 0
data123.loc[mask, cols] = (data123.loc[mask, cols].div(data123.loc[mask, "고객비중합"], axis=0) * 100)
data123["보정합"] = data123[cols].sum(axis=1).round(1)

data123 = data123.drop(columns = ['고객비중합', '보정합'])

def compute_top_demo(df):
    demo_cols = [
        "남성 20대이하 고객 비중", "남성 30대 고객 비중", "남성 40대 고객 비중",
        "남성 50대 고객 비중", "남성 60대이상 고객 비중",
        "여성 20대이하 고객 비중", "여성 30대 고객 비중", "여성 40대 고객 비중",
        "여성 50대 고객 비중", "여성 60대이상 고객 비중"
    ]
    def top2(row):
        sorted_row = row.sort_values(ascending=False)
        return pd.Series({
            "Top1_demo": sorted_row.index[0],
            "Top1_share": sorted_row.iloc[0],
            "Top2_demo": sorted_row.index[1],
            "Top2_share": sorted_row.iloc[1]
        })
    tops = df[demo_cols].apply(top2, axis=1)
    return pd.concat([df, tops], axis=1)

def compute_concentration(df):
    demo_cols = [
        "남성 20대이하 고객 비중", "남성 30대 고객 비중", "남성 40대 고객 비중",
        "남성 50대 고객 비중", "남성 60대이상 고객 비중",
        "여성 20대이하 고객 비중", "여성 30대 고객 비중", "여성 40대 고객 비중",
        "여성 50대 고객 비중", "여성 60대이상 고객 비중"
    ]
    arr = df[demo_cols].to_numpy()
    df["Demo_HHI"] = np.sum(arr**2, axis=1)
    df["Demo_entropy"] = -np.sum(arr * np.log(arr + 1e-9), axis=1)
    return df

data123 = compute_top_demo(data123)
data123 = compute_concentration(data123)

def compute_inflow_profile(df):
    inflow_cols = ["거주 이용 고객 비율", "직장 이용 고객 비율", "유동인구 이용 고객 비율"]
    def inflow_tags(row):
        values = row[inflow_cols].fillna(0)
        major_idx = values.idxmax()
        major_val = values.max()
        sorted_vals = values.sort_values(ascending=False)
        margin = sorted_vals.iloc[0] - sorted_vals.iloc[1] if len(sorted_vals) > 1 else sorted_vals.iloc[0]
        return pd.Series({
            "Inflow_major": str(major_idx).replace(" 이용 고객 비율", ""),
            "Inflow_major_val": major_val,
            "Inflow_margin": margin
        })
    inflow_info = df.apply(inflow_tags, axis=1)
    return pd.concat([df, inflow_info], axis=1)

def compute_delivery_profile(df):
    df["Delivery_share"] = df["배달매출금액 비율"].fillna(0)
    df["Delivery_tier"] = pd.cut(
        df["Delivery_share"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low", "mid", "high"]
    )
    return df

data123 = compute_inflow_profile(data123)
data123 = compute_delivery_profile(data123)

def parse_interval(val: str):
    if pd.isna(val):
        return np.nan
    s = str(val)
    if "이하" in s:
        num = int(re.search(r"(\d+)", s).group(1))
        return num / 2
    if "초과" in s:
        num = int(re.search(r"(\d+)", s).group(1))
        return (num + 100) / 2
    match = re.search(r"(\d+)-(\d+)", s)
    if match:
        low, high = map(int, match.groups())
        return (low + high) / 2
    return np.nan

data123["매출금액 구간"] = data123["매출금액 구간"].apply(parse_interval)
data123["매출건수 구간"] = data123["매출건수 구간"].apply(parse_interval)
data123["유니크고객 수 구간"] = data123["유니크 고객 수 구간"].apply(parse_interval)
data123["객단가 구간"]  = data123["객단가 구간"].apply(parse_interval)

def compute_size_profile(df):
    for col in ["매출금액 구간", "매출건수 구간", "유니크 고객 수 구간", "객단가 구간"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    def scale_series(s):
        max_val = s.max()
        return (s - 0.5) / max_val if max_val > 0 else s
    scaled_sales = scale_series(df["매출금액 구간"])
    scaled_txn = scale_series(df["매출건수 구간"])
    scaled_unique = scale_series(df["유니크 고객 수 구간"])
    scaled_atv = scale_series(df["객단가 구간"])
    df["SizeScore"] = (
        0.40 * scaled_sales +
        0.25 * scaled_txn +
        0.25 * scaled_unique +
        0.10 * scaled_atv
    )
    if df["SizeScore"].nunique() == 1:
        df["SizeClass"] = "M"
    else:
        df["SizeClass"] = pd.qcut(df["SizeScore"], 3, labels=["S","M","L"], duplicates="drop")
    return df

def compute_lifecycle(df):
    df["가맹점 운영개월수 구간"] = pd.to_numeric(df["가맹점 운영개월수 구간"], errors="coerce")
    def lifecycle_map(x):
        if pd.isna(x):
            return "Unknown"
        elif x <= 6:
            return "New"
        elif x <= 18:
            return "Growth"
        else:
            return "Mature"
    df["Lifecycle"] = df["가맹점 운영개월수 구간"].apply(lifecycle_map)
    return df

data123 = compute_size_profile(data123)
data123 = compute_lifecycle(data123)

def compute_within_industry_percentiles_bins(df):
    for col in ["매출금액 구간", "매출건수 구간", "객단가 구간"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Sales_pct"] = df.groupby("업종")["매출금액 구간"].rank(pct=True, ascending=True)
    df["Txn_pct"]   = df.groupby("업종")["매출건수 구간"].rank(pct=True, ascending=True)
    df["ATV_pct"]   = df.groupby("업종")["객단가 구간"].rank(pct=True, ascending=True)
    bins = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    labels = ["상위10%", "상위10-25%", "상위25-50%", "상위50-75%", "상위75-90%", "하위10%"]
    df["Sales_bin"] = pd.cut(df["Sales_pct"], bins=bins, labels=labels, include_lowest=True)
    df["Txn_bin"]   = pd.cut(df["Txn_pct"], bins=bins, labels=labels, include_lowest=True)
    df["ATV_bin"]   = pd.cut(df["ATV_pct"], bins=bins, labels=labels, include_lowest=True)
    return df

data123 = compute_within_industry_percentiles_bins(data123)

def compute_loyalty_attrition(df):
    df["재방문 고객 비중"] = pd.to_numeric(df["재방문 고객 비중"], errors="coerce").fillna(0)
    df["신규 고객 비중"] = pd.to_numeric(df["신규 고객 비중"], errors="coerce").fillna(0)
    df["취소율 구간"] = pd.to_numeric(df["취소율 구간"], errors="coerce").fillna(0)
    cancel_scaled = (df["취소율 구간"] - 0.5) / df["취소율 구간"].max() if df["취소율 구간"].max() > 0 else 0
    df["Cancel_scaled"] = cancel_scaled
    df["Loyalty_Score"] = 100 * (0.7 * df["재방문 고객 비중"] - 0.3 * df["Cancel_scaled"])
    df["Attrition_Score"] = 100 * (0.6 * df["Cancel_scaled"] + 0.4 * df["신규 고객 비중"])
    return df

def compute_loyalty_attrition_percentiles(df):
    df["Loyalty_pct"] = df.groupby("업종")["Loyalty_Score"].rank(pct=True, ascending=False)
    df["Attrition_pct"] = df.groupby("업종")["Attrition_Score"].rank(pct=True, ascending=True)
    return df

data123 = compute_loyalty_attrition(data123)
data123 = compute_loyalty_attrition_percentiles(data123)

def compute_competition(df):
    for col in ["동일 업종 내 매출 순위 비율", "동일 상권 내 매출 순위 비율"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1.0)
    df["Comp_power"] = (df["동일 업종 내 매출 순위 비율"] + df["동일 상권 내 매출 순위 비율"]) / 2
    bins = [0, 33.3, 66.6, 100]
    labels = ["Strong", "Mid", "Weak"]
    df["Comp_tier"] = pd.cut(df["Comp_power"], bins=bins, labels=labels, include_lowest=True)
    return df

def compute_closure_env(df):
    for col in ["동일 업종 내 해지 가맹점 비중", "동일 상권 내 해지 가맹점 비중"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Closure_env_industry_tier"] = pd.qcut(
        df["동일 업종 내 해지 가맹점 비중"],
        q=3,
        labels=["Low", "Mid", "High"],
        duplicates="drop"
    )
    df["Closure_env_market_tier"] = pd.qcut(
        df["동일 상권 내 해지 가맹점 비중"],
        q=3,
        labels=["Low", "Mid", "High"],
        duplicates="drop"
    )
    return df

data123 = compute_competition(data123)
data123 = compute_closure_env(data123)

def add_open_close_features(df):
    df["개설월"] = pd.to_numeric(df["개설월"], errors="coerce")
    df["폐업월"] = pd.to_numeric(df["폐업월"], errors="coerce")
    df["기준년월"] = pd.to_numeric(df["기준년월"], errors="coerce")
    df["운영개월수_계산"] = (df["기준년월"] // 100 - df["개설월"] // 100) * 12 + (df["기준년월"] % 100 - df["개설월"] % 100)
    df["Lifecycle"] = pd.cut(
        df["운영개월수_계산"],
        bins=[-1, 12, 36, 999],
        labels=["New", "Growth", "Mature"]
    )
    df["Status"] = df.apply(
        lambda x: "closed_store" if pd.notna(x["폐업월"]) and x["폐업월"] <= x["기준년월"] else "active_store",
        axis=1
    )
    return df

data123 = add_open_close_features(data123)

use_cols = [
    "가맹점구분번호", "기준년월", "업종", "상권","가맹점명",
    "Top1_demo", "Top1_share", "Top2_demo", "Top2_share",
    "Demo_HHI", "Demo_entropy",
    "Inflow_major", "Inflow_major_val", "Inflow_margin",
    "Delivery_share", "Delivery_tier",
    "SizeScore", "SizeClass", "Lifecycle", "Status",
    "Sales_pct", "Sales_bin",
    "Txn_pct", "Txn_bin",
    "ATV_pct", "ATV_bin",
    "Loyalty_Score", "Loyalty_pct",
    "Attrition_Score", "Attrition_pct",
    "Comp_power", "Comp_tier",
    "Closure_env_industry_tier", "Closure_env_market_tier",
]

data_mid = data123[use_cols].copy()

score_cols = ["Loyalty_Score", "Attrition_Score", "Comp_power", "SizeScore", "Demo_HHI", "Demo_entropy"]

def zscore_to_100(df, cols):
    scaler_z = StandardScaler()
    scaler_mm = MinMaxScaler(feature_range=(0,100))
    z = scaler_z.fit_transform(df[cols])
    s = scaler_mm.fit_transform(z)
    for i, col in enumerate(cols):
        df[col] = s[:, i].round(1)
    return df

data_mid = zscore_to_100(data_mid, score_cols)

use_cols_final = [
    "가맹점구분번호", "기준년월", "업종", "상권","가맹점명",
    "Top1_demo", "Top1_share", "Top2_demo", "Top2_share",
    "Demo_HHI", "Demo_entropy",
    "Inflow_major", "Inflow_major_val", "Inflow_margin",
    "Delivery_share", "Delivery_tier",
    "SizeScore", "SizeClass", "Lifecycle", "Status",
    "Sales_bin", "Txn_bin", "ATV_bin",
    "Loyalty_Score", "Attrition_Score",
    "Comp_tier",
    "Closure_env_market_tier", "Closure_env_industry_tier",
]

data123_final = data_mid[use_cols_final].copy()

num_cols = ["Top1_share", "Top2_share", "Demo_HHI", "Demo_entropy"]

data123_final = data123_final.sort_values(["가맹점구분번호", "기준년월"])
data123_final[num_cols] = data123_final.groupby("가맹점구분번호")[num_cols].transform(
    lambda x: x.interpolate(method="linear")
)

fill_num_cols = ["Top1_share", "Top2_share", "Demo_HHI", "Demo_entropy"]
fill_cat_cols = ["Top1_demo", "Top2_demo", "Delivery_tier", "Comp_tier"]

data123_final[fill_num_cols] = data123_final[fill_num_cols].fillna(0)

for col in fill_cat_cols:
    if pd.api.types.is_categorical_dtype(data123_final[col]):
        data123_final[col] = data123_final[col].cat.add_categories(["no_data"]).fillna("no_data")
    else:
        data123_final[col] = data123_final[col].fillna("no_data")

def fix_delivery_tier(df):
    share = df["Delivery_share"].fillna(0)
    tier = pd.Series("no_delivery", index=df.index)
    tier[share == 0] = "low"
    tier[(share > 0) & (share <= 0.3)] = "low"
    tier[(share > 0.3) & (share <= 0.7)] = "mid"
    tier[share > 0.7] = "high"
    df["Delivery_tier"] = tier
    return df

data123_final = fix_delivery_tier(data123_final)

rename_map = {
    "Top1_demo": "주요고객군(1위)",
    "Top1_share": "주요고객군비중(1위)",
    "Top2_demo": "보조고객군(2위)",
    "Top2_share": "보조고객군비중(2위)",
    "Demo_HHI": "고객집중도_HHI",
    "Demo_entropy": "고객다양성_Entropy",
    "Inflow_major": "주요유입경로",
    "Inflow_major_val": "주요유입경로비율",
    "Inflow_margin": "유입경로1·2위차이",
    "Delivery_share": "배달매출비중",
    "Delivery_tier": "배달의존도등급",
    "SizeScore": "규모점수",
    "SizeClass": "규모구분",
    "Lifecycle": "생애주기단계",
    "Status": "매장상태",
    "Sales_bin": "매출구간(업종내)",
    "Txn_bin": "거래건수구간(업종내)",
    "ATV_bin": "객단가구간(업종내)",
    "Loyalty_Score": "충성도점수",
    "Attrition_Score": "이탈위험점수",
    "Comp_tier": "경쟁력등급",
    "Closure_env_market_tier": "폐업환경위험도등급(상권)",
    "Closure_env_industry_tier": "폐업환경위험도등급(업종)"
}

data123_final = data123_final.rename(columns=rename_map)

data123_final.to_csv("./data/data_final.csv", encoding="utf-8")