# -*- coding: utf-8 -*-
"""
環境監測綜合分析系統
功能：
1. 施工前 vs 施工期間差異檢定
2. 異常偵測矩陣，格內顯示 p-value（小數後 3 位）
3. Mann-Kendall 長期趨勢分析
4. 支援 <、ND、N.D.、-、－ 等環境監測常見資料格式
"""

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

try:
    import pymannkendall as mk
except ImportError:
    mk = None


# =========================================================
# 0. 全域設定
# =========================================================
st.set_page_config(page_title="環境監測綜合分析系統", layout="wide")

ALPHA = 0.05
ROUND_PRECISION = 6
BOOTSTRAP_N = 1000
PERIOD_PRE = "施工前"
PERIOD_DURING = "施工期間"

COLOR_MAP = {
    "gray": "#BDC3C7",
    "green": "#2ECC71",
    "red": "#E74C3C",
}

STATUS_CODE_MAP = {
    "gray": 0,
    "green": 1,
    "red": 2,
}

# =========================================================
# 1. 資料模板
# =========================================================
@st.cache_data
def get_excel_template() -> bytes:
    """產生標準 Excel 範本。"""
    output = io.BytesIO()

    data = {
        "測站": ["測站A", "測站A", "測站A", "測站A", "測站A"],
        "測項": ["pH值", "pH值", "pH值", "pH值", "pH值"],
        "日期": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        "時期": ["施工前", "施工前", "施工期間", "施工期間", "施工期間"],
        "數值": [7.2, 7.3, 7.1, 6.8, 6.5],
        "MDL": ["", "", "", "", ""],
        "法規下限": [6, 6, 6, 6, 6],
        "法規上限": [9, 9, 9, 9, 9],
        "單位": ["", "", "", "", ""],
    }

    df_sample = pd.DataFrame(data)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_sample.to_excel(writer, index=False, sheet_name="監測數據")
        worksheet = writer.sheets["監測數據"]
        for col in list("ABCDEFGHI"):
            worksheet.column_dimensions[col].width = 16

    return output.getvalue()

# =========================================================
# 2. 資料清理與檢查
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """欄位名稱去除前後空白。"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df
    
# =========================================================
# 2.1 儲存格資料轉成數值
# =========================================================
def parse_numeric_like(value):                                             # 將各種輸入格式轉換為數值(float), 主要將存儲格轉為數值, 其餘計算在process_censored_data處理
    if pd.isna(value):                                                     # 若為 NaN、None、空儲存格
        return np.nan                                                      # 直接回傳缺值
    if isinstance(value, (int, float, np.integer, np.floating)):           # 若本身已經是數值型態
        return round(float(value), ROUND_PRECISION)                        # 統一轉 float 並依設定小數位數四捨五入

    text = str(value).strip().upper()                                      # 轉成字串, 去除前後空白, 統一轉大寫

    if text in {"", "-", "－", "—", "ND", "N.D.", "NAN", "NONE"}:          # 常見缺值表示方式
        return np.nan                                                      # 統一視為缺值
    if text.startswith("<"):                                               # 偵測 "<0.02" 之類資料
        return np.nan                                                      # 不在此函式處理, 交由 process_censored_data() 處理

    text = text.replace(",", "")                                           # 移除千分位逗號, 例如：1,234.56 → 1234.56

    try:                                                                   # 嘗試轉換成 float
        return round(float(text), ROUND_PRECISION)                         # 成功則回傳數值
    except ValueError:                                                     # 無法轉換
        return np.nan                                                      # 回傳缺值
        
# =========================================================
# 2.2 處理ND、<MDL 等環境監測常見資料
# =========================================================       
def process_censored_data(row):                                            # 處理 ND、<MDL 等環境監測常見資料
    raw_value = row.get("數值", np.nan)                                    # 取得原始測值
    mdl_value = parse_numeric_like(
        row.get("MDL", np.nan)
    )                                                                      # 取得 MDL並轉成數值
   
    if pd.isna(raw_value):                                                 # 若測值本身為空
        return np.nan                                                      # 直接回傳缺值

    text = str(raw_value).strip().upper().replace(",", "")                 # 統一格式, 去空白, 轉大寫, 移除千分位
    if text in {"ND", "N.D.", "-", "－", "—"}:                             # 未檢出或缺值符號
        if pd.notna(mdl_value):                                            # MDL存在, 使用 MDL/2
            return round(
                mdl_value / 2,
                ROUND_PRECISION
            )
        return np.nan                                                      # MDL不存在則回傳缺值
    
    if text.startswith("<"):                                               # 偵測 <0.02
        try:                                                               # 去掉 "<", 嘗試取得數值部分
            censored_value = float(
                text.replace("<", "").strip()
            )
        
            return round(
                censored_value / 2,
                ROUND_PRECISION
            )                                                              # 使用檢量線濃度的一半

        except ValueError:                                                 # 若格式異常
            if pd.notna(mdl_value):                                        # 有MDL, 改用 MDL/2
                return round(
                    mdl_value / 2,
                    ROUND_PRECISION
                )
            return np.nan                                                 # 否則缺值

    parsed = parse_numeric_like(raw_value)                                # 一般數值處理
    if pd.notna(parsed):                                                  # 成功轉換
        return parsed                                                     # 回傳原數值
    return np.nan                                                         # 其餘無法辨識資料, 一律回傳缺值
# =========================================================
# 2.3 檢查必要欄位並完成資料前處理
# =========================================================       
def validate_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    required_cols = ["測站", "測項", "時期", "數值"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}")

    if "日期" not in df.columns:
        st.warning("⚠️ 缺少 `日期` 欄位，MK 趨勢分析將使用列順序作為臨時日期，建議補上實際日期。")
        df["日期"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    else:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")

    if "MDL" not in df.columns:
        df["MDL"] = np.nan

    for col in ["法規下限", "法規上限"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = df[col].apply(parse_numeric_like)

    if "單位" not in df.columns:
        df["單位"] = ""

    df["測站"] = df["測站"].astype(str).str.strip()
    df["測項"] = df["測項"].astype(str).str.strip()
    df["時期"] = df["時期"].astype(str).str.strip()
    df["單位"] = df["單位"].astype(str).str.strip()

    # =====================================================
    # 測項名稱與單位顯示標準化：處理上下標
    # =====================================================
    df["測項"] = df["測項"].replace({
        "PM2.5": "PM₂.₅",
        "PM10": "PM₁₀",
        "NO2": "NO₂",
        "SO2": "SO₂",
        "CO2": "CO₂",
        "O3": "O₃",
        "NH3-N": "NH₃-N",
        "NH4-N": "NH₄-N",
        "NO2-N": "NO₂-N",
        "NO3-N": "NO₃-N",
        "PO4-P": "PO₄-P"
    })

    df["單位"] = df["單位"].replace({
        "ug/m3": "μg/m³",
        "μg/m3": "μg/m³",
        "m3/min": "m³/min",
        "m3/day": "m³/day",
        "m2": "m²",
    })

    df["數值_原始"] = df["數值"].astype(str)
    df["數值"] = df.apply(process_censored_data, axis=1)

    df = df.dropna(subset=["數值", "日期"])
    return df

# =========================================================
# 3. 統計檢定
# =========================================================
def detect_worse_direction(item_name: str, diff: float) -> bool:
    """
    判斷變化是否朝不利方向。
    目前保留簡化邏輯：
    - DO / 溶氧量：下降較不利
    - pH：顯著變動即視為需注意
    - 其他：上升較不利
    """
    item_text = str(item_name).upper()

    if "溶氧" in item_text or "DO" in item_text:
        return diff < 0

    if "PH" in item_text:
        return True

    return diff > 0


def safe_shapiro(values: np.ndarray) -> tuple[bool, float]:
    """Shapiro-Wilk 常態性檢定；樣本數不足時回傳非常態。"""
    if len(values) < 3:
        return False, np.nan

    try:
        _, p_value = stats.shapiro(values)
        return p_value > ALPHA, float(p_value)
    except Exception:
        return False, np.nan


def bootstrap_mean_diff_ci(
    group_pre: np.ndarray,
    group_dur: np.ndarray,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[float, float]:
    """Bootstrap 平均差異 95% CI。"""
    if len(group_pre) == 0 or len(group_dur) == 0:
        return np.nan, np.nan

    boot_diffs = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_boot):
        s_pre = rng.choice(group_pre, len(group_pre), replace=True)
        s_dur = rng.choice(group_dur, len(group_dur), replace=True)
        boot_diffs.append(np.mean(s_dur) - np.mean(s_pre))

    return float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def perform_stats(df_sub: pd.DataFrame) -> dict[str, Any]:
    """施工前 vs 施工期間差異檢定。"""
    base_result = {
        "status": "gray",
        "status_text": "數據不足",
        "p_val": np.nan,
        "diff": np.nan,
        "test_method": "N/A",
        "mean_pre": np.nan,
        "mean_dur": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "unit": "",
        "lower_limit": np.nan,
        "upper_limit": np.nan,
        "n_pre": 0,
        "n_dur": 0,
    }

    if df_sub.empty:
        base_result["status_text"] = "無數據"
        return base_result

    item_name = df_sub["測項"].iloc[0]
    unit = df_sub["單位"].iloc[0] if "單位" in df_sub.columns and pd.notna(df_sub["單位"].iloc[0]) else ""
    lower_limit = df_sub["法規下限"].iloc[0] if "法規下限" in df_sub.columns else np.nan
    upper_limit = df_sub["法規上限"].iloc[0] if "法規上限" in df_sub.columns else np.nan

    group_pre = df_sub.loc[df_sub["時期"] == PERIOD_PRE, "數值"].dropna().to_numpy(dtype=float)
    group_dur = df_sub.loc[df_sub["時期"] == PERIOD_DURING, "數值"].dropna().to_numpy(dtype=float)

    base_result.update({
        "unit": unit,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "n_pre": len(group_pre),
        "n_dur": len(group_dur),
    })

    if len(group_pre) < 2 or len(group_dur) < 2:
        return base_result

    mean_pre = round(float(np.mean(group_pre)), ROUND_PRECISION)
    mean_dur = round(float(np.mean(group_dur)), ROUND_PRECISION)
    diff = mean_dur - mean_pre

    base_result.update({
        "mean_pre": mean_pre,
        "mean_dur": mean_dur,
        "diff": diff,
    })

    if np.isclose(mean_pre, mean_dur):
        p_val = 1.0
        test_method = "無變化(Constant)"
        ci_lower, ci_upper = diff, diff
    else:
        normal_pre, _ = safe_shapiro(group_pre)
        normal_dur, _ = safe_shapiro(group_dur)
        is_normal = normal_pre and normal_dur

        try:
            if is_normal:
                _, p_val = stats.ttest_ind(group_pre, group_dur, equal_var=False, nan_policy="omit")
                test_method = "Welch's t-test"
            else:
                _, p_val = stats.mannwhitneyu(group_pre, group_dur, alternative="two-sided")
                test_method = "Mann-Whitney U"
        except Exception:
            base_result["status_text"] = "計算錯誤"
            return base_result

        ci_lower, ci_upper = bootstrap_mean_diff_ci(group_pre, group_dur)

    is_significant = bool(p_val < ALPHA)
    is_worse = detect_worse_direction(item_name, diff)

    if is_significant:
        status = "red" if is_worse else "red"
        status_text = "具顯著變化"
    else:
        status = "green"
        status_text = "無顯著變化"

    base_result.update({
        "p_val": float(p_val),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "test_method": test_method,
        "status": status,
        "status_text": status_text,
    })

    return base_result
#====================================================================    
def format_test_method_code(test_method: Any) -> str:
    """將統計檢定方法轉換為矩陣顯示代碼。"""
    method_map = {
        "Welch's t-test": "W",
        "Mann-Whitney U": "U",
        "無變化(Constant)": "C",
        "N/A": "—",
    }

    if test_method is None or pd.isna(test_method):
        return "—"
    return method_map.get(str(test_method), "—")
#=====================================================================

def perform_mk_analysis(df_sub: pd.DataFrame) -> dict[str, Any] | None:
    """執行 Mann-Kendall 趨勢檢定。"""
    if mk is None:
        return None

    df_sub = df_sub.sort_values(by="日期")
    values = df_sub["數值"].dropna().to_numpy(dtype=float)

    if len(values) < 4:
        return None

    try:
        result = mk.original_test(values)
    except Exception:
        return None

    return {
        "trend": result.trend,
        "h": result.h,
        "p_val": float(result.p),
        "z": float(result.z),
        "tau": float(result.Tau),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "dates": df_sub["日期"].to_numpy(),
        "values": values,
        "unit": df_sub["單位"].iloc[0] if "單位" in df_sub.columns else "",
    }


# =========================================================
# 4. 圖表
# =========================================================
def format_value(value: Any) -> str:
    """將矩陣中的數值格式化至小數後 3 位。"""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"

def build_abnormal_matrix(res_df: pd.DataFrame) -> go.Figure:
    """建立異常偵測矩陣：每格顯示 p-value，小數後 3 位。"""
    plot_df = res_df.copy()
    plot_df["status_code"] = plot_df["status"].map(STATUS_CODE_MAP).fillna(0)

    annotations = []
    for _, row in plot_df.iterrows():
        if row["status"] == "gray" or pd.isna(row["p_val"]):
            cell_text = "N/A"
        else:
            pre_text = format_value(row.get("mean_pre", np.nan))
            dur_text = format_value(row.get("mean_dur", np.nan))
            
            method_code = format_test_method_code(
                row.get("test_method", "N/A"),
            )            
            cell_text = (f"{pre_text} → {dur_text}"
                         f"<br><i>p</i>={row['p_val']:.3f}｜{method_code}")

        font_color = "white" if row["status"] == "red" else "black"
        annotations.append(
            dict(
                x=row["測站"],
                y=row["測項"],
                text=cell_text,
                showarrow=False,
                font=dict(color=font_color, size=18),
                align="center",
            )
        )

    customdata = np.stack(
        [
            plot_df["status_text"].fillna(""),
            plot_df["p_val"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.6f}"),
            plot_df["test_method"].fillna("N/A"),
            plot_df["n_pre"].fillna(0).astype(int).astype(str),
            plot_df["n_dur"].fillna(0).astype(int).astype(str),
        ],
        axis=-1,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_df["status_code"],
            x=plot_df["測站"],
            y=plot_df["測項"],
            customdata=customdata,
            colorscale=[
                [0.00, COLOR_MAP["gray"]],
                [0.33, COLOR_MAP["gray"]],
                [0.33, COLOR_MAP["green"]],
                [0.66, COLOR_MAP["green"]],
                [0.66, COLOR_MAP["red"]],
                [1.00, COLOR_MAP["red"]],
            ],
            zmin=0,
            zmax=2,
            xgap=2,
            ygap=2,
            showscale=False,
            hovertemplate=(
                "測站: %{x}<br>"
                "測項: %{y}<br>"
                "狀態: %{customdata[0]}<br>"
                "p-value: %{customdata[1]}<br>"
                "檢定方法: %{customdata[2]}<br>"
                "施工前 n: %{customdata[3]}<br>"
                "施工期間 n: %{customdata[4]}"
                "<extra></extra>"
            ),
        )
    )

    legend_items = [
        ("數據不足", COLOR_MAP["gray"]),
        ("無顯著變化", COLOR_MAP["green"]),
        ("具顯著變化", COLOR_MAP["red"]),
    ]
    for label, color in legend_items:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=15, color=color),
                name=label,
            )
        )

    height = max(420, 45 * plot_df["測項"].nunique() + 120)
    fig.update_layout(
        annotations=annotations,
        height=height,
        plot_bgcolor="white",
        margin=dict(l=80, r=40, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=18, family="Arial", color="black"),
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=18, color="black")),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=18, color="black")),
    )

    return fig


def build_difference_detail_figure(t1_df: pd.DataFrame, res: pd.Series, item_name: str) -> go.Figure:
    """建立詳細差異圖。"""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(f"{item_name} 分佈", f"差異估計 ({res['test_method']})"),
    )

    p_color = COLOR_MAP["red"] if res["status"] == "red" else COLOR_MAP["green"]
    box_colors = {PERIOD_PRE: "gray", PERIOD_DURING: p_color}

    for period in [PERIOD_PRE, PERIOD_DURING]:
        sub = t1_df[t1_df["時期"] == period]
        if not sub.empty:
            fig.add_trace(
                go.Box(
                    y=sub["數值"],
                    x=sub["時期"],
                    name=period,
                    boxpoints="all",
                    jitter=0.5,
                    marker=dict(color=box_colors.get(period, "#3498DB")),
                    line=dict(color=box_colors.get(period, "#3498DB")),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    if pd.notna(res["upper_limit"]):
        fig.add_hline(y=res["upper_limit"], line_dash="dash", line_color="red", row=1, col=1)

    if pd.notna(res["lower_limit"]):
        fig.add_hline(y=res["lower_limit"], line_dash="dash", line_color="red", row=1, col=1)

    fig.add_hline(y=0, line_color="black", row=1, col=2)

    if res["test_method"] == "無變化(Constant)" or pd.isna(res["ci_lower"]) or pd.isna(res["ci_upper"]):
        fig.add_trace(
            go.Scatter(x=["差異"], y=[res["diff"]], mode="markers", marker=dict(color="black")),
            row=1,
            col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=["差異"],
                y=[res["diff"]],
                mode="markers",
                marker=dict(color="black"),
                error_y=dict(
                    type="data",
                    array=[res["ci_upper"] - res["diff"]],
                    arrayminus=[res["diff"] - res["ci_lower"]],
                ),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title_text=f"狀態: {res['status_text']} (p={res['p_val']:.3f})",
        title_font=dict(size=26, color="black"),
        font=dict(size=20, color="black"),
    )
    fig.update_xaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))
    fig.update_yaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))

    return fig


def build_mk_figure(mk_df: pd.DataFrame, mk_res: dict[str, Any], station: str, item: str) -> go.Figure:
    """建立 MK 趨勢圖。"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mk_res["dates"],
            y=mk_res["values"],
            mode="markers+lines",
            name="監測數值",
            marker=dict(color="#3498DB", size=8),
            line=dict(color="#AED6F1", width=1),
        )
    )

    x_nums = np.arange(len(mk_res["values"]))
    y_trend = mk_res["slope"] * x_nums + mk_res["intercept"]

    fig.add_trace(
        go.Scatter(
            x=mk_res["dates"],
            y=y_trend,
            mode="lines",
            name=f"Sen's slope = {mk_res['slope']:.3f}",
            line=dict(color="#E74C3C", width=3),
        )
    )

    limit_info = mk_df[["法規上限", "法規下限"]].iloc[0]
    if pd.notna(limit_info["法規上限"]):
        fig.add_hline(y=limit_info["法規上限"], line_dash="dash", line_color="red", annotation_text="上限")

    if pd.notna(limit_info["法規下限"]):
        fig.add_hline(y=limit_info["法規下限"], line_dash="dash", line_color="red", annotation_text="下限")

    unit = mk_res.get("unit", "")
    fig.update_layout(
        title=f"{station} - {item} 長期趨勢分析",
        title_font=dict(size=28, color="black"),
        yaxis_title=f"數值 ({unit})" if unit else "數值",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, font=dict(size=18)),
    )
    fig.update_xaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))
    fig.update_yaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))

    return fig


# =========================================================
# 5. Streamlit UI
# =========================================================
st.title("🛡️ 環境監測綜合分析系統")

st.sidebar.title("📁 資料匯入")
st.sidebar.download_button(
    label="📥 下載 Excel 範本",
    data=get_excel_template(),
    file_name="環境監測_分析範本.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded_file = st.sidebar.file_uploader("請上傳 Excel (xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 請先上傳資料。本系統支援 **施工前後差異檢定** 與 **Mann-Kendall 長期趨勢分析**。")
    st.stop()

if mk is None:
    st.error("缺少 `pymannkendall` 套件，請在 requirements.txt 加入 `pymannkendall`。")
    st.stop()

try:
    raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
    df = validate_and_prepare_data(raw_df)
except Exception as exc:
    st.error(f"讀取或清理資料失敗：{exc}")
    st.stop()

if df.empty:
    st.warning("資料清理後無有效數據，請檢查 `數值`、`日期` 欄位。")
    st.stop()

stations = sorted(df["測站"].dropna().unique())
items = sorted(df["測項"].dropna().unique())

tab1, tab2 = st.tabs(["📊 施工前後差異檢定", "📈 Mann-Kendall 長期趨勢"])


# =========================================================
# TAB 1: 施工前後差異檢定
# =========================================================
with tab1:
    st.markdown("### 施工前後差異分析")

    results = []
    for station in stations:
        for item in items:
            sub_df = df[(df["測站"] == station) & (df["測項"] == item)]
            if sub_df.empty:
                continue
            res = perform_stats(sub_df)
            res["測站"] = station
            res["測項"] = item
            results.append(res)

    res_df = pd.DataFrame(results)

    if res_df.empty:
        st.warning("無有效統計數據。")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 具顯著變化", len(res_df[res_df["status"] == "red"]))
    col2.metric("🟢 無顯著變化", len(res_df[res_df["status"] == "green"]))
    col3.metric("⚪ 數據不足", len(res_df[res_df["status"] == "gray"]))

    st.divider()

    st.subheader("異常偵測矩陣（格內顯示 p-value）")
    fig_h = build_abnormal_matrix(res_df)
    st.plotly_chart(fig_h, use_container_width=True)

    st.download_button(
        label="⬇️ 下載差異檢定結果 CSV",
        data=res_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="施工前後差異檢定結果.csv",
        mime="text/csv",
    )

    with st.expander("查看完整檢定結果表"):
        show_cols = [
            "測站", "測項", "status_text", "p_val", "test_method",
            "n_pre", "n_dur", "mean_pre", "mean_dur", "diff",
            "ci_lower", "ci_upper",
        ]
        st.dataframe(res_df[show_cols], use_container_width=True)

    st.divider()

    st.subheader("詳細檢定圖表")
    sc1, sc2 = st.columns(2)
    t1_st = sc1.selectbox("選擇測站", stations, key="t1_st")
    t1_it = sc2.selectbox("選擇測項", items, key="t1_it")

    t1_df = df[(df["測站"] == t1_st) & (df["測項"] == t1_it)]
    t1_res = res_df[(res_df["測站"] == t1_st) & (res_df["測項"] == t1_it)]

    if t1_res.empty:
        st.info("此測站/測項無結果。")
    else:
        res = t1_res.iloc[0]
        if res["status"] == "gray":
            st.info("數據不足，無法進行差異檢定。")
        else:
            fig_est = build_difference_detail_figure(t1_df, res, t1_it)
            st.plotly_chart(fig_est, use_container_width=True)


# =========================================================
# TAB 2: Mann-Kendall 趨勢分析
# =========================================================
with tab2:
    st.markdown("### Mann-Kendall 長期趨勢分析")
    st.markdown(
        """
        使用 **Mann-Kendall Test** 檢測時間序列是否存在顯著單調趨勢，
        並以 **Sen's Slope** 估計變化速率。
        """
    )

    mk_c1, mk_c2 = st.columns(2)
    mk_st = mk_c1.selectbox("選擇測站", stations, key="mk_st")
    mk_it = mk_c2.selectbox("選擇測項", items, key="mk_it")

    mk_df = df[(df["測站"] == mk_st) & (df["測項"] == mk_it)].copy()
    mk_res = perform_mk_analysis(mk_df)

    if mk_res is None:
        st.warning("⚠️ 數據點過少（< 4 筆）或 MK 檢定失敗，無法進行有效趨勢檢定。")
    else:
        trend_map = {
            "increasing": "📈 上升趨勢",
            "decreasing": "📉 下降趨勢",
            "no trend": "➡️ 無顯著趨勢",
        }
        trend_text = trend_map.get(mk_res["trend"], mk_res["trend"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("趨勢結果", trend_text)
        m2.metric("P-value", f"{mk_res['p_val']:.3f}", delta="顯著" if mk_res["p_val"] < ALPHA else None)
        m3.metric("Sen's Slope", f"{mk_res['slope']:.4f}")
        m4.metric("Kendall Tau", f"{mk_res['tau']:.3f}")

        fig_mk = build_mk_figure(mk_df, mk_res, mk_st, mk_it)
        st.plotly_chart(fig_mk, use_container_width=True)

        st.info(
            f"💡 **Sen's Slope 解讀**：`{mk_res['slope']:.4f}`，"
            f"代表每個採樣序列單位的數值變化約 `{mk_res['slope']:.4f}`。"
        )
