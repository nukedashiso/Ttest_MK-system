import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
# [新增] 引入 MK 檢定套件
try:
    import pymannkendall as mk
except ImportError:
    st.error("請先安裝 pymannkendall 套件: `pip install pymannkendall`")
    st.stop()

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="環境監測綜合分析系統", layout="wide")

# ==========================================
# 1. 資料處理核心邏輯
# ==========================================

def get_excel_template():
    """產生標準 Excel 範本 (新增日期欄位)"""
    output = io.BytesIO()
    data = {
        '測站': ['測站A', '測站A', '測站A', '測站A', '測站A'],
        '測項': ['pH值', 'pH值', 'pH值', 'pH值', 'pH值'],
        '日期': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'], # [新增]
        '時期': ['施工前', '施工前', '施工期間', '施工期間', '施工期間'],
        '數值': [7.2, 7.3, 7.1, 6.8, 6.5],
        'MDL':  ['', '', '', '', ''],
        '法規下限': [6, 6, 6, 6, 6],
        '法規上限': [9, 9, 9, 9, 9],
        '單位': ['', '', '', '', '']
    }
    df_sample = pd.DataFrame(data)
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='監測數據')
        worksheet = writer.sheets['監測數據']
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            worksheet.column_dimensions[col].width = 15
    return output.getvalue()

def process_censored_data(row):
    """處理 <, ND, －"""
    val = row['數值']
    mdl_raw = row['MDL'] # 讀取原始MDL值

    # 設置一個統一的精度，例如 6 位小數
    ROUND_PRECISION = 6 

    mdl = np.nan
    try:
        mdl_float = float(mdl_raw)
        mdl = np.round(mdl_float, ROUND_PRECISION) # 在這裡對MDL進行四捨五入
    except:
        mdl = np.nan
    
    if isinstance(val, (int, float)):
        # 對於正常數值也進行四捨五入，確保所有數值精度一致
        return np.round(float(val), ROUND_PRECISION)
    
    val_str = str(val).strip().upper()
    
    if "ND" in val_str or "N.D." in val_str:
        return mdl if pd.notna(mdl) else np.nan

    if val_str in ["-", "－"]:
        return mdl if pd.notna(mdl) else np.nan
            
    if "<" in val_str:
        try:
            num_text = val_str.replace("<", "").strip()
            if num_text:
                return np.round(float(num_text), ROUND_PRECISION) # 對 < 轉換的數值也四捨五入
            elif pd.notna(mdl):
                return mdl
            else:
                return np.nan
        except:
            return np.nan

    try:
        return np.round(float(val_str), ROUND_PRECISION) # 對其他可轉換數值也四捨五入
    except:
        return np.nan

# --- 原有的差異檢定邏輯 ---
def perform_stats(df_sub):
    """前後差異檢定 (t-test / Mann-Whitney)"""
    if df_sub.empty:
        return {'status': 'gray', 'status_text': '無數據', 'p_val': 1.0, 'diff': 0}
        
    group_pre = df_sub[df_sub['時期'] == '施工前']['數值'].dropna().values
    group_dur = df_sub[df_sub['時期'] == '施工期間']['數值'].dropna().values
    
    if len(group_pre) < 2 or len(group_dur) < 2:
        return {'status': 'gray', 'status_text': '數據不足', 'p_val': 1.0, 'diff': 0}

    lower_limit = df_sub['法規下限'].iloc[0] if '法規下限' in df_sub.columns else np.nan
    upper_limit = df_sub['法規上限'].iloc[0] if '法規上限' in df_sub.columns else np.nan
    unit = df_sub['單位'].iloc[0] if pd.notna(df_sub['單位'].iloc[0]) else ""
    item_name = df_sub['測項'].iloc[0]

    ROUND_DECIMALS = 6 # 與 process_censored_data 中使用的精度一致
    mean_pre = np.round(np.mean(group_pre), ROUND_DECIMALS)
    mean_dur = np.round(np.mean(group_dur), ROUND_DECIMALS)
    diff = mean_dur - mean_pre 
    
    if mean_pre == mean_dur: # 這裡的比較會更可靠
        p_val = 1.0
        test_method = "無變化(Constant)"
        is_normal = True    
    else:
        try:
            if len(group_pre) < 3 or len(group_dur) < 3:
                is_normal = False
            else:
                _, p_norm_pre = stats.shapiro(group_pre)
                _, p_norm_dur = stats.shapiro(group_dur)
                is_normal = (p_norm_pre > 0.05) and (p_norm_dur > 0.05)
        except:
            is_normal = False

        try:
            if is_normal:
                stat, p_val = stats.ttest_ind(group_pre, group_dur, equal_var=False)
                test_method = "Welch's t-test"
            else:
                stat, p_val = stats.mannwhitneyu(group_pre, group_dur)
                test_method = "Mann-Whitney"
        except:
            return {'status': 'gray', 'status_text': '計算錯誤', 'p_val': 1.0}

    # Bootstrap CI
    try:
        if test_method == "無變化(Constant)":
            ci_lower, ci_upper = diff, diff
        else:
            n_boot = 1000
            boot_diffs = []
            for _ in range(n_boot):
                s_pre = np.random.choice(group_pre, len(group_pre), replace=True)
                s_dur = np.random.choice(group_dur, len(group_dur), replace=True)
                boot_diffs.append(np.mean(s_dur) - np.mean(s_pre))
            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)
    except:
        ci_lower, ci_upper = diff, diff

    is_significant = p_val < 0.05
    if '溶氧量' in str(item_name) or 'DO' in str(item_name):
        is_worse = diff < 0 
    elif 'pH' in str(item_name):
        is_worse = True 
    else:
        is_worse = diff > 0 
    
    if is_significant:
        status = "red"
        status_text = "具顯著變化"
    else:
        status = "green"
        status_text = "無顯著變化"

    return {
        'mean_pre': mean_pre, 'mean_dur': mean_dur, 'diff': diff,
        'p_val': p_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'test_method': test_method, 'status': status, 'status_text': status_text,
        'unit': unit, 'lower_limit': lower_limit, 'upper_limit' : upper_limit
    }

# ==========================================
# [新增] Mann-Kendall 趨勢分析函數
# ==========================================
def perform_mk_analysis(df_sub):
    """
    執行 Mann-Kendall 趨勢檢定
    """
    # 確保依照日期排序
    df_sub = df_sub.sort_values(by='日期')
    vals = df_sub['數值'].dropna().values
    dates = df_sub['日期'].dropna().values
    
    if len(vals) < 4:
        return None # 數據太少無法分析

    # 執行 MK 檢定
    # trend: increasing, decreasing, no trend
    # h: True (if trend is present), False (no trend)
    # p: p-value
    # z: normalized test statistics
    # Tau: Kendall Tau
    # s: Mann-Kendal's score
    # var_s: Variance S
    # slope: Sen's slope
    # intercept: intercept
    try:
        result = mk.original_test(vals)
        return {
            'trend': result.trend,
            'h': result.h,
            'p_val': result.p,
            'slope': result.slope,
            'intercept': result.intercept,
            'dates': dates,
            'values': vals,
            'unit': df_sub['單位'].iloc[0] if '單位' in df_sub.columns else ""
        }
    except Exception as e:
        return None

# ==========================================
# 2. Sidebar
# ==========================================
st.sidebar.title("📁 資料匯入")
st.sidebar.download_button(
    label="📥 下載 Excel 範本 (含日期)",
    data=get_excel_template(),
    file_name="環境監測_MK範本.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
uploaded_file = st.sidebar.file_uploader("請上傳 Excel (xlsx)", type=["xlsx"])

# ==========================================
# 3. 主畫面邏輯
# ==========================================
st.title("🛡️ 環境監測綜合分析系統")

if uploaded_file is None:
    st.info("👈 請上傳資料。本系統支援 **前後差異檢定** 與 **MK 趨勢分析**。")
else:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = df.columns.str.strip()
        
        # 欄位檢查 (需包含日期)
        required_cols = ['測站', '測項', '時期', '數值']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ 缺少必要欄位。")
            st.stop()
        
        # 處理日期欄位 (若無則警告)
        if '日期' not in df.columns:
            st.warning("⚠️ 警告：Excel 中缺少 `日期` 欄位，MK 趨勢分析將依據 Excel 列順序進行，可能不準確。")
            df['日期'] = pd.to_datetime(df.index) # 假日期
        else:
            df['日期'] = pd.to_datetime(df['日期'], errors='coerce') # 強制轉為日期
            
        # 確保其他欄位存在
        if 'MDL' not in df.columns: df['MDL'] = np.nan
        for col in ['法規下限', '法規上限']:
            if col not in df.columns: df[col] = np.nan
            else: df[col] = pd.to_numeric(df[col], errors='coerce')
        if '單位' not in df.columns: df['單位'] = ""

        df['時期'] = df['時期'].astype(str).str.strip()
        df['數值_原始'] = df['數值'].astype(str)
        df['數值_清洗後'] = df.apply(process_censored_data, axis=1)
        
        # 移除無效值
        df['數值'] = df['數值_清洗後']
        df = df.dropna(subset=['數值', '日期']) # 日期無效也移除

        stations = sorted(df['測站'].unique())
        items = sorted(df['測項'].unique())

        # ==========================================
        # 分頁 (Tabs) 結構
        # ==========================================
        tab1, tab2 = st.tabs(["📊 施工前後差異檢定", "📈 Mann-Kendall 長期趨勢"])

        # ==========================================
        # TAB 1: 施工前後差異檢定 (原功能)
        # ==========================================
        with tab1:
            st.markdown("### 施工前後差異分析 (Difference Analysis)")
            
            # --- 批次運算 ---
            results = []
            for s in stations:
                for i in items:
                    sub_df = df[(df['測站']==s) & (df['測項']==i)]
                    if not sub_df.empty:
                        res = perform_stats(sub_df)
                        res['測站'] = s
                        res['測項'] = i
                        results.append(res)
            res_df = pd.DataFrame(results)

            if res_df.empty:
                st.warning("無有效統計數據。")
            else:
                # 燈號總覽
                c1, c2, c3 = st.columns(3)
                if 'status' in res_df.columns:
                    c1.metric("🔴 具顯著變化", len(res_df[res_df['status'] == 'red']))
                    c2.metric("🟢 無顯著變化", len(res_df[res_df['status'] == 'green']))
                    c3.metric("⚪ 數據不足", len(res_df[res_df['status'] == 'gray']))
                
                st.divider()
                
                # 矩陣圖
                st.subheader("異常偵測矩陣")
                status_map = {'gray': 0, 'green': 1, 'red': 2}
                res_df['status_code'] = res_df['status'].map(status_map)
                
                annotations = []
                for idx, row in res_df.iterrows():
                    symbol = ""
                    if row['status']=='gray': symbol="N/A"
                    elif row['p_val']<0.05: symbol="*"
                    annotations.append(dict(x=row['測站'], y=row['測項'], text=symbol, showarrow=False,
                                            font=dict(color='white' if row['status']=='red' else 'black', size=12)))
                
                color_map = {'gray': '#BDC3C7', 'green': '#2ECC71', 'red': '#E74C3C'}
                fig_h = go.Figure()
                fig_h.add_trace(go.Heatmap(
                    z=res_df['status_code'], x=res_df['測站'], y=res_df['測項'],
                    colorscale=[[0, color_map['gray']], [0.33, color_map['gray']],
                                [0.33, color_map['green']], [0.66, color_map['green']],
                                [0.66, color_map['red']], [1, color_map['red']]],
                    zmin=0, zmax=2, xgap=2, ygap=2, showscale=False,
                    hovertemplate="狀態: %{text}<extra></extra>", text=res_df['status_text']
                ))
                
                fig_h.update_layout(annotations=annotations, height=400, plot_bgcolor='white',
                                    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=20, color='black')), 
                                    yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=20, color='black')),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, # [新增] 設定圖例文字大小
                                                font=dict(size=20, family="Arial",color='black'))
                                   )
                
                # Legend
                legend_items = [('數據不足', color_map['gray']), ('無顯著變化', color_map['green']), ('具顯著變化', color_map['red'])]
                for l, c in legend_items:
                    fig_h.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color=c), name=l))
                fig_h.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_h, use_container_width=True)

                st.divider()

                # 詳細分析
                st.subheader("詳細檢定圖表")
                sc1, sc2 = st.columns(2)
                t1_st = sc1.selectbox("選擇測站 (差異檢定)", stations, key='t1_st')
                t1_it = sc2.selectbox("選擇測項 (差異檢定)", items, key='t1_it')
                
                t1_df = df[(df['測站']==t1_st) & (df['測項']==t1_it)]
                t1_res = res_df[(res_df['測站']==t1_st) & (res_df['測項']==t1_it)]
                
                if not t1_res.empty:
                    res = t1_res.iloc[0]
                    if res['status'] == 'gray':
                        st.info("數據不足。")
                    else:
                        fig_est = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], 
                                              subplot_titles=(f"{t1_it} 分佈", f"差異估計 ({res['test_method']})"))
                        
                        p_color = '#E74C3C' if res['status'] == 'red' else '#2ECC71'
                        colors = {'施工前': 'gray', '施工期間': p_color}
                        for p in ['施工前', '施工期間']:
                            sub = t1_df[t1_df['時期']==p]
                            if not sub.empty:
                                fig_est.add_trace(go.Box(y=sub['數值'], x=sub['時期'], name=p, boxpoints='all', jitter=0.5,
                                                       marker=dict(color=colors.get(p)), line=dict(color=colors.get(p)), showlegend=False), row=1, col=1)
                        
                        if pd.notna(res['upper_limit']):
                            fig_est.add_hline(y=res['upper_limit'], line_dash="dash", line_color="red", row=1, col=1)
                        if pd.notna(res['lower_limit']):
                            fig_est.add_hline(y=res['lower_limit'], line_dash="dash", line_color="red", row=1, col=1)

                        fig_est.add_hline(y=0, line_color="black", row=1, col=2)
                        if res['test_method'] == "無變化(Constant)":
                             fig_est.add_trace(go.Scatter(x=['差異'], y=[res['diff']], mode='markers'), row=1, col=2)
                        else:
                            fig_est.add_trace(go.Scatter(x=['差異'], y=[res['diff']], mode='markers', marker=dict(color='black'),
                                                       error_y=dict(type='data', array=[res['ci_upper']-res['diff']], arrayminus=[res['diff']-res['ci_lower']])), row=1, col=2)
                        
                        fig_est.update_layout(title_text=f"狀態: {res['status_text']} (P={res['p_val']:.4f})",
                                             title_font=dict(size=28, color="black"), font=dict(size=24, color="black"),
                                              annotations=[
                                dict(font_size=24, font_color="black") for _ in fig_est.layout.annotations
                            ])
                        fig_est.update_xaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=20, color="black"))
                        fig_est.update_yaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=20, color="black"))
                        st.plotly_chart(fig_est, use_container_width=True)

        # ==========================================
        # TAB 2: Mann-Kendall 趨勢分析 (新功能)
        # ==========================================
        with tab2:
            st.markdown("### Mann-Kendall 長期趨勢分析")
            st.markdown("""
            使用 **Mann-Kendall Test** 檢測時間序列是否存在顯著的單調趨勢，並使用 **Sen's Slope** 估計變化速率。
            *   適用於非母數（非常態分佈）數據。
            *   對離群值與未檢出值（ND）具有較好的容忍度。
            """)
            
            mk_c1, mk_c2 = st.columns(2)
            mk_st = mk_c1.selectbox("選擇測站 (趨勢分析)", stations, key='mk_st')
            mk_it = mk_c2.selectbox("選擇測項 (趨勢分析)", items, key='mk_it')
            
            # 準備數據
            mk_df = df[(df['測站']==mk_st) & (df['測項']==mk_it)]
            
            # 執行分析
            mk_res = perform_mk_analysis(mk_df)
            
            if mk_res is None:
                st.warning("⚠️ 數據點過少 (< 4 筆)，無法進行有效的 MK 趨勢檢定。")
            else:
                # 顯示指標
                m1, m2, m3, m4 = st.columns(4)
                
                trend_map = {'increasing': '📈 上升趨勢', 'decreasing': '📉 下降趨勢', 'no trend': '➡️ 無顯著趨勢'}
                trend_text = trend_map.get(mk_res['trend'], mk_res['trend'])
                color_delta = "off"
                if mk_res['trend'] == 'increasing': color_delta = "inverse" # 紅色
                if mk_res['trend'] == 'decreasing': color_delta = "normal"  # 綠色
                
                m1.metric("趨勢結果", trend_text)
                m2.metric("P-value", f"{mk_res['p_val']:.4f}", delta="顯著" if mk_res['p_val']<0.05 else None)
                m3.metric("Sen's Slope (斜率)", f"{mk_res['slope']:.4f}", help="代表單位時間內的變化量")
                m4.metric("Kendall Tau", f"{mk_res['h']}")

                # 繪製趨勢圖
                fig_mk = go.Figure()
                
                # 1. 原始數據點
                fig_mk.add_trace(go.Scatter(
                    x=mk_res['dates'], y=mk_res['values'],
                    mode='markers+lines',
                    name='監測數值',
                    marker=dict(color='#3498DB', size=8),
                    line=dict(color='#AED6F1', width=1)
                ))
                
                # 2. 趨勢線 (y = mx + c)
                # 注意：MK 的 slope 是針對時間單位的，繪圖時需要運算
                # 這裡使用簡單的線性回歸視覺化來輔助 Sen's slope 的概念
                if mk_res['trend'] != 'no trend':
                    # 計算趨勢線端點
                    x_nums = np.arange(len(mk_res['dates']))
                    y_trend = mk_res['slope'] * x_nums + mk_res['intercept']
                    
                    fig_mk.add_trace(go.Scatter(
                        x=mk_res['dates'], y=y_trend,
                        mode='lines',
                        name=f"趨勢線 (Slope={mk_res['slope']:.3f})",
                        line=dict(color='#E74C3C', width=3, dash='solid')
                    ))

                # 法規線
                limit_info = mk_df[['法規上限', '法規下限']].iloc[0]
                if pd.notna(limit_info['法規上限']):
                    fig_mk.add_hline(y=limit_info['法規上限'], line_dash="dash", line_color="red", annotation_text="上限")
                if pd.notna(limit_info['法規下限']):
                    fig_mk.add_hline(y=limit_info['法規下限'], line_dash="dash", line_color="red", annotation_text="下限")
                
                fig_mk.update_layout(
                    title=f"{mk_st} - {mk_it} 長期趨勢分析",title_font=dict(size=28, color="black"),
                    yaxis_title=f"數值 ({mk_res['unit']})",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.1, font=dict(size=18))
                )
                fig_mk.update_xaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))
                fig_mk.update_yaxes(title_font=dict(size=18, color="black"), tickfont=dict(size=18, color="black"))
                st.plotly_chart(fig_mk, use_container_width=True)
                
                st.info(f"💡 **Sen's Slope 解讀**：數值為 `{mk_res['slope']:.4f}`，代表每個採樣週期，數值平均變化約 `{mk_res['slope']:.4f}`。")

    except Exception as e:
        st.error(f"執行錯誤：{e}")





















