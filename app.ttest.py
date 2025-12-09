import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="環境監測統計檢定系統 (Excel版)", layout="wide")

# ==========================================
# 1. 工具函數：產生範本與統計核心
# ==========================================

def get_excel_template():
    """產生標準 Excel 範本供使用者下載"""
    output = io.BytesIO()
    # 建立範例資料
    data = {
        '測站': ['測站A', '測站A', '測站A', '測站A'],
        '測項': ['pH值', 'pH值', '噪音(dB)', '噪音(dB)'],
        '時期': ['施工前', '施工期間', '施工前', '施工期間'],
        '數值': [7.2, 7.5, 55.0, 60.2],
        '法規下限': [6.0, 6.0, '', ''],
        '法規上限': [9.0, 9.0, 65.0, 65.0],
        '單位': ['', '', 'dB', 'dB']
    }
    df_sample = pd.DataFrame(data)
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='監測數據')
        # 加入說明頁籤
        worksheet = writer.sheets['監測數據']
        worksheet.set_column('A:G', 15) # 設定欄寬
        
    return output.getvalue()

def perform_stats(df_sub):
    """
    執行統計檢定並回傳燈號狀態與統計數據 (邏輯與前版相同)
    """
    if df_sub.empty:
        return {'status': 'gray', 'status_text': '無數據', 'p_val': 1.0, 'diff': 0, 'test_method': 'N/A'}
        
    # 確保數值型別正確，並移除空值
    df_sub['數值'] = pd.to_numeric(df_sub['數值'], errors='coerce')
    df_sub = df_sub.dropna(subset=['數值'])
    
    group_pre = df_sub[df_sub['時期'] == '施工前']['數值'].values
    group_dur = df_sub[df_sub['時期'] == '施工期間']['數值'].values
    
    if len(group_pre) < 2 or len(group_dur) < 2:
        return {'status': 'gray', 'status_text': '數據不足', 'p_val': 1.0, 'diff': 0, 'test_method': '樣本不足'}

    # 取得法規與單位資訊 (處理可能的 NaN)
    lower_limit = df_sub['法規下限'].iloc[0]
    upper_limit = df_sub['法規上限'].iloc[0]
    unit = df_sub['單位'].iloc[0] if pd.notna(df_sub['單位'].iloc[0]) else ""
    item_name = df_sub['測項'].iloc[0]

    mean_pre = np.mean(group_pre)
    mean_dur = np.mean(group_dur)
    diff = mean_dur - mean_pre
    
    # 1. 常態性檢定
    try:
        if len(group_pre) < 3 or len(group_dur) < 3:
            is_normal = False
        else:
            _, p_norm_pre = stats.shapiro(group_pre)
            _, p_norm_dur = stats.shapiro(group_dur)
            is_normal = (p_norm_pre > 0.05) and (p_norm_dur > 0.05)
    except:
        is_normal = False

    # 2. 差異檢定
    try:
        if is_normal:
            stat, p_val = stats.ttest_ind(group_pre, group_dur, equal_var=False)
            test_method = "t-test (Welch)"
        else:
            stat, p_val = stats.mannwhitneyu(group_pre, group_dur)
            test_method = "Mann-Whitney U"
    except:
        return {'status': 'gray', 'status_text': '計算錯誤', 'p_val': 1.0, 'test_method': 'Error'}

    # 3. Bootstrap CI
    try:
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

    # 4. 燈號邏輯
    is_significant = p_val < 0.05
    
    # 方向性判斷
    if '溶氧量' in str(item_name) or 'DO' in str(item_name):
        is_worse = diff < 0 # 越低越差
    elif 'pH' in str(item_name):
        is_worse = True # pH 顯著波動視為變化
    else:
        is_worse = diff > 0 # 越高越差

    # 超標判斷
    is_violation = False
    if pd.notna(upper_limit) and mean_dur > upper_limit:
        is_violation = True
    if pd.notna(lower_limit) and mean_dur < lower_limit:
        is_violation = True
    
    status = "green"
    status_text = "正常"
    
    if is_violation:
        status = "red"
        status_text = "數值違規/超標"
    elif is_significant and is_worse:
        status = "yellow"
        status_text = "顯著變差 (預警)"
    else:
        status = "green"
        status_text = "無顯著異常"

    return {
        'mean_pre': mean_pre, 'mean_dur': mean_dur, 'diff': diff,
        'p_val': p_val, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'test_method': test_method, 'status': status, 'status_text': status_text,
        'lower_limit': lower_limit, 'upper_limit': upper_limit, 'unit': unit
    }

# ==========================================
# 2. Sidebar: 檔案上傳區
# ==========================================
st.sidebar.title("📁 資料匯入")

# 下載範本按鈕
st.sidebar.subheader("1. 下載範本")
st.sidebar.download_button(
    label="📥 下載 Excel 格式範本",
    data=get_excel_template(),
    file_name="環境監測數據範本.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# 上傳按鈕
st.sidebar.subheader("2. 上傳資料")
uploaded_file = st.sidebar.file_uploader("請上傳您的監測數據 (xlsx)", type=["xlsx"])

st.sidebar.info("""
**格式說明：**
請務必包含以下欄位：
- `測站`, `測項`, `時期`, `數值`
- `時期` 欄位請填寫 "施工前" 或 "施工期間"
""")

# ==========================================
# 3. 主畫面邏輯
# ==========================================
st.title("🛡️ 環境監測智能統計檢定系統 (Excel版)")

if uploaded_file is None:
    # 初始歡迎畫面
    st.info("👈 請從左側選單下載範本，填入數據後上傳以開始分析。")
    st.markdown("""
    ### 系統功能特色：
    1.  **自動判斷**：依據數據分佈自動選擇 t-test 或 Mann-Whitney U 檢定。
    2.  **法規檢核**：自動比對法規上下限，判斷是否超標。
    3.  **視覺化報告**：一鍵生成矩陣熱圖與詳細差異估計圖。
    """)
    
else:
    # 讀取並處理資料
    try:
        df = pd.read_excel(uploaded_file)
        
        # 簡單的欄位檢查
        required_columns = ['測站', '測項', '時期', '數值']
        if not all(col in df.columns for col in required_columns):
            st.error(f"❌ 格式錯誤：Excel 缺少必要欄位。請檢查是否包含：{required_columns}")
            st.stop()
            
        # 確保有法規欄位，若無則補 NaN
        for col in ['法規下限', '法規上限', '單位']:
            if col not in df.columns:
                df[col] = np.nan

        # 資料前處理
        df['時期'] = df['時期'].astype(str).str.strip() # 去除空白
        
        # 檢查是否有有效的時期標籤
        if not df['時期'].str.contains('施工前').any() or not df['時期'].str.contains('施工期間').any():
            st.warning("⚠️ 警告：`時期` 欄位中未偵測到 '施工前' 或 '施工期間'，系統可能無法進行比對。")

        # --- 計算統計 ---
        results = []
        stations = sorted(df['測站'].unique())
        items = sorted(df['測項'].unique())

        # 進度條 (若資料量大時有用)
        progress_bar = st.progress(0)
        total_tasks = len(stations) * len(items)
        counter = 0

        for s in stations:
            for i in items:
                sub_df = df[(df['測站']==s) & (df['測項']==i)]
                if not sub_df.empty:
                    res = perform_stats(sub_df)
                    res['測站'] = s
                    res['測項'] = i
                    results.append(res)
                
                counter += 1
                progress_bar.progress(counter / total_tasks)
        
        progress_bar.empty() # 清除進度條
        res_df = pd.DataFrame(results)

        # ====================
        # Dashboard 顯示區 (同前版邏輯)
        # ====================
        
        # 1. 交通號誌總覽
        st.subheader("1. 監測總覽")
        c1, c2, c3, c4 = st.columns(4)
        
        # 為了避免 KeyError，先檢查 status 是否存在
        if 'status' in res_df.columns:
            n_red = len(res_df[res_df['status'] == 'red'])
            n_yellow = len(res_df[res_df['status'] == 'yellow'])
            n_green = len(res_df[res_df['status'] == 'green'])
            n_gray = len(res_df[res_df['status'] == 'gray'])
        else:
            n_red, n_yellow, n_green, n_gray = 0, 0, 0, 0

        c1.metric("🔴 違規/超標", f"{n_red}", delta_color="inverse")
        c2.metric("🟡 顯著變差", f"{n_yellow}", delta_color="off")
        c3.metric("🟢 正常/改善", f"{n_green}")
        c4.metric("⚪ 數據不足", f"{n_gray}")

        st.divider()

        # 2. 熱力圖
        st.subheader("2. 異常偵測矩陣")
        
        if not res_df.empty:
            status_map = {'red': 2, 'yellow': 1, 'green': 0, 'gray': -1}
            res_df['status_code'] = res_df['status'].map(status_map)
            
            # P值標註
            annotations = []
            for index, row in res_df.iterrows():
                symbol = ""
                if row['status'] == 'gray': symbol = "N/A"
                elif row['p_val'] < 0.001: symbol = "***"
                elif row['p_val'] < 0.01: symbol = "**"
                elif row['p_val'] < 0.05: symbol = "*"
                
                annotations.append(dict(
                    x=row['測站'], y=row['測項'], text=symbol, showarrow=False,
                    font=dict(color='white' if row['status'] in ['red', 'green'] else 'black')
                ))

            colorscale = [
                [0.0, '#BDC3C7'], [0.25, '#BDC3C7'], # Gray
                [0.25, '#2ECC71'], [0.5, '#2ECC71'], # Green
                [0.5, '#F1C40F'], [0.75, '#F1C40F'], # Yellow
                [0.75, '#E74C3C'], [1.0, '#E74C3C']  # Red
            ]

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=res_df['status_code'], x=res_df['測站'], y=res_df['測項'],
                colorscale=colorscale, zmin=-1, zmax=2, xgap=2, ygap=2,
                hovertemplate="測站: %{x}<br>測項: %{y}<br>狀態: %{text}<extra></extra>",
                text=res_df['status_text']
            ))
            fig_heatmap.update_layout(annotations=annotations, height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("沒有產生任何統計結果，請檢查數據內容。")

        st.divider()

        # 3. 詳細分析
        st.subheader("3. 詳細檢定分析")
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            sel_station = st.selectbox("選擇測站", stations)
        with col_sel2:
            sel_item = st.selectbox("選擇測項", items)

        target_df = df[(df['測站']==sel_station) & (df['測項']==sel_item)]
        target_res = res_df[(res_df['測站']==sel_station) & (res_df['測項']==sel_item)]

        if not target_df.empty and not target_res.empty:
            res = target_res.iloc[0]
            if res['status'] == 'gray':
                st.info("此項目數據不足。")
            else:
                # 繪製 Estimation Plot
                fig_est = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                                      subplot_titles=(f"{sel_item} 原始數據", "平均差異 (95% CI)"))
                
                # 左圖 Boxplot
                colors = {'施工前': 'gray', '施工期間': '#E74C3C' if res['status'] in ['red', 'yellow'] else '#2ECC71'}
                for period in ['施工前', '施工期間']:
                    sub = target_df[target_df['時期']==period]
                    if not sub.empty:
                        fig_est.add_trace(go.Box(
                            y=sub['數值'], x=sub['時期'], name=period, boxpoints='all',
                            jitter=0.5, pointpos=-1.8, marker=dict(color=colors.get(period, 'blue')),
                            line=dict(color=colors.get(period, 'blue')), showlegend=False
                        ), row=1, col=1)

                # 法規線
                if pd.notna(res['upper_limit']):
                    fig_est.add_hline(y=res['upper_limit'], line_dash="dash", line_color="red", row=1, col=1)
                if pd.notna(res['lower_limit']):
                    fig_est.add_hline(y=res['lower_limit'], line_dash="dash", line_color="red", row=1, col=1)

                # 右圖 CI
                fig_est.add_hline(y=0, line_color="black", row=1, col=2)
                fig_est.add_trace(go.Scatter(
                    x=['差異'], y=[res['diff']], mode='markers', marker=dict(size=12, color='black'),
                    error_y=dict(type='data', array=[res['ci_upper']-res['diff']], 
                               arrayminus=[res['diff']-res['ci_lower']], thickness=2, width=10, color='black')
                ), row=1, col=2)

                fig_est.update_yaxes(title_text=f"數值 {res['unit']}", row=1, col=1)
                fig_est.update_layout(title_text=f"狀態: {res['status_text']} (P={res['p_val']:.4f})")
                st.plotly_chart(fig_est, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 讀取檔案時發生錯誤：{e}")
        st.warning("請確保您上傳的是有效的 Excel 檔，且格式與範本一致。")