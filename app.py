"""
回帰曲線グラフジェネレーター
統計サマリの表データからインタラクティブなHTMLグラフを生成

使い方:
1. streamlit run regression_graph_generator.py
2. スプシから統計サマリをコピペ
3. グラフ生成 → HTMLダウンロード

機能:
- 広告費 vs 新規UU グラフ
- 広告費 vs N-CPA グラフ（N-CPA = 広告費 ÷ 新規UU）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import io

st.set_page_config(
    page_title="回帰曲線グラフジェネレーター",
    page_icon="📈",
    layout="wide"
)

# セッション状態の初期化（ダウンロード後も状態を保持するため）
if 'fig_uu' not in st.session_state:
    st.session_state.fig_uu = None
if 'fig_ncpa' not in st.session_state:
    st.session_state.fig_ncpa = None
if 'graph_generated' not in st.session_state:
    st.session_state.graph_generated = False
if 'brand_count' not in st.session_state:
    st.session_state.brand_count = 0

st.title("📈 回帰曲線グラフジェネレーター")
st.markdown("統計サマリの表データからインタラクティブなグラフを生成します")

# サンプルデータ
sample_data = """出品者カテゴリー	広告費と広告新規UUの対数回帰式	決定係数(対数)	データ範囲 min x	データ範囲 max x	広告費と広告新規UUの線形回帰式	決定係数(線形)
ブランドA_カテゴリ1	y = 77.1095 * ln(x) + -656.0219	0.61	150	195023	y = 0.0013 * x + 54.4297	0.60
ブランドA_カテゴリ2	y = 365.3877 * ln(x) + -3853.9650	0.81	2198	833174	y = 0.0015 * x + 178.5103	0.83
ブランドA_カテゴリ3	y = 1051.4716 * ln(x) + -12066.0985	0.82	525	2850648	y = 0.0003 * x + 1977.5350	0.76"""

# 入力エリア
st.subheader("1️⃣ データ入力")
st.markdown("スプシの `✅️統計サマリ` シートからコピペしてください（ヘッダー行含む）")

data_input = st.text_area(
    "表データ（タブ区切り）",
    value=sample_data,
    height=200,
    help="スプレッドシートからコピーしたデータをそのまま貼り付けてください"
)

# グラフ設定
st.subheader("2️⃣ グラフ設定")
col1, col2 = st.columns(2)

with col1:
    graph_type = st.selectbox(
        "表示する回帰式",
        ["対数回帰（サチュレーションあり）", "線形回帰（サチュレーションなし）", "両方表示"]
    )

with col2:
    show_extrapolation = st.checkbox("外挿範囲を表示（点線）", value=True)

col3, col4 = st.columns(2)
with col3:
    extrapolation_ratio = st.slider(
        "外挿範囲の拡張倍率",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="データ最大値の何倍まで外挿するか（1.5 = 50%先まで）"
    )

# グラフタイトル
graph_title = st.text_input("グラフタイトル", value="ブランド別 SA広告費のサチュレーション")


def parse_log_equation(eq_str):
    """対数回帰式をパース: y = a * ln(x) + b"""
    match = re.search(r'y\s*=\s*([-\d.]+)\s*\*\s*ln\(x\)\s*\+\s*([-\d.]+)', eq_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def parse_linear_equation(eq_str):
    """線形回帰式をパース: y = a * x + b"""
    match = re.search(r'y\s*=\s*([-\d.]+)\s*\*\s*x\s*\+\s*([-\d.]+)', eq_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def generate_graph(df, graph_type, show_extrapolation, title, extrap_ratio=1.5):
    """Plotlyグラフを生成"""
    fig = go.Figure()

    # カラーパレット
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # 全体のX範囲を取得（外挿用に拡張）
    all_x_min = df['データ範囲 min x'].min()
    all_x_max = df['データ範囲 max x'].max() * extrap_ratio  # 拡張倍率を適用

    for i, (_, row) in enumerate(df.iterrows()):
        brand = row['出品者カテゴリー']
        x_min = row['データ範囲 min x']
        x_max = row['データ範囲 max x']
        color = colors[i % len(colors)]

        # 対数回帰
        if graph_type in ["対数回帰（サチュレーションあり）", "両方表示"]:
            a_log, b_log = parse_log_equation(row['広告費と広告新規UUの対数回帰式'])
            r2_log = row['決定係数(対数)']

            if a_log is not None:
                # データ範囲内（実線）
                x_data = np.linspace(x_min, x_max, 300)
                y_data = a_log * np.log(x_data) + b_log

                label = f"{brand} (R²={r2_log:.3f})" if graph_type != "両方表示" else f"{brand} 対数 (R²={r2_log:.3f})"
                legend_group = f"{brand}_log"  # 凡例グループ名

                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),
                    legendgroup=legend_group,
                    hovertemplate=f"<b>{brand}</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                ))

                # 外挿範囲（点線）
                if show_extrapolation:
                    if all_x_min < x_min:
                        x_ext_left = np.linspace(all_x_min, x_min, 100)
                        y_ext_left = a_log * np.log(x_ext_left) + b_log
                        fig.add_trace(go.Scatter(
                            x=x_ext_left, y=y_ext_left,
                            mode='lines',
                            name=f"{brand} (外挿)",
                            line=dict(color=color, width=1.5, dash='dash'),
                            opacity=0.5,
                            legendgroup=legend_group,
                            showlegend=False,
                            hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                        ))

                    if all_x_max > x_max:
                        x_ext_right = np.linspace(x_max, all_x_max, 100)
                        y_ext_right = a_log * np.log(x_ext_right) + b_log
                        fig.add_trace(go.Scatter(
                            x=x_ext_right, y=y_ext_right,
                            mode='lines',
                            name=f"{brand} (外挿)",
                            line=dict(color=color, width=1.5, dash='dash'),
                            opacity=0.5,
                            legendgroup=legend_group,
                            showlegend=False,
                            hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                        ))

        # 線形回帰
        if graph_type in ["線形回帰（サチュレーションなし）", "両方表示"]:
            a_lin, b_lin = parse_linear_equation(row['広告費と広告新規UUの線形回帰式'])
            r2_lin = row['決定係数(線形)']

            if a_lin is not None:
                x_data = np.linspace(x_min, x_max, 300)
                y_data = a_lin * x_data + b_lin

                label = f"{brand} (R²={r2_lin:.3f})" if graph_type != "両方表示" else f"{brand} 線形 (R²={r2_lin:.3f})"
                line_style = 'dot' if graph_type == "両方表示" else 'solid'
                legend_group = f"{brand}_lin"  # 凡例グループ名

                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash=line_style),
                    legendgroup=legend_group,
                    hovertemplate=f"<b>{brand}</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                ))

                # 外挿範囲（点線）
                if show_extrapolation:
                    if all_x_min < x_min:
                        x_ext_left = np.linspace(all_x_min, x_min, 100)
                        y_ext_left = a_lin * x_ext_left + b_lin
                        fig.add_trace(go.Scatter(
                            x=x_ext_left, y=y_ext_left,
                            mode='lines',
                            name=f"{brand} (外挿)",
                            line=dict(color=color, width=1.5, dash='dash'),
                            opacity=0.5,
                            legendgroup=legend_group,
                            showlegend=False,
                            hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                        ))

                    if all_x_max > x_max:
                        x_ext_right = np.linspace(x_max, all_x_max, 100)
                        y_ext_right = a_lin * x_ext_right + b_lin
                        fig.add_trace(go.Scatter(
                            x=x_ext_right, y=y_ext_right,
                            mode='lines',
                            name=f"{brand} (外挿)",
                            line=dict(color=color, width=1.5, dash='dash'),
                            opacity=0.5,
                            legendgroup=legend_group,
                            showlegend=False,
                            hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>新規UU: %{{y:,.0f}}<extra></extra>"
                        ))

    # レイアウト設定
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="月間SA広告費 (円)",
        yaxis_title="広告新規UU数",
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(r=250),
        template='plotly_white'
    )

    # X軸のフォーマット（0以上のみ表示）
    fig.update_xaxes(
        tickformat=",",
        gridcolor='lightgray',
        gridwidth=0.5,
        rangemode='tozero'
    )
    fig.update_yaxes(
        tickformat=",",
        gridcolor='lightgray',
        gridwidth=0.5,
        rangemode='nonnegative',
        range=[0, None]
    )

    return fig


def find_ncpa_minimum_log(a, b):
    """
    対数回帰のN-CPA最小点を求める
    N-CPA = x / (a * ln(x) + b) の最小点
    微分して0になる点: x = exp(1 - b/a)
    """
    if a <= 0:
        return None
    x_min = np.exp(1 - b / a)
    # 最小点でUUが正であることを確認
    uu_at_min = a * np.log(x_min) + b
    if uu_at_min > 0:
        return x_min
    return None


def generate_ncpa_graph(df, graph_type, show_extrapolation, title, extrap_ratio=1.5):
    """N-CPAグラフを生成（N-CPA = 広告費 ÷ 新規UU）- 単調増加部分のみ表示"""
    fig = go.Figure()

    # カラーパレット
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # 全体のX範囲を取得（外挿用に拡張）
    all_x_max = df['データ範囲 max x'].max() * extrap_ratio

    for i, (_, row) in enumerate(df.iterrows()):
        brand = row['出品者カテゴリー']
        x_min = row['データ範囲 min x']
        x_max = row['データ範囲 max x']
        color = colors[i % len(colors)]

        # 対数回帰からN-CPA算出（単調増加部分のみ）
        if graph_type in ["対数回帰（サチュレーションあり）", "両方表示"]:
            a_log, b_log = parse_log_equation(row['広告費と広告新規UUの対数回帰式'])
            r2_log = row['決定係数(対数)']

            if a_log is not None:
                # N-CPAの最小点を求める
                x_ncpa_min = find_ncpa_minimum_log(a_log, b_log)

                # 表示開始点: N-CPA最小点とデータ範囲の大きい方
                if x_ncpa_min is not None:
                    x_start = max(x_ncpa_min, x_min)
                else:
                    x_start = x_min

                # データ範囲内（実線）- 単調増加部分のみ
                if x_start < x_max:
                    x_data = np.linspace(x_start, x_max, 300)
                    y_uu = a_log * np.log(x_data) + b_log
                    y_ncpa = np.where(y_uu > 0, x_data / y_uu, np.nan)

                    label = f"{brand} (R²={r2_log:.3f})" if graph_type != "両方表示" else f"{brand} 対数 (R²={r2_log:.3f})"
                    legend_group = f"{brand}_ncpa_log"  # 凡例グループ名

                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_ncpa,
                        mode='lines',
                        name=label,
                        line=dict(color=color, width=2),
                        legendgroup=legend_group,
                        hovertemplate=f"<b>{brand}</b><br>広告費: %{{x:,.0f}}円<br>N-CPA: %{{y:,.0f}}円<extra></extra>"
                    ))

                    # 外挿範囲（点線）- 右側のみ（単調増加方向）
                    if show_extrapolation and all_x_max > x_max:
                        x_ext_right = np.linspace(x_max, all_x_max, 100)
                        y_uu_right = a_log * np.log(x_ext_right) + b_log
                        y_ncpa_right = np.where(y_uu_right > 0, x_ext_right / y_uu_right, np.nan)
                        fig.add_trace(go.Scatter(
                            x=x_ext_right, y=y_ncpa_right,
                            mode='lines',
                            name=f"{brand} (外挿)",
                            line=dict(color=color, width=1.5, dash='dash'),
                            opacity=0.5,
                            legendgroup=legend_group,
                            showlegend=False,
                            hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>N-CPA: %{{y:,.0f}}円<extra></extra>"
                        ))

        # 線形回帰からN-CPA算出
        # 線形の場合: N-CPA = x / (ax + b) は x→∞ で 1/a に収束（単調減少）
        # 実務的には対数回帰のみが意味を持つが、参考として表示
        if graph_type in ["線形回帰（サチュレーションなし）", "両方表示"]:
            a_lin, b_lin = parse_linear_equation(row['広告費と広告新規UUの線形回帰式'])
            r2_lin = row['決定係数(線形)']

            if a_lin is not None:
                x_data = np.linspace(x_min, x_max, 300)
                y_uu = a_lin * x_data + b_lin
                y_ncpa = np.where(y_uu > 0, x_data / y_uu, np.nan)

                label = f"{brand} (R²={r2_lin:.3f})" if graph_type != "両方表示" else f"{brand} 線形 (R²={r2_lin:.3f})"
                line_style = 'dot' if graph_type == "両方表示" else 'solid'
                legend_group = f"{brand}_ncpa_lin"  # 凡例グループ名

                fig.add_trace(go.Scatter(
                    x=x_data, y=y_ncpa,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2, dash=line_style),
                    legendgroup=legend_group,
                    hovertemplate=f"<b>{brand}</b><br>広告費: %{{x:,.0f}}円<br>N-CPA: %{{y:,.0f}}円<extra></extra>"
                ))

                # 外挿範囲（点線）- 右側のみ
                if show_extrapolation and all_x_max > x_max:
                    x_ext_right = np.linspace(x_max, all_x_max, 100)
                    y_uu_right = a_lin * x_ext_right + b_lin
                    y_ncpa_right = np.where(y_uu_right > 0, x_ext_right / y_uu_right, np.nan)
                    fig.add_trace(go.Scatter(
                        x=x_ext_right, y=y_ncpa_right,
                        mode='lines',
                        name=f"{brand} (外挿)",
                        line=dict(color=color, width=1.5, dash='dash'),
                        opacity=0.5,
                        legendgroup=legend_group,
                        showlegend=False,
                        hovertemplate=f"<b>{brand} (外挿)</b><br>広告費: %{{x:,.0f}}円<br>N-CPA: %{{y:,.0f}}円<extra></extra>"
                    ))

    # レイアウト設定
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="月間SA広告費 (円)",
        yaxis_title="N-CPA (円/UU)",
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
        margin=dict(r=250),
        template='plotly_white'
    )

    # X軸のフォーマット
    fig.update_xaxes(
        tickformat=",",
        gridcolor='lightgray',
        gridwidth=0.5,
        rangemode='tozero'
    )
    fig.update_yaxes(
        tickformat=",",
        gridcolor='lightgray',
        gridwidth=0.5,
        rangemode='tozero'
    )

    return fig


# グラフ生成ボタン
if st.button("📊 グラフ生成", type="primary"):
    try:
        # データをパース
        df = pd.read_csv(io.StringIO(data_input), sep='\t')

        # 必要なカラムの確認
        required_cols = ['出品者カテゴリー', 'データ範囲 min x', 'データ範囲 max x']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"必要なカラムが見つかりません: {missing_cols}")
        else:
            # 数値変換
            df['データ範囲 min x'] = pd.to_numeric(df['データ範囲 min x'], errors='coerce')
            df['データ範囲 max x'] = pd.to_numeric(df['データ範囲 max x'], errors='coerce')
            df['決定係数(対数)'] = pd.to_numeric(df['決定係数(対数)'], errors='coerce')
            df['決定係数(線形)'] = pd.to_numeric(df['決定係数(線形)'], errors='coerce')

            # グラフ生成してセッション状態に保存
            st.session_state.fig_uu = generate_graph(df, graph_type, show_extrapolation, graph_title, extrapolation_ratio)
            st.session_state.fig_ncpa = generate_ncpa_graph(df, graph_type, show_extrapolation, f"{graph_title} - N-CPA", extrapolation_ratio)
            st.session_state.graph_generated = True
            st.session_state.brand_count = len(df)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
        st.info("データの形式を確認してください。タブ区切りでヘッダー行を含める必要があります。")

# グラフが生成されていれば表示（ダウンロード後も維持される）
if st.session_state.graph_generated and st.session_state.fig_uu is not None:
    st.success(f"✅ {st.session_state.brand_count}ブランドのデータを読み込みました")

    # グラフ表示
    st.subheader("3️⃣ グラフプレビュー")

    tab1, tab2 = st.tabs(["📈 新規UUグラフ", "💰 N-CPAグラフ"])

    with tab1:
        st.plotly_chart(st.session_state.fig_uu, use_container_width=True)

    with tab2:
        st.markdown("**N-CPA = 広告費 ÷ 新規UU**")
        st.plotly_chart(st.session_state.fig_ncpa, use_container_width=True)

    # HTMLダウンロード
    st.subheader("4️⃣ ダウンロード")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        html_uu = st.session_state.fig_uu.to_html(include_plotlyjs=True, full_html=True)
        st.download_button(
            label="📥 新規UUグラフ (HTML)",
            data=html_uu,
            file_name="brand_regression_uu.html",
            mime="text/html"
        )

    with col_dl2:
        html_ncpa = st.session_state.fig_ncpa.to_html(include_plotlyjs=True, full_html=True)
        st.download_button(
            label="📥 N-CPAグラフ (HTML)",
            data=html_ncpa,
            file_name="brand_regression_ncpa.html",
            mime="text/html"
        )

    st.info("💡 ダウンロードしたHTMLファイルはブラウザで開くとインタラクティブに操作できます")

# 使い方
with st.expander("📖 使い方"):
    st.markdown("""
    ### 手順
    1. コホートSIMスプレッドシートの `✅️統計サマリ` シートを開く
    2. A〜G列のデータを選択してコピー（ヘッダー行含む）
    3. このツールの入力欄に貼り付け
    4. 「グラフ生成」ボタンをクリック
    5. HTMLファイルをダウンロード

    ### 必要なカラム
    - `出品者カテゴリー`: ブランド名
    - `広告費と広告新規UUの対数回帰式`: y = a * ln(x) + b 形式
    - `決定係数(対数)`: R²値
    - `データ範囲 min x`: 広告費の最小値
    - `データ範囲 max x`: 広告費の最大値
    - `広告費と広告新規UUの線形回帰式`: y = a * x + b 形式（オプション）
    - `決定係数(線形)`: R²値（オプション）
    """)
