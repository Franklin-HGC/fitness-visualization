"""
Fitness Vision - 基于健身数据的 Streamlit 可视化分析平台
支持 CSV/Excel 上传、数据预览清洗、自然语言交互生成图表
"""
import io
import json
import re

import anthropic
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────── 页面配置 ────────────────────────────
st.set_page_config(
    page_title="Fitness Vision - 健身数据可视化平台",
    page_icon="  ",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── 自定义样式 ────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1E88E5;
        text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 0.75rem; color: white; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 24px; border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def load_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    """根据文件类型读取上传文件为 DataFrame"""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("不支持的文件格式，请上传 CSV 或 Excel 文件。")
            return None
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """基础数据清洗：填充缺失值、去除明显异常"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            # IQR 去除极端异常
            q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            df[col] = df[col].clip(lower, upper)
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    return df


def data_overview(df: pd.DataFrame) -> dict:
    """生成数据概览信息"""
    return {
        "行数": len(df),
        "列数": len(df.columns),
        "字段列表": list(df.columns),
        "字段类型": {col: str(dt) for col, dt in df.dtypes.items()},
        "缺失值": df.isnull().sum().to_dict(),
        "缺失值比例": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "数值列统计": df.describe().round(2).to_dict(),
    }


# ══════════════════════════════════════════════════════════════════
#  自然语言 → 图表 (API + 本地回退)
# ══════════════════════════════════════════════════════════════════

CHART_SPEC_SYSTEM = """你是一个数据可视化专家。用户会给你一个数据集的列信息和一条自然语言指令，你需要返回一个 JSON 对象来描述要生成的图表。

数据集列信息:
{columns_info}

请严格按以下 JSON 格式返回（不要输出任何其他内容）:
{{
  "chart_type": "bar|line|scatter|pie|heatmap|histogram|box",
  "x": "X轴列名（饼图不需要）",
  "y": "Y轴列名（饼图用 values）",
  "color": "分组列名（可选，无分组填null）",
  "title": "图表标题",
  "agg": "聚合方式: mean|sum|count|none（可选，默认none）",
  "top_n": "仅展示前N个（可选，填null表示全部）"
}}

规则:
- 列名必须是数据集中存在的列名
- 如果指令不明确，选择最合理的默认
- chart_type 必须是上述7种之一
- 只返回 JSON，不要解释"""


def build_columns_info(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"- {col} (类型: {dtype}, 示例: {sample})")
    return "\n".join(lines)


def call_api_for_chart_spec(user_query: str, df: pd.DataFrame, api_key: str, model: str) -> dict | None:
    """调用 Anthropic Claude API 解析自然语言为图表规格"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        columns_info = build_columns_info(df)
        message = client.messages.create(
            model=model,
            max_tokens=512,
            system=CHART_SPEC_SYSTEM.format(columns_info=columns_info),
            messages=[{"role": "user", "content": user_query}],
        )
        text = message.content[0].text.strip()
        # 提取 JSON（兼容 markdown code block）
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception as e:
        st.warning(f"API 调用失败，切换到本地解析: {e}")
        return None


def local_parse_chart_spec(user_query: str, df: pd.DataFrame) -> dict | None:
    """基于关键词的本地自然语言解析（API 不可用时的回退方案）"""
    q = user_query.lower()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    # 检测图表类型
    chart_type = "bar"
    for kw, ct in [("散点", "scatter"), ("scatter", "scatter"),
                    ("折线", "line"), ("line", "line"),
                    ("饼", "pie"), ("pie", "pie"),
                    ("热力", "heatmap"), ("heat", "heatmap"),
                    ("直方", "histogram"), ("histogram", "histogram"),
                    ("箱线", "box"), ("box", "box")]:
        if kw in q:
            chart_type = ct
            break

    # 尝试匹配列名
    def find_col(keywords: list[str], candidates: list[str]) -> str | None:
        for kw in keywords:
            for c in candidates:
                if kw in c.lower():
                    return c
        return None

    x_col = y_col = color_col = None

    if chart_type == "pie":
        y_col = find_col(["calori", "热量", "卡路里", "value", "数量"], num_cols) or (num_cols[0] if num_cols else None)
        x_col = find_col(["type", "种类", "类型", "category", "品类", "workout"], cat_cols + num_cols) or (cat_cols[0] if cat_cols else None)
    elif chart_type == "heatmap":
        x_col = num_cols[0] if len(num_cols) > 0 else None
        y_col = num_cols[1] if len(num_cols) > 1 else None
    else:
        # X 轴: 优先分类列
        x_keywords = ["type", "种类", "类型", "gender", "性别", "age", "年龄",
                       "experience", "经验", "category", "品类", "workout"]
        x_col = find_col(x_keywords, all_cols) or (cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None))

        # Y 轴: 数值列
        y_keywords = ["calori", "热量", "卡路里", "bmi", "heart", "心率",
                       "duration", "时长", "fat", "脂肪", "water", "水分",
                       "weight", "体重", "revenue", "收入", "sales", "销量"]
        y_col = find_col(y_keywords, num_cols) or (num_cols[0] if num_cols else None)

    # Color 分组
    color_kw = ["gender", "性别", "group", "分组", "category", "类别", "experience", "经验"]
    color_col = find_col(color_kw, cat_cols)

    agg = "mean" if chart_type in ("bar", "line") and y_col else "none"
    titles = {"bar": "柱状图", "line": "折线图", "scatter": "散点图",
              "pie": "饼图", "heatmap": "热力图", "histogram": "直方图", "box": "箱线图"}

    return {
        "chart_type": chart_type,
        "x": x_col,
        "y": y_col,
        "color": color_col,
        "title": f"{y_col or ''} {titles.get(chart_type, '')}",
        "agg": agg,
        "top_n": None,
    }


def generate_chart(df: pd.DataFrame, spec: dict) -> go.Figure | None:
    """根据图表规格用 Plotly 生成交互式图表"""
    ct = spec.get("chart_type", "bar")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    title = spec.get("title", "")
    agg = spec.get("agg", "none")
    top_n = spec.get("top_n")

    try:
        plot_df = df.copy()

        # 聚合（若指定了 color 分组，groupby 需包含 color 列）
        if agg in ("mean", "sum", "count") and x and y and x in plot_df.columns and y in plot_df.columns:
            group_cols = [x]
            if color and color in plot_df.columns and color != x:
                group_cols.append(color)
            if agg == "count":
                plot_df = plot_df.groupby(group_cols)[y].count().reset_index()
            else:
                plot_df = plot_df.groupby(group_cols)[y].agg(agg).reset_index()

        # Top N
        if top_n and x and x in plot_df.columns:
            if y and y in plot_df.columns:
                plot_df = plot_df.nlargest(int(top_n), y)
            else:
                plot_df = plot_df.head(int(top_n))

        if ct == "bar":
            fig = px.bar(plot_df, x=x, y=y, color=color, title=title, template="plotly_white")
        elif ct == "line":
            fig = px.line(plot_df, x=x, y=y, color=color, title=title, template="plotly_white",
                          markers=True)
        elif ct == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color, title=title, template="plotly_white",
                             opacity=0.7)
        elif ct == "pie":
            names = x if x in plot_df.columns else None
            values = y if y in plot_df.columns else (plot_df.select_dtypes("number").columns[0] if not plot_df.select_dtypes("number").empty else None)
            fig = px.pie(plot_df, names=names, values=values, title=title, template="plotly_white")
        elif ct == "heatmap":
            num_df = plot_df.select_dtypes(include="number")
            corr = num_df.corr().round(2)
            fig = px.imshow(corr, text_auto=True, title=title or "相关性热力图",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        elif ct == "histogram":
            col_for_hist = y if y in plot_df.columns else x
            fig = px.histogram(plot_df, x=col_for_hist, color=color, title=title,
                               template="plotly_white", nbins=30)
        elif ct == "box":
            fig = px.box(plot_df, x=x, y=y, color=color, title=title, template="plotly_white")
        else:
            st.error(f"不支持的图表类型: {ct}")
            return None

        fig.update_layout(
            font=dict(family="Microsoft YaHei, Arial", size=14),
            title_font_size=18,
            height=500,
        )
        return fig

    except Exception as e:
        st.error(f"图表生成失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
#  Skill 设计说明书内容
# ══════════════════════════════════════════════════════════════════

SKILL_DESIGN = """
## 可视化迭代 Skill 设计说明书

### Skill 名称: FitnessVizAgent — 健身数据可视化迭代智能体

---

### 环节 1: 数据理解 (Data Understanding)

| 项目 | 内容 |
|------|------|
| **核心逻辑** | 解析数据集结构，提取字段名、类型、分布特征、缺失情况，形成数据画像 |
| **输入** | 原始 DataFrame |
| **输出** | 数据画像 JSON（含字段列表、类型映射、数值列统计、分类列唯一值、缺失值概况） |
| **实现思路** | 使用 pandas 的 `dtypes`、`describe()`、`nunique()`、`isnull().sum()` 自动扫描；对数值列计算偏度/峰度判断分布形态；将画像存入上下文供后续环节使用 |

---

### 环节 2: 图表选择 (Chart Selection)

| 项目 | 内容 |
|------|------|
| **核心逻辑** | 根据用户自然语言意图 + 数据画像，匹配最优图表类型和字段映射 |
| **输入** | 用户指令（文本）+ 数据画像 |
| **输出** | 图表规格 JSON（chart_type, x, y, color, title, agg, top_n） |
| **实现思路** | 两层策略：(1) 规则层——关键词匹配图表类型，字段名相似度匹配坐标轴；(2) LLM 层——将数据画像 + 指令送入大语言模型，返回结构化 JSON。规则层作为回退保证可用性。支持柱状图、折线图、散点图、饼图、热力图、直方图、箱线图 |

---

### 环节 3: 样式优化 (Style Optimization)

| 项目 | 内容 |
|------|------|
| **核心逻辑** | 基于图表类型和数据特征自动优化视觉样式 |
| **输入** | 图表规格 JSON + 原始数据 |
| **输出** | Plotly Figure 对象 |
| **实现思路** | (1) 配色方案：根据分类数量自动选择离散/连续色板；(2) 布局优化：自动调整图例位置、坐标轴标签旋转、边距；(3) 交互增强：hover 信息格式化、缩放/平移支持；(4) 中文字体适配 |

---

### 环节 4: 可解释输出 (Interpretable Output)

| 项目 | 内容 |
|------|------|
| **核心逻辑** | 对生成的图表进行自然语言解读，帮助用户理解数据洞察 |
| **输入** | 图表规格 + 聚合后的数据 |
| **输出** | 文字说明（关键发现、趋势描述、异常提示） |
| **实现思路** | 自动计算：(1) 极值定位（最大/最小值及其对应维度）；(2) 趋势判断（线性回归斜率方向）；(3) 分布描述（集中趋势 + 离散程度）。将计算结果用模板或 LLM 生成通俗易懂的解读文字，展示在图表下方 |

---

### 环节 5: 多轮迭代 (Multi-turn Iteration)

| 项目 | 内容 |
|------|------|
| **核心逻辑** | 保留历史交互上下文，支持用户对图表进行增量修改 |
| **输入** | 用户追加指令 + 上一轮图表规格 |
| **输出** | 更新后的图表规格 + 新图表 |
| **实现思路** | (1) 维护 session 内的对话历史列表，每轮将上一次的 spec 作为上下文传入 LLM；(2) 支持增量指令如"换用折线图""按性别分组""只看前10个"——LLM 只需修改 spec 中变化的字段；(3) 保留图表历史，用户可回溯对比；(4) 支持"重置"指令清空上下文重新开始 |

---

### 流程图

```
用户输入指令
     │
     ▼
┌──────────────┐    ┌──────────────┐
│ 数据理解      │───▶│ 图表选择      │
│ (画像生成)    │    │ (NL→Spec)    │
└──────────────┘    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ 样式优化      │
                    │ (Spec→Fig)   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ 可解释输出    │
                    │ (Fig→解读)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ 用户反馈      │──→ 回到"图表选择"（多轮迭代）
                    └──────────────┘
```
"""


# ══════════════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 标题区 ──
    st.markdown('<div class="main-header">Fitness Vision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于健身数据的智能可视化分析平台 — 上传数据，用自然语言探索洞察</div>',
                unsafe_allow_html=True)

    # ── 侧边栏: API 配置 & 数据上传 ──
    with st.sidebar:
        st.header("⚙️ 设置")
        api_key = st.text_input("Anthropic API Key", type="password",
                                help="输入你的 Anthropic API Key 以启用 AI 图表生成；留空则使用本地解析")
        model_choice = st.selectbox("模型", ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"], index=0)
        use_api = bool(api_key)

        st.markdown("---")
        st.header("  数据上传")
        uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx", "xls"])

        use_sample = False
        if uploaded_file is None:
            use_sample = st.checkbox("使用内置示例数据集 (Gym Members)", value=True)

        st.markdown("---")
        st.markdown("**支持的图表类型:** 柱状图、折线图、散点图、饼图、热力图、直方图、箱线图")

    # ── 加载数据 ──
    df = None
    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
    elif use_sample:
        try:
            df = pd.read_csv("data/gym_members.csv")
        except FileNotFoundError:
            st.error("示例数据集不存在，请先运行 `python generate_sample_data.py`")

    if df is None:
        st.info("  请在左侧上传数据文件或勾选使用示例数据集。")
        return

    # ── Tab 布局 ──
    tab1, tab2, tab3, tab4 = st.tabs(["  数据概览", "  智能可视化", "  Skill 设计说明", "  使用说明"])

    # ─────────── Tab 1: 数据概览 ───────────
    with tab1:
        st.subheader("数据集预览")

        overview = data_overview(df)

        # 指标卡片
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("数据行数", overview["行数"])
        c2.metric("字段数量", overview["列数"])
        total_missing = sum(overview["缺失值"].values())
        c3.metric("缺失值总数", total_missing)
        c4.metric("缺失率", f"{total_missing / (overview['行数'] * overview['列数']) * 100:.1f}%")

        st.markdown("#### 前 N 行数据")
        n_rows = st.slider("预览行数", 5, 50, 10, step=5, key="preview_rows")
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown("#### 字段类型与缺失值")
        type_df = pd.DataFrame({
            "字段名": list(overview["字段类型"].keys()),
            "数据类型": list(overview["字段类型"].values()),
            "缺失值数": list(overview["缺失值"].values()),
            "缺失率(%)": list(overview["缺失值比例"].values()),
        })
        st.dataframe(type_df, use_container_width=True)

        # 数据清洗
        st.markdown("---")
        st.subheader("  数据清洗")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("执行基础清洗", type="primary"):
                st.session_state["df_cleaned"] = clean_dataframe(df)
                st.success(f"清洗完成！缺失值已填充，极端异常值已处理。")
        with col_c2:
            if "df_cleaned" in st.session_state:
                if st.button("恢复原始数据"):
                    del st.session_state["df_cleaned"]
                    st.rerun()

        if "df_cleaned" in st.session_state:
            st.info("当前使用**清洗后**的数据进行可视化。")
            df = st.session_state["df_cleaned"]

    # ─────────── Tab 2: 智能可视化 ───────────
    with tab2:
        st.subheader("自然语言图表生成")

        # 预设示例
        st.markdown("**试试这些指令:**")
        example_cols = st.columns(4)
        examples = [
            "各运动类型的平均卡路里消耗柱状图",
            "年龄与BMI的散点图",
            "按性别分组的体脂率箱线图",
            "数值字段相关性热力图",
        ]
        for i, ex in enumerate(examples):
            if example_cols[i].button(ex, key=f"ex_{i}"):
                st.session_state["nl_query"] = ex

        user_query = st.text_area(
            "输入你的可视化指令:",
            value=st.session_state.get("nl_query", ""),
            height=80,
            placeholder='例如: "根据数据生成各运动类型卡路里消耗的柱状图"',
            key="nl_input",
        )

        if st.button("  生成图表", type="primary", disabled=not user_query.strip()):
            with st.spinner("正在分析指令并生成图表..."):
                if use_api:
                    spec = call_api_for_chart_spec(user_query, df, api_key, model_choice)
                    if spec is None:
                        spec = local_parse_chart_spec(user_query, df)
                    source = "API (Claude)" if spec else "本地解析"
                else:
                    spec = local_parse_chart_spec(user_query, df)
                    source = "本地关键词解析"

            if spec:
                st.success(f"解析成功（{source}） — 图表类型: {spec['chart_type']}")
                with st.expander("查看图表规格 JSON"):
                    st.json(spec)

                fig = generate_chart(df, spec)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # 自动生成简单解读
                    st.markdown("####   数据解读")
                    y_col = spec.get("y")
                    x_col = spec.get("x")
                    if y_col and y_col in df.columns:
                        col_data = df[y_col].dropna()
                        st.write(f"- **{y_col}** 范围: {col_data.min():.1f} ~ {col_data.max():.1f}，"
                                 f"均值: {col_data.mean():.1f}，中位数: {col_data.median():.1f}")
                        if x_col and x_col in df.columns:
                            grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False)
                            st.write(f"- 最高: **{grouped.index[0]}** ({grouped.iloc[0]:.1f})，"
                                     f"最低: **{grouped.index[-1]}** ({grouped.iloc[-1]:.1f})")

                    # 多轮迭代: 追加指令
                    st.markdown("---")
                    st.markdown("####   迭代优化")
                    follow_up = st.text_input('输入追加指令来调整图表（例如"换折线图""按性别分组"）:',
                                              key="follow_up")
                    if follow_up.strip():
                        merged_query = f"{user_query}。补充要求: {follow_up}"
                        if use_api:
                            new_spec = call_api_for_chart_spec(merged_query, df, api_key, model_choice)
                            if not new_spec:
                                new_spec = local_parse_chart_spec(merged_query, df)
                        else:
                            new_spec = local_parse_chart_spec(merged_query, df)
                        if new_spec:
                            new_fig = generate_chart(df, new_spec)
                            if new_fig:
                                st.plotly_chart(new_fig, use_container_width=True)
            else:
                st.error("无法解析指令，请尝试更明确的描述。")

    # ─────────── Tab 3: Skill 设计说明 ───────────
    with tab3:
        st.subheader("可视化迭代 Skill 设计说明书")
        st.markdown(SKILL_DESIGN)

    # ─────────── Tab 4: 使用说明 ───────────
    with tab4:
        st.subheader("使用说明")
        st.markdown("""
        ### 快速开始

        1. **上传数据** — 在左侧面板上传 CSV 或 Excel 文件，或使用内置示例数据集
        2. **查看数据** — 切换到「数据概览」标签页，检查数据维度、字段类型和缺失值
        3. **清洗数据** — 点击「执行基础清洗」处理缺失值和异常值
        4. **生成图表** — 切换到「智能可视化」，用自然语言描述你想要的图表
        5. **迭代优化** — 生成图表后可输入追加指令进行调整

        ### 自然语言指令示例

        | 指令 | 图表类型 |
        |------|----------|
        | 各运动类型的平均卡路里消耗柱状图 | 柱状图 |
        | 年龄与BMI的关系散点图 | 散点图 |
        | 按性别分组的体脂率箱线图 | 箱线图 |
        | 所有数值字段的相关性热力图 | 热力图 |
        | 运动类型占比饼图 | 饼图 |
        | 心率分布直方图 | 直方图 |
        | 不同经验等级的训练时长折线图 | 折线图 |

        ### API 配置

        - 在左侧输入 **Anthropic API Key** 启用 AI 智能解析
        - 不输入 Key 也可使用，系统会自动切换到本地关键词解析模式
        - 推荐使用 `claude-sonnet-4-6` 模型以获得最佳解析效果

        ### 数据要求

        - 支持 CSV 和 Excel (.xlsx/.xls) 格式
        - 数据应包含表头行
        - 建议包含至少 1 个数值列和 1 个分类列以获得最佳可视化效果
        """)


if __name__ == "__main__":
    main()
