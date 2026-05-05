# Fitness Vision

基于健身数据的 Streamlit 智能可视化分析平台。

## 功能特性

- **数据处理**: 支持 CSV/Excel 上传，数据预览（维度/字段类型/缺失值统计），基础数据清洗
- **自然语言可视化**: 通过自然语言指令生成交互式图表（柱状图、折线图、散点图、饼图、热力图等）
- **AI 驱动**: 集成 Anthropic Claude API 智能解析指令，无 API Key 时自动切换本地关键词解析
- **自动数据解读**: 生成图表后自动计算极值、均值等统计信息并展示文字分析
- **多轮迭代**: 支持增量修改指令，持续优化图表

## 数据集

内置示例数据集 `data/gym_members.csv`，包含 500 条健身会员运动记录，涵盖：

| 字段 | 说明 |
|------|------|
| Age | 年龄 |
| Gender | 性别 |
| Weight_kg | 体重(kg) |
| Height_cm | 身高(cm) |
| BMI | 身体质量指数 |
| Workout_Type | 运动类型 (Cardio/Strength/HIIT/Yoga/Stretching) |
| Session_Duration_hrs | 训练时长(小时) |
| Calories_Burned | 消耗卡路里 |
| Avg_Heart_Rate | 平均心率 |
| Max_Heart_Rate | 最大心率 |
| Fat_Percentage | 体脂率 |
| Water_Intake_liters | 饮水量(升) |
| Workout_Frequency_days_week | 每周训练天数 |
| Experience_Level | 经验等级 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成示例数据（可选，data/ 目录下已有预生成数据）

```bash
python generate_sample_data.py
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`。

## 使用说明

1. 在左侧边栏上传 CSV/Excel 文件，或勾选使用内置示例数据集
2. 在左侧输入 Anthropic API Key 启用 AI 智能解析（可选）
3. 切换到「数据概览」标签页查看数据结构
4. 点击「执行基础清洗」处理缺失值和异常值
5. 切换到「智能可视化」，输入自然语言指令生成图表
6. 生成图表后可输入追加指令进行迭代优化

## 支持的图表类型

- 柱状图 (Bar)
- 折线图 (Line)
- 散点图 (Scatter)
- 饼图 (Pie)
- 热力图 (Heatmap)
- 直方图 (Histogram)
- 箱线图 (Box)

## 技术栈

- [Streamlit](https://streamlit.io/) — Web 应用框架
- [Plotly](https://plotly.com/python/) — 交互式图表
- [Pandas](https://pandas.pydata.org/) — 数据处理
- [Anthropic Claude API](https://www.anthropic.com/) — 自然语言解析

## License

MIT
