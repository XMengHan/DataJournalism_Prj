# 📊 Zhang Xuefeng Major Recommendation Analysis System

# 张雪峰专业推荐综合分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
 [![BERT](https://img.shields.io/badge/BERT-Sentiment%20Analysis-green.svg)](https://huggingface.co/transformers)
 [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://vip.51kaixin.cloud/c/LICENSE)
 [![Status](https://img.shields.io/badge/Status-Active-success.svg)](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d)

> 🎓 **A comprehensive data journalism project analyzing the gap between social media sentiment and employment reality in Chinese university majors**
>
> 基于真实BERT模型的社交媒体舆情分析 vs 官方就业数据对比研究

------

## 📋 Table of Contents | 目录

- [🎯 Project Overview](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-project-overview)
- [✨ Key Features](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-key-features)
- [🏗️ Project Structure](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#️-project-structure)
- [🔧 Installation](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-installation)
- [🚀 Quick Start](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-quick-start)
- [📊 Data Sources](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-data-sources)
- [🤖 Models & Algorithms](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-models--algorithms)
- [📈 Visualizations](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-visualizations)
- [🔮 LSTM Prediction Module](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-lstm-prediction-module)
- [📄 Output Files](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-output-files)
- [🛠️ Technical Stack](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#️-technical-stack)
- [📝 License](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-license)
- [🙏 Acknowledgments](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-acknowledgments)

------

## 🎯 Project Overview

This project investigates **whether popular career advice influencer Zhang Xuefeng's major recommendations align with actual employment data**. We combine:

- 🗣️ **Social Media Sentiment Analysis**: Using real BERT models to analyze comments from Zhihu, Weibo, and Bilibili
- 📊 **Official Employment Data**: Real employment rates, salaries, and career outcomes for 88+ majors
- 🎨 **Data Visualization**: Interactive bubble charts, heatmaps, and comprehensive reports
- 🔮 **LSTM Time Series Prediction**: Forecasting future trends in major popularity and employment

### Research Questions

1. Which majors have the highest positive sentiment on social media?
2. How does social media sentiment correlate with actual employment rates?
3. Which majors are **overrated** (high sentiment, low employment)?
4. Which majors are **underrated** (low sentiment, high employment)?
5. What is the ROI of pursuing a master's degree by major?

------

## ✨ Key Features

### 🤖 Real BERT Sentiment Analysis

- **Model**: `uer/roberta-base-finetuned-jd-binary-chinese`
- **GPU Accelerated**: Supports CUDA for faster processing
- **Batch Processing**: Analyzes 10,000+ comments efficiently
- **Confidence Scoring**: Each sentiment prediction includes confidence level

### 📊 Comprehensive Data Integration

- **88+ Majors**: Complete coverage of undergraduate majors
- **Multi-platform Comments**: Zhihu, Weibo, Bilibili
- **Official Employment Data**: Bachelor & Master employment rates, salaries, industry distribution
- **AI Replacement Risk**: Automation probability by major

### 🎨 Rich Visualizations

- **Bubble Charts**: Sentiment vs Employment scatter plots
- **Heatmaps**: Major recommendation rankings
- **Bar Charts**: Salary comparisons, employment improvements
- **Interactive HTML**: Plotly-based dynamic charts

### 🔮 LSTM Prediction

- **Bidirectional LSTM**: Time series forecasting
- **Multi-feature**: Employment rate, sentiment index, discussion volume
- **12-month Forecast**: Predict future trends for top majors

------

## 🏗️ Project Structure

```
DataJournalism_MajorAnalysis/
├── 📁 data/
│   ├── raw/                          # 原始数据
│   │   ├── zhihu/csv/               # 知乎评论数据
│   │   ├── weibo/csv/               # 微博评论数据
│   │   └── bili/csv/                # B站评论数据
│   └── processed/                    # 处理后的数据
│       └── comprehensive_major_data.csv  # 88个专业综合数据
│
├── 📁 scripts/
│   └── analysis/
│       ├── BERT.py                  # 🔥 主分析脚本（BERT + 可视化）
│       └── lstm_prediction.py       # 🔮 LSTM时序预测模块
│
├── 📁 output/                        # 输出目录（自动生成）
│   ├── figures/                     # 📊 所有图表
│   │   ├── 01_platform_distribution.png
│   │   ├── 02_bert_sentiment_distribution.png
│   │   ├── 03_recommendation_ranking.png
│   │   ├── 04_sentiment_comparison.png
│   │   ├── 05_bubble_chart.png      # 🎯 核心气泡图
│   │   ├── 06_salary_comparison.png
│   │   └── 07_employment_improvement.png
│   │
│   ├── tables/                      # 📋 数据表
│   │   ├── 01_bert_analyzed_comments.csv
│   │   ├── 02_major_sentiment_summary.csv
│   │   └── 03_integrated_sentiment_employment.csv
│   │
│   └── lstm_predictions/            # 🔮 LSTM预测结果
│       ├── 计算机科学与技术_training.png
│       ├── 计算机科学与技术_forecast.png
│       └── all_majors_comparison.png
│
├── 📄 README.md                      # 本文件
├── 📄 requirements.txt               # Python依赖
└── 📄 LICENSE                        # MIT许可证
```

------

## 🔧 Installation

### Prerequisites | 环境要求

- Python 3.8+
- pip package manager
- (Optional) CUDA-enabled GPU for faster BERT inference

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/DataJournalism_MajorAnalysis.git
cd DataJournalism_MajorAnalysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt Content

```txt
# Core Data Science
pandas>=1.5.0
numpy>=1.23.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0

# Machine Learning - BERT
transformers>=4.25.0
torch>=2.0.0

# Machine Learning - LSTM
tensorflow>=2.10.0

# Utilities
scikit-learn>=1.2.0
```

------

## 🚀 Quick Start

### 1️⃣ Run Main Analysis (BERT + Visualizations)

```bash
python scripts/analysis/BERT.py
```

**What it does:**

1. ✅ Loads comments from Zhihu, Weibo, Bilibili
2. ✅ Runs BERT sentiment analysis on 10,000+ comments
3. ✅ Extracts major mentions and aggregates sentiment
4. ✅ Integrates with official employment data
5. ✅ Generates 7 comprehensive visualizations
6. ✅ Exports CSV tables for further analysis

**Expected Output:**

```
======================================================================
张雪峰专业推荐 - 舆情分析 vs 就业现实 综合分析系统
======================================================================

📥 Loading Real Comment Data...
  ✅ Zhihu: 5,432 comments
  ✅ Weibo: 3,891 comments
  ✅ Bilibili: 2,103 comments

🤖 Loading BERT Model...
  ✅ BERT Model Loaded (Device: GPU)

🔄 Analyzing 11,426 comments with BERT...
  Progress: 11,426/11,426 (100.0%)
  ✅ BERT analysis completed!

📊 BERT Sentiment Analysis Results:
positive    5234
neutral     4102
negative    2090

🔍 Extracting major mentions from comments...
  ✅ Extracted 8,932 major mentions

📊 Aggregating sentiment by major...
  ✅ Aggregated 45 majors

🔗 Integrating sentiment and employment data...
  ✅ Integrated 38 majors

======================================================================
📊 Creating Visualizations
======================================================================

✅ Figure 1: Platform Distribution
✅ Figure 2: BERT Sentiment Distribution
✅ Figure 3: Recommendation Ranking
✅ Figure 4: Sentiment Comparison

🎨 Creating Enhanced Visualizations
✅ Figure 5: Bubble Chart
✅ Figure 6: Salary Comparison
✅ Figure 7: Employment Improvement

✅ ANALYSIS COMPLETED SUCCESSFULLY!
======================================================================
```

### 2️⃣ Run LSTM Prediction (Optional)

```bash
python scripts/analysis/lstm_prediction.py
```

**What it does:**

1. ✅ Loads integrated data from main analysis
2. ✅ Builds Bidirectional LSTM model
3. ✅ Trains on historical trends (36 months)
4. ✅ Forecasts next 12 months for top 5 majors
5. ✅ Generates prediction visualizations

------

## 📊 Data Sources

### 1. Social Media Comments (社交媒体评论)

| Platform       | Comments    | Date Range | Format |
| -------------- | ----------- | ---------- | ------ |
| 知乎 (Zhihu)   | ~5,400      | 2024-2025  | CSV    |
| 微博 (Weibo)   | ~3,900      | 2024-2025  | CSV    |
| B站 (Bilibili) | ~2,100      | 2024-2025  | CSV    |
| **Total**      | **~11,400** | -          | -      |

**Data Fields:**

- `content/text`: Comment text
- `like_count/likes`: Number of likes
- `platform`: Source platform
- `created_time`: Timestamp

### 2. Employment Data (就业数据)

**File**: `data/processed/comprehensive_major_data.csv`

**88 Majors Coverage**:

- 计算机科学与技术, 软件工程, 人工智能
- 临床医学, 金融学, 会计学
- 电子信息工程, 自动化, 机械工程
- ... (complete list in data file)

**Data Fields:**

```
专业, 本科就业率, 本科月薪, 学科门类, 红绿牌,
硕士就业率, 硕士月薪, 学历薪资溢价率%, 学历就业率提升%, 本硕薪资差
```

**Example Row:**

```csv
计算机科学与技术,0.69,6500,工科,普通,0.74,8775.0,35.0,5.0,2275.0
```

------

## 🤖 Models & Algorithms

### 1. BERT Sentiment Analysis

**Model**: `uer/roberta-base-finetuned-jd-binary-chinese`

- Pre-trained on Chinese e-commerce reviews
- Binary classification: Positive/Negative
- Threshold-based neutral detection

**Sentiment Classification Logic:**

```python
if positive_prob > 0.6:
    sentiment = 'positive'
elif negative_prob > 0.6:
    sentiment = 'negative'
else:
    sentiment = 'neutral'
```

### 2. Recommendation Score Algorithm

```python
match_score = (
    positive_rate * confidence * 
    (comment_count / max_comments) * 100
)
```

Where:

- `positive_rate`: % of positive comments
- `confidence`: Average BERT confidence
- `comment_count`: Discussion volume

### 3. Deviation Classification

```python
if sentiment_index > 20 and employment_rate < 85:
    category = 'Overrated'  # 被高估
elif sentiment_index < 0 and employment_rate > 88:
    category = 'Underrated'  # 被低估
else:
    category = 'Matched'  # 匹配
```

### 4. LSTM Architecture

```python
Model: Sequential
├── Bidirectional LSTM (64 units, return_sequences=True)
├── Dropout (0.3)
├── LSTM (32 units)
├── Dropout (0.2)
├── Dense (16 units, ReLU)
├── Dropout (0.2)
└── Dense (n_features, output)

Optimizer: Adam
Loss: Huber (robust to outliers)
Callbacks: EarlyStopping, ReduceLROnPlateau
```

------

## 📈 Visualizations

### Figure 1: Platform Distribution
<img src=".\visualizations\charts\01_platform_distribution.png" style="zoom:200%;" />

- Shows comment distribution across Zhihu, Weibo, Bilibili

### Figure 2: BERT Sentiment Distribution
<img src=".\visualizations\charts\02_bert_sentiment_distribution.png" style="zoom:200%;" />

- Pie chart of overall sentiment
- Bar chart of confidence scores

### Figure 3: Top 15 Majors Recommendation Ranking
<img src=".\visualizations\charts\03_recommendation_ranking.png" style="zoom:200%;" />

- Horizontal bar chart sorted by recommendation score

### Figure 4: Sentiment Comparison
<img src=".\visualizations\charts\04_sentiment_comparison.png" style="zoom:200%;" />

- Positive vs Negative rates for top majors

### 🎯 Figure 5: Bubble Chart (Core Visualization)

<img src=".\visualizations\plot1_bubble_chart.png" style="zoom:200%;" />

**Features:**

- **X-axis**: Official employment rate (%)
- **Y-axis**: Social media sentiment index
- **Bubble size**: Discussion volume
- **Color coding**:
  - 🟢 Green: Matched (sentiment = reality)
  - 🔴 Red: Overrated (high sentiment, low employment)
  - 🔵 Blue: Underrated (low sentiment, high employment)
- **Annotations**: Each major labeled
- **Quadrants**: Clearly marked zones

### Figure 6: Salary Comparison
<img src=".\visualizations\charts\06_salary_comparison.png" style="zoom:200%;" />

- Bachelor vs Master salary comparison
- Education premium rate ranking

### Figure 7: Employment Improvement
<img src=".\visualizations\charts\07_employment_improvement.png" style="zoom:200%;" />


- Employment rate boost from pursuing master's degree

------

## 🔮 LSTM Prediction Module

### Features

- **Lookback Window**: 12 months
- **Forecast Horizon**: 12 months ahead
- **Multi-variate**: Employment rate + Sentiment + Discussion volume
- **Model Evaluation**: MAE, RMSE, R² metrics

### Usage Example

```python
from lstm_prediction import run_lstm_prediction

# Load your integrated data
df_integrated = pd.read_csv('./output/tables/03_integrated_sentiment_employment.csv')

# Run prediction for top majors
results = run_lstm_prediction(
    df_integrated,
    target_feature='本科就业率',
    lookback=12
)
```

### Output Files

```
output/lstm_predictions/
├── 计算机科学与技术_training.png      # Training loss/MAE curves
├── 计算机科学与技术_forecast.png       # 12-month forecast
├── 计算机科学与技术_evaluation.png     # Model performance
└── all_majors_comparison.png           # Multi-major comparison
```

------

## 📄 Output Files

### 📊 Tables (CSV)

| File                                     | Description                      | Rows    | Columns                                                      |
| ---------------------------------------- | -------------------------------- | ------- | ------------------------------------------------------------ |
| `01_bert_analyzed_comments.csv`          | All comments with BERT sentiment | ~11,400 | text, sentiment, confidence, platform, likes                 |
| `02_major_sentiment_summary.csv`         | Aggregated sentiment by major    | ~45     | major, positive_rate, negative_rate, sentiment_index, comment_count, recommendation_score |
| `03_integrated_sentiment_employment.csv` | Combined sentiment + employment  | ~38     | major, sentiment_index, 本科就业率, 本科月薪, deviation_type, etc. |

### 📈 Figures (PNG)

All figures saved at **300 DPI** for publication quality.

| File                                 | Size         | Type                 |
| ------------------------------------ | ------------ | -------------------- |
| `01_platform_distribution.png`       | 10×6 inches  | Bar chart            |
| `02_bert_sentiment_distribution.png` | 14×6 inches  | Pie + Bar            |
| `03_recommendation_ranking.png`      | 12×8 inches  | Horizontal bar       |
| `04_sentiment_comparison.png`        | 14×8 inches  | Grouped bar          |
| `05_bubble_chart.png`                | 18×12 inches | Scatter (bubble)     |
| `06_salary_comparison.png`           | 16×7 inches  | Bar + Horizontal bar |
| `07_employment_improvement.png`      | 12×8 inches  | Horizontal bar       |

------

## 🛠️ Technical Stack

### Languages & Frameworks

- **Python 3.8+**: Core language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Machine Learning

- **PyTorch**: BERT model backend
- **Transformers (Hugging Face)**: BERT implementation
- **TensorFlow/Keras**: LSTM models
- **scikit-learn**: Evaluation metrics

### Visualization

- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Data Processing

- **Regular Expressions**: Major extraction
- **Unicode Normalization**: Text cleaning

------

## 🎯 Key Findings (Sample)

### Top 5 Recommended Majors (by Sentiment)

1. 🥇 **计算机科学与技术** (Recommendation Score: 87.3)
2. 🥈 **人工智能** (Score: 84.1)
3. 🥉 **软件工程** (Score: 81.5)
4. **数据科学与大数据技术** (Score: 78.9)
5. **电子信息工程** (Score: 76.2)

### Overrated Majors (High Sentiment, Low Employment)

- ⚠️ **生物工程**: Sentiment Index +25, Employment Rate 62%
- ⚠️ **环境工程**: Sentiment Index +18, Employment Rate 68%

### Underrated Majors (Low Sentiment, High Employment)

- 💎 **土木工程**: Sentiment Index -12, Employment Rate 91%
- 💎 **护理学**: Sentiment Index -8, Employment Rate 95%

------

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](https://vip.51kaixin.cloud/c/LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

------

## 🙏 Acknowledgments

### Data Sources

- **Zhihu, Weibo, Bilibili**: Social media comment data
- **教育部**: Official employment statistics
- **麦可思研究院**: Major employment quality reports

### Models

- **Hugging Face**: BERT model hosting
- **UER Team**: Chinese RoBERTa pre-training

### Inspiration

- **Zhang Xuefeng (张雪峰)**: Career counselor whose recommendations sparked this research

### Tools & Libraries

- PyTorch, TensorFlow, Transformers, Plotly, Matplotlib

------

## 📧 Contact & Contributing

### Author

- **Name**: [Your Name]
- **Email**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [@yourusername](https://github.com/yourusername)

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

------

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/DataJournalism_MajorAnalysis&type=Date)](https://star-history.com/#yourusername/DataJournalism_MajorAnalysis&Date)

------

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{zhang_major_analysis_2025,
  author = {MengXiaohan@whu.edu.cn},
  title = {Zhang Xuefeng Major Recommendation Analysis: Sentiment vs Reality},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/DataJournalism_MajorAnalysis}
}
```

------

**Made with ❤️ for Data Journalism**

[⬆ Back to Top](https://vip.51kaixin.cloud/c/7a7d811c-8b3c-4cd8-b327-19a74259d27d#-zhang-xuefeng-major-recommendation-analysis-system)
