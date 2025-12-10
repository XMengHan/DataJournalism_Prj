"""
舆情推荐 vs 现实就业率 - 多维度对比可视化
独立脚本：直接读取 output/tables 中的数据绘图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 字体配置
import platform
import os

system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']

plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./output/figures', exist_ok=True)

print("="*70)
print("📊 Sentiment vs Employment - Multi-Dimensional Visualization")
print("="*70 + "\n")

# ==================== 数据加载 ====================

def load_data():
    """加载已有的分析数据"""
    
    print("📥 Loading data from output/tables...")
    
    # 加载整合数据
    try:
        df_integrated = pd.read_csv('./output/tables/04_integrated_sentiment_employment.csv')
        print(f"  ✅ Integrated data: {len(df_integrated)} majors")
    except Exception as e:
        print(f"  ❌ Failed to load integrated data: {e}")
        df_integrated = None
    
    # 加载专业舆情汇总
    try:
        df_sentiment = pd.read_csv('./output/tables/02_major_sentiment_summary.csv')
        print(f"  ✅ Sentiment summary: {len(df_sentiment)} majors")
    except Exception as e:
        print(f"  ❌ Failed to load sentiment data: {e}")
        df_sentiment = None
    
    print()
    return df_integrated, df_sentiment


# ==================== 图表生成 ====================

def create_all_charts(df_integrated, df_sentiment, top_n=20):
    """生成所有舆情 vs 就业对比图表"""
    
    if df_integrated is None or len(df_integrated) == 0:
        print("⚠️ No integrated data available")
        return
    
    # 取 Top N 专业（按讨论量排序），但不超过实际数据量
    actual_top_n = min(top_n, len(df_integrated))
    df_top = df_integrated.nlargest(actual_top_n, 'mention_count').copy()
    
    # 确保必要字段存在
    if 'heat_index' not in df_top.columns:
        max_mentions = df_top['mention_count'].max()
        df_top['heat_index'] = (df_top['mention_count'] / max_mentions) * 100
    
    if 'sentiment_ratio' not in df_top.columns:
        df_top['sentiment_ratio'] = df_top.apply(
            lambda x: x['positive_rate'] / max(x['negative_rate'], 1), axis=1
        )
    
    print(f"📊 Creating charts for Top {actual_top_n} majors...\n")
    
    # ========== 图1: 综合对比仪表盘（4合1） ==========
    print("Creating Figure 1: Comprehensive Dashboard...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # --- 1a: 散点图 - 舆情指数 vs 就业率 ---
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        df_top['本科就业率'],
        df_top['sentiment_index'],
        s=df_top['heat_index'] * 10,
        c=df_top['本科月薪'],
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    for idx, row in df_top.iterrows():
        ax1.annotate(
            row['major'],
            (row['本科就业率'], row['sentiment_index']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Monthly Salary (CNY)', fontsize=10)
    
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=df_top['本科就业率'].median(), color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Employment Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sentiment Index (Pos% - Neg%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Sentiment vs Employment Rate\n(Bubble=Heat, Color=Salary)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # --- 1b: 双Y轴柱状图 ---
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    x = np.arange(len(df_top))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, df_top['recommendation_score'], width, 
                    label='Recommendation Score', color='#3498db', alpha=0.8)
    ax2.set_ylabel('Recommendation Score', fontsize=12, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    
    bars2 = ax2_twin.bar(x + width/2, df_top['本科就业率'], width,
                         label='Employment Rate', color='#e74c3c', alpha=0.8)
    ax2_twin.set_ylabel('Employment Rate (%)', fontsize=12, color='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_top['major'], rotation=45, ha='right', fontsize=8)
    ax2.set_title('B. Recommendation Score vs Employment Rate', 
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # --- 1c: 热力图 ---
    ax3 = axes[1, 0]
    
    heatmap_cols = ['sentiment_index', '本科就业率', '本科月薪', 'mention_count', 
                    'positive_rate', 'negative_rate', 'sentiment_ratio']
    heatmap_cols = [col for col in heatmap_cols if col in df_top.columns]
    
    corr_matrix = df_top[heatmap_cols].corr()
    
    rename_map = {
        'sentiment_index': 'Sentiment Index',
        '本科就业率': 'Employment Rate',
        '本科月薪': 'Salary',
        'mention_count': 'Discussion Volume',
        'positive_rate': 'Positive Rate',
        'negative_rate': 'Negative Rate',
        'sentiment_ratio': 'Pos/Neg Ratio'
    }
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                ax=ax3, fmt='.2f', linewidths=0.5,
                annot_kws={'fontsize': 9})
    ax3.set_title('C. Correlation Heatmap', fontsize=13, fontweight='bold')
    
    # --- 1d: 堆叠柱状图 ---
    ax4 = axes[1, 1]
    
    df_sorted = df_top.sort_values('positive_rate', ascending=True)
    y_pos = np.arange(len(df_sorted))
    
    ax4.barh(y_pos, df_sorted['positive_rate'], color='#2ecc71', 
             label='Positive', alpha=0.8)
    ax4.barh(y_pos, df_sorted['neutral_rate'], left=df_sorted['positive_rate'],
             color='#95a5a6', label='Neutral', alpha=0.8)
    ax4.barh(y_pos, df_sorted['negative_rate'], 
             left=df_sorted['positive_rate'] + df_sorted['neutral_rate'],
             color='#e74c3c', label='Negative', alpha=0.8)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(df_sorted['major'], fontsize=9)
    ax4.set_xlabel('Percentage (%)', fontsize=12)
    ax4.set_title('D. Sentiment Distribution by Major', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.set_xlim(0, 100)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Sentiment vs Employment: Multi-Dimensional Analysis Dashboard (Top {actual_top_n})',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./output/figures/sentiment_employment_dashboard.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✅ Figure 1: Dashboard saved")
    
    # ========== 图2: 雷达图（交互式） ==========
    print("Creating Figure 2: Radar Chart...")
    
    # 限制雷达图显示的专业数量为10个
    radar_n = min(10, len(df_top))
    df_radar = df_top.head(radar_n).copy()
    
    metrics = ['sentiment_index', '本科就业率', '本科月薪', 'heat_index', 'sentiment_ratio']
    metrics = [m for m in metrics if m in df_radar.columns]
    
    df_normalized = df_radar.copy()
    for col in metrics:
        min_val = df_radar[col].min()
        max_val = df_radar[col].max()
        if max_val > min_val:
            df_normalized[col + '_norm'] = (df_radar[col] - min_val) / (max_val - min_val) * 100
        else:
            df_normalized[col + '_norm'] = 50
    
    categories = ['Sentiment', 'Employment', 'Salary', 'Heat', 'Pos/Neg Ratio']
    categories = categories[:len(metrics)]
    
    fig = go.Figure()
    
    # 使用足够多的颜色
    color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
    
    for i, (idx, row) in enumerate(df_normalized.iterrows()):
        values = [row.get(m + '_norm', 50) for m in metrics]
        values.append(values[0])
        
        color = color_palette[i % len(color_palette)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row['major'],
            line=dict(color=color, width=2),
            fill='toself',
            fillcolor=color,
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=dict(
            text=f'<b>Top {radar_n} Majors: Multi-Dimensional Radar Chart</b>',
            font=dict(size=16)
        ),
        width=1000,
        height=800,
        showlegend=True
    )
    
    fig.write_html('./output/figures/radar_chart.html')
    print("  ✅ Figure 2: Radar Chart saved")
    
    # ========== 图3: 偏差分析图 ==========
    print("Creating Figure 3: Deviation Analysis...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if 'sentiment_rank' not in df_top.columns:
        df_top['sentiment_rank'] = df_top['sentiment_index'].rank(ascending=False)
    if 'employment_rank' not in df_top.columns:
        df_top['employment_rank'] = df_top['本科就业率'].rank(ascending=False)
    
    df_top['rank_diff'] = df_top['sentiment_rank'] - df_top['employment_rank']
    df_plot = df_top.sort_values('rank_diff')
    
    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in df_plot['rank_diff']]
    
    y_pos = np.arange(len(df_plot))
    bars = ax.barh(y_pos, df_plot['rank_diff'], color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['major'], fontsize=10)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Rank Difference (Sentiment Rank - Employment Rank)', fontsize=12, fontweight='bold')
    ax.set_title('Deviation Analysis: Sentiment Rank vs Employment Rank\n' +
                '(Positive = Overrated, Negative = Underrated)',
                fontsize=14, fontweight='bold')
    
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        value = row['rank_diff']
        label_x = value + (0.5 if value >= 0 else -0.5)
        ax.text(label_x, i, f'{value:+.1f}', va='center', fontsize=9, fontweight='bold')
    
    ax.text(0.02, 0.98, '← Underrated',
           transform=ax.transAxes, fontsize=10, color='#2ecc71', 
           verticalalignment='top', fontweight='bold')
    ax.text(0.98, 0.98, 'Overrated →',
           transform=ax.transAxes, fontsize=10, color='#e74c3c',
           verticalalignment='top', horizontalalignment='right', fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output/figures/deviation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Figure 3: Deviation Analysis saved")
    
    # ========== 图4: 气泡矩阵图（交互式） ==========
    print("Creating Figure 4: Bubble Matrix...")
    
    fig = px.scatter(
        df_top,
        x='本科就业率',
        y='本科月薪',
        size='heat_index',
        color='sentiment_index',
        text='major',
        color_continuous_scale='RdYlGn',
        size_max=60,
        title='<b>Salary vs Employment Rate Matrix</b><br>' +
              '<sub>Bubble Size = Discussion Heat | Color = Sentiment Index</sub>',
        labels={
            '本科就业率': 'Employment Rate (%)',
            '本科月薪': 'Monthly Salary (CNY)',
            'sentiment_index': 'Sentiment',
            'heat_index': 'Heat'
        }
    )
    
    fig.update_traces(textposition='top center', textfont_size=9)
    fig.update_layout(width=1200, height=800)
    
    fig.add_hline(y=df_top['本科月薪'].median(), line_dash='dash', 
                  line_color='gray', opacity=0.5)
    fig.add_vline(x=df_top['本科就业率'].median(), line_dash='dash',
                  line_color='gray', opacity=0.5)
    
    fig.write_html('./output/figures/bubble_matrix.html')
    print("  ✅ Figure 4: Bubble Matrix saved")
    
    # ========== 图5: 双轴折线图 ==========
    print("Creating Figure 5: Dual-Axis Trend Chart...")
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    df_trend = df_top.sort_values('本科就业率', ascending=False)
    x = np.arange(len(df_trend))
    
    color1 = '#2980b9'
    ax1.plot(x, df_trend['本科就业率'], 'o-', color=color1, linewidth=2.5, 
             markersize=10, label='Employment Rate')
    ax1.fill_between(x, df_trend['本科就业率'], alpha=0.2, color=color1)
    ax1.set_ylabel('Employment Rate (%)', fontsize=12, color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([df_trend['本科就业率'].min() - 5, df_trend['本科就业率'].max() + 5])
    
    ax2 = ax1.twinx()
    color2 = '#c0392b'
    ax2.plot(x, df_trend['sentiment_index'], 's--', color=color2, linewidth=2.5,
             markersize=10, label='Sentiment Index')
    ax2.fill_between(x, df_trend['sentiment_index'], alpha=0.2, color=color2)
    ax2.set_ylabel('Sentiment Index', fontsize=12, color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    for i, (idx, row) in enumerate(df_trend.iterrows()):
        ax1.annotate(f"¥{int(row['本科月薪'])}", 
                    (i, row['本科就业率']),
                    xytext=(0, 15), textcoords='offset points',
                    fontsize=8, ha='center', alpha=0.7)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_trend['major'], rotation=45, ha='right', fontsize=9)
    ax1.set_xlabel('Major (Sorted by Employment Rate)', fontsize=12, fontweight='bold')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    
    plt.title(f'Employment Rate vs Sentiment Index Trend (Top {actual_top_n}, with Salary)',
             fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./output/figures/trend_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Figure 5: Trend Chart saved")
    
    # ========== 图6: Sunburst 层级图 ==========
    print("Creating Figure 6: Sunburst Chart...")
    
    df_sunburst = df_top.copy()
    
    def categorize(row):
        if row['sentiment_index'] > 20 and row['本科就业率'] > 88:
            return 'High Sentiment + High Employment'
        elif row['sentiment_index'] > 20 and row['本科就业率'] <= 88:
            return 'High Sentiment + Low Employment'
        elif row['sentiment_index'] <= 20 and row['本科就业率'] > 88:
            return 'Low Sentiment + High Employment'
        else:
            return 'Low Sentiment + Low Employment'
    
    df_sunburst['category'] = df_sunburst.apply(categorize, axis=1)
    
    fig = px.sunburst(
        df_sunburst,
        path=['category', 'major'],
        values='mention_count',
        color='sentiment_index',
        color_continuous_scale='RdYlGn',
        title='<b>Major Classification: Sentiment × Employment</b>'
    )
    
    fig.update_layout(width=900, height=900)
    fig.write_html('./output/figures/sunburst_chart.html')
    print("  ✅ Figure 6: Sunburst Chart saved")
    
    # ========== 图7: 箱线图对比 ==========
    print("Creating Figure 7: Box Plot Comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if 'deviation_type' not in df_integrated.columns:
        def classify_deviation(row):
            if row['sentiment_index'] > 20 and row['本科就业率'] < 85:
                return 'Overrated'
            elif row['sentiment_index'] < 0 and row['本科就业率'] > 88:
                return 'Underrated'
            else:
                return 'Matched'
        df_integrated['deviation_type'] = df_integrated.apply(classify_deviation, axis=1)
    
    categories = ['Matched', 'Overrated', 'Underrated']
    colors_box = {'Matched': '#2ecc71', 'Overrated': '#e74c3c', 'Underrated': '#3498db'}
    
    # 筛选存在的类别
    valid_cats = [cat for cat in categories if cat in df_integrated['deviation_type'].values]
    
    if not valid_cats:
        print("  ⚠️ No valid categories for box plot")
    else:
        # 就业率分布
        data_emp = [df_integrated[df_integrated['deviation_type'] == cat]['本科就业率'].dropna() 
                    for cat in valid_cats]
        
        if data_emp and all(len(d) > 0 for d in data_emp):
            bp1 = axes[0].boxplot(data_emp, labels=valid_cats, patch_artist=True)
            for patch, cat in zip(bp1['boxes'], valid_cats):
                patch.set_facecolor(colors_box[cat])
                patch.set_alpha(0.7)
            axes[0].set_ylabel('Employment Rate (%)', fontsize=12)
            axes[0].set_title('A. Employment Rate Distribution', fontsize=12, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
        
        # 薪资分布
        data_salary = [df_integrated[df_integrated['deviation_type'] == cat]['本科月薪'].dropna()
                       for cat in valid_cats]
        if data_salary and all(len(d) > 0 for d in data_salary):
            bp2 = axes[1].boxplot(data_salary, labels=valid_cats, patch_artist=True)
            for patch, cat in zip(bp2['boxes'], valid_cats):
                patch.set_facecolor(colors_box[cat])
                patch.set_alpha(0.7)
            axes[1].set_ylabel('Monthly Salary (CNY)', fontsize=12)
            axes[1].set_title('B. Salary Distribution', fontsize=12, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
        
        # 情感指数分布
        data_sent = [df_integrated[df_integrated['deviation_type'] == cat]['sentiment_index'].dropna()
                     for cat in valid_cats]
        if data_sent and all(len(d) > 0 for d in data_sent):
            bp3 = axes[2].boxplot(data_sent, labels=valid_cats, patch_artist=True)
            for patch, cat in zip(bp3['boxes'], valid_cats):
                patch.set_facecolor(colors_box[cat])
                patch.set_alpha(0.7)
            axes[2].set_ylabel('Sentiment Index', fontsize=12)
            axes[2].set_title('C. Sentiment Index Distribution', fontsize=12, fontweight='bold')
            axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Distribution Comparison: Matched vs Overrated vs Underrated',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/figures/boxplot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Figure 7: Box Plot saved")
    
    print("\n" + "="*70)
    print("✅ All charts generated successfully!")
    print("="*70)
    print("\n📁 Output files:")
    print("  - sentiment_employment_dashboard.png (综合仪表盘)")
    print("  - radar_chart.html (雷达图-交互式)")
    print("  - deviation_analysis.png (偏差分析)")
    print("  - bubble_matrix.html (气泡矩阵-交互式)")
    print("  - trend_comparison.png (趋势对比)")
    print("  - sunburst_chart.html (层级图-交互式)")
    print("  - boxplot_comparison.png (箱线图对比)")
    print()


# ==================== 主函数 ====================

def main():
    """主执行流程"""
    
    # 加载数据
    df_integrated, df_sentiment = load_data()
    
    if df_integrated is None:
        print("❌ Cannot proceed without integrated data")
        return
    
    # 生成所有图表
    create_all_charts(df_integrated, df_sentiment, top_n=20)
    
    print("🎉 Done!")


if __name__ == "__main__":
    main()

