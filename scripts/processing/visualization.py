"""
第三步：核心可视化图表生成
功能：生成5张核心图表 + 1个交互式评估器
运行时间：约5分钟
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    df = pd.read_csv('data/reference/major_data_with_ai_index.csv')
    return df


def plot1_bubble_chart(df):
    """
    图表1：四维气泡图
    X轴：本科月薪，Y轴：抗AI指数
    气泡大小：学历薪资溢价率，颜色：学科门类
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 颜色映射
    categories = df['学科门类'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    for category in categories:
        subset = df[df['学科门类'] == category]
        ax.scatter(
            subset['本科月薪'],
            subset['本科抗AI指数'],
            s=subset['学历薪资溢价率%'] * 10,  # 气泡大小
            c=[color_map[category]],
            alpha=0.6,
            edgecolors='white',
            linewidth=1.5,
            label=category
        )
    
    # 标注重点专业
    highlight_majors = [
        '人工智能', '临床医学', '会计学', '法学', '音乐表演'
    ]
    for _, row in df[df['专业'].isin(highlight_majors)].iterrows():
        ax.annotate(
            row['专业'],
            (row['本科月薪'], row['本科抗AI指数']),
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )
    
    ax.set_xlabel('本科月薪 (元)', fontsize=14, fontweight='bold')
    ax.set_ylabel('抗AI指数 (0-1)', fontsize=14, fontweight='bold')
    ax.set_title('专业就业质量四维全景图\n(气泡大小=学历溢价率)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(title='学科门类', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/plot1_bubble_chart.png', dpi=300, bbox_inches='tight')
    print("✅ 图表1已生成: visualizations/plot1_bubble_chart.png")
    plt.close()


def plot2_heatmap(df):
    """
    图表2：学科门类热力图
    行：学科门类，列：关键指标
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算学科平均值
    metrics = ['本科就业率', '本科月薪', '本科抗AI指数', '学历薪资溢价率%']
    heatmap_data = df.groupby('学科门类')[metrics].mean()
    
    # 标准化数据（0-1）便于比较
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data),
        index=heatmap_data.index,
        columns=heatmap_data.columns
    )
    
    # 重命名列为中文
    heatmap_normalized.columns = ['就业率', '月薪', '抗AI指数', '学历溢价率']
    
    sns.heatmap(
        heatmap_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        linewidths=1,
        cbar_kws={'label': '标准化得分 (0-1)'},
        ax=ax
    )
    
    ax.set_title('各学科门类综合表现热力图\n(数值越高越优)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('学科门类', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/plot2_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 图表2已生成: visualizations/plot2_heatmap.png")
    plt.close()


def plot3_degree_premium(df):
    """
    图表3：学历溢价率对比（本科 vs 硕士）
    双轴柱状图
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 选取TOP15专业
    top15 = df.nlargest(15, '学历薪资溢价率%').sort_values('学历薪资溢价率%')
    
    x = np.arange(len(top15))
    width = 0.35
    
    # 左轴：薪资对比
    bars1 = ax1.bar(x - width/2, top15['本科月薪'], width, 
                    label='本科月薪', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, top15['硕士月薪'], width,
                    label='硕士月薪', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('月薪 (元)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('专业', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top15['专业'], rotation=45, ha='right')
    ax1.legend(loc='upper left')
    
    # 右轴：溢价率
    ax2 = ax1.twinx()
    line = ax2.plot(x, top15['学历薪资溢价率%'], 'o-', 
                    color='#2ecc71', linewidth=2, markersize=8,
                    label='学历溢价率')
    ax2.set_ylabel('学历薪资溢价率 (%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    # 标注最高值
    max_idx = top15['学历薪资溢价率%'].idxmax()
    max_row = top15.loc[max_idx]
    ax2.annotate(
        f"{max_row['学历薪资溢价率%']:.1f}%",
        xy=(list(top15.index).index(max_idx), max_row['学历薪资溢价率%']),
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
    )
    
    plt.title('学历最值钱的15个专业\n(本硕薪资对比 + 溢价率)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('visualizations/plot3_degree_premium.png', dpi=300, bbox_inches='tight')
    print("✅ 图表3已生成: visualizations/plot3_degree_premium.png")
    plt.close()


def plot4_risk_distribution(df):
    """
    图表4：AI风险等级分布（本科 vs 硕士）
    堆叠柱状图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 本科风险分布
    bachelor_risk = df['本科AI风险等级'].value_counts().reindex(['低风险', '中风险', '高风险'])
    colors_risk = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax1.pie(bachelor_risk, labels=bachelor_risk.index, autopct='%1.1f%%',
            colors=colors_risk, startangle=90, textprops={'fontsize': 12})
    ax1.set_title('本科生AI替代风险分布', fontsize=14, fontweight='bold')
    
    # 硕士风险分布
    master_risk = df['硕士AI风险等级'].value_counts().reindex(['低风险', '中风险', '高风险'])
    
    ax2.pie(master_risk, labels=master_risk.index, autopct='%1.1f%%',
            colors=colors_risk, startangle=90, textprops={'fontsize': 12})
    ax2.set_title('硕士生AI替代风险分布', fontsize=14, fontweight='bold')
    
    plt.suptitle('学历对AI风险的保护效应', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('visualizations/plot4_risk_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 图表4已生成: visualizations/plot4_risk_distribution.png")
    plt.close()


def plot5_redgreen_badge(df):
    """
    图表5：红绿牌专业真相对比
    雷达图
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 筛选红绿牌数据
    badge_data = df[df['红绿牌'] != '普通'].groupby('红绿牌').agg({
        '本科就业率': 'mean',
        '本科月薪': lambda x: x.mean() / 10000,  # 转为万元
        '本科抗AI指数': 'mean',
        '学历薪资溢价率%': lambda x: x.mean() / 100,  # 转为0-1
    })
    
    # 设置雷达图
    categories = ['就业率', '月薪(万)', '抗AI指数', '学历溢价']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 绘制绿牌
    if '绿牌' in badge_data.index:
        values_green = badge_data.loc['绿牌'].values.tolist()
        values_green += values_green[:1]
        ax.plot(angles, values_green, 'o-', linewidth=2, label='绿牌专业', color='#2ecc71')
        ax.fill(angles, values_green, alpha=0.25, color='#2ecc71')
    
    # 绘制红牌
    if '红牌' in badge_data.index:
        values_red = badge_data.loc['红牌'].values.tolist()
        values_red += values_red[:1]
        ax.plot(angles, values_red, 'o-', linewidth=2, label='红牌专业', color='#e74c3c')
        ax.fill(angles, values_red, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('红绿牌专业综合对比\n(标准化指标)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/plot5_redgreen_badge.png', dpi=300, bbox_inches='tight')
    print("✅ 图表5已生成: visualizations/plot5_redgreen_badge.png")
    plt.close()


def generate_interactive_tool(df):
    """
    生成交互式专业评估器（HTML）
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>专业风险评估器 | 数据新闻</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Microsoft YaHei', sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 32px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .search-box {{
            margin: 30px 0;
            text-align: center;
        }}
        #searchInput {{
            width: 60%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #3498db;
            border-radius: 10px;
            outline: none;
        }}
        #searchInput:focus {{
            border-color: #2980b9;
            box-shadow: 0 0 10px rgba(52, 152, 219, 0.3);
        }}
        .result-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            display: none;
        }}
        .metric {{
            display: inline-block;
            width: 23%;
            text-align: center;
            padding: 20px;
            margin: 5px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin: 5px;
        }}
        .risk-low {{ background: #2ecc71; color: white; }}
        .risk-medium {{ background: #f39c12; color: white; }}
        .risk-high {{ background: #e74c3c; color: white; }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin-top: 20px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 专业风险智能评估器</h1>
        <p class="subtitle">基于88个专业的真实就业数据 + AI替代风险模型</p>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="输入专业名称，如：计算机科学与技术" 
                   list="majorList">
            <datalist id="majorList">
                {"".join([f'<option value="{major}">' for major in df['专业'].values])}
            </datalist>
        </div>
        
        <div id="resultCard" class="result-card">
            <h2 id="majorName" style="color: #2c3e50; margin-bottom: 20px;"></h2>
            
            <div id="metrics"></div>
            
            <div style="margin-top: 30px;">
                <h3 style="color: #2c3e50;">🔍 风险评级</h3>
                <div id="riskBadges" style="margin: 15px 0;"></div>
            </div>
            
            <div class="recommendation">
                <h3 style="color: #856404; margin-bottom: 10px;">💡 智能建议</h3>
                <p id="recommendation"></p>
            </div>
        </div>
    </div>
    
    <script>
        const data = {df.to_json(orient='records', force_ascii=False)};
        
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            const query = e.target.value;
            const major = data.find(m => m.专业 === query);
            
            if (major) {{
                showResult(major);
            }}
        }});
        
        function showResult(major) {{
            document.getElementById('resultCard').style.display = 'block';
            document.getElementById('majorName').textContent = major.专业 + ' (' + major.学科门类 + ')';
            
            // 显示指标
            document.getElementById('metrics').innerHTML = `
                <div class="metric">
                    <div class="metric-label">本科就业率</div>
                    <div class="metric-value">${{(major.本科就业率 * 100).toFixed(1)}}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">本科月薪</div>
                    <div class="metric-value">¥${{major.本科月薪}}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">抗AI指数</div>
                    <div class="metric-value">${{major.本科抗AI指数}}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">学历溢价</div>
                    <div class="metric-value">${{major['学历薪资溢价率%']}}%</div>
                </div>
            `;
            
            // 显示风险标签
            const riskClass = major.本科AI风险等级 === '低风险' ? 'risk-low' : 
                            (major.本科AI风险等级 === '中风险' ? 'risk-medium' : 'risk-high');
            document.getElementById('riskBadges').innerHTML = `
                <span class="risk-badge ${{riskClass}}">本科：${{major.本科AI风险等级}}</span>
                <span class="risk-badge risk-low">硕士：${{major.硕士AI风险等级}}</span>
            `;
            
            // 生成建议
            let advice = '';
            if (major.本科抗AI指数 < 0.5) {{
                advice = '⚠️ 该专业AI替代风险较高，建议：1）发展复合技能 2）考虑读研深造 3）关注新兴交叉领域';
            }} else if (major['学历薪资溢价率%'] > 40) {{
                advice = '✅ 该专业学历价值高，强烈建议考研，硕士薪资比本科高' + major['学历薪资溢价率%'] + '%';
            }} else {{
                advice = '👍 该专业综合表现良好，本科就业即可，也可根据个人职业规划选择深造';
            }}
            document.getElementById('recommendation').textContent = advice;
        }}
    </script>
</body>
</html>
"""
    
    with open('outputs/interactive_tool.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ 交互工具已生成: outputs/interactive_tool.html")


def main():
    """主函数"""
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    print("="*70)
    print("📊 第三步：生成核心可视化")
    print("="*70)
    
    # 加载数据
    print("\n⏳ 正在加载数据...")
    df = load_data()
    
    # 生成5张核心图表
    print("\n⏳ 正在生成图表...")
    plot1_bubble_chart(df)
    plot2_heatmap(df)
    plot3_degree_premium(df)
    plot4_risk_distribution(df)
    plot5_redgreen_badge(df)
    
    # 生成交互工具
    print("\n⏳ 正在生成交互式评估器...")
    generate_interactive_tool(df)
    
    print("\n" + "="*70)
    print("✅ 第三步完成！")
    print(f"📁 所有文件已保存到 visualizations/ 文件夹")
    print("👉 下一步：运行 step4_ml_models.py（BERT+LSTM深度分析）")
    print("="*70)


if __name__ == "__main__":
    main()
