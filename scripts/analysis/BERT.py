"""
张雪峰专业推荐综合分析系统 - 完整版（含Contents分析）
真实BERT模型 + 真实数据 + 完整对比分析
新增：微博/知乎 contents + B站 videos 数据纳入分析
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

# BERT模型
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    BERT_AVAILABLE = True
    print("✅ BERT libraries loaded")
except ImportError:
    BERT_AVAILABLE = False
    print("❌ Please install: pip install transformers torch")
    exit()

# 字体配置
import platform
import os

system = platform.system()
if system == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    FONT_PATH = 'C:\\Windows\\Fonts\\msyh.ttc'
elif system == 'Darwin':
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti']
    FONT_PATH = '/System/Library/Fonts/STHeiti Light.ttc'
else:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./output/figures', exist_ok=True)
os.makedirs('./output/tables', exist_ok=True)

print("\n" + "="*70)
print("张雪峰专业推荐 - 舆情分析 vs 就业现实 综合分析系统")
print("="*70 + "\n")

# ==================== PART 1: 数据加载（含Contents） ====================

class RealDataLoader:
    """加载真实数据（评论 + 内容）"""
    
    def __init__(self):
        self.base_path = './data/raw/'
        self.comments = {}
        self.contents = {}
        
    def load_all_comments(self):
        """加载所有平台评论"""
        
        print("📥 Loading Real Comment Data...")
        
        # 知乎评论
        try:
            zhihu = pd.read_csv(f'{self.base_path}zhihu/search_comments_2025-12-09.csv')
            zhihu['platform'] = 'Zhihu'
            zhihu['data_type'] = 'comment'
            self.comments['zhihu'] = zhihu
            print(f"  ✅ Zhihu Comments: {len(zhihu):,} records")
        except Exception as e:
            print(f"  ⚠️ Zhihu comments failed: {e}")
        
        # 微博评论
        try:
            weibo = pd.read_csv(f'{self.base_path}weibo/search_comments_2025-12-09.csv')
            weibo['platform'] = 'Weibo'
            weibo['data_type'] = 'comment'
            self.comments['weibo'] = weibo
            print(f"  ✅ Weibo Comments: {len(weibo):,} records")
        except Exception as e:
            print(f"  ⚠️ Weibo comments failed: {e}")
        
        # B站评论
        try:
            bili = pd.read_csv(f'{self.base_path}bili/search_comments_2025-12-09.csv')
            bili['platform'] = 'Bilibili'
            bili['data_type'] = 'comment'
            self.comments['bili'] = bili
            print(f"  ✅ Bilibili Comments: {len(bili):,} records")
        except Exception as e:
            print(f"  ⚠️ Bilibili comments failed: {e}")
        
        return self.comments
    
    def load_all_contents(self):
        """加载所有平台的内容/帖子数据"""
        
        print("\n📥 Loading Content/Post Data...")
        
        # 知乎内容
        try:
            zhihu_content = pd.read_csv(f'{self.base_path}zhihu/search_contents_2025-12-09.csv')
            zhihu_content['platform'] = 'Zhihu'
            zhihu_content['data_type'] = 'content'
            self.contents['zhihu'] = zhihu_content
            print(f"  ✅ Zhihu Contents: {len(zhihu_content):,} records")
        except Exception as e:
            print(f"  ⚠️ Zhihu contents failed: {e}")
        
        # 微博内容
        try:
            weibo_content = pd.read_csv(f'{self.base_path}weibo/search_contents_2025-12-09.csv')
            weibo_content['platform'] = 'Weibo'
            weibo_content['data_type'] = 'content'
            self.contents['weibo'] = weibo_content
            print(f"  ✅ Weibo Contents: {len(weibo_content):,} records")
        except Exception as e:
            print(f"  ⚠️ Weibo contents failed: {e}")
        
        # B站视频
        try:
            bili_videos = pd.read_csv(f'{self.base_path}bili/search_videos_2025-12-09.csv')
            bili_videos['platform'] = 'Bilibili'
            bili_videos['data_type'] = 'video'
            self.contents['bili'] = bili_videos
            print(f"  ✅ Bilibili Videos: {len(bili_videos):,} records")
        except Exception as e:
            print(f"  ⚠️ Bilibili videos failed: {e}")
        
        return self.contents
    
    def standardize_comments(self):
        """标准化评论数据"""
        
        print("\n🔄 Standardizing comment data...")
        
        unified = []
        
        for platform, df in self.comments.items():
            df_copy = df.copy()
            
            # 统一文本字段
            text_fields = ['content', 'comment_text', 'text', 'comment_content']
            for field in text_fields:
                if field in df_copy.columns:
                    df_copy['text'] = df_copy[field].fillna('')
                    break
            
            if 'text' not in df_copy.columns:
                df_copy['text'] = ''
            
            # 统一点赞字段
            like_fields = ['like_count', 'likes', 'digg_count', 'attitudes_count', 'liked_count']
            for field in like_fields:
                if field in df_copy.columns:
                    df_copy['likes'] = pd.to_numeric(df_copy[field], errors='coerce').fillna(0)
                    break
            
            if 'likes' not in df_copy.columns:
                df_copy['likes'] = 0
            
            # 选择核心字段
            df_copy = df_copy[['text', 'likes', 'platform', 'data_type']].copy()
            unified.append(df_copy)
        
        df_all = pd.concat(unified, ignore_index=True)
        
        # 数据清洗
        df_all = df_all[
            (df_all['text'].notna()) &
            (df_all['text'].str.len() > 10) &
            (df_all['text'] != '')
        ].copy()
        
        df_all['text_length'] = df_all['text'].str.len()
        
        print(f"  ✅ Total valid comments: {len(df_all):,}")
        print(f"  Platform distribution:\n{df_all['platform'].value_counts()}\n")
        
        return df_all
    
    def standardize_contents(self):
        """标准化内容/帖子数据"""
        
        print("🔄 Standardizing content data...")
        
        unified = []
        
        for platform, df in self.contents.items():
            df_copy = df.copy()
            
            # 根据平台选择文本字段
            if platform == 'zhihu':
                # 知乎：合并 title + content_text + desc
                title = df_copy.get('title', pd.Series([''] * len(df_copy))).fillna('')
                content_text = df_copy.get('content_text', pd.Series([''] * len(df_copy))).fillna('')
                desc = df_copy.get('desc', pd.Series([''] * len(df_copy))).fillna('')
                df_copy['text'] = title.astype(str) + ' ' + content_text.astype(str) + ' ' + desc.astype(str)
                
                # 点赞数
                df_copy['likes'] = pd.to_numeric(df_copy.get('voteup_count', 0), errors='coerce').fillna(0)
                
            elif platform == 'weibo':
                # 微博：使用 content 字段
                df_copy['text'] = df_copy.get('content', pd.Series([''] * len(df_copy))).fillna('')
                df_copy['likes'] = pd.to_numeric(df_copy.get('liked_count', 0), errors='coerce').fillna(0)
                
            elif platform == 'bili':
                # B站视频：合并 title + desc
                title = df_copy.get('title', pd.Series([''] * len(df_copy))).fillna('')
                desc = df_copy.get('desc', pd.Series([''] * len(df_copy))).fillna('')
                df_copy['text'] = title.astype(str) + ' ' + desc.astype(str)
                df_copy['likes'] = pd.to_numeric(df_copy.get('liked_count', 0), errors='coerce').fillna(0)
            
            # 选择核心字段
            df_copy = df_copy[['text', 'likes', 'platform', 'data_type']].copy()
            unified.append(df_copy)
        
        if not unified:
            print("  ⚠️ No content data found")
            return pd.DataFrame()
        
        df_all = pd.concat(unified, ignore_index=True)
        
        # 数据清洗
        df_all = df_all[
            (df_all['text'].notna()) &
            (df_all['text'].str.len() > 10) &
            (df_all['text'] != '')
        ].copy()
        
        df_all['text_length'] = df_all['text'].str.len()
        
        print(f"  ✅ Total valid contents: {len(df_all):,}")
        print(f"  Platform distribution:\n{df_all['platform'].value_counts()}")
        print(f"  Data type distribution:\n{df_all['data_type'].value_counts()}\n")
        
        return df_all
    
    def merge_all_data(self, df_comments, df_contents):
        """合并评论和内容数据"""
        
        print("🔗 Merging comments and contents...")
        
        df_all = pd.concat([df_comments, df_contents], ignore_index=True)
        
        print(f"  ✅ Total merged records: {len(df_all):,}")
        print(f"  By data type:\n{df_all['data_type'].value_counts()}")
        print(f"  By platform:\n{df_all['platform'].value_counts()}\n")
        
        return df_all


def load_employment_data():
    """加载就业数据"""
    
    print("📥 Loading Employment Data...")
    
    try:
        df_emp = pd.read_csv('./data/processed/comprehensive_major_data.csv')
        print(f"  ✅ Loaded {len(df_emp)} majors' employment data")
        print(f"  Columns: {df_emp.columns.tolist()}\n")
        return df_emp
    except Exception as e:
        print(f"  ❌ Failed to load employment data: {e}")
        return None


# ==================== PART 2: 真实BERT情感分析 ====================

class RealBERTAnalyzer:
    """真实的BERT中文情感分析"""
    
    def __init__(self):
        print("🤖 Loading BERT Model...")
        
        try:
            # 使用经过微调的中文情感分析模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                "uer/roberta-base-finetuned-jd-binary-chinese"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "uer/roberta-base-finetuned-jd-binary-chinese"
            )
            
            # 检查GPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"  ✅ BERT Model Loaded (Device: {device_name})\n")
            
        except Exception as e:
            print(f"  ❌ BERT loading failed: {e}")
            self.model = None
    
    def predict_single(self, text):
        """预测单条文本"""
        
        if self.model is None:
            return 'neutral', 0.5
        
        try:
            # 截断并tokenize
            text = str(text)[:512]
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                   padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # 解析结果
            positive_prob = probs[0][1].item()
            negative_prob = probs[0][0].item()
            
            if positive_prob > 0.6:
                return 'positive', positive_prob
            elif negative_prob > 0.6:
                return 'negative', negative_prob
            else:
                return 'neutral', max(positive_prob, negative_prob)
        
        except Exception as e:
            return 'neutral', 0.5
    
    def batch_predict(self, texts, batch_size=32):
        """批量预测"""
        
        results = []
        total = len(texts)
        
        print(f"🔄 Analyzing {total:,} texts with BERT...")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict_single(text) for text in batch]
            results.extend(batch_results)
            
            if (i + batch_size) % 500 == 0 or i + batch_size >= total:
                progress = min(i + batch_size, total)
                print(f"  Progress: {progress:,}/{total:,} ({progress/total*100:.1f}%)")
        
        print("  ✅ BERT analysis completed!\n")
        return results


# ==================== PART 3: 专业提取与匹配 ====================

# 113个本科专业关键词库（扩展版）
MAJOR_KEYWORDS = {
    '计算机科学与技术': ['计算机', 'CS', '软件', '程序', '码农', 'IT', '编程', '代码', '计科'],
    '软件工程': ['软件工程', '软工', '开发', '程序员'],
    '电子信息工程': ['电子信息', '电信', '通信', '信号', '电子工程'],
    '临床医学': ['临床', '医学', '医生', '医师', '学医', '医学生'],
    '金融学': ['金融', '投资', '银行', '证券', '基金', '金融学'],
    '会计学': ['会计', '财务', '审计', 'CPA', '财会'],
    '法学': ['法学', '法律', '律师', '司法', '法考', '法硕'],
    '土木工程': ['土木', '建筑', '施工', '工程', '土建', '土木工程'],
    '机械工程': ['机械', '制造', '机电', '机械工程', '机械设计'],
    '电气工程': ['电气', '电力', '强电', '电气工程', '电工'],
    '自动化': ['自动化', '控制', '自动控制'],
    '通信工程': ['通信工程', '通信', '5G', '网络通信'],
    '师范类': ['师范', '教育', '教师', '老师', '当老师', '教育学'],
    '护理学': ['护理', '护士', '护理学'],
    '英语': ['英语', '英文', '翻译', '外语', '英语专业'],
    '新闻学': ['新闻', '传播', '媒体', '记者', '新闻学', '传媒'],
    '生物工程': ['生物', '生工', '生化', '天坑', '生物工程', '生科'],
    '化学类': ['化学', '化工', '化学工程'],
    '材料类': ['材料', '高分子', '材料科学', '材料工程'],
    '环境工程': ['环境', '环工', '环保', '环境工程'],
    '经济学': ['经济学', '经济', '宏观', '微观'],
    '工商管理': ['工商管理', '管理学', '企业管理', 'MBA'],
    '市场营销': ['市场营销', '营销', '销售'],
    '人力资源': ['人力资源', 'HR', '人事'],
    '建筑学': ['建筑学', '建筑设计', '建筑师'],
    '数学': ['数学', '数学专业', '应用数学', '数学系'],
    '物理学': ['物理', '物理学', '物理系'],
    '心理学': ['心理', '心理学', '心理咨询'],
    '汉语言文学': ['汉语言', '中文', '文学', '中文系', '汉语'],
    '历史学': ['历史', '历史学', '考古'],
    '哲学': ['哲学', '哲学专业'],
    '艺术设计': ['设计', '艺术设计', '平面设计', 'UI'],
    '音乐': ['音乐', '音乐专业', '声乐'],
    '美术': ['美术', '绑定', '美术生', '画画'],
    '体育': ['体育', '体育专业', '体育生'],
    '农学': ['农学', '农业', '种植'],
    '兽医': ['兽医', '动物医学', '宠物医生'],
    '药学': ['药学', '制药', '药剂'],
    '中医学': ['中医', '中医学', '中药'],
    '口腔医学': ['口腔', '牙医', '口腔医学'],
    '人工智能': ['人工智能', 'AI', '机器学习', '深度学习'],
    '数据科学': ['数据科学', '大数据', '数据分析'],
    '网络安全': ['网络安全', '信息安全', '网安'],
    '航空航天': ['航空航天', '飞行器', '航天'],
}


def extract_majors_from_text(df):
    """从文本中提取专业提及"""
    
    print("🔍 Extracting major mentions from texts...")
    
    def find_majors(text):
        text_lower = str(text).lower()
        found = []
        for major, keywords in MAJOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                found.append(major)
        return found if found else ['未明确提及']
    
    df['mentioned_majors'] = df['text'].apply(find_majors)
    
    # 展开为多行
    df_expanded = df.explode('mentioned_majors')
    df_expanded = df_expanded[df_expanded['mentioned_majors'] != '未明确提及']
    
    print(f"  ✅ Extracted {len(df_expanded):,} major mentions\n")
    
    return df_expanded


def aggregate_sentiment_by_major(df):
    """按专业聚合情感分析结果"""
    
    print("📊 Aggregating sentiment by major...")
    
    major_sentiment = df.groupby('mentioned_majors').agg({
        'sentiment': lambda x: {
            'positive_rate': (x == 'positive').sum() / len(x) * 100,
            'negative_rate': (x == 'negative').sum() / len(x) * 100,
            'neutral_rate': (x == 'neutral').sum() / len(x) * 100
        },
        'confidence': 'mean',
        'text': 'count',
        'likes': 'sum'
    }).reset_index()
    
    # 展开情感字典
    major_sentiment['positive_rate'] = major_sentiment['sentiment'].apply(lambda x: x['positive_rate'])
    major_sentiment['negative_rate'] = major_sentiment['sentiment'].apply(lambda x: x['negative_rate'])
    major_sentiment['neutral_rate'] = major_sentiment['sentiment'].apply(lambda x: x['neutral_rate'])
    major_sentiment = major_sentiment.drop('sentiment', axis=1)
    
    # 重命名列
    major_sentiment = major_sentiment.rename(columns={
        'mentioned_majors': 'major',
        'text': 'mention_count',
        'likes': 'total_likes'
    })
    
    # 计算综合指标
    major_sentiment['sentiment_index'] = (
        major_sentiment['positive_rate'] - major_sentiment['negative_rate']
    )
    
    max_mentions = major_sentiment['mention_count'].max()
    major_sentiment['recommendation_score'] = (
        major_sentiment['positive_rate'] * 
        major_sentiment['confidence'] * 
        (major_sentiment['mention_count'] / max_mentions) * 100
    )
    
    print(f"  ✅ Aggregated {len(major_sentiment)} majors\n")
    
    return major_sentiment.sort_values('recommendation_score', ascending=False)


def aggregate_sentiment_by_major_and_type(df):
    """按专业和数据类型分别聚合（用于对比分析）"""
    
    print("📊 Aggregating sentiment by major and data type...")
    
    result = df.groupby(['mentioned_majors', 'data_type']).agg({
        'sentiment': lambda x: (x == 'positive').sum() / len(x) * 100,
        'confidence': 'mean',
        'text': 'count',
        'likes': 'sum'
    }).reset_index()
    
    result = result.rename(columns={
        'mentioned_majors': 'major',
        'sentiment': 'positive_rate',
        'text': 'mention_count',
        'likes': 'total_likes'
    })
    
    print(f"  ✅ Aggregated {len(result)} major-type combinations\n")
    
    return result


# ==================== PART 4: 数据整合 ====================

def integrate_sentiment_and_employment(df_sentiment, df_employment):
    """整合舆情数据和就业数据 - 完全匹配版"""
    
    print("🔗 Integrating sentiment and employment data...")
    print(f"  Sentiment columns: {df_sentiment.columns.tolist()}")
    print(f"  Employment columns: {df_employment.columns.tolist()}")
    
    # 创建专业名称映射
    major_mapping = {}
    
    for sent_major in df_sentiment['major'].unique():
        # 直接精确匹配
        if sent_major in df_employment['专业'].values:
            major_mapping[sent_major] = sent_major
        else:
            # 模糊匹配
            for emp_major in df_employment['专业'].values:
                # 双向包含匹配
                if (sent_major in str(emp_major)) or (str(emp_major) in sent_major):
                    major_mapping[sent_major] = emp_major
                    break
    
    print(f"  ✅ Successfully mapped {len(major_mapping)} majors:")
    for k, v in list(major_mapping.items())[:5]:
        print(f"     {k} → {v}")
    
    # 应用映射
    df_sentiment['employment_major'] = df_sentiment['major'].map(major_mapping)
    
    # 合并数据
    df_merged = pd.merge(
        df_sentiment,
        df_employment,
        left_on='employment_major',
        right_on='专业',
        how='inner'
    )
    
    if len(df_merged) == 0:
        print("  ⚠️ No matches found!")
        return None
    
    # 数据类型转换
    df_merged['本科就业率'] = pd.to_numeric(df_merged['本科就业率'], errors='coerce') * 100
    df_merged['本科月薪'] = pd.to_numeric(df_merged['本科月薪'], errors='coerce')
    df_merged['硕士就业率'] = pd.to_numeric(df_merged['硕士就业率'], errors='coerce') * 100
    df_merged['硕士月薪'] = pd.to_numeric(df_merged['硕士月薪'], errors='coerce')
    df_merged['学历薪资溢价率%'] = pd.to_numeric(df_merged['学历薪资溢价率%'], errors='coerce')
    
    # 计算排名
    df_merged['sentiment_rank'] = df_merged['sentiment_index'].rank(ascending=False)
    df_merged['employment_rank'] = df_merged['本科就业率'].rank(ascending=False)
    df_merged['deviation_score'] = abs(df_merged['sentiment_rank'] - df_merged['employment_rank'])
    
    # 分类标签
    def classify_deviation(row):
        if row['sentiment_index'] > 20 and row['本科就业率'] < 85:
            return 'Overrated'
        elif row['sentiment_index'] < 0 and row['本科就业率'] > 88:
            return 'Underrated'
        else:
            return 'Matched'
    
    df_merged['deviation_type'] = df_merged.apply(classify_deviation, axis=1)
    
    print(f"  ✅ Integrated {len(df_merged)} majors")
    print(f"  Deviation distribution:\n{df_merged['deviation_type'].value_counts()}\n")
    
    return df_merged


# ==================== PART 5: 核心可视化 ====================

def create_enhanced_visualizations(df_integrated):
    """创建增强版可视化"""
    
    if df_integrated is None or len(df_integrated) == 0:
        print("⚠️ No integrated data available")
        return
    
    print("="*70)
    print("🎨 Creating Enhanced Visualizations")
    print("="*70 + "\n")
    
    # ========== 图5: 静态气泡图 ==========
    print("Creating bubble chart...")
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    color_map = {
        'Matched': '#2ecc71',
        'Overrated': '#e74c3c',
        'Underrated': '#3498db'
    }
    
    marker_map = {
        'Matched': 'o',
        'Overrated': '^',
        'Underrated': 's'
    }
    
    for deviation_type in ['Matched', 'Overrated', 'Underrated']:
        df_type = df_integrated[df_integrated['deviation_type'] == deviation_type]
        
        if len(df_type) > 0:
            ax.scatter(
                df_type['本科就业率'],
                df_type['sentiment_index'],
                s=df_type['mention_count'] * 3,
                c=color_map[deviation_type],
                alpha=0.6,
                edgecolors='black',
                linewidth=2,
                marker=marker_map[deviation_type],
                label=deviation_type,
                zorder=3
            )
            
            for idx, row in df_type.iterrows():
                ax.annotate(
                    row['major'],
                    (row['本科就业率'], row['sentiment_index']),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor=color_map[deviation_type],
                        alpha=0.3,
                        edgecolor='black',
                        linewidth=0.8
                    ),
                    zorder=4
                )
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.axvline(x=85, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.plot([65, 100], [-40, 50], 'k--', linewidth=2, alpha=0.3, 
            label='Ideal Match', zorder=2)
    
    ax.text(72, 45, 'HIGH Sentiment\nLOW Employment\n(Overrated)', 
           fontsize=12, alpha=0.6, fontstyle='italic', ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.2))
    
    ax.text(93, -35, 'LOW Sentiment\nHIGH Employment\n(Underrated)', 
           fontsize=12, alpha=0.6, fontstyle='italic', ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498db', alpha=0.2))
    
    ax.text(93, 40, 'HIGH Sentiment\nHIGH Employment\n(Ideal)', 
           fontsize=12, alpha=0.6, fontstyle='italic', ha='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.2))
    
    ax.set_xlabel('Official Employment Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Social Media Sentiment Index\n(Positive% - Negative%)', 
                 fontsize=14, fontweight='bold')
    ax.set_title('Zhang Xuefeng Major Recommendation: Sentiment vs Reality\n' +
                'Bubble Chart Analysis (BERT + Employment Data + Contents)',
                fontsize=17, fontweight='bold', pad=20)
    
    ax.set_xlim(65, 100)
    ax.set_ylim(-50, 60)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, zorder=0)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95, 
             edgecolor='black', shadow=True)
    
    plt.tight_layout()
    plt.savefig('./output/figures/05_bubble_chart.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Figure 5: Bubble Chart")
    
    # ========== 图6: 薪资对比 ==========
    print("Creating salary comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    df_plot = df_integrated.sort_values('本硕薪资差', ascending=False).head(12)
    
    x = np.arange(len(df_plot))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, df_plot['本科月薪'], width, 
                       label='Bachelor', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = axes[0].bar(x + width/2, df_plot['硕士月薪'], width,
                       label='Master', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9)
    
    axes[0].set_ylabel('Monthly Salary (CNY)', fontsize=12)
    axes[0].set_title('Salary Comparison: Bachelor vs Master', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_plot['major'], rotation=45, ha='right')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    df_plot2 = df_integrated.sort_values('学历薪资溢价率%', ascending=False).head(12)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_plot2)))
    bars = axes[1].barh(df_plot2['major'], df_plot2['学历薪资溢价率%'], 
                        color=colors, alpha=0.8, edgecolor='black')
    
    axes[1].set_xlabel('Salary Premium Rate (%)', fontsize=12)
    axes[1].set_title('Education Premium: Master vs Bachelor', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(df_plot2.iterrows()):
        axes[1].text(row['学历薪资溢价率%'], i, f" {row['学历薪资溢价率%']:.0f}%",
                    va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/figures/06_salary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 6: Salary Comparison")
    
    # ========== 图7: 就业率提升 ==========
    print("Creating employment improvement chart...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_plot3 = df_integrated.sort_values('学历就业率提升%', ascending=True).head(15)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_plot3['学历就业率提升%']]
    bars = ax.barh(df_plot3['major'], df_plot3['学历就业率提升%'], 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Employment Rate Improvement (%)', fontsize=12)
    ax.set_title('Employment Rate Boost: Master vs Bachelor', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(df_plot3.iterrows()):
        ax.text(row['学历就业率提升%'], i, f" {row['学历就业率提升%']:.1f}%",
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/figures/07_employment_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 7: Employment Improvement")
    
    print("\n" + "="*70)
    print("✅ All enhanced visualizations completed!")
    print("="*70 + "\n")


def create_all_visualizations(df_all, df_sentiment, df_integrated):
    """生成所有核心图表"""
    
    print("="*70)
    print("📊 Creating Visualizations")
    print("="*70 + "\n")
    
    # 图1: 平台和数据类型分布
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 按平台分布
    platform_counts = df_all['platform'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = axes[0].bar(platform_counts.index, platform_counts.values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_title('Distribution by Platform', fontsize=15, fontweight='bold')
    axes[0].set_ylabel('Number of Records')
    axes[0].grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # 按数据类型分布
    type_counts = df_all['data_type'].value_counts()
    colors_type = ['#9b59b6', '#3498db', '#e74c3c']
    bars2 = axes[1].bar(type_counts.index, type_counts.values, color=colors_type[:len(type_counts)], alpha=0.8, edgecolor='black')
    axes[1].set_title('Distribution by Data Type', fontsize=15, fontweight='bold')
    axes[1].set_ylabel('Number of Records')
    axes[1].grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/figures/01_platform_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 1: Platform & Type Distribution")
    
    # 图2: BERT情感分布（按数据类型）
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, data_type in enumerate(df_all['data_type'].unique()):
        df_type = df_all[df_all['data_type'] == data_type]
        sentiment_counts = df_type['sentiment'].value_counts()
        
        axes[i].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=colors_pie, shadow=True, startangle=90,
                   textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[i].set_title(f'{data_type.title()} Sentiment\n(n={len(df_type):,})', 
                         fontsize=12, fontweight='bold')
    
    plt.suptitle('BERT Sentiment Distribution by Data Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./output/figures/02_bert_sentiment_by_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 2: BERT Sentiment by Data Type")
    
    # 图3: 专业推荐度排名
    top_majors = df_sentiment.head(15)
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_majors)))
    bars = ax3.barh(top_majors['major'], top_majors['recommendation_score'], color=colors, edgecolor='black')
    
    ax3.set_xlabel('Recommendation Score', fontsize=12)
    ax3.set_title('Top 15 Majors - Social Media Recommendation Score\n(Based on BERT Sentiment Analysis - Comments + Contents)',
                 fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_majors.iterrows()):
        ax3.text(row['recommendation_score'], i, f" {row['recommendation_score']:.1f}",
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/figures/03_recommendation_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 3: Recommendation Ranking")
    
    # 图4: 情感指数对比
    top15 = df_sentiment.head(15)
    
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top15))
    width = 0.35
    
    ax4.bar(x - width/2, top15['positive_rate'], width, label='Positive %', 
           color='#2ecc71', alpha=0.8)
    ax4.bar(x + width/2, top15['negative_rate'], width, label='Negative %',
           color='#e74c3c', alpha=0.8)
    
    ax4.set_ylabel('Percentage', fontsize=12)
    ax4.set_title('Top 15 Majors - Positive vs Negative Sentiment Rate',
                 fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(top15['major'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/figures/04_sentiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 4: Sentiment Comparison")
    
    # 图5: 舆情 vs 就业（如果有就业数据）
    if df_integrated is not None and len(df_integrated) > 0:
        employment_col = '本科就业率' if '本科就业率' in df_integrated.columns else 'employment_rate'
        
        if employment_col in df_integrated.columns:
            fig5 = px.scatter(
                df_integrated,
                x=employment_col,
                y='sentiment_index',
                size='mention_count',
                color='positive_rate',
                hover_data=['major'],
                text='major',
                color_continuous_scale='RdYlGn',
                title='<b>Social Media Sentiment vs Official Employment Rate</b><br>' +
                      '<sub>Size = Discussion Volume | Color = Positive Sentiment Rate | Data = Comments + Contents</sub>',
                labels={
                    employment_col: 'Official Employment Rate (%)',
                    'sentiment_index': 'Social Media Sentiment Index',
                    'positive_rate': 'Positive %'
                }
            )
            
            fig5.update_traces(textposition='top center', textfont_size=8)
            fig5.update_layout(width=1400, height=800)
            
            fig5.write_html('./output/figures/05_sentiment_vs_employment.html')
            print("✅ Figure 5: Sentiment vs Employment (Interactive)")
    
    print("\n" + "="*70)
    print("✅ All basic visualizations completed!")
    print("="*70 + "\n")


def create_content_vs_comment_comparison(df_by_type):
    """创建评论 vs 内容对比图"""
    
    print("📊 Creating Content vs Comment Comparison...")
    
    # 透视表
    pivot = df_by_type.pivot_table(
        index='major',
        columns='data_type',
        values='positive_rate',
        aggfunc='first'
    ).reset_index()
    
    # 选择同时有评论和内容数据的专业
    valid_cols = [col for col in ['comment', 'content', 'video'] if col in pivot.columns]
    if len(valid_cols) < 2:
        print("  ⚠️ Not enough data types for comparison")
        return
    
    pivot = pivot.dropna(subset=valid_cols[:2])
    
    if len(pivot) < 5:
        print("  ⚠️ Not enough majors for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(pivot))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, col in enumerate(valid_cols):
        if col in pivot.columns:
            ax.bar(x + i*width, pivot[col], width, label=col.title(), 
                  color=colors[i], alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Positive Sentiment Rate (%)', fontsize=12)
    ax.set_title('Sentiment Comparison: Comments vs Contents vs Videos\n(by Major)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot['major'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./output/figures/08_content_vs_comment.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 8: Content vs Comment Comparison")


# ==================== PART 6: 导出数据表 ====================

def export_all_tables(df_all, df_sentiment, df_integrated, df_by_type):
    """导出所有数据表"""
    
    print("💾 Exporting data tables...\n")
    
    # Table 1: 原始数据（带BERT结果）
    df_all.to_csv('./output/tables/01_bert_analyzed_all_data.csv', 
                  index=False, encoding='utf-8-sig')
    print("✅ Table 1: BERT Analyzed All Data (Comments + Contents)")
    
    # Table 2: 专业舆情汇总
    df_sentiment.to_csv('./output/tables/02_major_sentiment_summary.csv',
                       index=False, encoding='utf-8-sig')
    print("✅ Table 2: Major Sentiment Summary")
    
    # Table 3: 按数据类型分组的舆情
    df_by_type.to_csv('./output/tables/03_sentiment_by_major_and_type.csv',
                     index=False, encoding='utf-8-sig')
    print("✅ Table 3: Sentiment by Major and Data Type")
    
    # Table 4: 整合数据（如果有）
    if df_integrated is not None and len(df_integrated) > 0:
        df_integrated.to_csv('./output/tables/04_integrated_sentiment_employment.csv',
                           index=False, encoding='utf-8-sig')
        print("✅ Table 4: Integrated Analysis")
    
    print()


# ==================== 主函数 ====================

def main():
    """主执行流程"""
    
    # Step 1: 加载所有数据
    loader = RealDataLoader()
    
    # 加载评论
    loader.load_all_comments()
    df_comments = loader.standardize_comments()
    
    # 加载内容/帖子
    loader.load_all_contents()
    df_contents = loader.standardize_contents()
    
    # 合并所有数据
    df_all = loader.merge_all_data(df_comments, df_contents)
    
    # Step 2: BERT情感分析
    analyzer = RealBERTAnalyzer()
    sentiment_results = analyzer.batch_predict(df_all['text'].tolist())
    
    df_all['sentiment'] = [r[0] for r in sentiment_results]
    df_all['confidence'] = [r[1] for r in sentiment_results]
    
    print("📊 BERT Sentiment Analysis Results:")
    print(df_all['sentiment'].value_counts())
    print(f"\nAverage Confidence: {df_all['confidence'].mean():.3f}")
    print(f"\nBy Data Type:")
    print(df_all.groupby('data_type')['sentiment'].value_counts())
    print()
    
    # Step 3: 提取专业提及
    df_with_majors = extract_majors_from_text(df_all)
    
    # Step 4: 按专业聚合（总体）
    df_sentiment = aggregate_sentiment_by_major(df_with_majors)
    
    # Step 4b: 按专业和数据类型聚合（用于对比）
    df_by_type = aggregate_sentiment_by_major_and_type(df_with_majors)
    
    print("📊 Top 10 Recommended Majors (by sentiment):")
    print(df_sentiment[['major', 'recommendation_score', 'positive_rate', 'mention_count']].head(10))
    print()
    
    # Step 5: 加载就业数据
    df_employment = load_employment_data()
    
    # Step 6: 整合数据
    df_integrated = None
    if df_employment is not None:
        df_integrated = integrate_sentiment_and_employment(df_sentiment, df_employment)
    
    # Step 7: 生成基础可视化
    create_all_visualizations(df_all, df_sentiment, df_integrated)
    
    # Step 8: 生成增强可视化
    create_enhanced_visualizations(df_integrated)
    
    # Step 8b: 生成内容 vs 评论对比图
    create_content_vs_comment_comparison(df_by_type)
    
    # Step 9: 导出数据表
    export_all_tables(df_all, df_sentiment, df_integrated, df_by_type)
    
    # 最终总结
    print("="*70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📊 Total records analyzed: {len(df_all):,}")
    print(f"   - Comments: {len(df_all[df_all['data_type']=='comment']):,}")
    print(f"   - Contents: {len(df_all[df_all['data_type']=='content']):,}")
    print(f"   - Videos: {len(df_all[df_all['data_type']=='video']):,}")
    print(f"🤖 BERT model used: uer/roberta-base-finetuned-jd-binary-chinese")
    print(f"📈 Majors extracted: {len(df_sentiment)}")
    print(f"📁 Output directory: ./output/")
    print("="*70 + "\n")
    
    return df_all, df_sentiment, df_integrated, df_by_type


# ==================== 执行 ====================

if __name__ == "__main__":
    df_all, df_sentiment, df_integrated, df_by_type = main()

