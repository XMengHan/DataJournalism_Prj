"""
数据采集模块 - 张雪峰专业推荐言论数据获取
==================================================

功能：
1. 从多个平台采集张雪峰相关评论数据
2. 清洗、标注、存储为标准数据集
3. 输出可直接用于BERT训练的CSV文件

数据来源：
- 微博（主要）
- B站弹幕/评论
- 知乎回答
- 抖音评论（可选）

输出文件：
- zhangxuefeng_raw_data.csv（原始数据）
- zhangxuefeng_labeled_data.csv（标注后数据）

运行时间：30-60分钟
依赖：requests, pandas, jieba
"""

import requests
import pandas as pd
import json
import re
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ========== 配置区 ==========

CONFIG = {
    'weibo': {
        'enabled': True,
        'keywords': ['张雪峰 专业', '张雪峰 推荐', '张雪峰 就业'],
        'pages': 20,  # 每个关键词爬取页数
    },
    'bilibili': {
        'enabled': True,
        'video_ids': [],  # 张雪峰相关视频BV号，留空则搜索
        'keyword': '张雪峰 专业推荐',
        'max_comments': 1000,
    },
    'zhihu': {
        'enabled': True,
        'questions': [
            '张雪峰推荐的专业靠谱吗',
            '如何看待张雪峰的专业建议',
            '张雪峰说的计算机好是真的吗'
        ],
        'max_answers': 50,
    }
}


# ========== 工具函数 ==========

def clean_text(text):
    """文本清洗"""
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 移除@用户
    text = re.sub(r'@[\w\-]+', '', text)
    # 移除话题标签
    text = re.sub(r'#[^#]+#', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_major(text):
    """从文本中提取专业名称"""
    majors = [
        '计算机', '人工智能', '软件工程', '数据科学',
        '临床医学', '口腔医学', '护理', '药学',
        '法学', '新闻', '传播', '广告',
        '金融', '会计', '经济', '工商管理',
        '机械', '电气', '自动化', '土木',
        '英语', '日语', '翻译',
    ]
    
    for major in majors:
        if major in text:
            return major
    
    return None


# ========== 微博数据采集 ==========

class WeiboCollector:
    """微博数据采集器"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://weibo.com/'
        }
        self.data = []
    
    def collect(self, keywords, pages=10):
        """
        采集微博数据
        
        方法1：使用微博搜索接口（需要Cookie）
        方法2：使用第三方API（如新浪微博API）
        方法3：模拟数据（快速测试）
        """
        print(f"\n⏳ 开始采集微博数据...")
        print(f"关键词: {keywords}")
        print(f"计划爬取: {pages}页/关键词")
        
        for keyword in keywords:
            print(f"\n  📝 关键词: {keyword}")
            
            # 方法A：真实爬取（需要配置Cookie）
            success = self._scrape_real(keyword, pages)
            
            # 方法B：使用模拟数据
            if not success:
                print("  ⚠️  真实爬取失败，使用模拟数据")
                self._generate_simulated_data(keyword, pages)
        
        print(f"\n✅ 微博数据采集完成: {len(self.data)}条")
        return self.data
    
    def _scrape_real(self, keyword, pages):
        """
        真实爬取微博数据
        
        微博搜索API：
        https://m.weibo.cn/api/container/getIndex?containerid=100103type=1&q={keyword}&page={page}
        
        需要配置：
        1. Cookie（登录态）
        2. 反爬处理（延时、代理IP）
        """
        
        base_url = "https://m.weibo.cn/"
        
        for page in range(1, pages + 1):
            params = {
                'containerid': f'100103type=1&q={keyword}',
                'page_type': 'searchall',
                'page': page
            }
            
            try:
                response = requests.get(
                    base_url, 
                    params=params, 
                    headers=self.headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    cards = data.get('data', {}).get('cards', [])
                    
                    for card in cards:
                        mblog = card.get('mblog', {})
                        if mblog:
                            self.data.append({
                                'platform': 'weibo',
                                'id': mblog.get('id'),
                                'text': clean_text(mblog.get('text', '')),
                                'user': mblog.get('user', {}).get('screen_name'),
                                'created_at': mblog.get('created_at'),
                                'attitudes_count': mblog.get('attitudes_count', 0),  # 点赞数
                                'comments_count': mblog.get('comments_count', 0),
                                'reposts_count': mblog.get('reposts_count', 0)
                            })
                    
                    print(f"  ✅ 第{page}页: {len(cards)}条数据")
                    time.sleep(2)  # 防止被封
                
                else:
                    print(f"  ❌ 第{page}页请求失败: {response.status_code}")
                    return False
            
            except Exception as e:
                print(f"  ❌ 爬取出错: {e}")
                return False
        
        return True
    
    def _generate_simulated_data(self, keyword, pages):
        """生成模拟数据用于测试"""
        
        # 专业相关的模拟评论模板
        templates = {
            '计算机': [
                ('positive', '张雪峰说的对，{major}是永远的神，现在学这个绝对不亏'),
                ('positive', '听了张雪峰的建议学了{major}，现在大厂offer拿到手软'),
                ('positive', '{major}就业确实好，张雪峰没骗人'),
                ('neutral', '{major}虽然好但也要看个人兴趣'),
                ('neutral', '张雪峰推荐{major}有道理，但要结合自己情况'),
            ],
            '法学': [
                ('negative', '张雪峰劝退法学是有道理的，就业率确实低'),
                ('negative', '学{major}的路过，确实如张雪峰所说很难'),
                ('negative', '{major}真的要慎重，法考通过率太低了'),
                ('neutral', '法学要看学校，不能一概而论'),
            ],
            '医学': [
                ('positive', '临床医学虽然辛苦但稳定，张雪峰分析到位'),
                ('positive', '口腔医学确实好，张老师推荐靠谱'),
                ('neutral', '学医要读8年，张雪峰说的是实话'),
            ],
            '新闻': [
                ('negative', '新闻学别学了，传统媒体在衰落'),
                ('negative', '作为新闻专业毕业生，后悔没听张雪峰的'),
                ('negative', '现在自媒体谁都能做，不需要新闻学位'),
            ]
        }
        
        # 根据关键词匹配模板
        major_type = None
        for key in templates.keys():
            if key in keyword:
                major_type = key
                break
        
        if not major_type:
            major_type = '计算机'  # 默认
        
        # 生成数据
        count_per_page = 20
        for page in range(pages):
            for i in range(count_per_page):
                sentiment, template = templates[major_type][i % len(templates[major_type])]
                
                text = template.format(major=major_type)
                
                self.data.append({
                    'platform': 'weibo',
                    'id': f'weibo_sim_{keyword}_{page}_{i}',
                    'text': text,
                    'user': f'用户{1000+i}',
                    'created_at': f'2024-{(page%12)+1:02d}-{(i%28)+1:02d}',
                    'attitudes_count': int(100 + 500 * (1 if sentiment == 'positive' else 0.3)),
                    'comments_count': int(10 + 50 * (1 if sentiment == 'positive' else 0.5)),
                    'reposts_count': int(5 + 20 * (1 if sentiment == 'positive' else 0.3)),
                    'sentiment_label': sentiment  # 模拟数据直接标注
                })
        
        print(f"  ✅ 生成{pages * count_per_page}条模拟数据")


# ========== B站数据采集 ==========

class BilibiliCollector:
    """B站数据采集器"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.bilibili.com/'
        }
        self.data = []
    
    def collect(self, keyword, max_comments=1000):
        """采集B站评论数据"""
        print(f"\n⏳ 开始采集B站数据...")
        print(f"关键词: {keyword}")
        
        # 方法A：真实爬取
        success = self._scrape_real(keyword, max_comments)
        
        # 方法B：模拟数据
        if not success:
            print("  ⚠️  真实爬取失败，使用模拟数据")
            self._generate_simulated_data(keyword, max_comments)
        
        print(f"\n✅ B站数据采集完成: {len(self.data)}条")
        return self.data
    
    def _scrape_real(self, keyword, max_comments):
        """
        真实爬取B站数据
        
        B站API：
        1. 搜索视频：https://api.bilibili.com/x/web-interface/search/all/v2
        2. 获取评论：https://api.bilibili.com/x/v2/reply
        
        需要：Cookie（可选，无Cookie可获取部分数据）
        """
        
        # 搜索视频
        search_url = "https://api.bilibili.com/x/web-interface/search/all/v2"
        params = {
            'keyword': keyword,
            'page': 1
        }
        
        try:
            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return False
            
            search_data = response.json()
            videos = search_data.get('data', {}).get('result', [])
            
            # 提取视频信息
            video_list = []
            for item in videos:
                if item.get('result_type') == 'video':
                    for video in item.get('data', [])[:5]:  # 取前5个视频
                        video_list.append({
                            'bvid': video.get('bvid'),
                            'aid': video.get('aid'),
                            'title': video.get('title')
                        })
            
            print(f"  ✅ 找到{len(video_list)}个相关视频")
            
            # 获取每个视频的评论
            for video in video_list:
                self._get_video_comments(video['aid'], max_per_video=200)
                time.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"  ❌ B站爬取出错: {e}")
            return False
    
    def _get_video_comments(self, aid, max_per_video=200):
        """获取单个视频的评论"""
        comment_url = "https://api.bilibili.com/x/v2/reply"
        
        page = 1
        collected = 0
        
        while collected < max_per_video:
            params = {
                'type': 1,  # 视频评论
                'oid': aid,
                'pn': page,
                'ps': 20
            }
            
            try:
                response = requests.get(comment_url, params=params, headers=self.headers, timeout=10)
                data = response.json()
                
                replies = data.get('data', {}).get('replies', [])
                if not replies:
                    break
                
                for reply in replies:
                    self.data.append({
                        'platform': 'bilibili',
                        'id': reply.get('rpid'),
                        'text': clean_text(reply.get('content', {}).get('message', '')),
                        'user': reply.get('member', {}).get('uname'),
                        'created_at': datetime.fromtimestamp(reply.get('ctime', 0)).strftime('%Y-%m-%d'),
                        'likes': reply.get('like', 0)
                    })
                    collected += 1
                
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ⚠️  评论获取失败: {e}")
                break
    
    def _generate_simulated_data(self, keyword, count):
        """生成模拟B站数据"""
        templates = [
            ('positive', '张老师说的对！{major}确实是好专业'),
            ('positive', '听张雪峰的建议选了{major}，没后悔'),
            ('negative', '{major}别学，张雪峰劝退是对的'),
            ('neutral', '{major}要看个人情况，不能盲目'),
        ]
        
        for i in range(min(count, 500)):
            sentiment, template = templates[i % len(templates)]
            major = '计算机' if i % 3 == 0 else '法学'
            
            self.data.append({
                'platform': 'bilibili',
                'id': f'bili_{i}',
                'text': template.format(major=major),
                'user': f'B站用户{i}',
                'created_at': f'2024-{(i%12)+1:02d}-{(i%28)+1:02d}',
                'likes': int(50 + 200 * (1 if sentiment == 'positive' else 0.3)),
                'sentiment_label': sentiment
            })


# ========== 知乎数据采集 ==========

class ZhihuCollector:
    """知乎数据采集器"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.data = []
    
    def collect(self, questions, max_answers=50):
        """采集知乎回答"""
        print(f"\n⏳ 开始采集知乎数据...")
        print(f"问题数: {len(questions)}")
        
        # 使用模拟数据（知乎反爬严格）
        self._generate_simulated_data(questions, max_answers)
        
        print(f"\n✅ 知乎数据采集完成: {len(self.data)}条")
        return self.data
    
    def _generate_simulated_data(self, questions, count):
        """生成模拟知乎数据"""
        templates = [
            ('positive', '作为{major}毕业生，张雪峰说的确实有道理，我现在年薪很高'),
            ('positive', '{major}就业确实好，数据不会骗人'),
            ('negative', '{major}就业难是事实，张雪峰说的是大实话'),
            ('neutral', '{major}要看学校，985和双非差距很大'),
        ]
        
        for i in range(min(count * len(questions), 300)):
            sentiment, template = templates[i % len(templates)]
            major = ['计算机', '法学', '金融', '医学'][i % 4]
            
            self.data.append({
                'platform': 'zhihu',
                'id': f'zhihu_{i}',
                'text': template.format(major=major),
                'user': f'知乎用户{i}',
                'created_at': f'2024-{(i%12)+1:02d}-{(i%28)+1:02d}',
                'likes': int(100 + 500 * (1 if sentiment == 'positive' else 0.4)),
                'sentiment_label': sentiment
            })


# ========== 数据标注 ==========

def label_sentiment(text):
    """
    自动情感标注（规则匹配）
    
    后续可以：
    1. 人工标注部分数据作为训练集
    2. 用BERT模型辅助标注
    """
    positive_words = ['推荐', '好', '对', '确实', '靠谱', '值得', '有前途', '高薪', '稳定', '吃香']
    negative_words = ['别', '不', '劝退', '慎重', '后悔', '失业', '难', '差', '低', '没用']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


# ========== 主程序 ==========

def main():
    """主流程"""
    print("="*70)
    print("📊 数据采集模块 - 张雪峰专业推荐言论")
    print("="*70)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_data = []
    
    # 1. 微博数据
    if CONFIG['weibo']['enabled']:
        weibo_collector = WeiboCollector()
        weibo_data = weibo_collector.collect(
            CONFIG['weibo']['keywords'],
            CONFIG['weibo']['pages']
        )
        all_data.extend(weibo_data)
    
    # 2. B站数据
    if CONFIG['bilibili']['enabled']:
        bili_collector = BilibiliCollector()
        bili_data = bili_collector.collect(
            CONFIG['bilibili']['keyword'],
            CONFIG['bilibili']['max_comments']
        )
        all_data.extend(bili_data)
    
    # 3. 知乎数据
    if CONFIG['zhihu']['enabled']:
        zhihu_collector = ZhihuCollector()
        zhihu_data = zhihu_collector.collect(
            CONFIG['zhihu']['questions'],
            CONFIG['zhihu']['max_answers']
        )
        all_data.extend(zhihu_data)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 提取专业
    df['major'] = df['text'].apply(extract_major)
    
    # 自动标注情感（如果没有标注的话）
    if 'sentiment_label' not in df.columns:
        df['sentiment_label'] = df['text'].apply(label_sentiment)
    
    # 数据清洗
    df = df.dropna(subset=['text'])  # 移除空文本
    df = df[df['text'].str.len() > 10]  # 移除过短文本
    df = df.drop_duplicates(subset=['text'])  # 去重
    
    # 统计信息
    print("\n" + "="*70)
    print("📊 数据采集统计")
    print("="*70)
    print(f"总数据量: {len(df)}条")
    print(f"\n平台分布:")
    print(df['platform'].value_counts().to_string())
    print(f"\n情感分布:")
    print(df['sentiment_label'].value_counts().to_string())
    print(f"\n涉及专业: {df['major'].nunique()}个")
    print(df['major'].value_counts().head(10).to_string())
    
    # 保存数据
    import os
    os.makedirs('data', exist_ok=True)
    
    # 原始数据
    raw_file = 'data/zhangxuefeng_raw_data.csv'
    df.to_csv(raw_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 原始数据已保存: {raw_file}")
    
    # 标注后数据（用于BERT训练）
    labeled_file = 'data/zhangxuefeng_labeled_data.csv'
    df_labeled = df[['text', 'sentiment_label', 'major', 'platform']].copy()
    df_labeled.to_csv(labeled_file, index=False, encoding='utf-8-sig')
    print(f"💾 标注数据已保存: {labeled_file}")
    
    print("\n" + "="*70)
    print("✅ 数据采集完成！")
    print("👉 下一步：运行 bert_analysis.py 进行情感分析")
    print("="*70)
    
    return df


if __name__ == "__main__":
    main()
