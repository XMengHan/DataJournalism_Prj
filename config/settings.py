# 全局配置
PROJECT_ROOT = "/path/to/DataJournalism_MajorAnalysis"

# 数据路径
RAW_DATA_PATH = f"{PROJECT_ROOT}/data/raw"
PROCESSED_DATA_PATH = f"{PROJECT_ROOT}/data/processed"

# 爬虫配置
CRAWL_DELAY = 2  # 请求间隔(秒)
MAX_RETRIES = 3  # 最大重试次数

# 分析参数
TOP_N_MAJORS = 50  # 重点分析专业数量
SALARY_NORMALIZE = True  # 是否标准化薪资
