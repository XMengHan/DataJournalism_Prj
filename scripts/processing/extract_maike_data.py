"""
extract_maike_data.py
从麦可思PDF报告中提取结构化数据
"""

import pdfplumber
import pandas as pd
import re

def extract_employment_data(pdf_path):
    """提取就业率等核心数据"""
    
    data = {
        'major': [],
        'employment_rate': [],
        'salary': [],
        'satisfaction': []
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            
            # 提取表格数据
            for table in tables:
                # 根据表格特征提取
                pass
    
    return pd.DataFrame(data)

# 执行提取
df_employment = extract_employment_data('./data/raw/麦可思2024就业报告.pdf')
df_employment.to_csv('./data/processed/employment_official.csv', index=False)
