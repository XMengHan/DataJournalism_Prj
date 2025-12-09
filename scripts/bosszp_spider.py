from selenium import webdriver
from bs4 import BeautifulSoup
import mysql
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import random


class Spider(object):

    def __init__(self):
        # 创建数据库对象
        self.__sql = mysql.MySql()
        
        # 配置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        # 添加UA
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
        # 添加其他反爬参数
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            service = Service(ChromeDriverManager().install())
            self.__driver = webdriver.Chrome(service=service, options=chrome_options)
            # 执行CDP命令
            self.__driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
        except Exception as e:
            print(f"Browser initialization error: {e}")
            raise e
        
        # 隐式等待
        self.__driver.implicitly_wait(20)
        
        # 关键词
        self.__keyword = ['Android']

    def __del__(self):
        # 关闭无头浏览器，减少内存损耗
        self.__driver.quit()

    # 设置爬取关键词
    def setKeyword(self, keyword):
        self.__keyword = []
        if isinstance(keyword, list):
            self.__keyword = keyword
        else:
            var = str(keyword)
            var.strip()
            if " " in var:
                keyword_list = var.split(' ')
                self.__keyword = keyword_list
            else:
                self.__keyword.append(var)

    # 获取所有关键词
    def getKeyword(self):
        return self.__keyword

    # 爬虫方法
    def run(self):

        print(">>>开始获取...")

        # 城市json
        cities = [{"name": "北京", "code": 101010100, "url": "/beijing/"},
                  {"name": "上海", "code": 101020100, "url": "/shanghai/"},
                  {"name": "广州", "code": 101280100, "url": "/guangzhou/"},
                  {"name": "深圳", "code": 101280600, "url": "/shenzhen/"},
                  {"name": "杭州", "code": 101210100, "url": "/hangzhou/"},
                  {"name": "天津", "code": 101030100, "url": "/tianjin/"},
                  {"name": "西安", "code": 101110100, "url": "/xian/"},
                  {"name": "苏州", "code": 101190400, "url": "/suzhou/"},
                  {"name": "武汉", "code": 101200100, "url": "/wuhan/"},
                  {"name": "厦门", "code": 101230200, "url": "/xiamen/"},
                  {"name": "长沙", "code": 101250100, "url": "/changsha/"},
                  {"name": "成都", "code": 101270100, "url": "/chengdu/"},
                  {"name": "郑州", "code": 101180100, "url": "/zhengzhou/"},
                  {"name": "重庆", "code": 101040100, "url": "/chongqing/"},
                  {"name": "佛山", "code": 101280800, "url": "/foshan/"},
                  {"name": "合肥", "code": 101220100, "url": "/hefei/"},
                  {"name": "济南", "code": 101120100, "url": "/jinan/"},
                  {"name": "青岛", "code": 101120200, "url": "/qingdao/"},
                  {"name": "南京", "code": 101190100, "url": "/nanjing/"},
                  {"name": "东莞", "code": 101281600, "url": "/dongguan/"},
                  {"name": "福州", "code": 101230100, "url": "/fuzhou/"}]
        # 总记录数
        all_count = 0
        # 关键词爬取
        for key in self.__keyword:
            print('>>>当前获取关键词: "{}"'.format(key))
            # 单个关键词爬取记录数
            key_count = 0
            # 每个城市爬取
            for city in cities:
                print('>>>当前获取城市: "{}"'.format(city['name']))
                # 记录每个城市爬取数据数目
                city_count = 0
                # 只获取前十页
                urls = ['https://www.zhipin.com/c{}/?query={}&page={}&ka=page-{}'
                            .format(city['code'], key, i, i) for i in range(1, 11)]
                # 逐条解析
                for url in urls:
                    try:
                        # 随机延时
                        time.sleep(random.uniform(2, 5))
                        
                        self.__driver.get(url)
                        # 添加随机滚动
                        self.__random_scroll()
                        
                        html = self.__driver.page_source
                        bs = BeautifulSoup(html, 'html.parser')
                        # 获取搜索框，用于判断是否被异常检测
                        flag = bs.find_all('div', {'class': 'inner home-inner'})
                        # 主要信息获取
                        job_all = bs.find_all('div', {"class": "job-primary"})

                        # ip检测 ip失效时中断程序
                        while True:
                            if not flag:
                                print(">>>当前IP已失效...更换IP后继续...")
                                choice = input(">>>继续(y/other):")
                                if choice == 'y' or 'Y':
                                    print(">>>继续获取...")
                                    # 重新解析当前连接
                                    self.__driver.get(url)
                                    html = self.__driver.page_source
                                    bs = BeautifulSoup(html, 'html.parser')
                                    flag = bs.find_all('div', {'class': 'inner home-inner'})
                                    job_all = bs.find_all('div', {"class": "job-primary"})

                                    if not flag:
                                        print(">>>无效ip...")
                                        continue
                                    else:
                                        break
                                else:
                                    print(">>>程序结束...")
                                    exit(0)
                            else:
                                break

                        # 解析页面
                        for job in job_all:
                            # 工作名称
                            job_name = job.find('span', {"class": "job-name"}).get_text()
                            # 工作地点
                            job_place = job.find('span', {'class': "job-area"}).get_text()
                            # 工作公司
                            job_company = job.find('div', {'class': 'company-text'}).find('h3', {'class': "name"}).get_text()
                            # 公司规模
                            job_scale = job.find('div', {'class': 'company-text'}).find('p').get_text()
                            # 工作薪资
                            job_salary = job.find('span', {'class': 'red'}).get_text()
                            # 工作学历
                            job_education = job.find('div', {'class': 'job-limit'}).find('p').get_text()[-2:]
                            # 工作经验
                            job_experience = job.find('div', {'class': 'job-limit'}).find('p').get_text()
                            # 工作标签
                            job_label = job.find('a', {'class': 'false-link'}).get_text()
                            # 技能要求
                            job_skill = job.find('div', {'class': 'tags'}).get_text().replace("\n", " ").strip()
                            # 福利
                            job_welfare = job.find('div', {'class': 'info-desc'}).get_text().replace("，", " ").strip()

                            #职位类型 追加
                            type=key

                            # 数据存储
                            self.__sql.saveData(job_name, job_place, job_company, job_scale, job_salary, job_education,
                                                 job_experience,
                                                 job_label,
                                                 job_skill,
                                                 job_welfare,type)
                            # 统计记录数
                            print(job_name, job_place, job_company, job_scale, job_salary, job_education,
                                                 job_experience,
                                                 job_label,
                                                 job_skill,
                                                 job_welfare)
                            city_count = city_count + 1
                        key_count = key_count + city_count
                    except Exception as e:
                        print(f"Error while scraping: {e}")
                        continue
                all_count = all_count + key_count

                print('>>>城市: "{}" 获取完成...获取数据: {} 条'.format(city['name'], city_count))
            print('>>>关键词: "{}" 获取完成...获取数据: {} 条'.format(key, key_count))
        print(">>>全部关键词获取完成...共获取 {} 条数据".format(all_count))

    def __random_scroll(self):
        """随机滚动页面"""
        total_height = self.__driver.execute_script("return document.body.scrollHeight")
        for i in range(3):
            target_height = random.randint(0, total_height)
            self.__driver.execute_script(f"window.scrollTo(0, {target_height});")
            time.sleep(random.uniform(0.5, 1.5))
