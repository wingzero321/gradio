import akshare as ak
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import gradio as gr

def init_database():
    """初始化数据库，创建必要的表"""
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # 创建股票基本信息表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stocks (
        code TEXT PRIMARY KEY,
        name TEXT,
        update_time TIMESTAMP
    )
    ''')
    
    # 创建股票交易数据表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_daily (
        code TEXT,
        trade_date DATE,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        amount REAL,
        PRIMARY KEY (code, trade_date)
    )
    ''')
    
    conn.commit()
    conn.close()

def download_stock_list():
    """下载上交所股票列表"""
    try:
        # 使用akshare获取上交所股票列表
        stock_info = ak.stock_info_sh_name_code()
        
        # 检查并重命名列名
        if '证券代码' in stock_info.columns:
            stock_info = stock_info.rename(columns={
                '证券代码': 'code',
                '证券简称': 'name'
            })
        
        conn = sqlite3.connect('stock_data.db')
        # 只保存需要的列
        stock_info = stock_info[['code', 'name']]
        stock_info['update_time'] = datetime.now()
        stock_info.to_sql('stocks', conn, if_exists='replace', index=False)
        conn.close()
        
        return True, f"成功下载{len(stock_info)}支股票的基本信息"
    except Exception as e:
        return False, f"下载股票列表失败：{str(e)}"

def download_stock_data(progress=gr.Progress()):
    """下载所有股票的90天交易数据"""
    global is_downloading
    is_downloading = True
    
    try:
        conn = sqlite3.connect('stock_data.db')
        stocks = pd.read_sql("SELECT code FROM stocks", conn)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        total_stocks = len(stocks)
        success_count = 0
        
        for idx, row in progress.tqdm(stocks.iterrows(), total=total_stocks):
            if not is_downloading:
                conn.close()
                return False, f"下载已中断。成功下载{success_count}/{total_stocks}支股票的交易数据"
                
            try:
                code = row['code']
                stock_data = ak.stock_zh_a_hist(symbol=code, 
                                              start_date=start_date.strftime('%Y%m%d'),
                                              end_date=end_date.strftime('%Y%m%d'))
                
                if not stock_data.empty:
                    # 检查并适配数据列
                    expected_columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
                    if all(col in stock_data.columns for col in expected_columns):
                        # 重命名主要的交易数据列
                        column_mapping = {
                            '日期': 'trade_date',
                            '开盘': 'open',
                            '最高': 'high',
                            '最低': 'low',
                            '收盘': 'close',
                            '成交量': 'volume',
                            '成交额': 'amount'
                        }
                        stock_data = stock_data.rename(columns=column_mapping)
                        stock_data['code'] = code
                        
                        # 只保存需要的列
                        columns_to_save = ['code', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                        stock_data[columns_to_save].to_sql('stock_daily', conn, if_exists='append', index=False)
                        
                        success_count += 1
                
            except Exception as e:
                print(f"下载股票 {code} 数据失败：{str(e)}")
                continue
        
        conn.close()
        is_downloading = False
        return True, f"成功下载{success_count}/{total_stocks}支股票的交易数据"
    
    except Exception as e:
        is_downloading = False
        return False, f"下载交易数据失败：{str(e)}"

def view_stock_list():
    """查看股票列表数据"""
    try:
        conn = sqlite3.connect('stock_data.db')
        stocks = pd.read_sql("SELECT * FROM stocks", conn)
        conn.close()
        return str(stocks.head(10))  # 显示前10条记录
    except Exception as e:
        return f"查询股票列表失败：{str(e)}"

def view_stock_daily():
    """查看交易数据"""
    try:
        conn = sqlite3.connect('stock_data.db')
        daily_data = pd.read_sql("SELECT * FROM stock_daily LIMIT 10", conn)  # 限制显示10条记录
        conn.close()
        return str(daily_data)
    except Exception as e:
        return f"查询交易数据失败：{str(e)}"

def create_download_tab():
    """创建数据下载标签页"""
    with gr.Column() as download_tab:
        gr.Markdown("## 股票数据下载")
        
        with gr.Row():
            download_list_btn = gr.Button("下载股票列表")
            download_data_btn = gr.Button("下载交易数据")
        
        output_info = gr.Textbox(label="下载状态", interactive=False)
        
        # 添加数据查看功能
        gr.Markdown("## 数据查看")
        with gr.Row():
            view_list_btn = gr.Button("查看股票列表")
            view_daily_btn = gr.Button("查看交易数据")
        
        data_view = gr.Textbox(label="数据预览", interactive=False, lines=10)
        
        # 初始化数据库
        init_database()
        
        # 绑定按钮事件
        download_list_btn.click(fn=download_stock_list, outputs=output_info)
        download_data_btn.click(fn=download_stock_data, outputs=output_info)
        view_list_btn.click(fn=view_stock_list, outputs=data_view)
        view_daily_btn.click(fn=view_stock_daily, outputs=data_view)
    
    return download_tab