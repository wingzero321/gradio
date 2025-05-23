import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gradio as gr

def calculate_kdj(df, n=9, m1=3, m2=3):
    """计算单个股票的KDJ值"""
    df = df.sort_values('trade_date')  # 确保按日期排序
    
    # 计算RSV
    low_list = df['low'].rolling(window=n, min_periods=1).min()
    high_list = df['high'].rolling(window=n, min_periods=1).max()
    
    rsv = np.where(high_list == low_list,
                   50,
                   (df['close'] - low_list) / (high_list - low_list) * 100)
    
    # 计算K值
    k = pd.Series(50.0, index=df.index)
    for i in range(1, len(df)):
        k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
    
    # 计算D值
    d = pd.Series(50.0, index=df.index)
    for i in range(1, len(df)):
        d[i] = (2/3) * d[i-1] + (1/3) * k[i]
    
    # 计算J值
    j = 3 * k - 2 * d
    
    return k.iloc[-1], d.iloc[-1], j.iloc[-1]

def find_oversold_stocks():
    """查找超跌股票（J值≤-10）"""
    try:
        conn = sqlite3.connect('stock_data.db')
        
        # 获取所有股票代码
        stocks = pd.read_sql("SELECT DISTINCT code FROM stock_daily", conn)
        
        results = []
        
        # 获取当前日期
        now = datetime.now()
        date_limit = (now - timedelta(days=30)).strftime('%Y-%m-%d')  # 获取近30天数据
        
        # 遍历每只股票计算KDJ
        for _, row in stocks.iterrows():
            code = row['code']
            
            # 获取该股票的最近交易数据
            query = """
            SELECT code, trade_date, open, high, low, close, volume
            FROM stock_daily 
            WHERE code = ? AND trade_date >= ?
            ORDER BY trade_date
            """
            df = pd.read_sql(query, conn, params=(code, date_limit))
            
            if len(df) >= 9:  # 确保有足够数据计算KDJ
                k, d, j = calculate_kdj(df)
                
                if j <= -10:  # 筛选J值小于等于-10的股票
                    # 获取股票名称
                    stock_name = pd.read_sql(
                        "SELECT name FROM stocks WHERE code = ?",
                        conn,
                        params=(code,)
                    )
                    name = stock_name['name'].iloc[0] if not stock_name.empty else "未知"
                    
                    results.append({
                        '股票代码': code,
                        '股票名称': name,
                        'K值': round(k, 2),
                        'D值': round(d, 2),
                        'J值': round(j, 2),
                        '收盘价': round(df['close'].iloc[-1], 2),
                        '交易日期': df['trade_date'].iloc[-1]
                    })
        
        conn.close()
        
        # 按J值从小到大排序
        results = sorted(results, key=lambda x: x['J值'])
        
        # 转换为DataFrame
        if results:
            df_results = pd.DataFrame(results)
            return df_results
        else:
            # 返回空DataFrame但保持列名
            return pd.DataFrame(columns=['股票代码', '股票名称', 'K值', 'D值', 'J值', '收盘价', '交易日期'])
            
    except Exception as e:
        print(f"查询失败：{str(e)}")
        return pd.DataFrame()

def create_discovery_tab():
    """创建股票发现标签页"""
    with gr.Column() as discovery_tab:
        gr.Markdown("## 超跌股票发现")
        gr.Markdown("### 查找J值≤-10的潜在超跌股票")
        
        discover_btn = gr.Button("开始发现", variant="primary")
        result_output = gr.DataFrame(
            headers=['股票代码', '股票名称', 'K值', 'D值', 'J值', '收盘价', '交易日期'],
            label="发现结果",
            interactive=False
        )
        
        # 绑定按钮事件
        discover_btn.click(
            fn=find_oversold_stocks,
            outputs=result_output
        )
    
    return discovery_tab
