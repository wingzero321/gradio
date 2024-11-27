import gradio as gr
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sqlite3

def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ计算函数"""
    try:
        # 计算RSV
        low_list = low.rolling(window=n, min_periods=1).min()
        high_list = high.rolling(window=n, min_periods=1).max()
        
        # 初始化序列
        rsv = pd.Series(index=close.index, dtype=float)
        k = pd.Series(index=close.index, dtype=float)
        d = pd.Series(index=close.index, dtype=float)
        j = pd.Series(index=close.index, dtype=float)
        
        # 计算RSV
        for i in range(len(close)):
            if high_list.iloc[i] != low_list.iloc[i]:
                rsv.iloc[i] = (close.iloc[i] - low_list.iloc[i]) / (high_list.iloc[i] - low_list.iloc[i]) * 100
            else:
                rsv.iloc[i] = rsv.iloc[i-1] if i > 0 else 50.0
        
        # 计算KDJ
        k.iloc[0] = 50.0
        d.iloc[0] = 50.0
        
        for i in range(1, len(close)):
            k.iloc[i] = (2.0 * k.iloc[i-1] + rsv.iloc[i]) / 3.0
            d.iloc[i] = (2.0 * d.iloc[i-1] + k.iloc[i]) / 3.0
            j.iloc[i] = 3.0 * k.iloc[i] - 2.0 * d.iloc[i]
        
        return k, d, j
    except Exception as e:
        print(f"KDJ calculation error: {str(e)}")
        return pd.Series(50, index=close.index), pd.Series(50, index=close.index), pd.Series(50, index=close.index)

def plot_stock_kline(stock_code="000001"):
    """K线图绘制函数"""
    try:
        if not stock_code:
            return None, "Please input stock code"
        
        stock_code = stock_code.strip()
        
        # 获取日期范围
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        # 获取股票名称和数据
        if stock_code == "000001":
            stock_name = "SSE Index"
            df = ak.stock_zh_index_daily(symbol="sh000001")
            # 只保留需要的时间范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]
        else:
            stock_info = ak.stock_zh_a_spot_em()
            stock_name = stock_info[stock_info['代码'] == stock_code]['名称'].values[0]
            df = ak.stock_zh_a_hist(symbol=stock_code, 
                                   start_date=start_date,
                                   end_date=end_date,
                                   adjust="qfq")
            print(df)
        
        if df.empty:
            return None, "No data available"
        
        # 确保日期索引格式正确
        if not isinstance(df.index, pd.DatetimeIndex):
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        
        # 计算KDJ
        k, d, j = calculate_kdj(df['最高'], df['最低'], df['收盘'])
        
        # 获取最后一天的数据
        last_date = df.index[-1].strftime('%Y-%m-%d')
        last_k = round(float(k.iloc[-1]), 2)
        last_d = round(float(d.iloc[-1]), 2)
        last_j = round(float(j.iloc[-1]), 2)
        last_close = round(float(df['收盘'].iloc[-1]), 2)
        last_volume = round(float(df['成交量'].iloc[-1])/10000, 2)
        
        # 创建图表
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(12, 10))
        
        # K线图
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        title = f'Stock: {stock_code} \nClose: {last_close} Volume: {last_volume}K\nKDJ(9,3,3) K:{last_k} D:{last_d} J:{last_j}'
        ax1.set_title(title, fontsize=12, pad=15)
        
        # 绘制K线
        for i in range(len(df)):
            color = 'red' if df['收盘'].iloc[i] >= df['开盘'].iloc[i] else 'green'
            ax1.bar(i, df['收盘'].iloc[i] - df['开盘'].iloc[i],
                   bottom=df['开盘'].iloc[i],
                   color=color,
                   width=0.8)
            ax1.plot([i, i], [df['最低'].iloc[i], df['最高'].iloc[i]],
                    color=color,
                    linewidth=1)
        
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([d.strftime('%Y-%m-%d') for d in df.index], rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_ylabel('Price')
        
        # 成交量图
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=1)
        ax2.set_title('Volume (K)', fontsize=10, pad=10)
        
        for i in range(len(df)):
            color = 'red' if df['收盘'].iloc[i] >= df['开盘'].iloc[i] else 'green'
            ax2.bar(i, df['成交量'].iloc[i]/10000, color=color, width=0.8)
        
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([d.strftime('%Y-%m-%d') for d in df.index], rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_ylabel('Volume(K)')
        
        # KDJ图
        ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
        ax3.set_title('KDJ Indicator', fontsize=10, pad=10)
        
        ax3.plot(range(len(df)), k, 'b-', label='K', linewidth=1)
        ax3.plot(range(len(df)), d, 'y-', label='D', linewidth=1)
        ax3.plot(range(len(df)), j, 'r-', label='J', linewidth=1)
        
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels([d.strftime('%Y-%m-%d') for d in df.index], rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='upper right')
        ax3.set_ylabel('Value')
        
        plt.tight_layout()
        
        # 准备状态信息
        status_msg = f"""
        Stock: {stock_code} ({stock_name})
        Date: {last_date}
        Close: {last_close}
        Volume: {last_volume}K
        KDJ Index:
        - K: {last_k}
        - D: {last_d}
        - J: {last_j}
        """
        
        # 确保返回两个值：图表和状态信息
        return fig, status_msg
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # 出错时也要返回两个值
        return None, f"Error: {str(e)}"

def get_stock_choices():
    """从数据库获取股票列表"""
    try:
        conn = sqlite3.connect('stock_data.db')
        df = pd.read_sql("SELECT code, name FROM stocks", conn)
        conn.close()
        
        # 将代码和名称组合成选项列表：格式如 "600000 浦发银行"
        choices = [f"{row['code']} {row['name']}" for _, row in df.iterrows()]
        if not choices:
            choices = ["请先下载股票数据"]
        return choices
    except Exception as e:
        print(f"获取股票列表失败：{str(e)}")
        return ["数据库错误，请检查数据下载"]

def create_query_tab():
    """创建股票查询标签页"""
    with gr.Column() as query_tab:
        gr.Markdown("## 股票数据查询")
        
        with gr.Row():
            with gr.Column(scale=4):
                stock_choices = get_stock_choices()
                input_code = gr.Dropdown(
                    choices=stock_choices,
                    value=None,
                    label="选择股票",
                    interactive=True,
                    filterable=True
                )
            
            with gr.Column(scale=2):
                # 移除 interactive 参数的显示
                query_btn = gr.Button("查询", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=4):
                plot_output = gr.Plot(label="Stock Chart")
            with gr.Column(scale=1):
                info_output = gr.Textbox(
                    label="Stock Info",
                    lines=6,
                    show_label=True
                )
        
        def query_stock(selected):
            if not selected or selected in ["请先下载股票数据", "数据库错误，请检查数据下载"]:
                return None, "请先选择有效的股票"
            stock_code = selected.split()[0]
            return plot_stock_kline(stock_code)
        
        # 使用 lambda 函数直接控制按钮状态
        def update_button_status(value):
            if not value or value in ["请先下载股票数据", "数据库错误，请检查数据下载"]:
                return gr.Button(interactive=False)
            return gr.Button(interactive=True)
        
        # 绑定事件
        input_code.change(
            fn=update_button_status,
            inputs=input_code,
            outputs=query_btn
        )
        
        query_btn.click(
            fn=query_stock,
            inputs=input_code,
            outputs=[plot_output, info_output]
        )
    
    return query_tab

def query_stock_data(stock_code):
    """查询股票数据"""
    try:
        # 从选项中提取股票代码（格式："600000 浦发银行" -> "600000"）
        code = stock_code.split()[0]
        
        conn = sqlite3.connect('stock_data.db')
        # 获取该股票的交易数据
        query = """
        SELECT trade_date, open, high, low, close, volume, amount 
        FROM stock_daily 
        WHERE code = ? 
        ORDER BY trade_date DESC
        """
        df = pd.read_sql(query, conn, params=(code,))
        conn.close()
        
        if df.empty:
            return "未找到该股票的交易数据"
            
        # ... 处理数据和绘图的代码 ...
        
    except Exception as e:
        return f"查询失败：{str(e)}"