import gradio as gr
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
        
        if df.empty:
            return None, "No data available"
        
        # 确保日期索引
        if not isinstance(df.index, pd.DatetimeIndex):
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
        title = f'Stock: {stock_code} ({stock_name})\nClose: {last_close} Volume: {last_volume}K\nKDJ(9,3,3) K:{last_k} D:{last_d} J:{last_j}'
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
        ax1.set_xticklabels(df.index.strftime('%m-%d'), rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_ylabel('Price')
        
        # 成交量图
        ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=1)
        ax2.set_title('Volume (K)', fontsize=10, pad=10)
        
        for i in range(len(df)):
            color = 'red' if df['收盘'].iloc[i] >= df['开盘'].iloc[i] else 'green'
            ax2.bar(i, df['成交量'].iloc[i]/10000, color=color, width=0.8)
        
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df.index.strftime('%m-%d'), rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_ylabel('Volume(K)')
        
        # KDJ图
        ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1)
        ax3.set_title('KDJ Indicator', fontsize=10, pad=10)
        
        ax3.plot(range(len(df)), k, 'b-', label='K', linewidth=1)
        ax3.plot(range(len(df)), d, 'y-', label='D', linewidth=1)
        ax3.plot(range(len(df)), j, 'r-', label='J', linewidth=1)
        
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df.index.strftime('%m-%d'), rotation=45)
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

def get_stock_list():
    """获取股票列表"""
    try:
        # 获取A股列表
        stock_list = ak.stock_zh_a_spot_em()
        # 构建选项列表：代码 - 名称
        choices = [f"{row['代码']} - {row['名称']}" for _, row in stock_list.iterrows()]
        # 添加上证指数
        choices.insert(0, "000001 - 上证指数")
        return choices
    except Exception as e:
        print(f"Error getting stock list: {str(e)}")
        return ["000001 - 上证指数"]  # 返回默认选项

def create_query_tab():
    """创建股票查询标签页"""
    with gr.Column() as query_tab:
        gr.Markdown("### Real-time Stock Analysis")
        
        # 获取股票列表
        stock_choices = get_stock_list()
        
        with gr.Row():
            with gr.Column(scale=4):
                # 修改下拉菜单参数
                input_code = gr.Dropdown(
                    choices=stock_choices,
                    value=stock_choices[0],  # 默认选择第一个（上证指数）
                    label="Select Stock",
                    interactive=True,
                    filterable=True  # 确保只使用 filterable，不要使用 search_keywords
                )
            with gr.Column(scale=1):
                query_btn = gr.Button("Query", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=4):
                plot_output = gr.Plot(label="Stock Chart")
            with gr.Column(scale=1):
                info_output = gr.Textbox(
                    label="Stock Info",
                    lines=6,
                    show_label=True
                )
        
        # 修改查询函数以处理下拉菜单的值
        def query_stock(selected):
            # 从选择的字符串中提取股票代码
            stock_code = selected.split(' - ')[0]
            return plot_stock_kline(stock_code)
        
        # 绑定事件
        input_code.change(
            fn=query_stock,
            inputs=input_code,
            outputs=[plot_output, info_output]
        )
        query_btn.click(
            fn=query_stock,
            inputs=input_code,
            outputs=[plot_output, info_output]
        )
    
    return query_tab