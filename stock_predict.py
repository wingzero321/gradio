 import gradio as gr
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from datetime import datetime, timedelta

def get_stock_choices():
    # 从数据库获取股票列表
    conn = sqlite3.connect('stock.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code, name FROM stock_daily ORDER BY code")
    stocks = cursor.fetchall()
    conn.close()
    return [f"{code} {name}" for code, name in stocks]

def predict_stock(stock_code):
    # 连接数据库
    conn = sqlite3.connect('stock.db')
    cursor = conn.cursor()
    
    # 提取股票代码
    stock_code = stock_code.split()[0]
    
    # 获取历史数据
    query = """
            SELECT trade_date, close 
            FROM stock_daily 
            WHERE code = ? 
            ORDER BY trade_date ASC
            """
    cursor.execute(query, (stock_code,))
    data = cursor.fetchall()
    conn.close()
    
    # 检查数据量
    min_required_data = 60
    if len(data) < min_required_data:
        return f"错误：当前数据量（{len(data)}条）不足以进行预测。需要至少{min_required_data}条数据。"
    
    # 数据预处理
    dates = [row[0] for row in data]
    prices = np.array([row[1] for row in data]).reshape(-1, 1)
    
    # 归一化数据
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prices)
    
    # 准备训练数据
    sequence_length = 60  # 使用过去60天的数据
    future_days = 7     # 预测未来7天
    
    X = []
    y = []
    
    for i in range(len(data) - sequence_length - future_days + 1):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length:i + sequence_length + future_days])
    
    X = np.array(X)
    y = np.array(y)
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # 构建模型
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(25),
        Dense(7)  # 预测未来7天
    ])
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    
    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    # 预测未来7天
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    
    future_pred = model.predict(last_sequence)
    future_pred = scaler.inverse_transform(future_pred)
    
    # 生成未来日期
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=x+1)).strftime('%Y-%m-%d') for x in range(7)]
    
    # 创建结果数据框
    results = []
    for date, price in zip(future_dates, future_pred[0]):
        results.append([date, round(float(price), 2)])
    
    return results

def create_predict_tab():
    with gr.Column() as predict_tab:
        gr.Markdown("## 股票预测")
        stock_choices = get_stock_choices()
        stock_selector = gr.Dropdown(
            choices=stock_choices, 
            label="选择股票",
            info="请选择要预测的股票"
        )
        predict_btn = gr.Button("预测未来7天走势", variant="primary")
        output = gr.DataFrame(
            headers=["日期", "预测收盘价"],
            label="预测结果",
            info="注意：预测结果仅供参考，不构成投资建议"
        )
        
        # 添加说明信息
        gr.Markdown("""
        ### 使用说明
        1. 从下拉列表选择要预测的股票
        2. 点击"预测未来7天走势"按钮
        3. 系统将显示未来7天的预测收盘价
        
        ### 注意事项
        - 预测基于历史数据，不考虑新闻、政策等外部因素
        - 预测结果仅供参考，实际投资请谨慎决策
        - 至少需要60天的历史数据才能进行预测
        """)
        
        predict_btn.click(
            fn=predict_stock,
            inputs=[stock_selector],
            outputs=[output]
        )
        
        return predict_tab

if __name__ == "__main__":
    with gr.Blocks() as demo:
        create_predict_tab()
    demo.launch()