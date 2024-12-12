import gradio as gr
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import time

def get_stock_choices():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT code FROM stock_daily ORDER BY code")
    stocks = cursor.fetchall()
    conn.close()
    return [code[0] for code in stocks]

def predict_stock(stock_code):
    # 连接数据库
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # 提取股票代码
    stock_code = stock_code.split()[0]
    
    # 获取历史数据
    query = """
            SELECT trade_date, open, close 
            FROM stock_daily 
            WHERE code = ? 
            ORDER BY trade_date DESC
            LIMIT 7
            """
    cursor.execute(query, (stock_code,))
    historical_data = cursor.fetchall()
    
    # 获取预测所需的收盘价数据
    query = """
            SELECT trade_date, close 
            FROM stock_daily 
            WHERE code = ? 
            ORDER BY trade_date ASC
            """
    cursor.execute(query, (stock_code,))
    prediction_data = cursor.fetchall()
    conn.close()
    
    # 检查数据量
    sequence_length = 50
    future_days = 3
    
    if len(prediction_data) < sequence_length + future_days:
        return [["错误", f"数据量不足（当前{len(prediction_data)}条），需要至少{sequence_length + future_days}条数据"]]
    
    # 预测未来价格的代码保持不变
    dates = [row[0] for row in prediction_data]
    prices = np.array([float(row[1]) for row in prediction_data]).reshape(-1, 1)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prices)
    
    X = []
    y = []
    
    for i in range(len(scaled_data) - sequence_length - future_days + 1):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length:i + sequence_length + future_days])
    
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(25),
        Dense(3)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.05, verbose=0)
    
    last_sequence = scaled_data[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    
    future_pred = model.predict(last_sequence)
    future_pred = scaler.inverse_transform(future_pred)
    
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=x+1)).strftime('%Y-%m-%d') for x in range(3)]
    
    # 准备结果数据
    results = []
    
    # 添加历史数据
    for date, open_price, close_price in historical_data:
        results.append([
            date,
            round(float(open_price), 2),
            round(float(close_price), 2),
            "历史"
        ])
    
    # 添加预测数据
    for date, price in zip(future_dates, future_pred[0]):
        results.append([
            date,
            None,  # 开盘价为空
            round(float(price), 2),
            "预测"
        ])
    
    return results

def create_predictions_table():
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # 删除表（如果存在）
        cursor.execute("DROP TABLE IF EXISTS stock_predictions")
        
        # 创建预测结果表，添加最新收盘价和涨跌幅字段
        cursor.execute("""
        CREATE TABLE stock_predictions (
            code TEXT,
            predict_date TEXT,
            predicted_close REAL,
            last_close REAL,
            change_rate REAL,
            prediction_time TEXT,
            PRIMARY KEY (code, predict_date)
        )
        """)
        
        conn.commit()
        print("成功创建预测结果表")
    except Exception as e:
        print(f"创建表时出错: {str(e)}")
    finally:
        conn.close()

def batch_predict_all_stocks():
    try:
        # 确保表存在
        create_predictions_table()
        
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        
        # 获取前10个股票代码
        cursor.execute("SELECT DISTINCT code FROM stock_daily ORDER BY code LIMIT 10")
        all_stocks = cursor.fetchall()
        
        start_time = time.time()
        total_stocks = len(all_stocks)
        processed = 0
        failed = []
        
        print(f"开始处理前{total_stocks}支股票...")
        prediction_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for stock in all_stocks:
            stock_code = stock[0]
            print(f"正在处理股票: {stock_code}")
            try:
                # 获取历史数据
                cursor.execute("""
                    SELECT trade_date, close 
                    FROM stock_daily 
                    WHERE code = ? 
                    ORDER BY trade_date ASC
                    """, (stock_code,))
                data = cursor.fetchall()
                
                if len(data) < 53:
                    failed.append((stock_code, "数据量不足"))
                    print(f"股票 {stock_code} 数据量不足，跳过")
                    continue
                
                # 获取最新收盘价
                last_close = float(data[-1][1])
                
                dates = [row[0] for row in data]
                prices = np.array([float(row[1]) for row in data]).reshape(-1, 1)
                
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(prices)
                
                sequence_length = 50
                X = []
                y = []
                
                for i in range(len(scaled_data) - sequence_length - 3 + 1):
                    X.append(scaled_data[i:(i + sequence_length)])
                    y.append(scaled_data[i + sequence_length:i + sequence_length + 3])
                
                X = np.array(X)
                y = np.array(y)
                
                from tensorflow.keras.layers import Input
                from tensorflow.keras.models import Model
                
                input_layer = Input(shape=(sequence_length, 1))
                lstm1 = LSTM(50, activation='relu', return_sequences=True)(input_layer)
                lstm2 = LSTM(50, activation='relu')(lstm1)
                dense1 = Dense(25)(lstm2)
                output_layer = Dense(3)(dense1)
                
                model = Model(inputs=input_layer, outputs=output_layer)
                model.compile(optimizer='adam', loss='mse')
                
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                last_sequence = scaled_data[-sequence_length:]
                last_sequence = last_sequence.reshape((1, sequence_length, 1))
                future_pred = model.predict(last_sequence, verbose=0)
                future_pred = scaler.inverse_transform(future_pred)
                
                last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
                future_dates = [(last_date + timedelta(days=x+1)).strftime('%Y-%m-%d') for x in range(3)]
                
                # 存储预测结果，包括最新收盘价和涨跌幅
                for date, price in zip(future_dates, future_pred[0]):
                    # 计算涨跌幅
                    change_rate = ((float(price) - last_close) / last_close) * 100
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO stock_predictions 
                        (code, predict_date, predicted_close, last_close, change_rate, prediction_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (stock_code, date, float(price), last_close, change_rate, prediction_time))
                
                processed += 1
                print(f"成功处理股票 {stock_code}")
                print(f"最新收盘价: {last_close:.2f}")
                print(f"预测价格: {future_pred[0][0]:.2f}, 涨跌幅: {((future_pred[0][0] - last_close) / last_close * 100):.2f}%")
                conn.commit()
                    
            except Exception as e:
                failed.append((stock_code, str(e)))
                print(f"处理股票 {stock_code} 时出错: {str(e)}")
                
        conn.commit()
        
    except Exception as e:
        print(f"批量处理时出错: {str(e)}")
        return {"error": str(e)}
    finally:
        conn.close()
    
    elapsed_time = time.time() - start_time
    
    result = {
        "processed": processed,
        "failed": len(failed),
        "time": round(elapsed_time/60, 2),
        "failed_stocks": failed,
        "success_stocks": [stock[0] for stock in all_stocks if stock[0] not in [f[0] for f in failed]]
    }
    
    print("\n处理完成!")
    print(f"成功处理: {result['processed']} 支股票")
    print(f"失败: {result['failed']} 支股票")
    print(f"总耗时: {result['time']} 分钟")
    
    return result

def create_predict_tab():
    with gr.Column() as predict_tab:
        gr.Markdown("## 股票预测")
        
        with gr.Row():
            # 单只股票预测
            with gr.Column():
                stock_choices = get_stock_choices()
                stock_selector = gr.Dropdown(
                    choices=stock_choices, 
                    label="选择股票",
                    info="请选择要预测的股票"
                )
                predict_btn = gr.Button("预测未来3天走势", variant="primary")
                output = gr.DataFrame(
                    headers=["日期", "开盘价", "收盘价", "类型"],
                    label="历史数据与预测结果"
                )
            
            # 批量预测
            with gr.Column():
                batch_predict_btn = gr.Button("批量预测所有股票", variant="secondary")
                batch_output = gr.JSON(label="批量预测结果")
        
        # 添加说明信息
        gr.Markdown("""
        ### 使用说明
        1. 单只股票预测：
           - 从下拉列表选择要预测的股票
           - 点击"预测未来3天走势"按钮
           - 显示过去7天数据和未来3天预测
        
        2. 批量预测：
           - 点击"批量预测所有股票"按钮
           - 系统将处理所有股票并保存预测结果
           - 预测结果将保存到数据库中
        
        ### 注意事项
        - 预测基于历史数据，不考虑新闻、政策等外部因素
        - 预测结果仅供参考，实际投资请谨慎决策
        - 至少需要60天的历史数据才能进行预测
        - 批量预测可能需要较长时间，请耐心等待
        """)
        
        predict_btn.click(
            fn=predict_stock,
            inputs=[stock_selector],
            outputs=[output]
        )
        
        batch_predict_btn.click(
            fn=batch_predict_all_stocks,
            inputs=[],
            outputs=[batch_output]
        )
        
        return predict_tab

if __name__ == "__main__":
    create_predictions_table()  # 确保表存在
    with gr.Blocks() as demo:
        create_predict_tab()
    demo.launch()