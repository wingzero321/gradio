import gradio as gr
from stock_query import create_query_tab
from stock_download import create_download_tab
from stock_discovery import create_discovery_tab
from stock_predict import create_predict_tab

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Stock Analysis Platform")
    
    with gr.Tabs() as tabs:
        # 第一个标签页：股票数据获取
        with gr.Tab("Data Download"):
            create_download_tab()
        
        # 第二个标签页：股票查询
        with gr.Tab("Stock Query"):
            create_query_tab()
        
        # 第三个标签页：优质股票发现
        with gr.Tab("Stock Discovery"):
            create_discovery_tab()
        
        # 第四个标签页：股票预测
        with gr.Tab("Stock Prediction"):
            create_predict_tab()

if __name__ == "__main__":
    iface.launch()
