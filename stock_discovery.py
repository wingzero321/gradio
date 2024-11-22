import gradio as gr
import akshare as ak
import pandas as pd

def discover_good_stocks():
    """发现优质股票"""
    try:
        stocks = ak.stock_zh_a_spot_em()
        filtered_stocks = stocks[
            (stocks['市盈率'] > 0) & 
            (stocks['市盈率'] < 30) &
            (stocks['市净率'] > 0) &
            (stocks['市净率'] < 5)
        ]
        result_df = filtered_stocks[['代码', '名称', '最新价', '市盈率', '市净率', '总市值']].head(20)
        return result_df
    except Exception as e:
        return pd.DataFrame()

def create_discovery_tab():
    """创建股票发现标签页"""
    gr.Markdown("""
    ### Quality Stock Discovery
    Discover potential investment opportunities based on:
    - Financial Indicators
    - Technical Analysis
    - Market Sentiment
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Filtering Criteria")
            with gr.Row():
                pe_min = gr.Number(label="Min P/E", value=0)
                pe_max = gr.Number(label="Max P/E", value=50)
            with gr.Row():
                pb_min = gr.Number(label="Min P/B", value=0)
                pb_max = gr.Number(label="Max P/B", value=10)
            with gr.Row():
                market_cap_min = gr.Number(label="Min Market Cap (Billion)", value=0)
                market_cap_max = gr.Number(label="Max Market Cap (Billion)", value=1000)
            
            discover_btn = gr.Button("Start Discovery", variant="primary")
    
    with gr.Row():
        discovery_output = gr.Dataframe(
            headers=["Code", "Name", "Price", "P/E", "P/B", "Market Cap"],
            label="Discovery Results"
        )
    
    discover_btn.click(
        fn=discover_good_stocks,
        outputs=discovery_output
    )