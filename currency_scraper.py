import requests
import pandas as pd
from datetime import datetime, timedelta
from odps import ODPS
import os

def get_yesterday_partition():
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime('%Y%m%d')

def fetch_exchange_rates():
    # 使用 USD 作为基准货币
    base_currency = "USD"
    url = f"https://open.er-api.com/v6/latest/{base_currency}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print("无法获取数据，状态码:", response.status_code)
        return
    
    data = response.json()
    
    if data["result"] == "success":
        rates = data["rates"]
        timestamp = datetime.fromtimestamp(data["time_last_update_unix"]).strftime('%Y-%m-%d %H:%M:%S')
        return rates, timestamp
    else:
        print("API 返回错误")
        return None, None

def save_to_maxcompute(rates, timestamp):
    if rates is None:
        return
        
    # 配置 MaxCompute 连接
    access_id = os.getenv('ODPS_ACCESS_ID')
    access_key = os.getenv('ODPS_ACCESS_KEY')
    project = os.getenv('ODPS_PROJECT')
    endpoint = os.getenv('ODPS_ENDPOINT')
    
    if not all([access_id, access_key, project, endpoint]):
        print("请设置 MaxCompute 连接所需的环境变量")
        return
        
    odps = ODPS(access_id, access_key, project, endpoint=endpoint)
    table = odps.get_table('dwd_fetch_exchange_rates_i_d')
    
    # 准备分区信息
    partition = get_yesterday_partition()
    partition_spec = f'log_date={partition}'
    
    # 检查并删除已存在的分区数据
    if table.exist_partition(partition_spec):
        print(f"删除已存在的分区数据: {partition_spec}")
        table.delete_partition(partition_spec, if_exists=True)
    
    # 使用 table.new_record() 创建记录
    with table.open_writer(partition=f'log_date={partition}', create_partition=True) as writer:
        for currency, rate in rates.items():
            record = table.new_record()
            record['currency'] = currency
            record['exchange_rate'] = float(rate)
            record['updated_time'] = timestamp
            writer.write(record)
    
    print(f"数据已写入 MaxCompute 表 dwd_fetch_exchange_rates_i_d, 分区: {partition}")

def save_to_csv(rates, timestamp, filename='exchange_rates.csv'):
    if rates is None:
        return
        
    df = pd.DataFrame(rates.items(), columns=['Currency', 'Exchange Rate'])
    df['Updated Time'] = timestamp
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"汇率信息已保存到 {filename}")

if __name__ == "__main__":
    rates, timestamp = fetch_exchange_rates()
    if rates:
        save_to_csv(rates, timestamp)
        save_to_maxcompute(rates, timestamp)