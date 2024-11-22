# 股票数据下载与分析系统

这是一个基于 Python 的股票数据下载与分析系统，使用 Gradio 构建用户界面，通过 AKShare 获取股票数据。

## 功能特点

1. 数据下载
   - 获取上海证券交易所的股票列表
   - 下载股票最近90天的交易数据
   - 支持中断下载操作
   - 实时显示下载进度

2. 数据存储
   - 使用 SQLite 数据库存储数据
   - 股票基本信息表 (stocks)
   - 股票日线数据表 (stock_daily)

3. 数据查看
   - 查看股票列表
   - 查看交易数据

## 安装依赖
bash
pip install akshare pandas gradio
## 数据库结构

1. stocks 表（股票基本信息）
   - code: 股票代码
   - name: 股票名称
   - update_time: 更新时间

2. stock_daily 表（股票交易数据）
   - code: 股票代码
   - trade_date: 交易日期
   - open: 开盘价
   - high: 最高价
   - low: 最低价
   - close: 收盘价
   - volume: 成交量
   - amount: 成交额

## 使用说明

1. 启动程序
2. 数据下载步骤
   - 点击"下载股票列表"获取最新的股票信息
   - 点击"下载交易数据"获取90天交易数据
   - 如需中断下载，点击"中断下载"按钮

3. 数据查看
   - 点击"查看股票列表"查看已下载的股票信息
   - 点击"查看交易数据"查看已下载的交易数据

## 注意事项

1. 首次运行会自动创建数据库和相关表
2. 下载交易数据可能需要较长时间
3. 确保网络连接正常
4. 数据来源为 AKShare 接口

## 技术栈

- Python 3.x
- Gradio：构建Web界面
- AKShare：获取股票数据
- SQLite：数据存储
- Pandas：数据处理
## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

MIT License