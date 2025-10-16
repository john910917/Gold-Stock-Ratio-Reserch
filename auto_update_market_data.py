import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import time

class MarketDataUpdater:
    """市場數據自動更新器"""
    
    def __init__(self):
        # 定義要爬取的指數
        self.indices = {
            '^GSPC': '^GSPC_data.csv',  # S&P 500
            '^IXIC': '^IXIC_data.csv',  # NASDAQ
            '^DJI': '^DJI_data.csv'  ,    # Dow Jones
            'GC=F' : '黃金期貨歷史數據75~25106.csv' # 黃金期貨
        }
        
        # 只保留這些欄位
        self.required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # 數據起始日期
        self.start_date = '1995-01-01'
    
    def get_last_date(self, filename):
        """獲取CSV文件中的最後日期"""
        if not os.path.exists(filename):
            return None
        
        try:
            df = pd.read_csv(filename)
            if len(df) > 0 and 'Date' in df.columns:
                last_date = pd.to_datetime(df['Date'].iloc[-1])
                return last_date
            return None
        except Exception as e:
            print(f"讀取 {filename} 錯誤: {e}")
            return None
    
    def download_data(self, symbol, start_date, end_date):
        """下載指定期間的數據"""
        try:
            print(f"正在下載 {symbol} 從 {start_date} 到 {end_date}...")
            
            # 下載數據
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if len(df) == 0:
                print(f"  [!] 沒有新數據")
                return None
            
            # 重置索引，將Date作為列
            df.reset_index(inplace=True)
            
            # 只保留需要的欄位
            available_columns = [col for col in self.required_columns if col in df.columns]
            df = df[available_columns]
            
            # 格式化日期
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            print(f"  [+] 成功下載 {len(df)} 筆記錄")
            return df
            
        except Exception as e:
            print(f"  [!] 下載 {symbol} 失敗: {e}")
            return None
    
    def update_csv(self, symbol, filename):
        """更新CSV文件"""
        print(f"\n{'='*60}")
        print(f"更新 {symbol}")
        print(f"{'='*60}")
        
        # 獲取最後日期
        last_date = self.get_last_date(filename)
        
        if last_date is None:
            # 文件不存在或為空，下載全部數據
            print(f"文件不存在，下載完整歷史數據...")
            start_date = self.start_date
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            df_new = self.download_data(symbol, start_date, end_date)
            
            if df_new is not None:
                df_new.to_csv(filename, index=False)
                print(f"[+] 已創建新文件: {filename}")
                print(f"    數據期間: {df_new['Date'].iloc[0]} 到 {df_new['Date'].iloc[-1]}")
                print(f"    總記錄數: {len(df_new)}")
        else:
            # 文件存在，只下載新數據
            print(f"現有數據最後日期: {last_date.strftime('%Y-%m-%d')}")
            
            # 計算需要更新的起始日期（最後日期的下一天）
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 檢查是否需要更新
            if start_date >= end_date:
                print(f"[+] 數據已是最新，無需更新")
                return
            
            print(f"下載新數據: {start_date} 到 {end_date}")
            
            df_new = self.download_data(symbol, start_date, end_date)
            
            if df_new is not None and len(df_new) > 0:
                # 讀取現有數據
                df_existing = pd.read_csv(filename)
                
                # 標準化現有數據的日期格式
                df_existing['Date'] = pd.to_datetime(df_existing['Date'], utc=True, errors='coerce')
                df_existing['Date'] = df_existing['Date'].dt.strftime('%Y-%m-%d')
                
                # 合併數據
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # 去重（基於Date）
                df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
                
                # 排序
                df_combined['Date'] = pd.to_datetime(df_combined['Date'])
                df_combined = df_combined.sort_values('Date')
                df_combined['Date'] = df_combined['Date'].dt.strftime('%Y-%m-%d')
                
                # 保存
                df_combined.to_csv(filename, index=False)
                
                print(f"[+] 已更新文件: {filename}")
                print(f"    新增記錄: {len(df_new)}")
                print(f"    總記錄數: {len(df_combined)}")
                print(f"    數據期間: {df_combined['Date'].iloc[0]} 到 {df_combined['Date'].iloc[-1]}")
    
    def update_all(self):
        """更新所有指數數據"""
        print("="*60)
        print("市場數據自動更新系統")
        print(f"更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for symbol, filename in self.indices.items():
            try:
                self.update_csv(symbol, filename)
                success_count += 1
                time.sleep(1)  # 避免請求過快
            except Exception as e:
                print(f"[!] 更新 {symbol} 時發生錯誤: {e}")
                fail_count += 1
        
        print(f"\n{'='*60}")
        print("更新完成")
        print(f"{'='*60}")
        print(f"成功: {success_count} 個指數")
        print(f"失敗: {fail_count} 個指數")
        print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def check_data_status(self):
        """檢查所有數據文件的狀態"""
        print("="*60)
        print("數據狀態檢查")
        print("="*60)
        
        for symbol, filename in self.indices.items():
            print(f"\n{symbol}:")
            
            if not os.path.exists(filename):
                print(f"  狀態: 文件不存在")
                print(f"  建議: 需要下載完整數據")
            else:
                df = pd.read_csv(filename)
                last_date = pd.to_datetime(df['Date'].iloc[-1])
                # 確保都是naive datetime
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                days_old = (datetime.now() - last_date).days
                
                print(f"  文件: {filename}")
                print(f"  記錄數: {len(df)}")
                print(f"  起始日期: {df['Date'].iloc[0]}")
                print(f"  最後日期: {df['Date'].iloc[-1]}")
                print(f"  數據延遲: {days_old} 天")
                
                if days_old > 1:
                    print(f"  狀態: [!] 需要更新")
                else:
                    print(f"  狀態: [+] 最新")

def main():
    """主函數"""
    import sys
    
    updater = MarketDataUpdater()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'check':
            # 檢查數據狀態
            updater.check_data_status()
        elif command == 'update':
            # 更新數據
            updater.update_all()
        else:
            print("未知命令")
            print("使用方式:")
            print("  python auto_update_market_data.py check   - 檢查數據狀態")
            print("  python auto_update_market_data.py update  - 更新數據")
    else:
        # 默認執行更新
        updater.update_all()

if __name__ == "__main__":
    main()
