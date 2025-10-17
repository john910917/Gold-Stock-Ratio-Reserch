import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge
import pickle
import warnings
import sys
import io
import re
from datetime import datetime
warnings.filterwarnings('ignore')

# 設置輸出編碼
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MarketRiskIndexCalculator:
    """純市場風險指數計算器（不依賴個人配置）"""
    
    def __init__(self):
        self.state_risk_scores = {
            0: 10,  # 風險偏好期：低風險
            1: 50,  # 謹慎觀望期：中風險
            2: 90   # 避險主導期：高風險
        }
        
        self.state_names = {
            0: '風險偏好期',
            1: '謹慎觀望期',
            2: '避險主導期'
        }
        
        self.state_name_to_id = {
            '風險偏好期': 0,
            '謹慎觀望期': 1,
            '避險主導期': 2
        }
    
    def parse_hmm_report(self, report_path='hmm_latest_report.txt'):
        """
        解析HMM報告文件，提取關鍵數據
        
        Parameters:
        -----------
        report_path : str
            報告文件路徑
        
        Returns:
        --------
        dict : 包含所有必要數據的字典
        """
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {}
            
            # 1. 提取報告日期
            report_date_match = re.search(r'報告日期:\s*(\d{4}-\d{2}-\d{2})', content)
            if report_date_match:
                data['report_date'] = report_date_match.group(1)
            
            # 2. 提取狀態名稱
            state_match = re.search(r'狀態名稱:\s*(.+)', content)
            if state_match:
                state_name = state_match.group(1).strip()
                data['state_name'] = state_name
                data['current_state'] = self.state_name_to_id.get(state_name, 2)
            
            # 3. 提取狀態機率（從機率分布中找到當前狀態）
            # 尋找帶有 "← 當前" 標記的行
            prob_match = re.search(r'([^\n]+):\s+([\d.]+)%\s*←\s*當前', content)
            if prob_match:
                data['state_prob'] = float(prob_match.group(2))
            
            # 4. 提取狀態持續天數
            duration_match = re.search(r'當前狀態已持續:\s*(\d+)\s*個交易日', content)
            if duration_match:
                data['state_duration_days'] = int(duration_match.group(1))
            
            # 5. 提取預警信號數量
            warning_match = re.search(r'活躍預警信號:\s*(\d+)\s*個', content)
            if warning_match:
                data['warning_count'] = int(warning_match.group(1))
            
            # 6. 提取30天價格變化
            gold_change_match = re.search(r'黃金期貨:\s*([+-]?[\d.]+)%', content)
            if gold_change_match:
                data['gold_change_30d'] = float(gold_change_match.group(1))
            
            sp500_change_match = re.search(r'S&P 500:\s*([+-]?[\d.]+)%', content)
            if sp500_change_match:
                data['sp500_change_30d'] = float(sp500_change_match.group(1))
            
            ratio_change_match = re.search(r'黃金/S&P比率:\s*([+-]?[\d.]+)%', content)
            if ratio_change_match:
                data['ratio_change_30d'] = float(ratio_change_match.group(1))
            
            # 7. 驗證所有必要數據是否都已提取
            required_keys = ['current_state', 'state_prob', 'ratio_change_30d', 
                           'gold_change_30d', 'sp500_change_30d', 'warning_count', 
                           'state_duration_days']
            
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"報告解析失敗，缺少以下數據：{missing_keys}")
            
            return data
            
        except FileNotFoundError:
            print(f"❌ 錯誤：找不到報告文件 {report_path}")
            return None
        except Exception as e:
            print(f"❌ 解析報告時發生錯誤：{e}")
            return None
    
    def calculate_state_risk(self, current_state, state_prob):
        """
        計算HMM狀態風險分數（0-100）
        
        Parameters:
        -----------
        current_state : int
            當前狀態 (0=風險偏好, 1=謹慎觀望, 2=避險主導)
        state_prob : float
            當前狀態的機率 (0-100)
        
        Returns:
        --------
        dict : 狀態風險詳情
        """
        base_risk = self.state_risk_scores[current_state]
        
        # 根據狀態機率調整（機率越高越可信）
        confidence = state_prob / 100
        adjusted_risk = base_risk * confidence
        
        return {
            'base_risk': base_risk,
            'confidence': confidence,
            'adjusted_risk': adjusted_risk,
            'state_name': self.state_names[current_state]
        }
    
    def calculate_ratio_change_risk(self, ratio_change_30d):
        """
        計算比率變化風險（0-100）
        
        Parameters:
        -----------
        ratio_change_30d : float
            30天比率變化百分比
        
        Returns:
        --------
        dict : 比率變化風險詳情
        """
        # 比率上升越多，風險越高
        if ratio_change_30d >= 20:
            risk = 100
            level = "極高"
        elif ratio_change_30d >= 15:
            risk = 80
            level = "很高"
        elif ratio_change_30d >= 10:
            risk = 60
            level = "偏高"
        elif ratio_change_30d >= 5:
            risk = 40
            level = "中等"
        elif ratio_change_30d >= 0:
            risk = 20
            level = "偏低"
        else:
            # 比率下降代表股市相對強勢
            risk = max(10 + ratio_change_30d * 2, 0)
            level = "低"
        
        return {
            'risk': risk,
            'level': level,
            'change_pct': ratio_change_30d,
            'interpretation': f"比率{'上升' if ratio_change_30d > 0 else '下降'}{abs(ratio_change_30d):.2f}%"
        }
    
    def calculate_volatility_risk(self, gold_change_30d, sp500_change_30d):
        """
        計算市場波動風險（0-100）
        
        Parameters:
        -----------
        gold_change_30d : float
            黃金30天變化百分比
        sp500_change_30d : float
            S&P 500 30天變化百分比
        
        Returns:
        --------
        dict : 波動風險詳情
        """
        # 計算兩者的波動差異（避險資產vs風險資產）
        divergence = abs(gold_change_30d - sp500_change_30d)
        
        # 當黃金大漲而股市不動或下跌，代表避險情緒高
        hedge_premium = gold_change_30d - sp500_change_30d
        
        # 基於分化程度計算風險
        if divergence >= 20:
            base_risk = 80
        elif divergence >= 15:
            base_risk = 60
        elif divergence >= 10:
            base_risk = 40
        elif divergence >= 5:
            base_risk = 20
        else:
            base_risk = 10
        
        # 如果黃金表現遠優於股市，增加風險
        if hedge_premium > 10:
            risk = min(base_risk * 1.5, 100)
            sentiment = "強避險"
        elif hedge_premium > 5:
            risk = min(base_risk * 1.2, 100)
            sentiment = "避險"
        elif hedge_premium < -10:
            risk = base_risk * 0.7
            sentiment = "強風險偏好"
        elif hedge_premium < -5:
            risk = base_risk * 0.85
            sentiment = "風險偏好"
        else:
            risk = base_risk
            sentiment = "中性"
        
        return {
            'risk': risk,
            'divergence': divergence,
            'hedge_premium': hedge_premium,
            'sentiment': sentiment,
            'gold_change': gold_change_30d,
            'sp500_change': sp500_change_30d
        }
    
    def calculate_warning_signal_risk(self, warning_count):
        """
        計算預警信號風險（0-100）
        
        Parameters:
        -----------
        warning_count : int
            預警信號數量
        
        Returns:
        --------
        dict : 預警風險詳情
        """
        # 每個信號20分，最高100分
        risk = min(warning_count * 20, 100)
        
        if warning_count >= 5:
            level = "極度危險"
        elif warning_count >= 4:
            level = "高度危險"
        elif warning_count >= 3:
            level = "危險"
        elif warning_count >= 2:
            level = "警戒"
        elif warning_count >= 1:
            level = "注意"
        else:
            level = "正常"
        
        return {
            'risk': risk,
            'count': warning_count,
            'level': level
        }
    
    def calculate_duration_risk(self, state_duration_days, current_state):
        """
        計算狀態持續時間風險（0-100）
        
        Parameters:
        -----------
        state_duration_days : int
            當前狀態持續天數
        current_state : int
            當前狀態
        
        Returns:
        --------
        dict : 持續時間風險詳情
        """
        if current_state == 2:  # 避險主導期
            # 持續越久風險越高（危機深化）
            if state_duration_days >= 50:
                risk = 90
                phase = "深度危機"
            elif state_duration_days >= 22:
                risk = 70
                phase = "危機持續"
            elif state_duration_days >= 10:
                risk = 50
                phase = "危機初期"
            elif state_duration_days >= 5:
                risk = 40
                phase = "警戒期"
            else:
                risk = 30
                phase = "剛進入"
        
        elif current_state == 1:  # 謹慎觀望期
            # 持續越久可能醞釀轉變
            if state_duration_days >= 22:
                risk = 60
                phase = "長期觀望"
            elif state_duration_days >= 10:
                risk = 50
                phase = "持續觀望"
            elif state_duration_days >= 5:
                risk = 40
                phase = "觀望中"
            else:
                risk = 30
                phase = "剛進入"
        
        else:  # 風險偏好期
            # 持續越久越安全
            if state_duration_days >= 50:
                risk = 10
                phase = "長期繁榮"
            elif state_duration_days >= 22:
                risk = 15
                phase = "持續上漲"
            elif state_duration_days >= 10:
                risk = 20
                phase = "牛市中"
            else:
                risk = 25
                phase = "剛恢復"
        
        return {
            'risk': risk,
            'duration': state_duration_days,
            'phase': phase
        }
    
    def calculate_market_risk_index(self, current_state, state_prob, 
                                    ratio_change_30d, gold_change_30d, 
                                    sp500_change_30d, warning_count, 
                                    state_duration_days):
        """
        計算綜合市場風險指數（0-100）
        
        Returns:
        --------
        dict : 完整的市場風險評估
        """
        # 1. HMM狀態風險（權重35%）
        state_risk = self.calculate_state_risk(current_state, state_prob)
        
        # 2. 比率變化風險（權重25%）
        ratio_risk = self.calculate_ratio_change_risk(ratio_change_30d)
        
        # 3. 市場波動風險（權重20%）
        volatility_risk = self.calculate_volatility_risk(gold_change_30d, sp500_change_30d)
        
        # 4. 預警信號風險（權重15%）
        warning_risk = self.calculate_warning_signal_risk(warning_count)
        
        # 5. 持續時間風險（權重5%）
        duration_risk = self.calculate_duration_risk(state_duration_days, current_state)
        
        # 計算綜合指數
        market_risk_index = (
            state_risk['adjusted_risk'] * 0.35 +
            ratio_risk['risk'] * 0.25 +
            volatility_risk['risk'] * 0.20 +
            warning_risk['risk'] * 0.15 +
            duration_risk['risk'] * 0.05
        )
        
        # 判斷風險等級
        if market_risk_index >= 80:
            risk_level = "極高風險"
            color = "#8B0000"
            action = "極度危險！建議高度避險配置"
        elif market_risk_index >= 65:
            risk_level = "高風險"
            color = "#DC143C"
            action = "高度警戒！增加避險資產"
        elif market_risk_index >= 50:
            risk_level = "中高風險"
            color = "#FF6347"
            action = "保持警惕，適度避險"
        elif market_risk_index >= 35:
            risk_level = "中等風險"
            color = "#FFA500"
            action = "正常監控，平衡配置"
        elif market_risk_index >= 20:
            risk_level = "低風險"
            color = "#FFD700"
            action = "市場穩定，可積極配置"
        else:
            risk_level = "極低風險"
            color = "#32CD32"
            action = "市場樂觀，積極配置"
        
        return {
            'market_risk_index': market_risk_index,
            'risk_level': risk_level,
            'color': color,
            'action': action,
            'components': {
                'state_risk': state_risk,
                'ratio_risk': ratio_risk,
                'volatility_risk': volatility_risk,
                'warning_risk': warning_risk,
                'duration_risk': duration_risk
            },
            'weights': {
                'HMM狀態': 35,
                '比率變化': 25,
                '市場波動': 20,
                '預警信號': 15,
                '持續時間': 5
            }
        }
    
    def visualize_market_risk_index(self, risk_result):
        """視覺化市場風險指數"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35,
                             height_ratios=[1.5, 1, 1])
        
        # ==================== 1. 風險指數儀表板 ====================
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        score = risk_result['market_risk_index']
        color = risk_result['color']
        
        # 繪製儀表盤背景
        theta1, theta2 = 180, 0
        center = (0.5, 0.3)
        radius = 0.25
        
        # 背景半圓
        wedge_bg = Wedge(center, radius, theta1, theta2, 
                        facecolor='#ecf0f1', edgecolor='#95a5a6', linewidth=3)
        ax1.add_patch(wedge_bg)
        
        # 風險區域著色
        colors_zones = ['#32CD32', '#FFD700', '#FFA500', '#FF6347', '#DC143C', '#8B0000']
        zone_angles = [180, 150, 120, 90, 60, 30, 0]
        
        for i in range(len(colors_zones)):
            wedge = Wedge(center, radius, zone_angles[i+1], zone_angles[i],
                         facecolor=colors_zones[i], alpha=0.5, edgecolor='white', linewidth=2)
            ax1.add_patch(wedge)
        
        # 指針
        angle = 180 - (score / 100) * 180
        angle_rad = np.radians(angle)
        needle_length = radius * 0.85
        needle_x = center[0] + needle_length * np.cos(angle_rad)
        needle_y = center[1] + needle_length * np.sin(angle_rad)
        
        ax1.plot([center[0], needle_x], [center[1], needle_y], 
                color='black', linewidth=5, zorder=10)
        ax1.plot(center[0], center[1], 'ko', markersize=15, zorder=11)
        
        # 分數文字
        ax1.text(0.5, 0.75, '市場風險指數', fontsize=28, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax1.text(0.5, 0.12, f'{score:.1f}', fontsize=80, 
                fontweight='bold', ha='center', va='center', color=color)
        ax1.text(0.5, -0.05, risk_result['risk_level'], fontsize=22, 
                fontweight='bold', ha='center', color=color)
        
        # 刻度
        for val, angle in zip([0, 20, 40, 60, 80, 100], [180, 144, 108, 72, 36, 0]):
            angle_rad = np.radians(angle)
            x = center[0] + (radius + 0.03) * np.cos(angle_rad)
            y = center[1] + (radius + 0.03) * np.sin(angle_rad)
            ax1.text(x, y, str(val), fontsize=11, ha='center', va='center', 
                    fontweight='bold', color='#2c3e50')
        
        # ==================== 2. 風險分解（雷達圖風格）====================
        ax2 = fig.add_subplot(gs[1, 0])
        
        components = risk_result['components']
        categories = ['HMM\n狀態', '比率\n變化', '市場\n波動', '預警\n信號', '持續\n時間']
        values = [
            components['state_risk']['adjusted_risk'],
            components['ratio_risk']['risk'],
            components['volatility_risk']['risk'],
            components['warning_risk']['risk'],
            components['duration_risk']['risk']
        ]
        weights = list(risk_result['weights'].values())
        
        x = np.arange(len(categories))
        bars = ax2.bar(x, values, color=['#3498db', '#e67e22', '#9b59b6', '#e74c3c', '#1abc9c'],
                      alpha=0.7, edgecolor='black', linewidth=2)
        
        # 添加數值和權重
        for i, (bar, val, weight) in enumerate(zip(bars, values, weights)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}\n({weight}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax2.set_ylabel('風險分數 (0-100)', fontsize=11, fontweight='bold')
        ax2.set_title('風險組成分解', fontsize=14, fontweight='bold', pad=10)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # ==================== 3. HMM狀態詳情 ====================
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        state_info = components['state_risk']
        
        info_box = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                 boxstyle="round,pad=0.03",
                                 facecolor='#e8f4f8',
                                 edgecolor='#3498db',
                                 linewidth=3)
        ax3.add_patch(info_box)
        
        ax3.text(0.5, 0.85, '📊 HMM狀態分析', fontsize=14, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax3.text(0.5, 0.70, f"當前狀態：{state_info['state_name']}", 
                fontsize=12, ha='center', color='#34495e')
        ax3.text(0.5, 0.55, f"基礎風險：{state_info['base_risk']:.0f}/100", 
                fontsize=11, ha='center', color='#34495e')
        ax3.text(0.5, 0.40, f"狀態信心：{state_info['confidence']*100:.1f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax3.text(0.5, 0.25, f"調整風險：{state_info['adjusted_risk']:.1f}/100", 
                fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
        
        # ==================== 4. 比率變化詳情 ====================
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.axis('off')
        
        ratio_info = components['ratio_risk']
        
        info_box2 = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#fef5e7',
                                  edgecolor='#e67e22',
                                  linewidth=3)
        ax4.add_patch(info_box2)
        
        ax4.text(0.5, 0.85, '📈 比率變化分析', fontsize=14, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax4.text(0.5, 0.70, ratio_info['interpretation'], 
                fontsize=12, ha='center', color='#34495e')
        ax4.text(0.5, 0.55, f"風險等級：{ratio_info['level']}", 
                fontsize=11, ha='center', color='#34495e')
        ax4.text(0.5, 0.40, f"變化幅度：{ratio_info['change_pct']:+.2f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax4.text(0.5, 0.25, f"風險分數：{ratio_info['risk']:.1f}/100", 
                fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
        
        # ==================== 5. 市場波動詳情 ====================
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        vol_info = components['volatility_risk']
        
        info_box3 = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#f4ecf7',
                                  edgecolor='#9b59b6',
                                  linewidth=3)
        ax5.add_patch(info_box3)
        
        ax5.text(0.5, 0.85, '📊 市場波動分析', fontsize=14, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax5.text(0.5, 0.70, f"黃金30天：{vol_info['gold_change']:+.2f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax5.text(0.5, 0.58, f"S&P500 30天：{vol_info['sp500_change']:+.2f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax5.text(0.5, 0.46, f"分化程度：{vol_info['divergence']:.2f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax5.text(0.5, 0.34, f"避險溢價：{vol_info['hedge_premium']:+.2f}%", 
                fontsize=11, ha='center', color='#34495e')
        ax5.text(0.5, 0.22, f"市場情緒：{vol_info['sentiment']}", 
                fontsize=11, ha='center', color='#e74c3c', fontweight='bold')
        
        # ==================== 6. 預警信號詳情 ====================
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        warn_info = components['warning_risk']
        
        info_box4 = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#fadbd8',
                                  edgecolor='#e74c3c',
                                  linewidth=3)
        ax6.add_patch(info_box4)
        
        ax6.text(0.5, 0.85, '⚠️ 預警信號', fontsize=14, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax6.text(0.5, 0.65, f"信號數量：{warn_info['count']} 個", 
                fontsize=12, ha='center', color='#34495e')
        ax6.text(0.5, 0.50, f"風險等級：{warn_info['level']}", 
                fontsize=11, ha='center', color='#34495e')
        ax6.text(0.5, 0.35, f"風險分數：{warn_info['risk']:.1f}/100", 
                fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
        
        # ==================== 7. 持續時間分析 ====================
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        dur_info = components['duration_risk']
        
        info_box5 = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                                  boxstyle="round,pad=0.03",
                                  facecolor='#d5f4e6',
                                  edgecolor='#1abc9c',
                                  linewidth=3)
        ax7.add_patch(info_box5)
        
        ax7.text(0.5, 0.85, '⏱️ 持續時間', fontsize=14, 
                fontweight='bold', ha='center', color='#2c3e50')
        ax7.text(0.5, 0.65, f"持續天數：{dur_info['duration']} 天", 
                fontsize=12, ha='center', color='#34495e')
        ax7.text(0.5, 0.50, f"階段判斷：{dur_info['phase']}", 
                fontsize=11, ha='center', color='#34495e')
        ax7.text(0.5, 0.35, f"風險分數：{dur_info['risk']:.1f}/100", 
                fontsize=12, ha='center', color='#e74c3c', fontweight='bold')
        
        # 總標題和說明
        fig.suptitle('市場風險指數報告（純客觀統計）', fontsize=22, fontweight='bold', y=0.98)
        
        action_text = f'💡 市場研判：{risk_result["action"]}'
        fig.text(0.5, 0.02, action_text, ha='center', fontsize=13, 
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', 
                         edgecolor=color, linewidth=2))
        
        info_text = f'生成時間：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 數據來源：HMM模型 + 統計數據'
        fig.text(0.5, 0.005, info_text, ha='center', fontsize=9, 
                style='italic', color='gray')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        return fig
    
    def generate_text_report(self, risk_result, report_data):
        """生成文字格式的風險指數報告"""
        
        report_lines = []
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("                        市場風險指數報告")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"報告日期: {report_data.get('report_date', '未知')}")
        report_lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("一、市場風險評估")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("【風險指數】")
        report_lines.append(f"  綜合風險指數: {risk_result['market_risk_index']:.1f}/100")
        report_lines.append(f"  風險等級: {risk_result['risk_level']}")
        report_lines.append(f"  風險顏色: {risk_result['color']}")
        report_lines.append("")
        report_lines.append("【市場研判】")
        report_lines.append(f"  {risk_result['action']}")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("二、風險組成分析")
        report_lines.append("="*80)
        report_lines.append("")
        
        components = risk_result['components']
        weights = risk_result['weights']
        
        # 1. HMM狀態風險
        state_risk = components['state_risk']
        report_lines.append(f"【1. HMM狀態風險】（權重: {weights['HMM狀態']}%）")
        report_lines.append(f"  當前狀態: {state_risk['state_name']}")
        report_lines.append(f"  基礎風險: {state_risk['base_risk']:.0f}/100")
        report_lines.append(f"  狀態信心度: {state_risk['confidence']*100:.1f}%")
        report_lines.append(f"  調整後風險: {state_risk['adjusted_risk']:.1f}/100")
        report_lines.append("")
        
        # 2. 比率變化風險
        ratio_risk = components['ratio_risk']
        report_lines.append(f"【2. 比率變化風險】（權重: {weights['比率變化']}%）")
        report_lines.append(f"  {ratio_risk['interpretation']}")
        report_lines.append(f"  風險等級: {ratio_risk['level']}")
        report_lines.append(f"  變化幅度: {ratio_risk['change_pct']:+.2f}%")
        report_lines.append(f"  風險分數: {ratio_risk['risk']:.1f}/100")
        report_lines.append("")
        
        # 3. 市場波動風險
        vol_risk = components['volatility_risk']
        report_lines.append(f"【3. 市場波動風險】（權重: {weights['市場波動']}%）")
        report_lines.append(f"  黃金30天變化: {vol_risk['gold_change']:+.2f}%")
        report_lines.append(f"  S&P 500 30天變化: {vol_risk['sp500_change']:+.2f}%")
        report_lines.append(f"  市場分化程度: {vol_risk['divergence']:.2f}%")
        report_lines.append(f"  避險溢價: {vol_risk['hedge_premium']:+.2f}%")
        report_lines.append(f"  市場情緒: {vol_risk['sentiment']}")
        report_lines.append(f"  風險分數: {vol_risk['risk']:.1f}/100")
        report_lines.append("")
        
        # 4. 預警信號風險
        warn_risk = components['warning_risk']
        report_lines.append(f"【4. 預警信號風險】（權重: {weights['預警信號']}%）")
        report_lines.append(f"  活躍信號數量: {warn_risk['count']} 個")
        report_lines.append(f"  風險等級: {warn_risk['level']}")
        report_lines.append(f"  風險分數: {warn_risk['risk']:.1f}/100")
        report_lines.append("")
        
        # 5. 持續時間風險
        dur_risk = components['duration_risk']
        report_lines.append(f"【5. 持續時間風險】（權重: {weights['持續時間']}%）")
        report_lines.append(f"  狀態持續天數: {dur_risk['duration']} 天")
        report_lines.append(f"  階段判斷: {dur_risk['phase']}")
        report_lines.append(f"  風險分數: {dur_risk['risk']:.1f}/100")
        report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("三、風險分數詳細計算")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("【計算公式】")
        report_lines.append("  市場風險指數 = Σ(各項風險 × 權重)")
        report_lines.append("")
        report_lines.append("【計算過程】")
        report_lines.append(f"  HMM狀態風險:    {state_risk['adjusted_risk']:.1f} × 35% = {state_risk['adjusted_risk'] * 0.35:.1f}")
        report_lines.append(f"  比率變化風險:    {ratio_risk['risk']:.1f} × 25% = {ratio_risk['risk'] * 0.25:.1f}")
        report_lines.append(f"  市場波動風險:    {vol_risk['risk']:.1f} × 20% = {vol_risk['risk'] * 0.20:.1f}")
        report_lines.append(f"  預警信號風險:    {warn_risk['risk']:.1f} × 15% = {warn_risk['risk'] * 0.15:.1f}")
        report_lines.append(f"  持續時間風險:    {dur_risk['risk']:.1f} ×  5% = {dur_risk['risk'] * 0.05:.1f}")
        report_lines.append("  " + "-"*60)
        report_lines.append(f"  綜合風險指數:    {risk_result['market_risk_index']:.1f}/100")
        report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("四、風險等級說明")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("  極高風險 (80-100): 極度危險！建議高度避險配置")
        report_lines.append("  高風險   (65-79):  高度警戒！增加避險資產")
        report_lines.append("  中高風險 (50-64):  保持警惕，適度避險")
        report_lines.append("  中等風險 (35-49):  正常監控，平衡配置")
        report_lines.append("  低風險   (20-34):  市場穩定，可積極配置")
        report_lines.append("  極低風險 (0-19):   市場樂觀，積極配置")
        report_lines.append("")
        
        # 根據當前風險等級標記
        if risk_result['market_risk_index'] >= 80:
            report_lines.append(f"  >>> 當前等級: 極高風險 <<<")
        elif risk_result['market_risk_index'] >= 65:
            report_lines.append(f"  >>> 當前等級: 高風險 <<<")
        elif risk_result['market_risk_index'] >= 50:
            report_lines.append(f"  >>> 當前等級: 中高風險 <<<")
        elif risk_result['market_risk_index'] >= 35:
            report_lines.append(f"  >>> 當前等級: 中等風險 <<<")
        elif risk_result['market_risk_index'] >= 20:
            report_lines.append(f"  >>> 當前等級: 低風險 <<<")
        else:
            report_lines.append(f"  >>> 當前等級: 極低風險 <<<")
        report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("五、投資建議")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"【操作建議】")
        report_lines.append(f"  {risk_result['action']}")
        report_lines.append("")
        
        # 根據風險等級給出具體建議
        if risk_result['market_risk_index'] >= 65:
            report_lines.append("【配置建議】")
            report_lines.append("  1. 大幅增加黃金等避險資產配置（建議35-50%）")
            report_lines.append("  2. 降低股票倉位（建議50-65%）")
            report_lines.append("  3. 保持充足現金流動性")
            report_lines.append("  4. 密切關注市場變化")
            report_lines.append("")
            report_lines.append("【風險控制】")
            report_lines.append("  • 設置嚴格止損")
            report_lines.append("  • 避免高槓桿操作")
            report_lines.append("  • 分散投資組合")
        elif risk_result['market_risk_index'] >= 35:
            report_lines.append("【配置建議】")
            report_lines.append("  1. 保持平衡配置")
            report_lines.append("  2. 適度增加避險資產（建議20-35%）")
            report_lines.append("  3. 維持合理股票配置（建議65-80%）")
            report_lines.append("")
            report_lines.append("【風險控制】")
            report_lines.append("  • 保持正常風險管理")
            report_lines.append("  • 定期檢視投資組合")
        else:
            report_lines.append("【配置建議】")
            report_lines.append("  1. 可積極配置股票資產（建議85-95%）")
            report_lines.append("  2. 保持少量避險資產（建議5-15%）")
            report_lines.append("  3. 把握市場機會")
            report_lines.append("")
            report_lines.append("【風險控制】")
            report_lines.append("  • 保持基本風險意識")
            report_lines.append("  • 注意市場轉折信號")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("風險提示")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("  本報告基於HMM模型和統計數據的客觀分析，僅供參考，不構成投資建議。")
        report_lines.append("  投資有風險，決策需謹慎。建議結合其他技術指標和基本面分析進行綜合判斷。")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append(f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"數據來源: HMM市場狀態分析報告 ({report_data.get('report_date', '未知')})")
        report_lines.append("="*80)
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_to_history(self, risk_result, report_data, history_file='market_risk_index_history.csv'):
        """
        將風險指數記錄到歷史檔案
        
        Parameters:
        -----------
        risk_result : dict
            風險計算結果
        report_data : dict
            報告數據
        history_file : str
            歷史記錄檔案名稱
        """
        import os
        
        # 準備記錄數據
        record = {
            '日期': report_data.get('report_date', datetime.now().strftime('%Y-%m-%d')),
            '記錄時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '風險指數': round(risk_result['market_risk_index'], 2),
            '風險等級': risk_result['risk_level'],
            'HMM狀態': report_data.get('state_name', '未知'),
            '狀態機率': round(report_data.get('state_prob', 0), 2),
            'HMM狀態風險': round(risk_result['components']['state_risk']['adjusted_risk'], 2),
            '比率變化風險': round(risk_result['components']['ratio_risk']['risk'], 2),
            '市場波動風險': round(risk_result['components']['volatility_risk']['risk'], 2),
            '預警信號風險': round(risk_result['components']['warning_risk']['risk'], 2),
            '持續時間風險': round(risk_result['components']['duration_risk']['risk'], 2),
            '比率變化30天': round(report_data.get('ratio_change_30d', 0), 2),
            '黃金變化30天': round(report_data.get('gold_change_30d', 0), 2),
            'SP500變化30天': round(report_data.get('sp500_change_30d', 0), 2),
            '預警信號數': report_data.get('warning_count', 0),
            '狀態持續天數': report_data.get('state_duration_days', 0),
            '市場情緒': risk_result['components']['volatility_risk']['sentiment'],
            '操作建議': risk_result['action']
        }
        
        # 檢查文件是否存在
        file_exists = os.path.isfile(history_file)
        
        # 如果文件存在，檢查是否已有今天的記錄
        if file_exists:
            try:
                df_existing = pd.read_csv(history_file, encoding='utf-8')
                # 檢查今天是否已有記錄
                today_date = record['日期']
                if today_date in df_existing['日期'].values:
                    # 更新今天的記錄
                    df_existing = df_existing[df_existing['日期'] != today_date]
                    df_new = pd.DataFrame([record])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(history_file, index=False, encoding='utf-8-sig')
                    return 'updated'
                else:
                    # 追加新記錄
                    df_new = pd.DataFrame([record])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(history_file, index=False, encoding='utf-8-sig')
                    return 'appended'
            except Exception as e:
                print(f"⚠️ 讀取歷史檔案時發生錯誤：{e}")
                # 如果讀取失敗，創建新文件
                df_new = pd.DataFrame([record])
                df_new.to_csv(history_file, index=False, encoding='utf-8-sig')
                return 'created'
        else:
            # 創建新文件
            df_new = pd.DataFrame([record])
            df_new.to_csv(history_file, index=False, encoding='utf-8-sig')
            return 'created'

def main():
    """主函數：計算當前市場風險指數"""
    
    print("="*70)
    print("市場風險指數計算器（自動偵測最新報告）")
    print("="*70)
    
    calculator = MarketRiskIndexCalculator()
    
    # 自動讀取最新HMM報告
    print("\n🔍 正在讀取最新HMM報告...")
    report_data = calculator.parse_hmm_report('hmm_latest_report.txt')
    
    if report_data is None:
        print("❌ 無法讀取報告，程式結束")
        return
    
    print(f"✅ 成功讀取報告（日期：{report_data.get('report_date', '未知')}）")
    
    # 從報告中提取數據
    current_state = report_data['current_state']
    state_prob = report_data['state_prob']
    ratio_change_30d = report_data['ratio_change_30d']
    gold_change_30d = report_data['gold_change_30d']
    sp500_change_30d = report_data['sp500_change_30d']
    warning_count = report_data['warning_count']
    state_duration_days = report_data['state_duration_days']
    
    print(f"\n當前市場數據：")
    print(f"  HMM狀態：{calculator.state_names[current_state]}")
    print(f"  狀態機率：{state_prob:.2f}%")
    print(f"  比率變化(30天)：{ratio_change_30d:+.2f}%")
    print(f"  黃金變化(30天)：{gold_change_30d:+.2f}%")
    print(f"  S&P500變化(30天)：{sp500_change_30d:+.2f}%")
    print(f"  預警信號：{warning_count} 個")
    print(f"  狀態持續：{state_duration_days} 天")
    
    # 計算市場風險指數
    result = calculator.calculate_market_risk_index(
        current_state, state_prob, ratio_change_30d, 
        gold_change_30d, sp500_change_30d, warning_count, 
        state_duration_days
    )
    
    print(f"\n" + "="*70)
    print(f"市場風險指數：{result['market_risk_index']:.1f}/100")
    print(f"="*70)
    print(f"風險等級：{result['risk_level']}")
    print(f"市場研判：{result['action']}")
    
    print(f"\n風險組成分解：")
    print(f"  1. HMM狀態風險（35%）：{result['components']['state_risk']['adjusted_risk']:.1f}/100")
    print(f"  2. 比率變化風險（25%）：{result['components']['ratio_risk']['risk']:.1f}/100")
    print(f"  3. 市場波動風險（20%）：{result['components']['volatility_risk']['risk']:.1f}/100")
    print(f"  4. 預警信號風險（15%）：{result['components']['warning_risk']['risk']:.1f}/100")
    print(f"  5. 持續時間風險（5%）：{result['components']['duration_risk']['risk']:.1f}/100")
    
    # 生成文字報告
    print(f"\n" + "="*70)
    print("生成文字報告...")
    print("="*70)
    
    text_report = calculator.generate_text_report(result, report_data)
    
    # 保存報告（使用當日日期命名）
    today = datetime.now().strftime('%Y-%m-%d')
    output_file = f'market_risk_index_report_{today}.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text_report)
    
    print(f"\n[OK] 風險指數報告已保存：{output_file}")
    
    # 同時顯示在螢幕上
    print(text_report)
    
    # 不保存歷史記錄
    
    print("\n" + "="*70)
    print("風險指數計算完成！")
    print("="*70)
    print("\n【特點】")
    print("  ✓ 自動讀取最新HMM報告")
    print("  ✓ 生成詳細文字報告")
    print("  ✓ 完全客觀，基於HMM和統計數據")
    print("  ✓ 不需要輸入個人配置")
    print("  ✓ 可作為市場通用風險指標")
    print("  ✓ 實時反映市場風險水平")
    print("\n")

if __name__ == "__main__":
    main()

