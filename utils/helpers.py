import os
from dotenv import load_dotenv

def load_api_key():
    """環境変数からClaudeのAPIキーを読み込む"""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEYが設定されていません。.envファイルを確認してください。")
    return api_key

def format_travel_plan(destination, duration, activities, accommodations, transportation):
    """旅行プランをフォーマットする"""
    plan = f"""
    === {destination}旅行プラン ===
    
    【期間】{duration}
    
    【アクティビティ】
    {activities}
    
    【宿泊】
    {accommodations}
    
    【交通】
    {transportation}
    """
    return plan

# 既存のコードに以下を追加

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory

def create_memory():
    """会話履歴を管理するためのメモリを作成する"""
    # 直近の会話を保存するバッファメモリ
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )
    
    # 長い会話を要約するサマリーメモリ
    summary_memory = ConversationSummaryMemory(
        memory_key="conversation_summary",
        return_messages=False,
        input_key="input"
    )
    
    # 両方のメモリを組み合わせる
    memory = CombinedMemory(memories=[buffer_memory, summary_memory])
    
    return memory

# 既存のコードに以下を追加

class UserProfile:
    def __init__(self):
        self.preferences = {
            "destinations": [],  # 好きな旅行先
            "activities": [],    # 好きなアクティビティ
            "budget": None,      # 予算
            "travel_style": None # 旅行スタイル（例: 贅沢、節約、アドベンチャー）
        }
        self.past_trips = []     # 過去の旅行
    
    def update_preference(self, key, value):
        """ユーザーの好みを更新する"""
        if key in self.preferences:
            if isinstance(self.preferences[key], list):
                if value not in self.preferences[key]:
                    self.preferences[key].append(value)
            else:
                self.preferences[key] = value
    
    def add_past_trip(self, destination, date, notes=None):
        """過去の旅行を追加する"""
        self.past_trips.append({
            "destination": destination,
            "date": date,
            "notes": notes
        })
    
    def get_profile_summary(self):
        """ユーザープロファイルの要約を取得する"""
        summary = "ユーザープロファイル:\n"
        
        # 好みの情報
        summary += "【好み】\n"
        for key, value in self.preferences.items():
            if value:
                if key == "destinations":
                    summary += f"好きな旅行先: {', '.join(value)}\n" if value else ""
                elif key == "activities":
                    summary += f"好きなアクティビティ: {', '.join(value)}\n" if value else ""
                elif key == "budget":
                    summary += f"予算: {value}円\n"
                elif key == "travel_style":
                    summary += f"旅行スタイル: {value}\n"
        
        # 過去の旅行
        if self.past_trips:
            summary += "\n【過去の旅行】\n"
            for trip in self.past_trips:
                summary += f"- {trip['destination']} ({trip['date']})"
                if trip['notes']:
                    summary += f": {trip['notes']}"
                summary += "\n"
        
        return summary