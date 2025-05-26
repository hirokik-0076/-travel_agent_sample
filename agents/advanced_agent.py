from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

from tools.weather_tool import WeatherTool
from tools.hotel_tool import HotelTool
from utils.helpers import UserProfile

class AdvancedTravelAgent:
    def __init__(self, api_key):
        """高度な旅行エージェントの初期化（Claude用）"""
        self.llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        # ユーザープロファイルの初期化
        self.user_profile = UserProfile()
        
        # メモリの初期化
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # ツールの初期化
        self.tools = [
            Tool(
                name="WeatherTool",
                func=WeatherTool()._run,
                description="旅行先の天気情報を取得するツール。引数として都市名を指定してください。"
            ),
            Tool(
                name="HotelTool",
                func=HotelTool()._run,
                description="旅行先のホテル情報を検索するツール。引数として「都市名,予算(円)」の形式で指定してください。例: 東京,15000"
            ),
            Tool(
                name="UpdateUserProfile",
                func=self._update_user_profile,
                description="ユーザーの好みや過去の旅行情報を更新するツール。引数として「key:value」の形式で指定してください。例: destinations:京都"
            ),
            Tool(
                name="GetUserProfile",
                func=self._get_user_profile,
                description="ユーザープロファイルの情報を取得するツール。引数は必要ありません。"
            )
        ]
        
        # エージェントの初期化
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            agent_kwargs={
                "system_message": """あなたは旅行プランニングの専門家です。ユーザーの質問に答え、旅行プランを提案してください。
                ユーザーの好みや過去の旅行履歴を学習し、パーソナライズされた提案をしてください。
                必要に応じて、天気情報やホテル情報などのツールを使用してください。
                ユーザーの好みや過去の旅行情報は、UpdateUserProfileツールを使って更新し、GetUserProfileツールで取得できます。
                """,
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
            },
            verbose=True
        )
    
    def _update_user_profile(self, query):
        """ユーザープロファイルを更新するツール"""
        try:
            key, value = query.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key in ["destinations", "activities"]:
                self.user_profile.update_preference(key, value)
                return f"ユーザーの{key}を'{value}'に更新しました。"
            elif key == "budget":
                try:
                    budget = int(value)
                    self.user_profile.update_preference(key, budget)
                    return f"ユーザーの予算を{budget}円に更新しました。"
                except ValueError:
                    return "予算は数値で指定してください。"
            elif key == "travel_style":
                self.user_profile.update_preference(key, value)
                return f"ユーザーの旅行スタイルを'{value}'に更新しました。"
            elif key == "past_trip":
                try:
                    destination, date = value.split(',', 1)
                    self.user_profile.add_past_trip(destination.strip(), date.strip())
                    return f"ユーザーの過去の旅行（{destination}, {date}）を追加しました。"
                except ValueError:
                    return "過去の旅行は「目的地,日付」の形式で指定してください。"
            else:
                return f"不明なキー'{key}'です。有効なキーは 'destinations', 'activities', 'budget', 'travel_style', 'past_trip' です。"
        except ValueError:
            return "クエリの形式が正しくありません。「key:value」の形式で指定してください。"
    
    def _get_user_profile(self, _):
        """ユーザープロファイルを取得するツール"""
        return self.user_profile.get_profile_summary()
    
    def get_response(self, user_input):
        """ユーザー入力に対する応答を取得"""
        response = self.agent.run(user_input)
        
        # ユーザーの好みを自動的に抽出して更新
        self._extract_preferences(user_input)
        
        return response
    
    def _extract_preferences(self, user_input):
        """ユーザー入力から好みを抽出して更新（簡易的な実装）"""
        # 実際のアプリケーションでは、より高度なNLPを使用することをお勧めします
        
        # 旅行先の抽出（簡易的）
        destinations = ["東京", "大阪", "京都", "札幌", "那覇", "沖縄", "北海道", "福岡", "名古屋", "広島"]
        for dest in destinations:
            if dest in user_input:
                self.user_profile.update_preference("destinations", dest)
        
        # アクティビティの抽出（簡易的）
        activities = ["観光", "グルメ", "ショッピング", "温泉", "ハイキング", "ビーチ", "美術館", "博物館"]
        for act in activities:
            if act in user_input:
                self.user_profile.update_preference("activities", act)
        
        # 予算の抽出（簡易的）
        import re
        budget_match = re.search(r'予算[は]?(\d+)万?円', user_input)
        if budget_match:
            budget = int(budget_match.group(1))
            if "万" in budget_match.group(0):
                budget *= 10000
            else:
                budget *= 1000
            self.user_profile.update_preference("budget", budget)
        
        # 旅行スタイルの抽出（簡易的）
        styles = {
            "贅沢": ["贅沢", "高級", "ラグジュアリー"],
            "節約": ["節約", "安い", "格安", "バジェット"],
            "アドベンチャー": ["アドベンチャー", "冒険", "アクティブ"],
            "リラックス": ["リラックス", "のんびり", "ゆっくり"],
            "文化体験": ["文化", "歴史", "伝統"]
        }
        
        for style, keywords in styles.items():
            for keyword in keywords:
                if keyword in user_input:
                    self.user_profile.update_preference("travel_style", style)
                    break