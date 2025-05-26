from langchain_anthropic import ChatAnthropic
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import BaseTool

from tools.weather_tool import WeatherTool
from tools.hotel_tool import HotelTool
from utils.helpers import UserProfile

class MultiAgentSystem:
    def __init__(self, api_key):
        """マルチエージェントシステムの初期化（Claude用）"""
        # 共通のLLM
        self.llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-haiku-20240307",
            temperature=0.7
        )
        
        # ユーザープロファイル
        self.user_profile = UserProfile()
        
        # 共通ツール
        self.weather_tool = WeatherTool()
        self.hotel_tool = HotelTool()
        
        # エージェントの初期化
        self.coordinator = self._create_coordinator()
        self.researcher = self._create_researcher()
        self.planner = self._create_planner()
        self.budget_manager = self._create_budget_manager()
        
        # エージェント間の通信用メモリ
        self.shared_memory = {
            "research_results": "",
            "travel_plan": "",
            "budget_analysis": ""
        }
    
    def _create_coordinator(self):
        """コーディネーターエージェントの作成"""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        tools = [
            Tool(
                name="GetUserProfile",
                func=self._get_user_profile,
                description="ユーザープロファイルの情報を取得するツール。引数は必要ありません。"
            ),
            Tool(
                name="UpdateUserProfile",
                func=self._update_user_profile,
                description="ユーザーの好みや過去の旅行情報を更新するツール。引数として「key:value」の形式で指定してください。例: destinations:京都"
            ),
            Tool(
                name="AssignResearchTask",
                func=self._assign_research_task,
                description="リサーチエージェントにタスクを割り当てるツール。引数として調査すべき内容を指定してください。"
            ),
            Tool(
                name="AssignPlanningTask",
                func=self._assign_planning_task,
                description="プランナーエージェントにタスクを割り当てるツール。引数として計画すべき内容を指定してください。"
            ),
            Tool(
                name="AssignBudgetTask",
                func=self._assign_budget_task,
                description="予算管理エージェントにタスクを割り当てるツール。引数として予算分析すべき内容を指定してください。"
            ),
            Tool(
                name="GetResearchResults",
                func=self._get_research_results,
                description="リサーチエージェントの調査結果を取得するツール。引数は必要ありません。"
            ),
            Tool(
                name="GetTravelPlan",
                func=self._get_travel_plan,
                description="プランナーエージェントの旅行プランを取得するツール。引数は必要ありません。"
            ),
            Tool(
                name="GetBudgetAnalysis",
                func=self._get_budget_analysis,
                description="予算管理エージェントの予算分析を取得するツール。引数は必要ありません。"
            )
        ]
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": SystemMessage(content="""あなたは旅行プランニングシステムのコーディネーターです。
                ユーザーの要望を理解し、適切なエージェントにタスクを割り当ててください。
                まず、ユーザーの好みや要望を理解し、必要に応じてユーザープロファイルを更新してください。
                次に、リサーチエージェントに旅行先の情報収集を依頼し、その結果を取得してください。
                その後、プランナーエージェントに旅行プランの作成を依頼し、予算管理エージェントに予算の分析を依頼してください。
                最後に、すべての情報を統合して、ユーザーに最適な旅行プランを提案してください。
                """),
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
            },
            verbose=True
        )
    
    def _create_researcher(self):
        """リサーチエージェントの作成"""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        tools = [
            Tool(
                name="WeatherTool",
                func=self.weather_tool._run,
                description="旅行先の天気情報を取得するツール。引数として都市名を指定してください。"
            ),
            Tool(
                name="GetUserProfile",
                func=self._get_user_profile,
                description="ユーザープロファイルの情報を取得するツール。引数は必要ありません。"
            )
        ]
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": """あなたは旅行先の情報を収集するリサーチエージェントです。与えられたタスクに基づいて、旅行先の情報を収集してください。天気情報ツールを使用して、旅行先の天気情報を取得できます。また、ユーザープロファイルを参照して、ユーザーの好みに合った情報を収集してください。収集した情報は、観光スポット、グルメ、アクティビティ、ベストシーズンなどを含む、詳細かつ構造化された形式で提供してください。"""
            }
        )
    
    def _create_planner(self):
        """プランナーエージェントの作成"""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        tools = [
            Tool(
                name="HotelTool",
                func=self.hotel_tool._run,
                description="旅行先のホテル情報を検索するツール。引数として「都市名,予算(円)」の形式で指定してください。例: 東京,15000"
            ),
            Tool(
                name="GetUserProfile",
                func=self._get_user_profile,
                description="ユーザープロファイルの情報を取得するツール。引数は必要ありません。"
            ),
            Tool(
                name="GetResearchResults",
                func=self._get_research_results,
                description="リサーチエージェントの調査結果を取得するツール。引数は必要ありません。"
            )
        ]
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": """あなたは旅行プランを作成するプランナーエージェントです。与えられたタスクに基づいて、詳細な旅行プランを作成してください。ホテル検索ツールを使用して、適切な宿泊施設を提案できます。また、ユーザープロファイルとリサーチ結果を参照して、ユーザーの好みに合ったプランを作成してください。作成したプランは、日程ごとの詳細なスケジュール、宿泊施設、交通手段などを含む、構造化された形式で提供してください。"""
            }
        )
    
    def _create_budget_manager(self):
        """予算管理エージェントの作成"""
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        tools = [
            Tool(
                name="GetUserProfile",
                func=self._get_user_profile,
                description="ユーザープロファイルの情報を取得するツール。引数は必要ありません。"
            ),
            Tool(
                name="GetTravelPlan",
                func=self._get_travel_plan,
                description="プランナーエージェントの旅行プランを取得するツール。引数は必要ありません。"
            )
        ]
        
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            agent_kwargs={
                "system_message": """あなたは旅行の予算を管理する予算管理エージェントです。与えられたタスクに基づいて、旅行の予算分析を行ってください。ユーザープロファイルと旅行プランを参照して、予算の内訳と最適化案を提案してください。予算分析は、宿泊費、交通費、食費、アクティビティ費、その他の費用などを含む、詳細な内訳を提供してください。また、予算を節約するためのヒントや、予算を最大限に活用するための提案も含めてください。"""
            }
        )
    
    def _get_user_profile(self, _):
        """ユーザープロファイルを取得するツール"""
        return self.user_profile.get_profile_summary()
    
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
    
    def _assign_research_task(self, task):
        """リサーチエージェントにタスクを割り当てるツール"""
        response = self.researcher.run(task)
        self.shared_memory["research_results"] = response
        return "リサーチタスクが完了しました。GetResearchResultsツールで結果を取得できます。"
    
    def _assign_planning_task(self, task):
        """プランナーエージェントにタスクを割り当てるツール"""
        response = self.planner.run(task)
        self.shared_memory["travel_plan"] = response
        return "プランニングタスクが完了しました。GetTravelPlanツールで結果を取得できます。"
    
    def _assign_budget_task(self, task):
        """予算管理エージェントにタスクを割り当てるツール"""
        response = self.budget_manager.run(task)
        self.shared_memory["budget_analysis"] = response
        return "予算分析タスクが完了しました。GetBudgetAnalysisツールで結果を取得できます。"
    
    def _get_research_results(self, _):
        """リサーチエージェントの調査結果を取得するツール"""
        return self.shared_memory["research_results"]
    
    def _get_travel_plan(self, _):
        """プランナーエージェントの旅行プランを取得するツール"""
        return self.shared_memory["travel_plan"]
    
    def _get_budget_analysis(self, _):
        """予算管理エージェントの予算分析を取得するツール"""
        return self.shared_memory["budget_analysis"]
    
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
    
    def get_response(self, user_input):
        self._extract_preferences(user_input)
        response = self.coordinator.invoke({"input": user_input}, handle_parsing_errors=True)
        # 返り値がdict型でoutputが短い場合、chat_historyからAIの最後のcontentを返す
        if isinstance(response, dict):
            # outputが短すぎる場合はchat_historyからAIの最後のcontentを返す
            output = response.get('output', '')
            if output and len(output) > 30:
                return output
            # chat_historyからAIの最後のcontentを探す
            history = response.get('chat_history', [])
            for msg in reversed(history):
                if hasattr(msg, 'content') and hasattr(msg, '__class__') and msg.__class__.__name__ == 'AIMessage':
                    if msg.content and len(msg.content) > 30:
                        return msg.content
            # それでもなければoutputやcontentを返す
            if 'content' in response:
                return response['content']
            return str(response)
        return response
