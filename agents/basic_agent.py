from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class BasicTravelAgent:
    def __init__(self, api_key):
        """基本的な旅行エージェントの初期化（Claude用）"""
        self.llm = ChatAnthropic(
            api_key=api_key,
            model="claude-3-haiku-20240307",  # Claudeのモデル名例
            temperature=0.7
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template="""あなたは旅行プランニングの専門家です。ユーザーの質問に答え、旅行プランを提案してください。
            
            これまでの会話:
            {chat_history}
            
            ユーザー: {input}
            旅行アドバイザー:"""
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
    
    def get_response(self, user_input):
        """ユーザー入力に対する応答を取得"""
        response = self.chain.predict(input=user_input)
        return response