from utils.helpers import load_api_key
from agents.basic_agent import BasicTravelAgent
from agents.advanced_agent import AdvancedTravelAgent
from agents.multi_agent_system import MultiAgentSystem
import argparse

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='旅行プランニングアシスタント')
    parser.add_argument('--mode', choices=['basic', 'advanced', 'multi'], default='advanced',
                        help='エージェントモード（basic, advanced, または multi）')
    args = parser.parse_args()
    
    # APIキーの読み込み
    api_key = load_api_key()
    
    # エージェントの初期化
    if args.mode == 'basic':
        print("基本的な旅行エージェントを使用します。")
        agent = BasicTravelAgent(api_key)
    elif args.mode == 'multi':
        print("マルチエージェントシステムを使用します。")
        agent = MultiAgentSystem(api_key)
    else:
        print("高度な旅行エージェントを使用します。")
        agent = AdvancedTravelAgent(api_key)
    
    print("旅行プランニングアシスタントへようこそ！")
    print("終了するには 'exit' または 'quit' と入力してください。")
    
    while True:
        user_input = input("\nあなた: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("ありがとうございました。良い旅を！")
            break
        
        response = agent.get_response(user_input)
        print(f"\n旅行アドバイザー: {response}")

if __name__ == "__main__":
    main()
