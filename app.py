from flask import Flask, request, jsonify, render_template
from utils.helpers import load_api_key
from agents.basic_agent import BasicTravelAgent
from agents.advanced_agent import AdvancedTravelAgent
from agents.multi_agent_system import MultiAgentSystem

app = Flask(__name__)

# APIキーの読み込み
api_key = load_api_key()

# エージェントの初期化
basic_agent = BasicTravelAgent(api_key)
advanced_agent = AdvancedTravelAgent(api_key)
multi_agent = MultiAgentSystem(api_key)

# セッション管理用の辞書
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', 'default')
    agent_type = data.get('agent_type', 'advanced')
    
    # セッションの初期化
    if session_id not in sessions:
        sessions[session_id] = {
            'history': []
        }
    
    # エージェントの選択
    if agent_type == 'basic':
        agent = basic_agent
    elif agent_type == 'multi':
        agent = multi_agent
    else:
        agent = advanced_agent
    
    # 応答の取得
    response = agent.get_response(user_input)
    
    # 会話履歴の更新
    sessions[session_id]['history'].append({
        'user': user_input,
        'agent': response
    })
    
    return jsonify({
        'response': response,
        'session_id': session_id
    })

if __name__ == '__main__':
    app.run(debug=True)