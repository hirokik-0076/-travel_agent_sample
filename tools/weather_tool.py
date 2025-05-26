import requests
from langchain.tools import BaseTool

class WeatherTool(BaseTool):
    name: str = "weather_tool"
    description: str = "旅行先の天気情報を取得するツール。引数として都市名を指定してください。"
    
    def _run(self, city: str) -> str:
        """指定された都市の天気情報を取得する"""
        # 注: 実際のアプリケーションでは、OpenWeatherMapなどの実際のAPIを使用することをお勧めします
        # ここではデモのために簡易的な実装をしています
        
        # 実際のAPIを使用する場合のコード例:
        # api_key = os.getenv("WEATHER_API_KEY")
        # url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        # response = requests.get(url)
        # data = response.json()
        # return f"{city}の現在の天気: {data['weather'][0]['description']}, 気温: {data['main']['temp']}°C"
        
        # デモ用の簡易実装
        weather_data = {
            "東京": {"description": "晴れ", "temp": 25},
            "大阪": {"description": "曇り", "temp": 23},
            "京都": {"description": "小雨", "temp": 22},
            "札幌": {"description": "雪", "temp": 5},
            "那覇": {"description": "晴れ", "temp": 30},
            # 英語の都市名も対応
            "tokyo": {"description": "晴れ", "temp": 25},
            "osaka": {"description": "曇り", "temp": 23},
            "kyoto": {"description": "小雨", "temp": 22},
            "sapporo": {"description": "雪", "temp": 5},
            "naha": {"description": "晴れ", "temp": 30},
        }
        
        city_lower = city.lower()
        if city_lower in weather_data or city in weather_data:
            data = weather_data.get(city, weather_data.get(city_lower))
            return f"{city}の現在の天気: {data['description']}, 気温: {data['temp']}°C"
        else:
            return f"{city}の天気情報は見つかりませんでした。"
    
    def _arun(self, city: str):
        """非同期実行用（今回は使用しない）"""
        raise NotImplementedError("WeatherToolは非同期実行をサポートしていません。")