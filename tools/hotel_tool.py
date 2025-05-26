from langchain.tools import BaseTool

class HotelTool(BaseTool):
    name: str = "hotel_tool"
    description: str = "旅行先のホテル情報を検索するツール。引数として「都市名,予算(円)」の形式で指定してください。例: 東京,15000"
    
    def _run(self, query: str) -> str:
        """指定された都市と予算に基づいてホテル情報を検索する"""
        try:
            # クエリを都市名と予算に分割
            parts = query.split(',')
            if len(parts) != 2:
                return "クエリの形式が正しくありません。「都市名,予算」の形式で指定してください。例: 東京,15000"
            
            city = parts[0].strip()
            try:
                budget = int(parts[1].strip())
            except ValueError:
                return "予算は数値で指定してください。例: 東京,15000"
            
            # デモ用のホテルデータ
            hotel_data = {
                "東京": [
                    {"name": "ホテルメトロポリタン", "price": 20000, "rating": 4.5},
                    {"name": "相鉄フレッサイン", "price": 12000, "rating": 4.0},
                    {"name": "アパホテル", "price": 8000, "rating": 3.5},
                ],
                "大阪": [
                    {"name": "ホテルグランヴィア大阪", "price": 18000, "rating": 4.5},
                    {"name": "ホテルモントレ", "price": 13000, "rating": 4.2},
                    {"name": "ドーミーイン", "price": 9000, "rating": 3.8},
                ],
                "京都": [
                    {"name": "京都センチュリーホテル", "price": 22000, "rating": 4.7},
                    {"name": "三井ガーデンホテル", "price": 15000, "rating": 4.3},
                    {"name": "イビススタイルズ", "price": 10000, "rating": 3.9},
                ],
                # 英語の都市名も対応
                "tokyo": [
                    {"name": "Hotel Metropolitan", "price": 20000, "rating": 4.5},
                    {"name": "Sotetsu Fresa Inn", "price": 12000, "rating": 4.0},
                    {"name": "APA Hotel", "price": 8000, "rating": 3.5},
                ],
                "osaka": [
                    {"name": "Hotel Granvia Osaka", "price": 18000, "rating": 4.5},
                    {"name": "Hotel Monterey", "price": 13000, "rating": 4.2},
                    {"name": "Dormy Inn", "price": 9000, "rating": 3.8},
                ],
                "kyoto": [
                    {"name": "Kyoto Century Hotel", "price": 22000, "rating": 4.7},
                    {"name": "Mitsui Garden Hotel", "price": 15000, "rating": 4.3},
                    {"name": "Ibis Styles", "price": 10000, "rating": 3.9},
                ],
            }
            
            city_lower = city.lower()
            if city_lower in hotel_data or city in hotel_data:
                hotels = hotel_data.get(city, hotel_data.get(city_lower))
                # 予算内のホテルをフィルタリング
                affordable_hotels = [h for h in hotels if h["price"] <= budget]
                
                if not affordable_hotels:
                    return f"{city}で予算{budget}円以内のホテルは見つかりませんでした。予算を増やしてみてください。"
                
                # 結果をフォーマット
                result = f"{city}で予算{budget}円以内のホテル情報:\n\n"
                for hotel in affordable_hotels:
                    result += f"- {hotel['name']}: {hotel['price']}円/泊, 評価: {hotel['rating']}/5.0\n"
                
                return result
            else:
                return f"{city}のホテル情報は見つかりませんでした。"
        
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    def _arun(self, query: str):
        """非同期実行用（今回は使用しない）"""
        raise NotImplementedError("HotelToolは非同期実行をサポートしていません。")