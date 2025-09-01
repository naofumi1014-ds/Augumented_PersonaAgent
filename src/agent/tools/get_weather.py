
import requests
from datetime import date
from typing import List, Dict, Tuple



class FreeWeatherError(RuntimeError):
    """通信やレスポンス異常時に送出する例外"""
    IndentationError = "FreeWeatherError"

class GetWeather:
    def __init__(self, city: str, days: int ):
        self.city = city
        self.days = days
        self.lang = "ja"
    
    def _geocode(self) -> Tuple[float, float, str]:
        """
        都市名から (lat, lon, timezone) を返す。
        """
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": self.city, "language": self.lang, "count": 1}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            raise FreeWeatherError(f"ジオコーディング失敗: {r.text}")
        data = r.json()
        if not data.get("results"):
            # 結果が空の場合、指定された都市名を表示
            raise FreeWeatherError(f"都市が見つかりません: {self.city}")
        res = data["results"][0]
        return res["latitude"], res["longitude"], res["timezone"]

    # --- 2) 予報を取得（Open-Meteo Forecast API） ------------------------
    #   daily に weather_code, 気温最大/最小 を指定し、
    #   forecast_days で必要日数だけ取得する。timezone を渡すとローカル時刻で返る。
    #
    #   利用できる日次・時次変数は公式ドキュメントに一覧があります:contentReference[oaicite:1]{index=1}。
    def get_free_weather_forecast(
        self,
    ) -> List[Dict[str, str]]:
        """
        Open-Meteo で無料取得した天気予報を整形して返す。

        Returns
        -------
        list[dict]  例:
            [{'date': '2025-06-28', 'summary': '晴れ', 't_min': 23.1, 't_max': 29.5}, ...]
        """
        if not 1 <= self.days <= 16:
            raise ValueError("days は 1〜16 の範囲で指定してください")

        # 地理座標とタイムゾーンを取得
        lat, lon, tz = self._geocode()

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": self.days,
            "timezone": tz,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            raise FreeWeatherError(f"予報取得失敗: {r.text}")

        d = r.json()["daily"]
        # 天気コード→日本語マップ
        wcodes = self._weather_code_map(self.lang)

        out = []
        for idx, iso in enumerate(d["time"]):
            out.append(
                {
                    "date": iso,                              # YYYY-MM-DD
                    "summary": wcodes.get(d["weather_code"][idx], "不明"),
                    "t_min": d["temperature_2m_min"][idx],
                    "t_max": d["temperature_2m_max"][idx],
                }
            )
        return out

    # --- 3) 天気コード → 日本語／英語 ------------------
    def _weather_code_map(self, lang: str = "ja") -> Dict[int, str]:
        # （主要コードのみ抜粋。全コードは公式 README を参照）
        ja = {
            0: "快晴",
            1: "晴れ",
            2: "薄曇り",
            3: "曇り",
            45: "霧",
            48: "着氷性霧",
            51: "霧雨 (弱)",
            53: "霧雨 (中)",
            55: "霧雨 (強)",
            61: "雨 (弱)",
            63: "雨 (中)",
            65: "雨 (強)",
            71: "雪 (弱)",
            73: "雪 (中)",
            75: "雪 (強)",
            80: "にわか雨 (弱)",
            81: "にわか雨 (中)",
            82: "にわか雨 (強)",
            95: "雷雨",
        }
        return ja if lang.startswith("ja") else {}

    def run(self) -> List[Dict[str, str]]:
        """
        初期化時に指定した日数で天気予報を取得
        """
        return self.get_free_weather_forecast()

if __name__ == "__main__":
    for info in GetWeather(city="Tokyo", days=15).run():
        d = date.fromisoformat(info["date"]).strftime("%m/%d(%a)")
        print(f"{d}: {info['summary']}  {info['t_min']}°C–{info['t_max']}°C")