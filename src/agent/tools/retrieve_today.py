from datetime import date

def get_today_date():
    """
    今日の日付を YYYY-MM-DD 形式で取得する関数。
    :return: str - 今日の日付
    """
    today = date.today().isoformat()
    return today