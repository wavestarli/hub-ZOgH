"""
weather_backend.py — 天气查询后端（可循环调用版）

改进：
  1. 支持持续查询多个城市，直到用户退出
  2. 保持原有业务逻辑不变
  3. 添加友好的交互提示
"""

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        # Step 1：Geocoding — 城市名 → 经纬度
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")

        # Step 2：天气查询
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        # Step 3：格式化输出
        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = f"{country} {admin1} {city_name}".strip()

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)


def main():
    """主循环函数：持续接收用户输入并查询天气"""
    print("=" * 50)
    print("🌤️  天气查询工具（可循环使用）")
    print("=" * 50)
    print("输入城市名查询天气，输入 'q' 或 'exit' 退出程序\n")

    while True:
        try:
            user_input = input("请输入城市名称：").strip()

            # 检查退出条件
            if user_input.lower() in ('q', 'quit', 'exit', '退出'):
                print("感谢使用，再见！👋")
                break

            # 检查空输入
            if not user_input:
                print("⚠️  城市名不能为空，请重新输入\n")
                continue

            print(f"\n🔍 正在查询「{user_input}」的天气...\n")
            result = get_weather(user_input)
            print(result)
            print("\n" + "-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n\n程序已中断，再见！👋")
            break
        except Exception as e:
            print(f"❌ 发生错误：{e}\n")


if __name__ == "__main__":
    main()