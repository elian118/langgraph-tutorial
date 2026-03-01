import base64
import requests
from io import BytesIO
from PIL import Image as PILImage  # pip install pillow 필요


def visualize_graph(app):
    try:
        # 1. Mermaid 문법 및 이미지 URL 생성
        mermaid_syntax = app.get_graph().draw_mermaid()
        encoded_string = base64.b64encode(mermaid_syntax.encode('utf-8')).decode('utf-8')
        image_url = f"https://mermaid.ink/img/{encoded_string}"

        # 2. 이미지 데이터 다운로드
        response = requests.get(image_url)
        img = PILImage.open(BytesIO(response.content))

        # 3. 시스템 기본 이미지 뷰어로 띄우기
        img.show()
        print("그래프 이미지를 실행했습니다.")

    except Exception as e:
        print(f"시각화 실패: {e}")