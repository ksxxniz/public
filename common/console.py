"""명령줄 출력이 윈도에서 죽지 않게 하기.

윈도에서 출력을 파일이나 파이프로 돌리거나(> out.txt, | more) Git Bash 에서 실행하면
파이썬이 cp949 로 출력한다. cp949 에 없는 글자('—', '✓', '█', 제로폭 문자 등)를 만나면
명령이 UnicodeEncodeError 로 멈춘다. 콘솔에 바로 찍을 때는 문제가 없다.

명령의 main() 맨 앞에서 safe_stdio() 를 부르면 그런 글자만 ? 로 바꾸고 계속 출력한다.
"""

import sys


def safe_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # 바꿀 수 없는 출력(None, 테스트 캡처 등)
            pass
