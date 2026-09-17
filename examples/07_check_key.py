"""API 키가 유효한지만 확인한다. 답변을 생성하지 않으므로 과금이 없다.

    uv run python examples/07_check_key.py
"""

import re
import sys
from getpass import getpass

from openai import OpenAI

key = getpass("OpenAI API 키를 붙여넣고 Enter (화면에 보이지 않음): ").strip()
if not key:
    sys.exit("키를 입력하지 않았습니다.")

try:
    names = sorted(m.id for m in OpenAI(api_key=key, timeout=30, max_retries=0).models.list())
except Exception as e:
    # 401 응답에는 키 조각이 섞여 오므로 지우고 출력한다
    sys.exit("사용 불가 — %s: %s" % (type(e).__name__,
                                     re.sub(r"sk-[A-Za-z0-9\-_*]{8,}", "sk-***", str(e))[:200]))

print("유효합니다. 모델 %d개 접근 가능 (과금 없음)" % len(names))
print("gpt-4o-mini:", "사용 가능" if "gpt-4o-mini" in names else "목록에 없음")
