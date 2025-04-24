import re
import pandas as pd
import json
from pathlib import Path

EXCEL_DIR   = Path("data/excels")
OUTPUT_JSON = Path("data/categories.json")

all_records = []
seen = set()   # 이미 처리된 full_path 기록

for xlsx in EXCEL_DIR.glob("*.xlsx"):
    df = pd.read_excel(xlsx, dtype=str)
    path_col = next(
        col for col in df.columns
        if df[col].dropna().astype(str).str.contains(">").any()
    )
    for idx, row in df.iterrows():
        raw = str(row[path_col]).strip()
        if ">" not in raw or "\n" in raw or len(raw) > 100:
            continue

        m = re.match(r"\[(.*?)\]\s*(.*)", raw)
        if m:
            cat_id    = m.group(1)
            full_path = m.group(2).strip()
        else:
            cat_id    = f"{xlsx.stem}_{idx}"
            full_path = raw

        # 중복 체크: full_path 기준으로만
        if full_path in seen:
            continue
        seen.add(full_path)

        all_records.append({
            "id":        cat_id,
            "full_path": full_path
        })

data = {
    "coupang": all_records,
    "naver":   [],
    "others":  [{"id":"others_0", "full_path":"기타>기타"}]
}

OUTPUT_JSON.parent.mkdir(exist_ok=True, parents=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ categories.json 생성 완료: {OUTPUT_JSON} (총 {len(all_records)}개 레코드)")

