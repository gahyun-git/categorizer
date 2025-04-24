# app/classifier.py
import os, json
import google.generativeai as genai
from dotenv import load_dotenv
from app.categories import load_categories
from app.embedder import build_tfidf, embed_query, cosine_similarity

# Gemini API 키 설정
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 1회: 카테고리 로드 + TF-IDF 사전 구축
cats_by_market    = load_categories()
idf_by_market, tfidf_by_market = build_tfidf(cats_by_market)

def gemini_fallback(prompt: str) -> int:
    model = genai.GenerativeModel(model_name="chat-bison-001")
    choice = model.generate_content(prompt, temperature=0.0)
    return int(choice.strip())

def classify(market_id: str, product_text: str, image_desc: str = None) -> dict:
    # 텍스트 결합
    combined = product_text + (". 이미지 설명: "+image_desc if image_desc else "")
    # 로컬 TF-IDF 임베딩
    idf = idf_by_market.get(market_id, {})
    qvec = embed_query(combined, idf)

    # 후보 풀 & 유사도 계산
    pool = tfidf_by_market.get(market_id, [])
    cats = cats_by_market.get(market_id, [])
    sims = [cosine_similarity(qvec, doc) for doc in pool]

    # 최고점
    best_i, best_score = max(enumerate(sims), key=lambda x: x[1])
    best_cat = cats[best_i]["full_path"]
    category_id = cats[best_i]["id"]

    if best_score >= 0.73:
        return {"category": best_cat, "category_id": category_id, "score": best_score}

    print("##########################")
    print(best_score)
    print(category_id)
    print(best_cat)

    return {"category": best_cat, "category_id": category_id}

    # 애매할 때 Top-3만 Gemini 재검증 하던가 띄워주고 고르라하면될듯
    top3 = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]

    
    candidates = [cats[i]["full_path"] for i in top3]
    print("##########################")
    print(candidates)
    

    # 카테고리 선택 프롬프트 생성하고 잼민이 ㄱㄱ 
    # prompt = f"상품 정보: \"{combined}\"\n아래 번호만 골라주세요:\n"
    # for idx, full_path in enumerate(candidates,1):
    # prompt += f"{idx}) {full_path}\n"
    
    # choice = gemini_fallback(prompt)

    # return {"category": candidates[choice-1], "score": None}

    
