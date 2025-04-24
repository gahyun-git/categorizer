from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.classifier import classify

app = FastAPI(title="Categorizer")

class Item(BaseModel):
    market_id:    str
    product_text: str
    image_desc:   str = None

@app.post("/classify")
def api_classify(item: Item):
    try:
        return classify(item.market_id, item.product_text, item.image_desc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
