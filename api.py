from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Medical NER API",
    description="API profesional pentru extragerea entităților medicale din rapoarte clinice folosind Transformers.",
    version="1.0.0"
)

print("Se inițializează modelul biomedical pentru API (acest pas poate dura câteva secunde)...")
model_id = "Helios9/BioMed_NER"
medical_nlp = pipeline("ner", model=model_id, aggregation_strategy="simple")

class MedicalReport(BaseModel):
    text: str

def merge_adjacent_entities(entities, text):
    if not entities:
        return []
    
    merged = []
    current = entities[0].copy()
    
    for next_ent in entities[1:]:
        if current['word'].lower() == 'up' and current['entity_group'] == 'Age':
            current = next_ent.copy()
            continue
            
        if next_ent['entity_group'] == current['entity_group'] and (next_ent['start'] - current['end']) <= 2:
            current['end'] = next_ent['end']
            current['word'] = text[current['start']:current['end']]
            current['score'] = float((current['score'] + next_ent['score']) / 2)
        else:
            current['score'] = float(current['score'])
            merged.append(current)
            current = next_ent.copy()
            
    if not (current['word'].lower() == 'up' and current['entity_group'] == 'Age'):
        current['score'] = float(current['score'])
        merged.append(current)
        
    return merged

@app.post("/extract", summary="Extrage entități medicale dintr-un raport clinic")
def extract_entities(report: MedicalReport):
    if not report.text.strip():
        raise HTTPException(status_code=400, detail="Textul raportului nu poate fi gol.")
    
    try:
        raw_entities = medical_nlp(report.text)
        cleaned_entities = merge_adjacent_entities(raw_entities, report.text)
        
        # Formatăm entitățile pentru un JSON curat
        formatted_entities = [
            {
                "entity": ent["word"],
                "category": ent["entity_group"],
                "confidence": round(ent["score"], 4)
            }
            for ent in cleaned_entities
        ]
        
        return {
            "status": "success",
            "report": report.text,
            "entities_count": len(formatted_entities),
            "entities": formatted_entities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", summary="Health Check")
def root():
    return {"message": "Medical NER API este activ și funcțional!"}
