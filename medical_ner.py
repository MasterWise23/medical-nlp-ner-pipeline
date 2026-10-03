from transformers import pipeline

model_id = "Helios9/BioMed_NER"
print("Se inițializează modelul biomedical...")
medical_nlp = pipeline("ner", model=model_id, aggregation_strategy="simple")

medical_reports = [
    "Patient John Doe presents with acute chest pain and shortness of breath. Diagnosed with hypertension and prescribed Lisinopril 10mg.",
    "A 45-year-old female complains of severe headache, dizziness, and mild fever. Recommended rest and Ibuprofen 400mg.",
    "Follow-up examination for diabetes management. Blood glucose levels are stable. Continue Metformin treatment."
]

def merge_adjacent_entities(entities, text):
    if not entities:
        return []
    
    merged = []
    current = entities[0].copy()
    
    for next_ent in entities[1:]:
        # Ignorăm erorile izolate de tip 'up' din cuvinte compuse (ex: Follow-up)
        if current['word'].lower() == 'up' and current['entity_group'] == 'Age':
            current = next_ent.copy()
            continue
            
        # Dacă aparțin aceleiași categorii și sunt adiacente în text
        if next_ent['entity_group'] == current['entity_group'] and (next_ent['start'] - current['end']) <= 2:
            current['end'] = next_ent['end']
            current['word'] = text[current['start']:current['end']]
            current['score'] = (current['score'] + next_ent['score']) / 2
        else:
            merged.append(current)
            current = next_ent.copy()
            
    if not (current['word'].lower() == 'up' and current['entity_group'] == 'Age'):
        merged.append(current)
        
    return merged

def extract_and_clean(texts):
    print("\n--- Pipeline Hugging Face (Final & Optimizat) ---")
    for i, text in enumerate(texts, 1):
        print(f"\nRaport #{i}: {text}")
        raw_entities = medical_nlp(text)
        cleaned_entities = merge_adjacent_entities(raw_entities, text)
        
        if not cleaned_entities:
            print("  - Nicio entitate detectată.")
        for ent in cleaned_entities:
            print(f"  * {ent['word']} --> [Tip: {ent['entity_group']}] (Scor: {ent['score']:.2f})")

if __name__ == "__main__":
    extract_and_clean(medical_reports)
