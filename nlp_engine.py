
import spacy

# Load spaCy model once at startup
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import sys
    print("Error: The 'en_core_web_sm' model is not installed. Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def analyse_pos(sentence: str) -> dict:
    doc = nlp(sentence)
    
    # Step 1: input
    step_input = {
        "id": "input",
        "label": "Input",
        "sentence": sentence
    }
    
    # Step 2: tokenize
    tokens = [{"text": token.text, "index": i} for i, token in enumerate(doc)]
    step_tokenize = {
        "id": "tokenize",
        "label": "Tokenize",
        "tokens": tokens
    }
    
    # Step 3: pos_tags
    tagged = []
    counts = {}
    for token in doc:
        tagged.append({
            "text": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "explanation": spacy.explain(token.tag_) or "No explanation"
        })
        counts[token.pos_] = counts.get(token.pos_, 0) + 1
        
    step_pos_tags = {
        "id": "pos_tags",
        "label": "POS tags",
        "tagged": tagged
    }
    
    # Step 4: result
    step_result = {
        "id": "result",
        "label": "Result",
        "tagged": tagged,
        "counts": counts
    }
    
    return {
        "sentence": sentence,
        "task": "pos",
        "steps": [step_input, step_tokenize, step_pos_tags, step_result]
    }

def analyse_ner(sentence: str) -> dict:
    doc = nlp(sentence)
    
    # Step 1: input
    step_input = {
        "id": "input",
        "label": "Input",
        "sentence": sentence
    }
    
    # Step 2: tokenize
    tokens = [{"text": token.text, "index": i} for i, token in enumerate(doc)]
    step_tokenize = {
        "id": "tokenize",
        "label": "Tokenize",
        "tokens": tokens
    }
    
    # Step 3: bio_tags
    bio = []
    for token in doc:
        # Determine BIO tag
        bio_tag = "O"
        if token.ent_iob_ != "":
            if token.ent_iob_ == "O":
                bio_tag = "O"
            else:
                bio_tag = f"{token.ent_iob_}-{token.ent_type_}"
        bio.append({
            "text": token.text,
            "bio": bio_tag
        })
        
    step_bio_tags = {
        "id": "bio_tags",
        "label": "BIO tagging",
        "bio": bio
    }
    
    # Step 4: result
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "explanation": spacy.explain(ent.label_) or "No explanation",
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
        
    step_result = {
        "id": "result",
        "label": "Result",
        "sentence": sentence,
        "entities": entities
    }
    
    return {
        "sentence": sentence,
        "task": "ner",
        "steps": [step_input, step_tokenize, step_bio_tags, step_result]
    }
