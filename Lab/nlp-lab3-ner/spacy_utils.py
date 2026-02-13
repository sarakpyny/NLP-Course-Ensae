import spacy
from spacy.tokens import DocBin
from spacy.training import iob_to_biluo
from tqdm import tqdm
import re

def normalize_text(text):
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+;", ";", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+\!", "!", text)
    text = re.sub(r"\s+\?", "?", text)
    text = re.sub(r"\s+:", ":", text)
    text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
    return text

def save_text(data, output_path):
    print(f"Saving text to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(data):
            tokens = item['tokens']
            text = " ".join(tokens)
            text = normalize_text(text)
            f.write(f"{text}\n")
    print(f"Saved to {output_path}")

def save_bio(data, output_path):
    print(f"Saving BIO text to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(data):
            tokens = item['tokens']
            ner_tags = item['ner_tags']
            
            # Save original BIO tags
            for token, tag in zip(tokens, ner_tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")
    print(f"Saved to {output_path}")


def save_docbin(data, output_path):
    print(f"Creating {output_path} with {len(data)} examples...")
    db = DocBin()
    nlp = spacy.blank("fr")
    skipped_count = 0
    
    for item in tqdm(data):
        tokens = item['tokens']
        ner_tags = item['ner_tags']
        
        # Adminset seems to use BIO scheme convert to BILOU
        biluo_tags = iob_to_biluo(ner_tags)
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        
        try:
            ents = spacy.training.biluo_tags_to_spans(doc, biluo_tags)
            doc.ents = ents
            db.add(doc)
        except Exception as e:
            skipped_count += 1
            pass
            
    if skipped_count > 0:
        print(f"Skipped {skipped_count} documents due to alignment/tag issues.")
        
    db.to_disk(output_path)
    print(f"Saved to {output_path}")

def verify_spacy_conversion(file_path, nlp, index=0):
    print(f"\nVerifying {file_path} (Example {index}):")
    try:
        doc_bin = DocBin().from_disk(file_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        if 0 <= index < len(docs):
            doc = docs[index]
            print("Text:", doc.text)
            print("Entities:")
            for ent in doc.ents:
                print(f"  - {ent.text} ({ent.label_})")
            print("Tokens & Tags:")
            for token in doc:
                print(f"  {token.text}\t{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else f"  {token.text}\t{token.ent_iob_}")
        else:
            print(f"Index {index} out of bounds for {len(docs)} docs.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
