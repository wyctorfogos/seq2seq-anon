from transformers import pipeline, AutoTokenizer
import re


MODEL = "pierreguillou/ner-bert-large-cased-pt-lenerbr" # "celiudos/legal-bert-lgpd"
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
ner_pipe = pipeline("ner", model=MODEL, tokenizer=tokenizer, aggregation_strategy="first")


texto = "\nQUARTA CÂMARA CÍVEL\n\n\n \n\n\nAGRAVO DE INSTRUMENTO N.º 5007027-81.2022.8.08.0000\n\n\nAGRAVANTE: MINISTÉRIO PÚBLICO DO ESTADO DO ESPÍRITO SANTO\n\n\nAGRAVADOS: MUNICÍPIO DE MARECHAL FLORIANO E EDIA KLIPPEL LITTIG\n\n\nRELATOR: DESEMBARGADOR ARTHUR JOSÉ NEIVA DE ALMEIDA\n\n\n \n\n\n \n\n\nDECISÃO MONOCRÁTICA\n\n\n \n\n\nNos termos do art. 932, III, do Código de Processo Civil1 e, também, do art. 160 do Regimento Interno deste egrégio Tribunal de Justiça2, não conheço do recurso de agravo de instrumento, eis que prejudicado em razão da perda do objeto – sentença proferida – (id 4729528).\n\n\n \n\n\nPublique-se na íntegra, intimando-se as partes.\n\n\n \n\n\nApós o decurso do prazo recursal, remetam-se os autos à Comarca de origem, com as cautelas de estilo. \n\n\n \n\n\nVitória (ES), em 28 de junho de 2023.\n\n\n\n\n\n\nDESEMBARGADOR ARTHUR JOSÉ NEIVA DE ALMEIDA\n\n\nRELATOR\n\n\n1 Art.  932. Incumbe ao relator: (…) III – não conhecer de recurso  inadmissível, prejudicado ou que não tenha impugnado  especificamente os fundamentos da decisão recorrida.\n\n\n2 Art.  160 – Nos feitos cíveis, poderá o recorrente, a qualquer tempo,  sem anuência do recorrido, ou do litisconsorte, desistir do recurso  interposto, sendo este ato unilateral não receptício e  irretratável, que independe de homologação.\n"
entities = ner_pipe(texto)


anon_text = texto
offset = 0  

for ent in entities:
    start = ent['start'] + offset
    end = ent['end'] + offset
    label = ent['entity_group']
    
    placeholder = f"[{label}]"
    original = anon_text[start:end]
    
    
    anon_text = anon_text[:start] + placeholder + anon_text[end:]
    
    
    offset += len(placeholder) - len(original)


print("texto original:")
print(texto)
print("\ntexto anonimizado:")
print(anon_text)