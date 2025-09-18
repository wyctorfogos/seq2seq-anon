import pandas as pd
import spacy
import re
import os

def read_dataframe(path_folder_dir: str, sep: str = ",", column_names=["input"]) -> list:
    try:
        content = pd.read_csv(filepath_or_buffer=path_folder_dir, sep=sep, encoding="utf-8")
        return content[column_names[0]].tolist()
    except Exception as e:
        raise Exception(f"Erro ao ler o arquivo. Erro {e}")
    

def get_words_from_sentences(sentences: list, save_path: str = None, checkpoint_size: int = 1000) -> list:
    try:
        unique_words = set()
        new_words = []

        nlp = spacy.load("pt_core_news_sm")
        docs = nlp.pipe(sentences, n_process=-1, batch_size=8)

        print("Iniciando extração de palavras...")

        if save_path and os.path.exists(save_path):
            old_df = pd.read_csv(save_path, encoding="utf-8")
            unique_words.update(old_df["tokens"].astype(str).str.lower().tolist())

        for doc in docs:
            for token in doc:
                if not token.is_punct and not token.is_space and not token.like_num:
                    text = re.sub(r"\d", "*", token.text)
                    if "*" in text:
                        continue

                    word = token.text.lower()
                    if word not in unique_words:
                        unique_words.add(word)
                        new_words.append(word)

                        if save_path and len(new_words) >= checkpoint_size:
                            pd.DataFrame(new_words, columns=["tokens"]).to_csv(
                                save_path, sep=",", index=False, mode="a",
                                header=not os.path.exists(save_path), encoding="utf-8"
                            )
                            print(f"Checkpoint salvo: +{len(new_words)} palavras novas")
                            new_words = []

        if save_path and new_words:
            pd.DataFrame(new_words, columns=["tokens"]).to_csv(
                save_path, sep=",", index=False, mode="a",
                header=not os.path.exists(save_path), encoding="utf-8"
            )
            print(f"Checkpoint final salvo: +{len(new_words)} palavras novas")

        print(f"Extração concluída. Total de {len(unique_words)} palavras únicas encontradas.")
        return list(unique_words)

    except Exception as e:
        raise Exception(f"Erro ao extrair as palavras. Erro: {e}\n")


if __name__ == "__main__":
    csv_path_dir = "./data_seq2seq/to_test/pares-gpt-oss20b.csv"
    gotten_words_folder_path = "./data_seq2seq/to_test/palavras_unicas.csv"

    # (opcional) limpar arquivo anterior
    if os.path.exists(gotten_words_folder_path):
        os.remove(gotten_words_folder_path)

    lista_dos_textos = read_dataframe(path_folder_dir=csv_path_dir)
    palavras_unicas = get_words_from_sentences(
        sentences=lista_dos_textos,
        save_path=gotten_words_folder_path,
        checkpoint_size=500
    )

    df = pd.read_csv(gotten_words_folder_path, encoding="utf-8")
    df = df.drop_duplicates().sort_values("tokens")
    df.to_csv(gotten_words_folder_path, index=False, encoding="utf-8")
    print(f"Arquivo final salvo com {len(df)} palavras únicas.")
