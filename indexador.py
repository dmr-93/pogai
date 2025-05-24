import os
import pickle
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

DIRETORIO_TXT = 'textos'
ARQUIVO_INDEX = 'index.pkl'
ARQUIVO_LOG = 'index_log.pkl'

modelo = SentenceTransformer('all-MiniLM-L6-v2')

def carregar_frases(log_anterior):
    novas_frases = []
    novos_nomes = []
    novos_logs = {}

    for nome_arquivo in os.listdir(DIRETORIO_TXT):
        if not nome_arquivo.endswith('.txt'):
            continue

        caminho = os.path.join(DIRETORIO_TXT, nome_arquivo)
        modificado = os.path.getmtime(caminho)

        if log_anterior.get(nome_arquivo) == modificado:
            continue

        for cod in ['utf-8', 'latin-1', 'windows-1252']:
            try:
                with open(caminho, 'r', encoding=cod) as f:
                    texto = f.read()
                    break
            except UnicodeDecodeError:
                continue
        else:
            continue

        frases = sent_tokenize(texto)
        frases = [f.strip() for f in frases if f.strip()]
        novas_frases.extend(frases)
        novos_nomes.extend([nome_arquivo] * len(frases))
        novos_logs[nome_arquivo] = modificado

    return novos_nomes, novas_frases, novos_logs

def indexar():
    if os.path.exists(ARQUIVO_INDEX):
        with open(ARQUIVO_INDEX, 'rb') as f:
            nomes_ant, frases_ant, embeddings_ant = pickle.load(f)
    else:
        nomes_ant, frases_ant, embeddings_ant = [], [], []

    if os.path.exists(ARQUIVO_LOG):
        with open(ARQUIVO_LOG, 'rb') as f:
            log_antigo = pickle.load(f)
    else:
        log_antigo = {}

    novos_nomes, novas_frases, novos_logs = carregar_frases(log_antigo)

    if not novas_frases:
        print('Nada novo para indexar.')
        return

    print(f'Indexando {len(novas_frases)} novas frases...')
    novos_embeddings = modelo.encode(novas_frases)

    todos_nomes = nomes_ant + novos_nomes
    todas_frases = frases_ant + novas_frases
    todos_embeddings = list(embeddings_ant) + list(novos_embeddings)

    with open(ARQUIVO_INDEX, 'wb') as f:
        pickle.dump((todos_nomes, todas_frases, todos_embeddings), f)

    log_antigo.update(novos_logs)
    with open(ARQUIVO_LOG, 'wb') as f:
        pickle.dump(log_antigo, f)

    print('Indexação atualizada.')

if __name__ == '__main__':
    indexar()
