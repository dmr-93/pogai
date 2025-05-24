import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer('all-MiniLM-L6-v2')

def carregar_index():
    with open('index.pkl', 'rb') as f:
        return pickle.load(f)

def responder(pergunta, top_k=1):
    nomes, frases, embeddings = carregar_index()
    pergunta_vec = modelo.encode([pergunta])
    similaridades = cosine_similarity(pergunta_vec, embeddings).flatten()
    indices = similaridades.argsort()[-top_k:][::-1]

    for i in indices:
        print(f'\n{frases[i]}')

if __name__ == '__main__':
    while True:
        pergunta = input('Pergunta: ').strip()
        if pergunta.lower() in {'sair', 'exit'}:
            break
        responder(pergunta)
