from flask import Flask, request, render_template, session, redirect, url_for
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'segredo'

modelo = SentenceTransformer('all-MiniLM-L6-v2')

with open('index.pkl', 'rb') as f:
    nomes, frases, embeddings = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'historico' not in session:
        session['historico'] = []

    if request.method == 'POST':
        pergunta = request.form.get('pergunta', '').strip()
        if pergunta:
            pergunta_vec = modelo.encode([pergunta])
            similaridades = cosine_similarity(pergunta_vec, embeddings).flatten()
            i = similaridades.argmax()
            resposta = frases[i]
            session['historico'].insert(0, (pergunta, resposta))
            session.modified = True
        return redirect(url_for('index'))

    return render_template('index.html', historico=session['historico'])

if __name__ == '__main__':
    app.run(debug=True)
