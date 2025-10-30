import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(
    page_title="TF-IDF Demo",
    page_icon="🔍",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #F0FDF4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        margin: 0.5rem 0;
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-title">🔍 Demo de TF-IDF con Preguntas y Respuestas</h1>', unsafe_allow_html=True)

# Información inicial
st.markdown("""
<div class="info-box">
<h3>📝 Instrucciones</h3>
<p><strong>Cada línea se trata como un documento</strong> (puede ser una frase, un párrafo o un texto más largo).</p>
<p>⚠️ <strong>Importante:</strong> Los documentos y las preguntas deben estar en <strong>inglés</strong>, ya que el análisis está configurado para ese idioma.</p>
<p>La aplicación aplica normalización y <em>stemming</em> para que palabras como <span class="highlight">playing</span> y <span class="highlight">play</span> se consideren equivalentes.</p>
</div>
""", unsafe_allow_html=True)

# Ejemplo inicial en inglés
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Documentos de Entrada")
    text_input = st.text_area(
        "Escribe tus documentos (uno por línea, en inglés):",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
        height=150,
        help="Cada línea representa un documento separado"
    )

with col2:
    st.subheader("❓ Pregunta")
    question = st.text_input(
        "Escribe una pregunta (en inglés):", 
        "Who is playing?",
        help="La pregunta que quieres hacer sobre los documentos"
    )

# Inicializar stemmer para inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    # Pasar a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfabéticos
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenizar (palabras con longitud > 1)
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Botón de cálculo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Calcular TF-IDF y Buscar Respuesta", use_container_width=True):
        documents = [d.strip() for d in text_input.split("\n") if d.strip()]
        if len(documents) < 1:
            st.warning("⚠️ Ingresa al menos un documento.")
        else:
            # Vectorizador con stemming
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                stop_words="english",
                token_pattern=None
            )

            # Ajustar con documentos
            X = vectorizer.fit_transform(documents)

            # Mostrar matriz TF-IDF
            df_tfidf = pd.DataFrame(
                X.toarray(),
                columns=vectorizer.get_feature_names_out(),
                index=[f"📄 Doc {i+1}" for i in range(len(documents))]
            )

            st.markdown("### 📊 Matriz TF-IDF (stems)")
            st.dataframe(df_tfidf.round(3), use_container_width=True)

            # Vector de la pregunta
            question_vec = vectorizer.transform([question])

            # Similitud coseno
            similarities = cosine_similarity(question_vec, X).flatten()

            # Documento más parecido
            best_idx = similarities.argmax()
            best_doc = documents[best_idx]
            best_score = similarities[best_idx]

            # Resultado principal
            st.markdown("### 🎯 Resultado de la Búsqueda")
            st.markdown(f"""
            <div class="result-box">
            <h4>❓ <strong>Tu pregunta:</strong> {question}</h4>
            <h4>📄 <strong>Documento más relevante (Doc {best_idx+1}):</strong> {best_doc}</h4>
            <h4>⭐ <strong>Puntaje de similitud:</strong> {best_score:.3f}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Mostrar todas las similitudes
            sim_df = pd.DataFrame({
                "📊 Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "📝 Texto": documents,
                "⭐ Similitud": similarities.round(3)
            })
            
            st.markdown("### 📈 Puntajes de Similitud (Ordenados)")
            st.dataframe(
                sim_df.sort_values("⭐ Similitud", ascending=False),
                use_container_width=True
            )

            # Mostrar coincidencias de stems
            vocab = vectorizer.get_feature_names_out()
            q_stems = tokenize_and_stem(question)
            matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
            
            if matched:
                st.markdown("### 🔍 Stems Coincidentes")
                st.markdown(f"""
                <div class="metric-box">
                <p>Los siguientes stems de la pregunta están presentes en el documento elegido:</p>
                <p><strong>{', '.join(matched)}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No se encontraron stems coincidentes entre la pregunta y el documento.")

# Información adicional en el sidebar
with st.sidebar:
    st.markdown("### 💡 Sobre TF-IDF")
    st.markdown("""
    <div style='background-color: #F1F5F9; padding: 1rem; border-radius: 8px;'>
    <strong>TF-IDF</strong> (Term Frequency-Inverse Document Frequency) es una medida estadística que evalúa la importancia de una palabra en un documento.
    
    <br><br><strong>Componentes:</strong>
    <br>• <em>TF</em>: Frecuencia del término en el documento
    <br>• <em>IDF</em>: Frecuencia inversa en el corpus
    
    <br><br><strong>Stemming:</strong>
    <br>Reduce palabras a su raíz (ej: "playing" → "play")
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Ejemplos de Uso")
    st.write("""
    • Búsqueda de documentos
    • Sistemas de preguntas y respuestas
    • Análisis de similitud textual
    • Recuperación de información
    """)
