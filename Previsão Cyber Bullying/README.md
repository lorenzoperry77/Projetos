# Deteção de Cyberbullying com Processamento de Linguagem Natural (NLP) 🚫📱

Este projeto desenvolve uma solução automatizada de inteligência artificial capaz de analisar textos de interações sociais online e classificar se os mesmos contêm mensagens associadas a *cyberbullying* ou assédio moral. 

O objetivo é fornecer uma ferramenta preditiva robusta para a moderação de conteúdo em plataformas digitais, convertendo linguagem humana não-estruturada em dados matemáticos acionáveis através de técnicas de Processamento de Linguagem Natural (NLP).

## 📊 O Conjunto de Dados

O modelo foi treinado com base em dados extraídos da rede social **Formspring**, contendo perguntas e respostas textuais geradas por utilizadores. 
* **Estratégia de Enriquecimento (Data Augmentation):** Para mitigar o desequilíbrio natural de classes (uma vez que a esmagadora maioria dos comentários na internet é classificada como normal), foi integrado um conjunto de dados expandido (`Formspring_augmented.csv`). Esta abordagem forneceu ao algoritmo uma base significativamente mais rica de padrões de linguagem ofensiva, otimizando a exatidão e a capacidade de generalização do modelo no mundo real.

## 🚀 Metodologia e Pipeline de NLP

A manipulação de texto puro exige uma preparação rigorosa antes da introdução em modelos matemáticos. O pipeline construído de raiz engloba:

1. **Limpeza e Normalização de Texto:**
   * Conversão sistemática de todos os caracteres para minúsculas.
   * Remoção de ruído estrutural como tags HTML (ex: `<br>`), pontuação e numeração.
2. **Filtragem Semântica e Stemming:**
   * Eliminação de *Stop Words* (palavras muito comuns que não acrescentam valor de contexto, como preposições e artigos).
   * Aplicação do algoritmo *SnowballStemmer* para reduzir cada palavra ao seu radical (ex: "laughing" e "laughs" são normalizadas para o mesmo termo básico), diminuindo o tamanho do vocabulário e acelerando o processamento.
3. **Vetorização Numérica (TF-IDF):**
   * Utilização do *TfidfVectorizer* com análise de n-gramas (`ngram_range=(1, 2)`) para captar não só palavras isoladas, mas também termos compostos (ex: "you ugly"). O modelo calcula o peso estatístico de cada palavra com base na sua frequência local contra a raridade global no dataset.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Processamento de Linguagem Natural (NLP):** NLTK (Natural Language Toolkit)
* **Manipulação e Vetorização de Dados:** Pandas, NumPy, Scikit-Learn (TF-IDF)
* **Visualização Estatística:** Matplotlib, Seaborn
* **Modelos de Machine Learning:** Naive Bayes (MultinomialNB), Regressão Logística, Decision Tree, Random Forest

## 📈 Resultados e Conclusões

Durante a fase experimental, foram testados e comparados quatro algoritmos de classificação distintos. O desempenho foi ordenado pelo **F1-Score**, a métrica ideal para este cenário por balancear a precisão e a sensibilidade de deteção da classe minoritária (*cyberbullying*).

* **Desempenho dos Modelos:**
  * **Random Forest:** Alcançou o primeiro lugar com o melhor equilíbrio geral, registando um **F1-Score de aproximadamente 88.6%**, além de uma exatidão (*Accuracy*) global extremamente elevada.
  * **Regressão Logística e Naive Bayes:** Apresentaram excelentes tempos de execução e métricas competitivas, validando a escolha do TF-IDF.
* **Conclusão Prática:** O modelo final demonstrou grande maturidade ao identificar corretamente insultos e ameaças estruturadas em linguagem informal da internet. A eficácia foi validada através de uma Matriz de Confusão detalhada e de testes diretos com frases novas introduzidas em tempo real.

## 💻 Como Executar o Projeto

1. Clona este repositório para a tua máquina local.
2. Garante que tens todas as dependências instaladas através do terminal:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
3. Certifica-te de que o ficheiro enriquecido Formspring_augmented.csv se encontra na mesma pasta do teu notebook.
4. Abre o ficheiro do projeto no Jupyter Notebook ou JupyterLab e executa as células sequencialmente para observar a limpeza, o treino e o teste em direto.