# Previsão de Preços de Aluguer - AirBnB

Este projeto aplica técnicas avançadas de *Machine Learning* para prever os preços de alojamentos no AirBnB, com base em características como localização (coordenadas e bairro), tipo de quarto e disponibilidade.

O objetivo é fornecer uma ferramenta preditiva robusta que possa ajudar anfitriões a definir preços competitivos e apoiar investidores na análise do mercado imobiliário local.

## 🚀 Metodologia e Processamento de Dados
Para garantir a precisão do modelo, foi desenhado e implementado um pipeline rigoroso de preparação de dados:
* **Tratamento de Outliers:** Utilização do método estatístico IQR (Intervalo Interquartil) para identificar e remover propriedades com preços extremos, focando a aprendizagem no comportamento real do mercado comum.
* **Engenharia de Variáveis (Feature Engineering):** Aplicação de *One-Hot Encoding* a todas as variáveis categóricas, garantindo que os algoritmos interpretam corretamente a informação sem assumir hierarquias falsas.
* **Padrozinação(Scaling):** Normalização das variáveis numéricas contínuas para otimizar o treino e a convergência dos modelos de previsão.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost

## 📊 Modelação e Resultados
Durante a fase de desenvolvimento, foram avaliados e comparados múltiplos algoritmos de regressão (Linear, Ridge, Random Forest, Gradient Boosting e XGBoost) para identificar a melhor abordagem aos dados habitacionais.

O modelo final implementado foi o **XGBoost**, que apresentou o melhor desempenho geral, conseguindo explicar mais de metade da variância dos preços do mercado. A extração de *insights* (*Feature Importance*) deste algoritmo revelou que a **localização geográfica exata (Longitude e Latitude)** e o **tipo de alojamento (Casa inteira vs. Quarto partilhado)** são, de forma destacada, os fatores mais determinantes no valor final de uma estadia.

## 💻 Como Executar o Projeto
1. Clona este repositório para a tua máquina local.
2. Garante que tens as dependências instaladas (`pip install pandas numpy scikit-learn matplotlib seaborn xgboost`).
3. Certifica-te de que o ficheiro de dados `dataSP23.csv` está localizado na raiz do projeto.
4. Abre o ficheiro `PorjetoAirBnbRentPrice.ipynb` no Jupyter Notebook ou JupyterLab e executa as células de forma sequencial.