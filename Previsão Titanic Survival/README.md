# Previsão de Sobrevivência no Titanic 🚢

Este projeto aplica algoritmos de *Machine Learning* para resolver um problema clássico de **Classificação Binária**: prever a probabilidade de sobrevivência de um passageiro no naufrágio do Titanic, com base nos seus dados demográficos e informações de viagem.

O objetivo principal deste repositório é demonstrar um ciclo completo de Ciência de Dados, desde a análise exploratória e engenharia de características (*Feature Engineering*) até à comparação e avaliação de múltiplos algoritmos de classificação.

## 🚀 Metodologia e Preparação de Dados

Para extrair o máximo de sinal preditivo dos dados, foi desenvolvido um pipeline detalhado de pré-processamento:

* **Imputação Inteligente de Dados:** Tratamento de valores nulos na Idade (`Age`) utilizando a mediana agrupada por Classe e Sexo, em vez de uma média global, garantindo maior fidelidade estatística.
* **Feature Engineering:**
  * **Título:** Extração dos títulos de cortesia (Sr., Sra., Miss, etc.) a partir dos nomes completos para captar o estatuto social.
  * **Tamanho da Família e Companhia:** Criação das variáveis `FamilySize` (combinando irmãos/cônjuges e pais/filhos) e `IsAlone` (viajante solitário).
  * **Cabine:** Transformação de dados esparsos numa variável binária (`HasCabin`), indicando se o passageiro possuía uma cabine registada.
* **Transformação Categórica e Escalonamento:** Aplicação de *One-Hot Encoding* a variáveis nominais e *StandardScaler* para normalização dos dados antes da modelação.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Regressão Logística, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting), XGBoost

## 📊 Modelação e Resultados

Foram treinados e testados 7 algoritmos diferentes, avaliados com base em múltiplas métricas (Accuracy, Precision, Recall e F1-Score) para garantir que o desequilíbrio natural entre sobreviventes e vítimas fosse tido em conta.

* **O Modelo Vencedor:** O algoritmo **Random Forest** foi selecionado como o modelo final devido à sua elevada precisão geral e estabilidade no tratamento de interações complexas entre as variáveis. 
* **Otimização de Hiperparâmetros:** Foi conduzida uma experiência com `GridSearchCV` (através de *4-Fold Cross Validation*). Esta análise confirmou que o conjunto de teste específico favorecia o modelo base, oferecendo uma excelente lição sobre a variância entre *Cross-Validation* e *Holdout Sets*.
* **Insights (Feature Importance):** A extração da importância das variáveis do Random Forest revelou conclusões claras: o **Sexo** (mulheres com vantagem drástica de sobrevivência), o **Preço do Bilhete (Fare)** e o **Título Social** foram os determinantes matemáticos mais fortes para prever a sobrevivência. A eficácia da classificação foi comprovada pela visualização da Matriz de Confusão final.

## 💻 Como Executar o Projeto

1. Clona este repositório para a tua máquina local.
2. Garante que tens as dependências instaladas (`pip install pandas numpy matplotlib seaborn scikit-learn xgboost`).
3. Certifica-te de que o ficheiro de dados `titanic.csv` está localizado na raiz do projeto.
4. Abre o ficheiro `ProjetoTitanic.ipynb` num ambiente Jupyter (Notebook) e executa as células sequencialmente.