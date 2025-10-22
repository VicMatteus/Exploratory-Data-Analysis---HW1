Aqui está um arquivo README.md que explica o funcionamento do script:

# Análise Exploratória de Dados (EDA) e PCA

Este script é uma ferramenta abrangente para realizar uma Análise Exploratória de Dados (EDA) detalhada e uma Análise de Componentes Principais (PCA) em um conjunto de dados classificado, provavalmente relacionado à variabilidade da frequência cardíaca.

O fluxo de análise é dividido em etapas:

1.  **Importação e Informações Iniciais:** Carrega os dados e mostra uma visão geral (forma, distribuição das classes).
2.  **Análise Univariada:** Analisa cada variável individualmente (histogramas, boxplots, estatísticas).
3.  **Análise Univariada por Classe:** Repete a análise anterior, mas agrupando os dados pela classe-alvo.
4.  **Análise Bivariada:** Explora a relação entre pares de variáveis (heatmap de correlação, scatterplot matrix).
5.  **PCA (Análise de Componentes Principais):** Reduz a dimensionalidade dos dados para 2D e plota o resultado para verificar a separabilidade das classes.

-----

## ⚙️ Configuração e Parâmetros

Para que o script funcione corretamente, o conjunto de dados deve seguir uma estrutura específica:

  * **Localização do CSV:** O script espera que o arquivo CSV esteja localizado no seguinte caminho relativo:
    ```
    [pasta_do_script]/dataset/train_data/
    ```
  * **Nome do Arquivo:** O nome do arquivo esperado é:
    ```
    full_dataset_classified.csv
    ```

A linha de código responsável por isso na função `main()` é:

```python
train_ds_filepath = os.path.join(os.getcwd(), "dataset", "train_data", "full_dataset_classified.csv")
```

Se o seu arquivo tiver um nome ou local diferente, **altere esta linha** para corresponder ao seu ambiente.

-----

## 💬 Interações com o Usuário

O script é interativo e fará **três perguntas** no terminal durante sua execução. Isso é feito para evitar a geração de gráficos que podem ser computacionalmente pesados ou visualmente poluídos sem a sua permissão.

As perguntas são:

1.  **Análise Univariada por Classe (Boxplots):**

      * **Pergunta:** `Gerar boxplots da análise univariável relacionada a classe? *Cerca de 34 gráficos serão gerados.\nS = Sim\nQualquer outra coisa = Não`
      * **Ação:** Se você digitar `S` (ou `s`), o script irá gerar e exibir um boxplot para cada variável numérica, segmentado pelas classes (alta, media, repouso/baixa). Se você digitar qualquer outra tecla, ele pulará esta etapa de visualização (mas ainda imprimirá as estatísticas no console).

2.  **Análise Bivariada (Scatterplot Matrix):**

      * **Pergunta:** `Gerar scatterplot? Será demorado.\nS = Sim\nN = Não`
      * **Ação:** Se você digitar `S` (ou `s`), o script usará `plotly` para gerar uma matriz de dispersão interativa (scatterplot matrix) das variáveis listadas em `vars_list`. **Aviso:** Isso pode ser lento e abrirá uma nova aba no seu navegador.

3.  **Gráfico PCA (Scatterplot 2D):**

      * **Pergunta:** `Deseja visualizar o scatterplot do PCA? Poderá demorar um pouvo.\nS = Sim\nN = Não`
      * **Ação:** Se você digitar `S` (ou `s`), o script gerará um gráfico de dispersão 2D (PC1 vs PC2) dos dados transformados pelo PCA, com os pontos coloridos pela sua classe original.

-----

## 🛠️ Tecnologias Utilizadas

O script depende das seguintes bibliotecas Python:

  * **pandas:** Para manipulação e análise de dados.
  * **numpy:** Para cálculos numéricos e PCA.
  * **matplotlib:** Para a geração de gráficos estáticos (histogramas, boxplots, etc.).
  * **seaborn:** Para visualizações estatísticas mais avançadas (heatmap, scatterplot PCA).
  * **plotly:** Para a geração do scatterplot matrix interativo.
  * **scikit-learn (sklearn):** Especificamente o `StandardScaler` para padronização dos dados antes do PCA.

-----

## 🚀 Como Executar

1.  Certifique-se de ter todas as bibliotecas listadas acima instaladas:
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn
    ```
2.  Coloque seu arquivo `full_dataset_classified.csv` no local correto (`dataset/train_data/`).
3.  Execute o script Python no seu terminal:
    ```bash
    python nome_do_script.py
    ```
4.  Responda às perguntas interativas no terminal (`S` ou `N`) conforme desejado.