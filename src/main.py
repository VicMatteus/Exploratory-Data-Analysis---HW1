import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np

def end_section():
    separator = "'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'"
    print(separator, "\n")

def data_import(dataset_filepath):
    return pd.read_csv(dataset_filepath).drop(columns=["datasetId"])

def get_initial_info(df):
    print("Prévia dos dados:")
    print(f"{df.head()}\n")
    # df.info()
    print(f"Linhas x Colunas: {df.shape}\n")

    text = "Distribuição das classes\n"
    text += "- Valores absolutos:\n"
    text += f"alta : {df[df["classification"] == "alta"].shape[0]} \n"
    text += f"média: {df[df["classification"] == "media"].shape[0]} \n"
    text += f"repouso/baixa: {df[df["classification"] == "repouso/baixa"].shape[0]} \n"
    print(text)

    print("- Porcentagem")
    print(f"{df["classification"].value_counts(normalize=True)}\n")
    print(f"{df["condition"].value_counts(normalize=True)}\n")
    end_section()

def get_univariate_analysis(df):
    print("Análise univariada")
    print("- Média, desvio padrão e skewness")
    print(f"Skewness: \n{df.skew(numeric_only=True)}\n")
    print(f"Dados quantitativos: \n{df.describe().T}\n")
    print(f"Dados qualitativos: \n{df.describe(include=["object"]).T}\n")

    # df["classification"].value_counts().plot(kind="pie", autopct='%1.1f%%')
    # plt.title("Distribuição de classe: frequência cardíaca")
    # plt.ylabel("")
    # plt.show()

    # df["condition"].value_counts().plot(kind="pie", autopct='%1.1f%%')
    # plt.title("característica: condição")
    # plt.ylabel("")
    # plt.show()

    # Histogramas
    # df.hist(figsize=(15, 10), bins=20) #20 barras
    # plt.tight_layout() # Ajusta os gráficos para não sobrepor os títulos
    # plt.show()
    
    # Box-plots
    # df.boxplot(figsize=(15, 10), rot=90)
    # plt.show()
    end_section()

def get_class_related_univariate_analysis(df):
    print("Análise univariada relacionada à classe")
    # mesma análise, mas restrita às classes, portanto farei 3 para cada variável
    
    user_input = input("Gerar boxplots da análise univariável relacionada a classe? *Cerca de 34 gráficos serão gerados.\nS = Sim\nQualquer outra coisa = Não\n").upper()

    # iteraçãozinha que vai gerar os boxplots para cada coluna, uma por vez
    for column in df.columns:
        if df[column].dtype == "float64":
            print(f"Análise da varíavel: {column}")
            print(df.groupby("classification")[column].describe())
            # print(df.groupby("classification").describe())
 

            if user_input == "S":
                df.boxplot(column=column, by="classification", figsize=(10,6))
                plt.title(f"Boxplot de {column} por classe")
                plt.xlabel("Classe da Frequência Cardíaca")
                plt.ylabel(column)
                plt.show()

    end_section()

def get_bivariate_analysis(df):
    # Scatterplot, matriz de correlação tabular - heatmap
    vars_list = ["VLF", "LF", "LF_NU", "HF", "HF_PCT", "HF_NU", "LF_HF", "HF_LF", "higuci", "sampen"] #MEDIAN_RR; HR; MEAN_RR, SDRR_RMSSD_REL_RR
    
    print("Análise bivariada - parte 4")
    user_input = input("Gerar scatterplot? Será demorado.\nS = Sim\nN = Não\n").upper()
    if user_input == "S":
        print("Gerando Scatterplot...")
        # sns.pairplot(
        #     df,
        #     hue = "classification",
        #     diag_kind = "kde",
        #     vars=vars_list,
        #     corner = True
        # )
        # plt.show()

        # Alternativa para questões de desempenho
        fig = px.scatter_matrix(
            df,
            dimensions=vars_list,      # Equivalente a 'vars' -> pesa bastante na gpu quando a aba do navegador abre
            color="classification"     # colore as classes
        )
        fig.update_traces(showupperhalf=True)
        fig.show()

    # Heatmap da matriz de correlação
    print("Matriz de correlação - heatmap")
    corr_matrix = df.corr(numeric_only = True, )
    # print(f"Matriz de correlação:\n{corr_matrix}")

    plt.figure(figsize=(12,10))
    sns.heatmap(
        corr_matrix,
        annot = False,
        fmt = ".1f",
        cmap = "coolwarm"
    )
    plt.title("Heatmap da matriz")
    plt.show()

    end_section()

def get_pca(df):
    # Padronização e normalização; matriz de covariança; decomposição; principais componentes; scatterplot;
    features_df = df.drop(columns=["classification", "condition", "uuid"])
    target = df["classification"]

    feature_names = features_df.columns
    n_features = len(feature_names)

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    cov_matrix = np.cov(X_scaled, rowvar=False)

    # Decomposição
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

    # Seleção dos Componentes Principais
    eigen_pairs = []
    for i in range(len(eigen_values)):
        pair = (np.abs(eigen_values[i]), eigen_vectors[:, i])
        eigen_pairs.append(pair)
        
    # Ordenar
    eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)

    # Seleciona os dois principais
    pc1_vector = eigen_pairs[0][1].reshape(n_features, 1)
    pc2_vector = eigen_pairs[1][1].reshape(n_features, 1)

    matrix_w = np.hstack((pc1_vector, pc2_vector))

    X_pca = X_scaled.dot(matrix_w)

    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])

    # Adiciona a coluna classification de volta para a coloração
    pca_df = pd.concat([pca_df, target.reset_index(drop=True)], axis=1)

    user_input = input("Deseja visualizar o scatterplot do PCA? Poderá demorar um pouvo.\nS = Sim\nN = Não").upper()
    if user_input == "S":
        print("Gerando gráfico de dispersão PCA (PC1 vs PC2)...")

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x="PC1", 
            y="PC2", 
            hue="classification", 
            data=pca_df,
            alpha=0.7,
            s=50
        )

        plt.title("Análise de Componentes Principais (PCA) - 2 Componentes")
        plt.xlabel("Componente Principal 1 (PC1)")
        plt.ylabel("Componente Principal 2 (PC2)")
        plt.legend(title="Classe")
        plt.grid(True)
        plt.show()

    end_section()

def main():
    # Leitura e análise genérica inicial
    train_ds_filepath = os.path.join(os.getcwd(), "dataset", "train_data", "full_dataset_classified.csv")
    print(train_ds_filepath)
    df = data_import(train_ds_filepath)
    
    # Parte 1
    get_initial_info(df)

    # Análise univariável - parte 2
    get_univariate_analysis(df)

    # Análise univariada condicional a classe - parte 3
    get_class_related_univariate_analysis(df)

    # Análise bivariada - parte 4
    get_bivariate_analysis(df)

    # pca
    get_pca(df)

if __name__ == "__main__":
    main()
