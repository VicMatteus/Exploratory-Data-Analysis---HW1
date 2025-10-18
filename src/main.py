import pandas as pd
import os
import matplotlib.pyplot as plt

def data_import(dataset_filepath):
    return pd.read_csv(dataset_filepath)

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

def univariate_analysis(df):
    print("Análise univariada")
    print("- Média, desvio padrão e skewness")
    print(f"Dados quantitativos: \n{df.describe().T}\n")
    print(f"Dados qualitativos: \n{df.describe(include=["object"]).T}\n")

    df["classification"].value_counts().plot(kind="pie", autopct='%1.1f%%')
    plt.title("Distribuição de classe: frequência cardíaca")
    plt.ylabel("")
    plt.show()

    # Histogramas
    df.hist(figsize=(15, 10), bins=20) #20 barras
    plt.tight_layout() # Ajusta os gráficos para não sobrepor os títulos
    plt.show()
    
    # # Box-plots
    # df.boxplot(figsize=(15, 7), rot=45)
    # plt.show()

def main():
    # Leitura e análise genérica inicial
    train_ds_filepath = os.path.join(os.getcwd(), "dataset", "train_data", "full_dataset_classified.csv")
    print(train_ds_filepath)
    df = data_import(train_ds_filepath)
    get_initial_info(df)

    # Análise univariável
    univariate_analysis(df)

if __name__ == "__main__":
    main()
