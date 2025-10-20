import pandas as pd
import os
import matplotlib.pyplot as plt

def end_section():
    separator = "'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'"
    print(separator, "\n")

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
    print(f"{df["condition"].value_counts(normalize=True)}\n")
    end_section()

def univariate_analysis(df):
    print("Análise univariada")
    print("- Média, desvio padrão e skewness")
    print(f"Skewness: \n{df.skew(numeric_only=True)}\n")
    print(f"Dados quantitativos: \n{df.describe().T}\n")
    print(f"Dados qualitativos: \n{df.describe(include=["object"]).T}\n")

    df["classification"].value_counts().plot(kind="pie", autopct='%1.1f%%')
    plt.title("Distribuição de classe: frequência cardíaca")
    plt.ylabel("")
    plt.show()

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

def class_related_univariate_analysis(df):
    print("Análise univariada relacionada à classe")
    # mesma análise, mas restrita às classes, portanto farei 3 para cada variável
    
    user_input = input("Gerar boxplots da análise univariável relacionada a classe? *Cerca de 34 gráficos serão gerados.\n'S' = sim\nQualquer outra coisa = não\n")
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

def main():
    # Leitura e análise genérica inicial
    train_ds_filepath = os.path.join(os.getcwd(), "dataset", "train_data", "full_dataset_classified.csv")
    print(train_ds_filepath)
    df = data_import(train_ds_filepath)
    
    # Parte 1
    get_initial_info(df)

    # Análise univariável - parte 2
    univariate_analysis(df)

    # Análise univariada condicional a classe - parte 3
    class_related_univariate_analysis(df)

if __name__ == "__main__":
    main()
