import pandas as pd
import os

def data_import(dataset_filepath):
    return pd.read_csv(dataset_filepath)

def get_initial_info(df):
    df.head()
    df.info()
    print(df.shape)
    df.describe()

    text = "Distribuição das classes:\n"
    text += f"classe(alta) : {df[df["classification"] == "alta"].shape[0]} \n"
    text += f"classe(média): {df[df["classification"] == "media"].shape[0]} \n"
    text += f"classe(repouso/baixa): {df[df["classification"] == "repouso/baixa"].shape[0]} \n"
    print(text)

def univariate_analysis():
    print("Histograma, boxplot e (média, desvio padrão e skewness)")
    

def main():
    # Leitura e análise genérica inicial
    train_ds_filepath = os.path.join(os.getcwd(), "dataset", "train_data", "full_dataset_classified.csv")
    print(train_ds_filepath)
    df = data_import(train_ds_filepath)
    get_initial_info(df)

    # Análise univariável
    # loopar todas as colunas
    univariate_analysis()

if __name__ == "__main__":
    main()
