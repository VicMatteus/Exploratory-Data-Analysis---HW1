import pandas as pd
import sys

arquivo_de_entrada = 'full_dataset.csv'
arquivo_de_saida = 'full_dataset_classified.csv'
coluna_para_classificar = 'HR'
nova_coluna = 'classification'

try:
    print(f"Lendo o arquivo: {arquivo_de_entrada}...")
    df = pd.read_csv(arquivo_de_entrada)
    print(" -> Leitura concluída com sucesso!")

    # Verifica se a coluna 'hr' existe no DataFrame
    if coluna_para_classificar not in df.columns:
        print(f"ERRO: A coluna '{coluna_para_classificar}' não foi encontrada no arquivo. Verifique o nome da coluna.", file=sys.stderr)
        sys.exit(1)

    def classificar_hr(frequencia_cardiaca):
        # Converte para numérico, tratando possíveis erros
        hr_valor = pd.to_numeric(frequencia_cardiaca, errors='coerce')
        
        # Se não for um número (NaN), retorna uma categoria indefinida
        if pd.isna(hr_valor):
            return 'indefinido'
        elif hr_valor <= 60:
            return 'repouso/baixa'
        elif hr_valor < 100:
            return 'média'
        else:
            return 'alta'

    # 2. Aplica a função à coluna 'hr' para criar a nova coluna 'classification'
    print(f"Aplicando a classificação na coluna '{coluna_para_classificar}'...")
    df[nova_coluna] = df[coluna_para_classificar].apply(classificar_hr)
    print(" -> Classificação aplicada!")

    # 3. Mostra as primeiras linhas do DataFrame com a nova coluna para verificação
    print("\n--- Amostra do resultado ---")
    print(df[[coluna_para_classificar, nova_coluna]].head())
    print("--------------------------\n")

    # 4. Salva o DataFrame modificado em um novo arquivo CSV
    print(f"Salvando o resultado em '{arquivo_de_saida}'...")
    df.to_csv(arquivo_de_saida, index=False)
    
    print("Processo de classificação concluído com sucesso!")

except FileNotFoundError:
    print(f"ERRO: O arquivo de entrada '{arquivo_de_entrada}' não foi encontrado. Certifique-se de que ele está na mesma pasta que o script.", file=sys.stderr)
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}", file=sys.stderr)
