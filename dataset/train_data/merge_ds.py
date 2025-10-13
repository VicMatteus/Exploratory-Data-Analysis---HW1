import pandas as pd
import sys

# Configuração
nomes_dos_arquivos = ['frequency_domain_features_train.csv', 'heart_rate_non_linear_features_train.csv', 'time_domain_features_train.csv']
arquivo_de_saida = 'full_dataset.csv'
coluna_chave = 'uuid'

try:
    # Carrega o primeiro arquivo CSV, que servirá como base
    print(f"Lendo o arquivo base: {nomes_dos_arquivos[0]}...")
    df_mesclado = pd.read_csv(nomes_dos_arquivos[0])
    print(f" -> Sucesso! {len(df_mesclado)} linhas carregadas.")

    # Loop para mesclar os arquivos restantes (o segundo e o terceiro)
    for i in range(1, len(nomes_dos_arquivos)):
        nome_arquivo_atual = nomes_dos_arquivos[i]
        try:
            print(f"Lendo e mesclando o arquivo: {nome_arquivo_atual}...")
            
            # Carrega o próximo arquivo
            df_para_mesclar = pd.read_csv(nome_arquivo_atual)
            print(f" -> Sucesso! {len(df_para_mesclar)} linhas carregadas.")

            # 'how='inner'' garante que apenas as linhas com 'uuid' correspondentes
            df_mesclado = pd.merge(df_mesclado, df_para_mesclar, on=coluna_chave, how='inner')
            
            print(f" -> Mesclagem concluída. DataFrame agora tem {len(df_mesclado)} linhas.")

        except FileNotFoundError:
            print(f"ERRO: O arquivo '{nome_arquivo_atual}' não foi encontrado. Verifique o nome e o caminho.", file=sys.stderr)
            sys.exit(1)
        except KeyError:
            print(f"ERRO: A coluna chave '{coluna_chave}' não foi encontrada no arquivo '{nome_arquivo_atual}'.", file=sys.stderr)
            sys.exit(1)

    # Salva o DataFrame final em um novo arquivo CSV, sem o índice do pandas
    print(f"\nSalvando o resultado final em '{arquivo_de_saida}'...")
    df_mesclado.to_csv(arquivo_de_saida, index=False)
    
    print("✨ Processo concluído com sucesso!")

except FileNotFoundError:
    print(f"ERRO: O arquivo base '{nomes_dos_arquivos[0]}' não foi encontrado. O script não pode continuar.", file=sys.stderr)
except KeyError:
    print(f"ERRO: A coluna chave '{coluna_chave}' não foi encontrada no arquivo base '{nomes_dos_arquivos[0]}'.", file=sys.stderr)
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}", file=sys.stderr)