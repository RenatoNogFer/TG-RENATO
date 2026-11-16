import numpy as np
import os
import matplotlib.pyplot as plt

# Caminho raiz onde a pasta 'dataset_tcc_preprocessed' está
# (Aponte para a pasta GERAL, ele vai achar a subpasta sozinho)
SEARCH_ROOT = "/home/voxar-labs/Documentos/TCC-Renato/rPPG-Toolbox/dataset_tcc_preprocessed"

def find_and_inspect():
    print(f"Procurando arquivos .npy recursivamente em: {SEARCH_ROOT} ...")
    
    npy_files = []
    for root, dirs, files in os.walk(SEARCH_ROOT):
        for file in files:
            if file.endswith("_input0.npy"): # Pega apenas o primeiro chunk de cada vídeo
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        print("ERRO: Nenhum arquivo '_input0.npy' encontrado!")
        print("Verifique se o path está correto ou se o preprocessamento rodou.")
        return

    print(f"Encontrados {len(npy_files)} arquivos de vídeo processados.")
    
    # Analisa o primeiro arquivo encontrado
    target_file = npy_files[0]
    print(f"\n==========================================")
    print(f"ANALISANDO: {os.path.basename(target_file)}")
    print(f"Caminho: {target_file}")
    print(f"==========================================")

    try:
        data = np.load(target_file)
        # O formato esperado do BaseLoader é (N, H, W, 6) ou (N, 6, H, W)
        print(f"Shape: {data.shape}")
        print(f"Tipo: {data.dtype}")
        
        # Separa canais (Assumindo formato padrão do DeepPhys: Diff + Raw)
        # Geralmente os 3 primeiros são DiffNormalized e os 3 últimos são Raw Standardized
        if data.shape[-1] == 6:
            diff = data[..., :3]
            raw = data[..., 3:]
        elif data.shape[1] == 6: # Se estiver transposto
            diff = data[:, :3, ...]
            raw = data[:, 3:, ...]
        else:
            print("Aviso: Formato de canais estranho. Analisando tudo junto.")
            diff = data
            raw = data

        # ESTATÍSTICAS CRÍTICAS
        std_diff = np.std(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n--- Movimento (Diff) ---")
        print(f"Min: {np.min(diff):.5f}")
        print(f"Max: {np.max(diff):.5f}")
        print(f"Std (Variância): {std_diff:.5f}")
        
        print(f"\n--- Aparência (Raw) ---")
        print(f"Min: {np.min(raw):.5f}")
        print(f"Max: {np.max(raw):.5f}")
        
        # DIAGNÓSTICO
        if std_diff == 0 or np.max(diff) == np.min(diff):
            print("\n[FALHA] O sinal de movimento está MORTO (Zerado/Constante).")
            print("Causa: O corte do rosto falhou e gerou frames pretos, ou a normalização falhou.")
        elif np.isnan(np.sum(data)):
            print("\n[FALHA] O arquivo contém NaNs (Not a Number).")
        else:
            print("\n[SUCESSO] O arquivo parece conter dados válidos!")
            print("Se o teste ainda der métricas ruins, o problema é o modelo PURE_DeepPhys não casar com esses dados.")

    except Exception as e:
        print(f"Erro fatal ao ler arquivo: {e}")

if __name__ == "__main__":
    find_and_inspect()