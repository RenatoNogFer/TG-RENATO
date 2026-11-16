#!/bin/bash

# ==============================================================================
# SCRIPT DE AUTOMATIZAÇÃO DE EXPERIMENTOS TCC
# ==============================================================================

# --- Configurações ---
CONFIG_DIR="./configs/tcc_configs"           # Pasta onde você salvou os YAMLs
CACHE_DIR="dataset_tcc_preprocessed"         # Pasta de cache definida nos YAMLs (CACHED_PATH)
LOG_FILE="log_execucao_tcc_completo_teste_nan.txt"     # Arquivo onde tudo será salvo

# --- Inicialização ---
# Limpa o arquivo de log anterior (cria um novo vazio)
echo "=== INÍCIO DA BATERIA DE TESTES: $(date) ===" > "$LOG_FILE"
echo "Configurações serão lidas de: $CONFIG_DIR" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

# Verifica se a pasta de configs existe
if [ ! -d "$CONFIG_DIR" ]; then
    echo "[ERRO] A pasta $CONFIG_DIR não foi encontrada!" | tee -a "$LOG_FILE"
    exit 1
fi

# Ativa modo de parada em erro grave (opcional, comente se quiser que continue mesmo se um falhar)
# set -e 

# --- Loop Principal ---
# Itera sobre cada arquivo .yaml dentro da pasta
for config_file in "$CONFIG_DIR"/*.yaml; do
    
    # Verifica se encontrou arquivos (caso a pasta esteja vazia)
    [ -e "$config_file" ] || continue

    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo ">>> EXECUTANDO: $(basename "$config_file")" | tee -a "$LOG_FILE"
    echo ">>> DATA: $(date)" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"

    # 1. LIMPEZA DE CACHE (CRÍTICO)
    # Apaga a pasta de pré-processamento para garantir que o novo modelo
    # gere os dados no formato (H, W, Chunk) que ele precisa.
    if [ -d "$CACHE_DIR" ]; then
        echo "[INFO] Limpando cache antigo em: $CACHE_DIR..." | tee -a "$LOG_FILE"
        rm -rf "$CACHE_DIR"
    else
        echo "[INFO] Cache limpo (pasta não existia)." | tee -a "$LOG_FILE"
    fi

    # 2. EXECUÇÃO DO EXPERIMENTO
    # O comando '2>&1' redireciona erros (stderr) para a saída padrão (stdout)
    # O comando 'tee -a' mostra na tela E anexa no arquivo de log
    echo "[INFO] Iniciando python main.py..." | tee -a "$LOG_FILE"
    
    python main.py --config_file "$config_file" 2>&1 | tee -a "$LOG_FILE"

    # Verifica status de saída
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[SUCESSO] Finalizado: $(basename "$config_file")" | tee -a "$LOG_FILE"
    else
        echo "[FALHA] Erro ao executar: $(basename "$config_file")" | tee -a "$LOG_FILE"
        # Descomente a linha abaixo se quiser parar tudo ao encontrar um erro
        # exit 1 
    fi

    # Pequena pausa para garantir que I/O de disco termine
    sleep 2
done

echo "" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "BATERIA DE TESTES CONCLUÍDA: $(date)" | tee -a "$LOG_FILE"
echo "Log completo salvo em: $LOG_FILE" | tee -a "$LOG_FILE"