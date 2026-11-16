#!/bin/bash

# ==============================================================================
# SCRIPT DE DEBUG: PHYSNET E PHYSMAMBA APENAS
# ==============================================================================

# --- Configurações ---
CONFIG_DIR="./configs/tcc_configs"           
CACHE_DIR="dataset_tcc_preprocessed"         
LOG_FILE="log_debug_physnet_mamba.txt"       # Log separado para não misturar

# --- Inicialização ---
echo "=== INÍCIO DO DEBUG (PhysNet & Mamba): $(date) ===" > "$LOG_FILE"
echo "Lendo configs de: $CONFIG_DIR" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "[ERRO] A pasta $CONFIG_DIR não foi encontrada!" | tee -a "$LOG_FILE"
    exit 1
fi

# --- Loop Principal ---
for config_file in "$CONFIG_DIR"/*.yaml; do
    
    # Verifica existência
    [ -e "$config_file" ] || continue

    # ==========================================================
    # FILTRO DE DEBUG: Pula tudo que não for PhysNet ou Mamba
    # ==========================================================
    if [[ "$config_file" != *"PhysNet"* ]]; then
        # Silenciosamente ignora os outros
        continue
    fi
    # ==========================================================

    echo "" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"
    echo ">>> EXECUTANDO DEBUG: $(basename "$config_file")" | tee -a "$LOG_FILE"
    echo ">>> DATA: $(date)" | tee -a "$LOG_FILE"
    echo "########################################################" | tee -a "$LOG_FILE"

    # 1. LIMPEZA DE CACHE (OBRIGATÓRIA)
    if [ -d "$CACHE_DIR" ]; then
        echo "[INFO] Limpando cache antigo em: $CACHE_DIR..." | tee -a "$LOG_FILE"
        rm -rf "$CACHE_DIR"
    fi

    # 2. EXECUÇÃO
    echo "[INFO] Iniciando python main.py..." | tee -a "$LOG_FILE"
    
    python main.py --config_file "$config_file" 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[SUCESSO] Finalizado: $(basename "$config_file")" | tee -a "$LOG_FILE"
    else
        echo "[FALHA] Erro crítico em: $(basename "$config_file")" | tee -a "$LOG_FILE"
        # Opcional: Parar imediatamente se um falhar para você ver o erro
        exit 1 
    fi

    sleep 2
done

echo "" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
echo "DEBUG CONCLUÍDO: $(date)" | tee -a "$LOG_FILE"
echo "Log salvo em: $LOG_FILE" | tee -a "$LOG_FILE"