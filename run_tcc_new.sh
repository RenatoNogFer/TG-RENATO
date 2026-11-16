#!/bin/bash

# ==============================================================================
# SCRIPT DE AUTOMATIZAÇÃO DE EXPERIMENTOS TCC
# ==============================================================================

CONFIG_ROOT="./configs/tcc_configs"
CACHE_DIR="dataset_tcc_preprocessed"
RESULTS_ROOT="./evaluation_outputs"
LOG_ROOT="./logs_tcc"

MASTER_LOG="$LOG_ROOT/log_execucao_tcc_completo.txt"

mkdir -p "$RESULTS_ROOT"
mkdir -p "$LOG_ROOT"

echo "=== INÍCIO DA BATERIA DE TESTES: $(date) ===" > "$MASTER_LOG"
echo "Configurações serão lidas de: $CONFIG_ROOT" | tee -a "$MASTER_LOG"
echo "Resultados serão salvos em: $RESULTS_ROOT" | tee -a "$MASTER_LOG"
echo "--------------------------------------------------------" | tee -a "$MASTER_LOG"

if [ ! -d "$CONFIG_ROOT" ]; then
    echo "[ERRO] A pasta $CONFIG_ROOT não foi encontrada!" | tee -a "$MASTER_LOG"
    exit 1
fi

# Percorre YAMLs em subpastas:
# configs/tcc_configs/<PRETRAIN_DATASET>/TCC_<TECNICA>.yaml
find "$CONFIG_ROOT" -mindepth 2 -maxdepth 2 \( -name "*.yaml" -o -name "*.YAML" \) | sort | while read -r config_file; do

    PRETRAIN_DATASET=$(basename "$(dirname "$config_file")")

    config_base=$(basename "$config_file")
    config_base_no_ext="${config_base%.*}"

    # Remove prefixo TCC_ se existir
    MODEL_NAME="${config_base_no_ext#TCC_}"

    EXP_ID="${MODEL_NAME}_${PRETRAIN_DATASET}"

    EXP_OUTPUT_DIR="$RESULTS_ROOT/$EXP_ID"
    EXP_LOG="$LOG_ROOT/${EXP_ID}.log"
    COMPLETED_FILE="$EXP_OUTPUT_DIR/COMPLETED"

    echo "" | tee -a "$MASTER_LOG"
    echo "########################################################" | tee -a "$MASTER_LOG"
    echo ">>> EXPERIMENTO: $EXP_ID" | tee -a "$MASTER_LOG"
    echo ">>> MODELO/TÉCNICA: $MODEL_NAME" | tee -a "$MASTER_LOG"
    echo ">>> DATASET PRÉ-TREINO: $PRETRAIN_DATASET" | tee -a "$MASTER_LOG"
    echo ">>> CONFIG: $config_file" | tee -a "$MASTER_LOG"
    echo ">>> DATA: $(date)" | tee -a "$MASTER_LOG"
    echo "########################################################" | tee -a "$MASTER_LOG"

    # Se o marcador de conclusão existe, pula
    if [ -f "$COMPLETED_FILE" ]; then
        echo "[SKIP] Experimento já concluído: $EXP_ID" | tee -a "$MASTER_LOG"
        echo "[SKIP] Marcador encontrado: $COMPLETED_FILE" | tee -a "$MASTER_LOG"
        continue
    fi

    echo "[INFO] Experimento incompleto ou não realizado." | tee -a "$MASTER_LOG"
    echo "[INFO] Limpando saída anterior: $EXP_OUTPUT_DIR" | tee -a "$MASTER_LOG"
    echo "[INFO] Limpando cache: $CACHE_DIR" | tee -a "$MASTER_LOG"

    rm -rf "$EXP_OUTPUT_DIR"
    rm -rf "$CACHE_DIR"

    mkdir -p "$EXP_OUTPUT_DIR"

    # Variáveis lidas depois pelo metrics.py
    export TCC_EXPERIMENT_ID="$EXP_ID"
    export TCC_MODEL_NAME="$MODEL_NAME"
    export TCC_PRETRAIN_DATASET="$PRETRAIN_DATASET"
    export TCC_OUTPUT_DIR="$EXP_OUTPUT_DIR"
    export TCC_CONFIG_FILE="$config_file"

    echo "[INFO] TCC_EXPERIMENT_ID=$TCC_EXPERIMENT_ID" | tee -a "$MASTER_LOG"
    echo "[INFO] TCC_MODEL_NAME=$TCC_MODEL_NAME" | tee -a "$MASTER_LOG"
    echo "[INFO] TCC_PRETRAIN_DATASET=$TCC_PRETRAIN_DATASET" | tee -a "$MASTER_LOG"
    echo "[INFO] TCC_OUTPUT_DIR=$TCC_OUTPUT_DIR" | tee -a "$MASTER_LOG"
    echo "[INFO] Log individual: $EXP_LOG" | tee -a "$MASTER_LOG"

    python main.py --config_file "$config_file" 2>&1 | tee "$EXP_LOG"

    status=${PIPESTATUS[0]}

    if [ $status -eq 0 ]; then
        echo "[SUCESSO] Finalizado: $EXP_ID" | tee -a "$MASTER_LOG"

        # Pode ser criado pelo script por simplicidade.
        # Depois, se preferir, o metrics.py pode assumir essa responsabilidade.
        if [ ! -f "$COMPLETED_FILE" ]; then
            echo "completed_at=$(date)" > "$COMPLETED_FILE"
            echo "experiment_id=$EXP_ID" >> "$COMPLETED_FILE"
            echo "model_name=$MODEL_NAME" >> "$COMPLETED_FILE"
            echo "pretrain_dataset=$PRETRAIN_DATASET" >> "$COMPLETED_FILE"
            echo "config_file=$config_file" >> "$COMPLETED_FILE"
        fi
    else
        echo "[FALHA] Erro ao executar: $EXP_ID" | tee -a "$MASTER_LOG"
        echo "[INFO] O experimento será reexecutado do zero na próxima bateria." | tee -a "$MASTER_LOG"
        echo "[INFO] Verifique o log: $EXP_LOG" | tee -a "$MASTER_LOG"
    fi

    unset TCC_EXPERIMENT_ID
    unset TCC_MODEL_NAME
    unset TCC_PRETRAIN_DATASET
    unset TCC_OUTPUT_DIR
    unset TCC_CONFIG_FILE

    sleep 2
done

echo "" | tee -a "$MASTER_LOG"
echo "========================================================" | tee -a "$MASTER_LOG"
echo "BATERIA DE TESTES CONCLUÍDA: $(date)" | tee -a "$MASTER_LOG"
echo "Log completo salvo em: $MASTER_LOG" | tee -a "$MASTER_LOG"