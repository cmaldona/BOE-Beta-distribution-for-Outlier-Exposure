import os
import re
import pprint
import time
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from transformers import (
    RobertaModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BertModel,
    BertTokenizer,
    AdamW
)
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from functions import *
from data_scripts import get_setups

# Archivo para registrar tiempos de ejecución
log_file_path = os.path.join('', 'execution_time_log.txt')
log_file = open(log_file_path, 'a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_id = 0
total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
used_memory = torch.cuda.memory_allocated(gpu_id)
reserved_memory = torch.cuda.memory_reserved(gpu_id)

# Información sobre memoria de la GPU
print(f"Total memory: {total_memory / 1e9} GB")
print(f"Used memory: {used_memory / 1e9} GB")
print(f"Reserved memory: {reserved_memory / 1e9} GB")

# Carga de datos y registro del tiempo
data_load_start_time = time.time()
experiments = get_setups()
data_load_end_time = time.time()
log_file.write(f"Data loading time: {data_load_end_time - data_load_start_time} seconds\n")

# Configuración de parámetros
EPOCHS = 10
params_grid = ParameterGrid({
    'setup': list(experiments.items()),
    'structure': ['peft'],
    'seed': [103,137,23,199,97,7,178,493,91,3],
    'regularizer': [1],
    'divergence_metric': ['tv', 'kl'],
    'distribution': ['Beta'],
})

# Configuración de PEFT
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# Rutas base y archivo de resultados
base_path = 'baseline'
excel_file = os.path.join(base_path, 'exps_oe.xlsx')

# Ejecución de experimentos
for params in params_grid:
    print('params:\n')
    pprint.pprint(params)

    all_results = []
    setup = params['setup'][1]
    N = 0
    sheet_name = f'seed {params["seed"]}'

    # Obtener clases del dataset
    generalisation = []
    train_dataloader = setup['train_dataloader']
    labels = []
    for batch in train_dataloader:
        N += batch['input_ids'].shape[0]
        generalisation.extend(batch['generalisation'].numpy())
        labels.extend(batch['labels'].numpy())

    mask_id = np.array(generalisation) == 1
    k_classes = len(set(np.array(labels)[mask_id]))

    # Configuración del dispositivo y pérdida personalizada
    device = prepare_environment(seed=params['seed'], device="cuda")
    custom_loss_function = CustomLoss(
        loss_type=params['divergence_metric'],
        weight_regularizer=params['regularizer'],
        num_classes=k_classes,
        distribution=params['distribution']
    )

    model, optimizer = initialize_model(params['structure'], k_classes, peft_config, device)

    # Entrenamiento y evaluación
    flag_peft = True if params['structure'] == 'peft' else False
    start_time_run = time.time()
    val_metrics, train_metrics = train_evaluate(
        model,
        setup['train_dataloader'],
        setup['eval_dataloader'],
        optimizer,
        EPOCHS,
        flag_peft,
        custom_loss_function
    )
    end_time_run = time.time()
    total_time = end_time_run - start_time_run
    log_file.write(f"Training time for {params}: {total_time} seconds\n" + "--" * 20)

    print(f"Training time: {total_time} seconds")
    print(train_metrics)
    print(val_metrics)

    # Guardar resultados
    model_path_saved = save_results_v2(model, val_metrics, train_metrics, params, base_path, k_classes)
    params_str = ', '.join([f'({key}:{value})' for key, value in params.items()])

    metrics = eval_model(model, setup['eval_dataloader'])
    metrics['tech'] = 'OE+' + params['distribution']
    metrics['params'] = params_str
    all_results.append(metrics)

    # Guardar resultados en Excel
    if not os.path.exists(excel_file):
        new_df = pd.DataFrame(columns=list(metrics.keys()))
        for dic_data in all_results:
            new_df.loc[len(new_df)] = list(dic_data.values())
        new_df.to_excel(excel_file, sheet_name=sheet_name)
    else:
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            if sheet_name in writer.sheets:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            else:
                df = pd.DataFrame(columns=list(metrics.keys()))
            for dic_data in all_results:
                if not set(dic_data.keys()).issubset(df.columns):
                    raise ValueError(
                        f"Mismatch between dictionary keys and DataFrame columns.\nKeys: {dic_data.keys()}\nColumns: {df.columns}"
                    )
                df = df.reindex(columns=dic_data.keys())
                df.loc[len(df)] = list(dic_data.values())
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    torch.cuda.empty_cache()
