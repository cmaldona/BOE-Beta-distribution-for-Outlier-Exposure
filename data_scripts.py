import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
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
from datasets import load_dataset, ClassLabel, Dataset


""" Collate function personalizada para DataLoader. """
def custom_collate_fn(batch):
    # Transformación de los códigos de generalización
    transform_code = {'far-OOD': -2, 'near-OOD': -1, 'covariate-shift': 0, 'ID': 1}
    generalisation_values = torch.tensor([transform_code[ex["generalisation"]] for ex in batch])

    # Extrae 'input_ids' y 'attention_mask', y aplica padding manualmente si es necesario
    input_ids = pad_sequence([torch.tensor(ex['input_ids']) for ex in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(ex['attention_mask']) for ex in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([ex['labels'] for ex in batch])

    # Retorna un diccionario con los componentes del batch correctamente alineados y con padding
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'generalisation': generalisation_values
    }


""" Codifica las etiquetas del lote utilizando LabelEncoder. """
def encode_labels(batch, le):
    batch['labels'] = [le.transform([label])[0] if label in le.classes_ else -1 for label in batch['labels']]
    return batch


tokenizer = AutoTokenizer.from_pretrained('roberta-base', padding_side="right")

# key: path_df, value: max_l
dfs_data = {
    'setup1_clinc.csv': 30,
    'setup2_nc.csv': 50,
    'setup3_trec.csv': 40,
    'setup4_newsgroups.csv': 512
}

mapping_dataset_name = {
    0: 'Clinc150',
    1: 'NCv3',
    2: 'TREC',
    3: 'NG20'
}

BATCH_SIZE = 16


def get_setups():
    experiments = {}

    for i, (df_path, max_l) in enumerate(dfs_data.items()):

        df = pd.read_csv(df_path)
        # Elimina datos con solo números, interpretados como float/int
        df = df[df['data'].apply(lambda x: isinstance(x, str))]

        dataset = Dataset.from_pandas(df)

        # Verificar si la columna 'Unnamed: 0' existe antes de intentar eliminarla
        if 'Unnamed: 0' in dataset.column_names:
            dataset = dataset.remove_columns('Unnamed: 0')

        # Configuración de codificadores
        label_encoder = LabelEncoder()
        classes_list_without_ood = list(
            set([row['labels'] for row in dataset if row['generalisation'] not in ['near-OOD', 'far-OOD']]))
        label_encoder.fit(classes_list_without_ood)

        dataset = dataset.map(encode_labels, batched=True, fn_kwargs={'le': label_encoder})
        tokenize_function = lambda df: tokenizer(df['data'], padding='max_length', truncation=True, max_length=max_l)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['data'])

        df = tokenized_dataset.to_pandas()
        if '__index_level_0__' in df.columns:
            df.drop('__index_level_0__', axis=1, inplace=True, errors='ignore')

        train_dataset = Dataset.from_pandas(df[df['group'] == 'train'])
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn
        )
        eval_dataset = Dataset.from_pandas(df[df['group'] == 'validation'])
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn
        )
        test_dataset = Dataset.from_pandas(df[df['group'] == 'test'])
        test_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn
        )

        experiments[f'setup {mapping_dataset_name[i]}'] = {
            'train_dataloader': train_dataloader,
            'eval_dataloader': eval_dataloader,
            'test_dataloader': test_dataloader
        }

    return experiments

