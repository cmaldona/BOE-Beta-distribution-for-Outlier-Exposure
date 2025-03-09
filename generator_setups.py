import pandas as pd
import json
import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

# Configuración inicial
SEED = 42
np.random.seed(SEED)

# Funciones de utilidad
def asignar_grupo(df, train_frac=0.7, val_frac=0.1, test_frac=0.2):
    assert train_frac + val_frac + test_frac == 1, "Las fracciones deben sumar 1"
    n = len(df)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    grupos = ['train'] * train_size + ['validation'] * val_size + ['test'] * (n - train_size - val_size)
    np.random.shuffle(grupos)
    df['group'] = grupos

def asignar_grupos_con_tamanos_disjuntos(df, train_val_frac=0.7, test_frac=0.3):
    assert train_val_frac + test_frac == 1, "Las fracciones deben sumar 1"
    clases = df['labels'].value_counts().to_frame('count')
    total_samples = len(df)
    test_sample_goal = int(test_frac * total_samples)
    test_labels = []
    accumulated_test_samples = 0

    for label, count in clases.sample(frac=1).iterrows():
        if accumulated_test_samples + count['count'] <= test_sample_goal:
            test_labels.append(label)
            accumulated_test_samples += count['count']
        if accumulated_test_samples >= test_sample_goal:
            break

    df['group'] = np.where(df['labels'].isin(test_labels), 'test', None)
    remaining_indices = df[df['group'].isnull()].index.tolist()
    np.random.shuffle(remaining_indices)
    cutoff = int(train_val_frac * len(remaining_indices))
    train_indices = remaining_indices[:cutoff]
    val_indices = remaining_indices[cutoff:]
    df.loc[train_indices, 'group'] = 'train'
    df.loc[val_indices, 'group'] = 'validation'
    return df

# Procesamiento de dataset CLINC
with open('data_full.json', 'r') as f:
    data = json.load(f)

df_rows = []
for group in data.keys():
    for text, label in data[group]:
        data_type = 'near-OOD' if 'oos' in group else 'ID'
        real_group = group.replace('oos_', '')
        label = label if label != 'oos' else 'ood'
        df_rows.append({'data': text, 'labels': label, 'group': real_group, 'generalisation': data_type})

df_clinc = pd.DataFrame(df_rows)
df_clinc['group'] = df_clinc['group'].apply(lambda x: 'validation' if x == 'val' else x)

# FarOOD Train - SST2
dataset_sst2 = load_dataset("sst2", trust_remote_code=True)
n_far_ood_eval = df_clinc[df_clinc['group'] == 'validation'][df_clinc['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_sst2 = dataset_sst2['train'].shuffle(seed=42).select(range(n_far_ood_eval))
rows = [{'data': row['sentence'], 'labels': row['label'], 'group': 'train', 'generalisation': 'far-OOD'} for row in sampled_dataset_sst2]
df_far_ood = pd.DataFrame(rows, columns=['data', 'labels', 'group', 'generalisation'])
df_clinc = pd.concat([df_clinc, df_far_ood], ignore_index=True)

# FarOOD Eval - Yelp
yelp_dataset = load_dataset("yelp_review_full", trust_remote_code=True)
n_far_ood_train = df_clinc[df_clinc['group'] == 'train'][df_clinc['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset = yelp_dataset['train'].shuffle(seed=42).select(range(n_far_ood_train))
rows = [{'data': row['text'], 'labels': 'ood', 'group': 'validation', 'generalisation': 'far-OOD'} for row in sampled_dataset]
df_far_ood_train = pd.DataFrame(rows, columns=['data', 'labels', 'group', 'generalisation'])
df_clinc = pd.concat([df_clinc, df_far_ood_train], axis=0)
df_clinc.reset_index(drop=True, inplace=True)

# FarOOD Test - News
new_df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
new_df = new_df.sample(1000)[['category', 'headline']]
new_df.rename(columns={'category': 'labels', 'headline': 'data'}, inplace=True)
new_df['group'] = 'test'
new_df['generalisation'] = 'far-OOD'
df_clinc = pd.concat([df_clinc, new_df], ignore_index=True)

df_clinc.to_csv('setup1_clinc.csv')

# Procesamiento de dataset News Category
df_nc = pd.read_json('News_Category_Dataset_v3.json', lines=True)[['category', 'headline']]
df_nc.rename(columns={'category': 'labels', 'headline': 'data'}, inplace=True)
N_largest = 30
top_categorias = df_nc['labels'].value_counts().nlargest(N_largest).index
df_nc_id = df_nc[df_nc['labels'].isin(top_categorias)]
df_nc_id['generalisation'] = 'ID'
df_nc_ood = df_nc[~df_nc['labels'].isin(top_categorias)]
df_nc_ood['generalisation'] = 'near-OOD'
asignar_grupo(df_nc_id)
asignar_grupos_con_tamanos_disjuntos(df_nc_ood)
df_nc = pd.concat([df_nc_id, df_nc_ood], ignore_index=True)

# FarOOD Train - SNLI
dataset_snli = load_dataset("stanfordnlp/snli", trust_remote_code=True)
df_snli = pd.DataFrame({'data': dataset_snli['train']['hypothesis'], 'labels': dataset_snli['train']['label'], 'group': 'train', 'generalisation': 'far-OOD'})
n_far_ood_train = df_nc[df_nc['group'] == 'train'][df_nc['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_snli = df_snli.sample(n_far_ood_train)
df_nc = pd.concat([df_nc, sampled_dataset_snli], ignore_index=True)

# FarOOD Eval - Yelp
yelp_dataset = load_dataset("yelp_review_full", trust_remote_code=True)
df_yelp = pd.DataFrame({'data': yelp_dataset['train']['text'], 'labels': 'ood', 'group': 'validation', 'generalisation': 'far-OOD'})
n_far_ood_eval = df_nc[df_nc['group'] == 'validation'][df_nc['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_yelp = df_yelp.sample(n_far_ood_eval)
df_nc = pd.concat([df_nc, sampled_dataset_yelp], axis=0)
df_nc.reset_index(drop=True, inplace=True)

# FarOOD Test - SST2
sst2_dataset = load_dataset("sst2", trust_remote_code=True)
df_sst2 = pd.DataFrame({'data': sst2_dataset['train']['sentence'], 'labels': sst2_dataset['train']['label'], 'group': 'test', 'generalisation': 'far-OOD'})
n_far_ood_test = df_nc[df_nc['group'] == 'test'][df_nc['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_sst2 = df_sst2.sample(n_far_ood_test)
df_nc = pd.concat([df_nc, sampled_dataset_sst2], axis=0)
df_nc.reset_index(drop=True, inplace=True)

df_nc.to_csv('setup2_nc.csv')

# Procesamiento de dataset TREC
N_largest = 4
dataset_trec = load_dataset("trec", trust_remote_code=True)
df_trec = pd.DataFrame({'data': dataset_trec['train']['text'], 'labels': dataset_trec['train']['coarse_label']})
top_categorias = df_trec['labels'].value_counts().nlargest(N_largest).index
df_trec_id = df_trec[df_trec['labels'].isin(top_categorias)]
df_trec_id['generalisation'] = 'ID'
df_trec_ood = df_trec[~df_trec['labels'].isin(top_categorias)]
df_trec_ood['generalisation'] = 'near-OOD'
asignar_grupo(df_trec_id)
asignar_grupos_con_tamanos_disjuntos(df_trec_ood)
df_trec = pd.concat([df_trec_id, df_trec_ood], ignore_index=True)

# FarOOD Train - SST2
sst2_dataset = load_dataset("sst2", trust_remote_code=True)
df_sst2_train = pd.DataFrame({'data': sst2_dataset['train']['sentence'], 'labels': sst2_dataset['train']['label'], 'group': 'train', 'generalisation': 'far-OOD'})
n_far_ood_train = df_trec[df_trec['group'] == 'train'][df_trec['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_sst2 = df_sst2_train.sample(n_far_ood_train)
df_trec = pd.concat([df_trec, sampled_dataset_sst2], axis=0)
df_trec.reset_index(drop=True, inplace=True)

# FarOOD Eval - Yelp
yelp_dataset = load_dataset("yelp_review_full", trust_remote_code=True)
df_yelp_eval = pd.DataFrame({'data': yelp_dataset['train']['text'], 'labels': 'ood', 'group': 'validation', 'generalisation': 'far-OOD'})
n_far_ood_eval = df_trec[df_trec['group'] == 'validation'][df_trec['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_yelp = df_yelp_eval.sample(n_far_ood_eval)
df_trec = pd.concat([df_trec, sampled_dataset_yelp], axis=0)
df_trec.reset_index(drop=True, inplace=True)

# FarOOD Test - SNLI
dataset_snli = load_dataset("stanfordnlp/snli", trust_remote_code=True)
df_snli_test = pd.DataFrame({'data': dataset_snli['train']['hypothesis'], 'labels': dataset_snli['train']['label'], 'group': 'test', 'generalisation': 'far-OOD'})
n_far_ood_test = df_trec[df_trec['group'] == 'test'][df_trec['generalisation'] == 'near-OOD'].shape[0]
sampled_dataset_snli = df_snli_test.sample(n_far_ood_test)
df_trec = pd.concat([df_trec, sampled_dataset_snli], ignore_index=True)

df_trec.to_csv('setup3_trec.csv')

# Procesamiento de dataset 20 Newsgroups
N_largest = 15
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, random_state=SEED)
df_ng = pd.DataFrame({'data': newsgroups.data, 'labels': newsgroups.target})
df_ng['labels'] = df_ng['labels'].apply(lambda x: newsgroups.target_names[x])
top_categorias = df_ng['labels'].value_counts().nlargest(N_largest).index
df_ng_id = df_ng[df_ng['labels'].isin(top_categorias)]
df_ng_id['generalisation'] = 'ID'
df_ng_ood = df_ng[~df_ng['labels'].isin(top_categorias)]
df_ng_ood['generalisation'] = 'near-OOD'
asignar_grupo(df_ng_id)
asignar_grupos_con_tamanos_disjuntos(df_ng_ood)
df_ng_combined = pd.concat([df_ng_id, df_ng_ood], ignore_index=True)

# FarOOD Train & Eval - Reddit
reddit_dataset = load_dataset("reddit", split='train', streaming=True, trust_remote_code=True)
df_reddit = pd.DataFrame(reddit_dataset.take(20000))
df_reddit = pd.DataFrame({'data': df_reddit['content'], 'labels': df_reddit['subreddit']})
unique_labels = df_reddit['labels'].unique()
split_point = int(len(unique_labels) / 2)
train_labels = unique_labels[:split_point]
validation_labels = unique_labels[split_point:]
df_reddit_train = df_reddit[df_reddit['labels'].isin(train_labels)]
df_reddit_train['group'] = 'train'
df_reddit_train['generalisation'] = 'far-OOD'
df_reddit_validation = df_reddit[df_reddit['labels'].isin(validation_labels)]
df_reddit_validation['group'] = 'validation'
df_reddit_validation['generalisation'] = 'far-OOD'

n_far_ood_train = min(df_ng_combined[df_ng_combined['generalisation'] == 'near-OOD'].shape[0], df_reddit_train.shape[0])
sampled_dataset_reddit_train = df_reddit_train.sample(n_far_ood_train, random_state=SEED)
n_far_ood_validation = min(df_ng_combined[df_ng_combined['group'] == 'validation'][df_ng_combined['generalisation'] == 'near-OOD'].shape[0], df_reddit_validation.shape[0])
sampled_dataset_reddit_validation = df_reddit_validation.sample(n_far_ood_validation, random_state=SEED)

df_final_combined = pd.concat([df_ng_combined, sampled_dataset_reddit_train, sampled_dataset_reddit_validation], axis=0)
df_final_combined.reset_index(drop=True, inplace=True)

# FarOOD Test - IMDb
imdb_dataset = load_dataset("imdb", split='train', trust_remote_code=True)
df_imdb = pd.DataFrame({'data': imdb_dataset['text'], 'labels': imdb_dataset['label'], 'group': 'test', 'generalisation': 'far-OOD'})
n_far_ood_test = min(df_ng_combined[df_ng_combined['group'] == 'test'][df_ng_combined['generalisation'] == 'near-OOD'].shape[0], df_imdb.shape[0])
sampled_dataset_imdb = df_imdb.sample(n_far_ood_test, random_state=SEED)
df_final_combined = pd.concat([df_final_combined, sampled_dataset_imdb], ignore_index=True)

# Guardar el dataset combinado final
df_final_combined.to_csv('setup4_newsgroups.csv', index=False)

