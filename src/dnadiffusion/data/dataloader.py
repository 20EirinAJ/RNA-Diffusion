import os
import pickle
import random
import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from dnadiffusion.utils.utils import one_hot_encode

# ロガーの設定
logging.basicConfig(filename='logfile.log', level=logging.INFO)



def load_data(
    data_path: str = "Combined_4R_dataframe.txt",
    saved_data_path: str = "encode_data.pkl",
    subset_list: List = [
        "A_4R",
        "B_4R",
        """
        "GM12878_ENCLB441ZZZ",
        "hESCT0_ENCLB449ZZZ",
        "K562_ENCLB843GMH",
        "HepG2_ENCLB029COU",
        """
    ],
    limit_total_sequences: int = 0,
    num_sampling_to_compare_cells: int = 1000,
    load_saved_data: bool = False,
):
    # Preprocessing data
    if load_saved_data:
        with open(saved_data_path, "rb") as f:
            encode_data = pickle.load(f)

    else:
    # encode_data には、generate_motifs_and_fastas 関数で生成されたデータが格納されています。
    # このデータは辞書として保存されており、レーニング、テスト、およびシャッフルデータセットに関連する情報を含みます。
        encode_data = preprocess_data(
            input_csv=data_path,
            subset_list=subset_list,
            limit_total_sequences=limit_total_sequences,
            number_of_sequences_to_motif_creation=num_sampling_to_compare_cells,
        )

    # Splitting enocde data into train/test/shuffle
    # "train" キーでアクセスした後、さらに "motifs" キーでアクセスすることで、
    # トレーニングデータセットに関連するモチーフ情報にアクセスできる。

    # トレーニングデータセットのモチーフ情報にアクセス
    train_motifs = encode_data["train"]["motifs"]
    train_motifs_cell_specific = encode_data["train"]["final_subset_motifs"]

    # テストデータセットのモチーフ情報にアクセス
    test_motifs = encode_data["test"]["motifs"]
    test_motifs_cell_specific = encode_data["test"]["final_subset_motifs"]

    # シャッフルデータセットのモチーフ情報にアクセス
    shuffle_motifs = encode_data["train_shuffled"]["motifs"]
    shuffle_motifs_cell_specific = encode_data["train_shuffled"]["final_subset_motifs"]

    # データセットのモチーフ情報を収集しました

    # Creating sequence dataset
    # シーケンスデータセットを作成

    # トレーニングデータセットからデータを取得
    df = encode_data["train"]["df"]
    nucleotides = ["A", "C", "G", "T"]

    # DNAシーケンスをOne-Hotエンコード
    x_train_seq = np.array([one_hot_encode(x, nucleotides, 200) for x in df["sequence"] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tag_to_numeric = {x: n for n, x in enumerate(df["TAG"].unique(), 1)}
    numeric_to_tag = dict(enumerate(df["TAG"].unique(), 1))
    cell_types = list(numeric_to_tag.keys())
    x_train_cell_type = torch.tensor([tag_to_numeric[x] for x in df["TAG"]])

    # Collecting variables into a dict
    # 変数を辞書にまとめなおす。
    encode_data_dict = {
        "train_motifs": train_motifs,
        "train_motifs_cell_specific": train_motifs_cell_specific,
        "test_motifs": test_motifs,
        "test_motifs_cell_specific": test_motifs_cell_specific,
        "shuffle_motifs": shuffle_motifs,
        "shuffle_motifs_cell_specific": shuffle_motifs_cell_specific,
        "tag_to_numeric": tag_to_numeric,
        "numeric_to_tag": numeric_to_tag,
        "cell_types": cell_types,
        "X_train": X_train,
        "x_train_cell_type": x_train_cell_type,
    }

    """encode_data_dict には、以下のようなデータが格納されている。
    load_data関数が前処理して、データセットは結局、以下の2点のみを使用して作った
    ・シーケンスデータ
    ・TAG(各シーケンスがどのセルタイプにあったかを表す)

    :??_motifs: gimmescanを使って計算された全モチーフ
    :??_motifs_cell_specific: 各セルタイプに特有のモチーフ
    :tag_to_numeric: タグを数値に変換する辞書
    :numeric_to_tag: 数値をタグに変換する辞書
    :cell_types: セルタイプのリスト
    :X_train: OneHotしたトレーニングデータセットのシーケンスデータ
    :x_train_cell_type: トレーニングデータセットのセルタイプ
    """



    # シーケンスデータセットと関連情報を含む辞書を返す
    return encode_data_dict


def motifs_from_fasta(fasta: str):
    """外部サービスを使ってモチーフを計算する。
    generate_motifs_and_fastas 関数で使用される。

    :param fasta: _description_
    :type fasta: str
    :return: _description_
    :rtype: _type_
    """
    print("Computing Motifs....")

    # 外部サービスを使ってモチーフを計算する。
    os.system(f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 -n 20> train_results_motifs.bed")

    # 計算結果のファイルを読み込んで、モチーフ列を抽出する。
    df_results_seq_guime = pd.read_csv("train_results_motifs.bed", sep="\t", skiprows=5, header=None)


    print(df_results_seq_guime.head())

    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])

    # 各モチーフが何回出現したかカウントする。
    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_results_seq_guime_count_out = df_results_seq_guime[[0, "motifs"]].groupby("motifs").count()

    # どんなデータフレームか確認する。
    print(df_results_seq_guime_count_out.head())

    return df_results_seq_guime_count_out


def save_fasta(df: pd.DataFrame, name: str, num_sequences: int, seq_to_subset_comp: bool = False) -> str:
    """与えられたデータフレーム（df）からFASTAファイルを生成し、保存する。
    generate_motifs_and_fastas 関数で使用される。

    :param df: _description_
    :type df: pd.DataFrame
    :param name: _description_
    :type name: str
    :param num_sequences: _description_
    :type num_sequences: int
    :param seq_to_subset_comp: _description_, defaults to False
    :type seq_to_subset_comp: bool, optional
    :return: _description_
    :rtype: str
    """


    fasta_path = f"{name}.fasta"
    # 生成するFASTAファイルを書き込みモードで開きます。これにより、ファイルへの書き込みが可能。
    save_fasta_file = open(fasta_path, "w")

    # データフレーム df の行数を取得し、num_to_sample として保持
    # この値はサンプリングするシーケンスの数を制御
    num_to_sample = df.shape[0]

    # Subsetting sequences
    # num_sequences が非ゼロかつ seq_to_subset_comp が True の場合に、
    # サンプリングするシーケンス数を num_sequences に設定します。
    # それ以外の場合はデータフレーム内のすべてのシーケンスを使用します。
    if num_sequences and seq_to_subset_comp:
        num_to_sample = num_sequences

    # Sampling sequences
    print(f"Sampling {num_to_sample} sequences")

    # fastaファイルに書き込む文字列を生成する。
    # FASTA形式のエントリーを作成しています。各エントリーは > で始まり、
    # シーケンス名やタグ、実際のシーケンスデータを含みます。
    write_fasta_component = "\n".join(
        df[["sequence_id", "sequence", "TAG"]]
        .head(num_to_sample)
        .apply(lambda x: f">{x[0]}_TAG_{x[2]}\n{x[1]}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    return fasta_path


def generate_motifs_and_fastas(
    df: pd.DataFrame, name: str, num_sequences: int, subset_list: Optional[List] = None
) -> Dict[str, Any]:
    """与えられたデータフレーム（df）からモチーフとFASTAファイルを生成する関数。
    preprocess_data 関数で使用される。

    :param df: preprocess_data 関数で生成されたデータフレーム。訓練、テスト、シャッフル
    :type df: pd.DataFrame
    :param name: _description_
    :type name: str
    :param num_sequences: _description_
    :type num_sequences: int
    :param subset_list: _description_, defaults to None
    :type subset_list: Optional[List], optional
    :return: _description_
    :rtype: Dict[str, Any]
    """

    print("Generating Motifs and Fastas...", name)
    print("---" * 10)



    # Saving fasta
    if subset_list:
        fasta_path = save_fasta(df, f"{name}_{'_'.join([str(c) for c in subset_list])}", num_sequences)
    else:
        fasta_path = save_fasta(df, name, num_sequences)
    

    # motifs_from_fasta 関数を使って、FASTAファイルからモチーフを計算します。
    motifs = motifs_from_fasta(fasta_path)

    # Generating subset specific motifs
    final_subset_motifs = {}

    #  データフレーム df を "TAG" 列でグループ化し、各サブセットに対して以下の処理を行います。
    for comp, v_comp in df.groupby("TAG"):
        print(comp)
        c_fasta = save_fasta(v_comp, f"{name}_{comp}", num_sequences, seq_to_subset_comp=True)
        final_subset_motifs[comp] = motifs_from_fasta(c_fasta)

    # 辞書型で返す。
    return {
        "fasta_path": fasta_path,                   # FASTAファイルのパス
        "motifs": motifs,                           # モチーフ
        "final_subset_motifs": final_subset_motifs, # サブセット特有のモチーフ
        "df": df,                                   # データフレーム
    }

# Function to perform stratified sampling based on proportions
def stratified_sampling(df, test_prop, train_prop):
    # Split into test and temp (train + train_shuffled) datasets
    df_temp, df_test = train_test_split(df, test_size=test_prop, random_state=42, shuffle=True)
    
    # Calculate the proportion of train data relative to (train + train_shuffled)
    actual_train_prop = train_prop / (1 - test_prop)
    
    # Split temp into train and train_shuffled datasets
    df_train, df_train_shuffled = train_test_split(df_temp, test_size=(1 - actual_train_prop), random_state=42, shuffle=True)
    
    return df_train, df_train_shuffled, df_test


def preprocess_data(
    input_csv: str,
    subset_list: Optional[List] = None,
    limit_total_sequences: Optional[int] = None,
    number_of_sequences_to_motif_creation: int = 1000,
    save_output: bool = True,
):
    # Reading the csv file
    df = pd.read_csv(input_csv, sep="\t")

    # Subsetting the dataframe
    # サブセットリストにあるタグのみを抽出する。
    if subset_list:
        print(" or ".join([f"TAG == {c}" for c in subset_list]))
        df = df.query(" or ".join([f'TAG == "{c}" ' for c in subset_list]))
        print("Subsetting...")

    # Limiting the total number of sequences
    # シーケンス総数の制限
    if limit_total_sequences > 0:
        print(f"Limiting total sequences to {limit_total_sequences}")
        df = df.sample(limit_total_sequences)

    # このshuffled_dfを作成する工程を少し変更する。

    """
    # Creating train/test/shuffle groups
    # chr 列が "chr1" であるすべての行を抽出して新しいデータフレーム df_test を作成します。
    df_test = df[df["chr"] == "chr1"].reset_index(drop=True)

    # chr 列が "chr2" であるすべての行を抽出して新しいデータフレーム df_train_shuffled を作成します。
    df_train_shuffled = df[df["chr"] == "chr2"].reset_index(drop=True)

    # chr 列が "chr1" または "chr2" でないすべての行を抽出して新しいデータフレーム df_train を作成します。
    df_train = df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True)

    # df_train_shuffled の sequence 列のDNA配列をランダムに並び替えます。
    df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )


    """
    # Filter data by TAG to perform stratified sampling
    # アプタマーのデータに対応した抽出方法
    df_A_4R = df[df['TAG'] == 'A_4R']
    df_B_4R = df[df['TAG'] == 'B_4R']

    # Define the proportion s for the splits
    # train,test,train_shuffled に分割する割合を定義する。
    test_prop = 0.09
    train_prop = 0.08
    train_shuffle_prop = 0.81 # Remaining proportion will automatically become this

    # Perform stratified sampling for each TAG
    df_train_A_4R, df_train_shuffled_A_4R, df_test_A_4R = stratified_sampling(df_A_4R, test_prop, train_prop)
    df_train_B_4R, df_train_shuffled_B_4R, df_test_B_4R = stratified_sampling(df_B_4R, test_prop, train_prop)

    # Combine the TAG-specific datasets to create the final datasets
    df_train = pd.concat([df_train_A_4R, df_train_B_4R]).reset_index(drop=True)
    df_train_shuffled = pd.concat([df_train_shuffled_A_4R, df_train_shuffled_B_4R]).reset_index(drop=True)
    df_test = pd.concat([df_test_A_4R, df_test_B_4R]).reset_index(drop=True)

    # Shuffle the 'sequence' column in df_train_shuffled
    df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )

    '''
    df_train_shuffledについて
    バックグラウンドモデルがあれば、観測されたデータ（例えば、特定のモチーフが特定の領域に集中している、
    あるいは特定のタンパク質間に強い相互作用があるなど）が、
    単なる偶然によるものなのか、それとも何らかの生物学的な意味を持つのかを統計的に評価できます。
    '''

    # Getting motif information from the sequences
    # 各データフレームについて、モチーフとFASTAファイルを生成する。
    # train,test,train_shuffled には辞書型が返されている。
    train = generate_motifs_and_fastas(df_train, "train", number_of_sequences_to_motif_creation, subset_list)
    test = generate_motifs_and_fastas(df_test, "test", number_of_sequences_to_motif_creation, subset_list)
    train_shuffled = generate_motifs_and_fastas(
        df_train_shuffled,
        "train_shuffled",
        number_of_sequences_to_motif_creation,
        subset_list,
    )

    # 生成されたデータをロガーに記録
    logging.info(f"Train data: {train}")

    # 辞書に辞書を格納する。
    combined_dict = {"train": train,                    # トレーニングデータセットに関する情報が格納された辞書
                     "test": test,                      # テストデータセットに関する情報が格納された辞書
                     "train_shuffled": train_shuffled   # シャッフルデータセットに関する情報が格納された辞書
                     }

    # Writing to pickle
    if save_output:
        # Saving all train, test, train_shuffled dictionaries to pickle
        with open("encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y
