from accelerate import Accelerator

from dnadiffusion.data.dataloader import load_data
from dnadiffusion.models.diffusion import Diffusion
from dnadiffusion.models.unet import UNet
from dnadiffusion.utils.train_util import TrainLoop


def train():
    accelerator = Accelerator(split_batches=True, log_with=["wandb"], mixed_precision="bf16")

    # データの読み込み。load_data関数はdnadiffusion/data/dataloader.pyに定義されている。
    # train,test,shuffleのそれぞれのモチーフ、セルタイプ特有のモチーフ、セルタイプなどが格納されている。
    data = load_data(
        data_path="data/Combined_4R_dataframe.txt",
        saved_data_path="dnadiffusion/data/encode_data.pkl",

        # サブセットリスト。読み込むデータには以下2つのタグが含まれる。それを全て指定して訓練データとしている。
        subset_list=[
            "A_4R",
            "B_4R",
        ],
        # 0にすると全てのデータを使う。1000にすると1000個のデータを使う。
        limit_total_sequences=0,

        # dnadiffusion/data/dataloader.pyのsave_fasta関数を参照。
        num_sampling_to_compare_cells=1000,
        load_saved_data=True,
    )

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
    )

    diffusion = Diffusion(
        unet,
        timesteps=50,
    )

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        epochs=1000,
        log_step_show=50,
        sample_epoch=50,
        save_epoch=50,
        model_name="model_200k_A_B_4R",
        image_size=200,
        num_sampling_to_compare_cells=1000,
        batch_size=480,
    ).train_loop()


if __name__ == "__main__":
    train()
