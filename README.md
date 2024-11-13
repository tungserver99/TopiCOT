# TopiCOT: Neural Topic Model Aligning with Pretrained Clustering and Optimal Transport


## Preparing libraries
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    wandb
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./topmost/evaluations/pametto.jar
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./data/wikipedia/ as an external reference corpus.

## Usage
To train and evaluate our model, run the following command:

> python main.py --model TopiCOT --dataset 20NG --num_topics 50 --beta_temp 0.2 --num_groups 20 --epochs 500 --device cuda --lr 0.002 --lr_scheduler StepLR --dropout 0.2 --batch_size 200 --lr_step_size 125 --use_pretrainWE --weight_ECR 200.0 --alpha_ECR 20.0 --weight_DCR 40.0 --weight_TCR 150.0

For demonstration of inference process, follow [this notebook](inferdemo.ipynb).

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.
