from topmost.utils import config, log, miscellaneous, seed
import topmost
import os
import numpy as np
import scipy
import torch

RESULT_DIR = 'results'
DATA_DIR = 'data'

if __name__ == "__main__":
    # parse args, setup logger
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_logging_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()
    
    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))

    # if args.dataset in ['20NG']:
    #     read_labels = True
    read_labels = True

    # load a preprocessed dataset
    if args.model in ['TopiCOT']:
        dataset = topmost.data.BasicDatasetHandler(
            os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
            as_tensor=True, contextual_embed=True, batch_size=args.batch_size)

    # create a model
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()

    # create a model
    if args.model == 'TopiCOT':
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      doc_embedding=dataset.train_contextual_embed,
                                                      num_topics=args.num_topics,
                                                      num_groups=args.num_groups,
                                                      num_data=len(dataset.train_texts),
                                                      dropout=args.dropout,
                                                      pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                                                      weight_loss_ECR=args.weight_ECR,
                                                      alpha_ECR=args.alpha_ECR,
                                                      weight_loss_DCR=args.weight_DCR,
                                                      alpha_DCR=args.alpha_DCR,
                                                      weight_loss_TCR=args.weight_TCR,
                                                      alpha_TCR=args.alpha_TCR,
                                                      weight_loss_InfoNCE=args.weight_loss_InfoNCE,
                                                      beta_temp=args.beta_temp)
    if args.model == 'TopiCOT':
        model.weight_loss_ECR = args.weight_ECR
    model = model.to(args.device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)

    # train the model
    trainer.train(dataset)
    
    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)

    # argmax of train and test theta
    train_theta_argmax = train_theta.argmax(axis=1)
    test_theta_argmax = test_theta.argmax(axis=1)

    # evaluate topic diversity
    TD_10 = topmost.evaluations.compute_topic_diversity(
        top_words_10, _type="TD")
    print(f"TD_10: {TD_10:.5f}")
    logger.info(f"TD_10: {TD_10:.5f}")

    TD_15 = topmost.evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")
    logger.info(f"TD_15: {TD_15:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = topmost.evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

    # evaluate the C_V coherences
    TC_15_list, TC_15 = topmost.evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")
    logger.info(f"TC_15: {TC_15:.5f}")
    logger.info(f'TC_15 list: {TC_15_list}')
