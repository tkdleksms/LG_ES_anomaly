import argparse


# boolean argument 받기 위한 함수
# string 형태의 'True' or 'False' 받으면 bool로 변환
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# config
def load_config():
    # parser 선언
    parser = argparse.ArgumentParser()

    # Data loader 생성을 위한 argument
    parser.add_argument("--root_path", type=str, default='../yahoo_S5/A1Benchmark/',
                        help="이상치 탐지를 수행할 데이터의 경로를 입력하세요. ")
    parser.add_argument("--data_name", type=str, default='real_17.csv',
                        help="이상치 탐지를 수행할 데이터의 이름을 입력하세요. 예) 'real_1.csv'")
    parser.add_argument("--num_features", type=int, default=1)
    parser.add_argument("--make_plot", type=str2bool, default=True)
    parser.add_argument("--test_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--normal", type=str2bool, default=True,
                        help="데이터를 정규화하려면 True를 입력하세요. 전체 데이터를 train data의 min, max값을 사용해 [0, 1] 구간의 값으로 정규화합니다. ")
    parser.add_argument("--window_size", type=int, default=48,
                        help="window size를 입력하세요. 시계열 데이터를 window size만큼 잘라서 저장합니다.")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size를 입력하세요. ")
    parser.add_argument("--slide_size", type=int, default=1,
                        help="slide size를 입력하세요. window size만큼 sequence를 잘라서, slide size만큼의 시점씩 밀어서 데이터를 구축합니다.")
    parser.add_argument("--forecast", type=str2bool, default=True)
    parser.add_argument("--forecast_step", type=int, default=1)

    # For training
    parser.add_argument("--training", type=str2bool, default=True, help="Model을 새로 training하려면 True를 입력하세요. ")
    parser.add_argument("--checkpoint", type=str, default='best', help="best or specific ckpt")
    parser.add_argument("--cuda", type=str2bool, default=True, help="GPU를 사용하려면 True를 입력하세요. ")
    parser.add_argument("--n_feature", type=int, default=1, help = "이상치 탐지를 수행할 데이터의 변수 개수를 입력하세요. ")
    parser.add_argument("--manualSeed", type=int, default=42, help="manualSeed를 변경하려면 양수를 입력하세요. ")
    parser.add_argument("--lr", type=float, default=0, help="learning rate를 입력히세요.")
    parser.add_argument("--epochs", type=int, default=700, help="training을 위한 iteration 횟수를 입력하세요. ")
    parser.add_argument("--step_size", type=int, default=5, help="A hyperparameter for learing rate scheduler")
    parser.add_argument("--gamma", type=float, default=1.0, help="A hyperparameter for learing rate scheduler")
    parser.add_argument("--version", type=int, default=0,
                        help="새로 training 시, saved_model에 이미 존재하는 version의 다음 version으로 저장됩니다. 이미 학습된 model의 checkpoint를 가져와 test하려면 test할 version을 입력하세요. 이때, training은 False로 지정해야 합니다. ")
    # Gradient Clipping
    parser.add_argument("--clip", type=str2bool, default=False, help="Gradient Clipping")
    parser.add_argument("--max_norm", type=float, default=1.0, help="max_norm for gradient clipping")
    
    # Early stopping
    parser.add_argument("--early_stopping", type=str2bool, default=True, help="Early stopping")
    parser.add_argument("--patience", type=int, default=200, help="stop until val_loss not improved n times")
    parser.add_argument("--save_last", type=str2bool, default=False, help="save checkpoint at last epoch")
    parser.add_argument("--min_epoch", type=int, default=0)

    # For Transformer
    parser.add_argument("--feature_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--activation", type=str, default='None')
    parser.add_argument("--initrange", type=float, default=0.1)

    # For model
    parser.add_argument("--model", type=str, default='STOC')
    parser.add_argument("--trend_learning", type=str, default=False)

    # For Testing
    parser.add_argument("--eval_plot", type=str2bool, default=True)
    parser.add_argument("--anomaly_plot", type=str2bool, default=True)
    parser.add_argument("--shuffle", type=str2bool, default=False)
    parser.add_argument("--pred_one", type=str2bool, default=False)

    # to save
    parser.add_argument("--experiment_name", type=str, default='STOC_test')

    args = parser.parse_args()
    return args
