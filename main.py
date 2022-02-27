import warnings
from args import Args
from utils.train import train
from utils.util import *
from utils.data_loader import *
from model.acgcn_mmp import ACGCN_MMP
from model.acgcn_sub import ACGCN_SUB
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def main(args):

    device = args['DEVICE']
    random_seed = args['RANDOM_SEED']

    ### Make dataset ###
    data = pd.read_csv('./data/' + args['TARGET_ID'] + '_mmps.csv')
    data['label'] = data['label'].astype(int)
    label = data['label']

    counter = label.value_counts()
    tot = counter.sum()
    class_weight = [tot / (2 * i) for i in counter]
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed, stratify=label)

    if args['MODEL'] == 'acgcn-mmp':
        model = ACGCN_MMP(args).to(device)
        train_loader = ACGCN_MMP_Dataset(args, train_data, True)
        test_loader = ACGCN_MMP_Dataset(args, test_data, False)

    elif args['MODEL'] == 'acgcn-sub':
        model = ACGCN_SUB(args).to(device)
        train_loader = ACGCN_SUB_Dataset(args, train_data, True)
        test_loader = ACGCN_SUB_Dataset(args, test_data, False)
    
    y_actual = get_actual_label(test_loader)
    y_proba = train(args, model, train_loader, test_loader, class_weight)

    print_metrics(y_proba, y_actual)

if __name__ == '__main__':

    args = Args().params
    
    random_seed = args['RANDOM_SEED']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    main(args)
