import argparse
from trainer import *
from vae_quant import train_VAE

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

############### Rep done ################
def main(args, seed):
    set_seed(seed)  # Change seed here
    DEVICE = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    # print('device:', DEVICE, 'dataset:', args.dataset)
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    model, optimizers, schedulers, criterion, classifier, criterion_cls, optimizer_cls = setup(args, DEVICE)

    # Train the VAE model with unlabelled large dataset, skip this step if the models is already trained
    # if not os.path.isfile(args.save+'/checkpt-0000.pth'):
    #     vae_model = train_VAE(train_loaders, args, DEVICE)   --> VAE is not the focus

    best_pretrain_model = train(train_loaders, val_loader, model, DEVICE, optimizers, schedulers, criterion, args)

    best_pretrain_model = test(test_loader, best_pretrain_model, DEVICE, criterion, args)

    ############################################################################################################

    trained_backbone = lock_backbone(best_pretrain_model, args)
    setattr(args, 'cases', 'subject') # Fine tune the models in the limited labelled data with the same target subject/domain
    train_loaders, val_loader, test_loader = setup_dataloaders(args)
    best_lincls = train_lincls(train_loaders, val_loader, trained_backbone, classifier, DEVICE, optimizer_cls, criterion_cls, args)
    miF,maF,maC,consistency = test_lincls(test_loader, trained_backbone, best_lincls, DEVICE, criterion_cls, args, plt=args.plt)
    delete_files(args)
    return miF, maF, maC, consistency
    # remove saved intermediate models
    values = np.array(all_metrics)
    mean = np.mean(values,0)
    print('Mean Acc: {}, Mean F1: {} '.format(mean[0],mean[1]))
    
    import pdb;pdb.set_trace();