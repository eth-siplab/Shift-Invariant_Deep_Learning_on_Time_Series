import os
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def tsne(latent, y_ground_truth, save_dir):
    """
        Plot t-SNE embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    y_ground_truth = y_ground_truth.cpu().detach().numpy()
    tsne = TSNE(n_components=2, verbose=0, perplexity=5, n_iter=300)
    tsne_results = tsne.fit_transform(latent)
    plt.figure(figsize=(4,3))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns.set_style("white")
    sns_plot = sns.scatterplot(
        x=tsne_results[:,0], y=tsne_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("flare", num_labels),
        alpha = 0.75,
        s = 50
        )
    sns_figure = sns_plot.get_figure()
    # sns_figure.savefig('ucihar-tsne.svgz',dpi=300)
    sns_figure.savefig(save_dir, dpi=300)
    # sns_figure.savefig('ucihar-tsne.pdf',dpi=300)
    # If you want to save high res pdf.
    # plt.savefig('save_dir.pdf', 
    #        dpi=300)
    ### If you want matlab
    # from scipy.io import savemat
    # mdic = {"x": tsne_results[:,0], "y" : tsne_results[:,1], "label": "experiment"}
    # savemat("matlab_tsne.mat", mdic)

def mds(latent, y_ground_truth, save_dir):
    """
        Plot MDS embeddings of the features
    """
    latent = latent.cpu().detach().numpy()
    mds = MDS(n_components=2)
    mds_results = mds.fit_transform(latent)
    plt.figure(figsize=(16,10))
    set_y = set(y_ground_truth)
    num_labels = len(set_y)
    sns_plot = sns.scatterplot(
        x=mds_results[:,0], y=mds_results[:,1],
        hue=y_ground_truth,
        palette=sns.color_palette("hls", num_labels),
        # data=df_subset,
        legend="full",
        alpha=0.5
        )

    sns_plot.get_figure().savefig(save_dir)



def sim_heatmap(similarity_tensor,target, args):
    """
        Plot similarity heatmap
    """
    import pdb;pdb.set_trace();
    data_dict = args.dataset + '_ablation.npz'

    abs_diff_matrix = torch.abs(target.view(-1, 1) - target.view(1, -1)).detach().cpu().numpy()
    similarity_tensor = similarity_tensor.detach().cpu().numpy()
    x1, y1 = abs_diff_matrix.flatten(), similarity_tensor.flatten()
    # Get the indices of the main diagonal
    diagonal_indices = np.arange(0, len(x1), x1.shape[0] + 1)
    # Remove the main diagonal elements
    x1 = np.delete(x1, diagonal_indices)
    y1 = np.delete(y1, diagonal_indices)
    # Check if the file already exists
    if os.path.exists(data_dict):
        loaded_data = np.load(data_dict)
        abs_diff_matrix = loaded_data['matrix1']
        similarity_tensor = loaded_data['matrix2']
        x1 = np.concatenate((x1, abs_diff_matrix.flatten()))
        y1 = np.concatenate((y1, similarity_tensor.flatten()))
    else:
        np.savez(data_dict, matrix1=abs_diff_matrix, matrix2=similarity_tensor)
        
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x1, y=y1, cmap="Blues", fill=True, thresh=0, levels=100)
    # Show the plot
    plt.show()
    import pdb;pdb.set_trace();
    #plt.savefig('small_sim.pdf',dpi=300)


def metrics_TR(seed_metric):
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    all_trgs = np.concatenate([s[0] for s in seed_metric])
    all_scores = np.vstack([s[1] for s in seed_metric])
    all_prds = np.concatenate([s[2] for s in seed_metric])
    auc = roc_auc_score(all_trgs, all_scores, multi_class='ovr')
    accuracy = accuracy_score(all_trgs, all_prds)
    f1 = f1_score(all_trgs, all_prds, average='macro')
    return auc*100, accuracy*100, f1*100