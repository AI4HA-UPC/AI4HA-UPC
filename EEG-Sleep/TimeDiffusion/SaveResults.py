import os
import hydra
from omegaconf import DictConfig, OmegaConf, SCMode
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
import os
from numpy.fft import fft, ifft

import pandas as pd

from ai4ha.util import (
    instantiate_from_config,
    fix_paths,
    experiment_name_diffusion,
    config_logger,
)
from ai4ha.util.data import load_dataset


logger = config_logger(level=1)


def main(config):
    logger.info(
        f"-------------------------------------------------------------------")
    BASE_DIR = f"{config['exp_dir']}/logs/{config['name']}"
    dir = config["dataset"]["train"]["params"]["dir"].split("/")[-1]
    os.makedirs(f"0results/{dir}", exist_ok=True)
    padding = config["dataset"]["train"]["params"]["padding"]
    padalgo = config["dataset"]["train"]["params"]["padalgo"]
    if "-s" in padalgo:
        paddingl = padding // 2
    else:
        paddingl = 0
    length = config["model"]["params"]["sample_size"] - padding

    config["dataset"]["train"]["params"]["padding"] = 0
    config["dataset"]["train"]["params"]["padalgo"] = "zero"
    logger.info(f"Loading dataset")
    ddir = config["dataset"]["train"]["params"]["dir"].split("/")[-1]
    config["dataset"]["train"]["params"]["dir"] = f'/home/bejar/ssdstorage/{ddir}'

    orig = instantiate_from_config(config["dataset"]["train"])

    ldata = [d[0].squeeze() for d in orig]
    sdata = np.array(ldata)
    datal = np.array([int(d[1]) for d in orig])

    model = config["name"]
    sname = (config["samples"]["samples_gen"] //
             config["samples"]["sample_batch_size"]) + 1

    logger.info(f"Loading samples")
    samples = np.load(f"{BASE_DIR}/samples/sampled_data_{model}_{sname}.npz")
    nclasses = config["dataset"]["nclasses"]
    os.makedirs(f"0results/{dir}/{model}", exist_ok=True)
    channels = config["model"]["params"]["in_channels"]
    lab = samples["classes"]
    if channels == 1:
        logger.info(f"Saving samples")
        ch = 0
        samp = samples["samples"][:, ch, paddingl:paddingl + length]
        fig = plt.figure(figsize=(20, 30))
        d = 0
        for i in range(10):
            plt.subplot(10, 1, i + 1)
            plt.plot(samp[i + d], c="r")
            plt.title(lab[i + d])

        fig.savefig(f"0results/{dir}/{model}/samples_{ch}.png")

        logger.info(f"Saving samples FFT spectra")
        nfreq = 500

        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(8)
        for lsel in range(nclasses):
            sdatas = sdata[datal == lsel]

            mean_freq = np.zeros((nfreq))

            for n in range(sdatas.shape[0]):
                mean_freq += np.abs(fft(sdatas[n])[:nfreq]) / length

            mean_freq /= sdatas.shape[0]

            samps = samp[lab == lsel].squeeze()

            mean_freq2 = np.zeros((nfreq))
            for n in range(samps.shape[0]):
                mean_freq2 += np.abs(fft(samps[n])[:nfreq]) / length

            mean_freq2 /= samps.shape[0]

            sp1 = fig.add_subplot(nclasses, 2, lsel * 2 + 1)
            vmax = np.max([np.max(mean_freq), np.max(mean_freq2)])
            vmin = np.min([np.min(mean_freq), np.min(mean_freq2)])
            plt.ylim(0, vmax)
            plt.plot(mean_freq, c="r")
            plt.plot(mean_freq2, c="b")
            sp1 = fig.add_subplot(5, 2, lsel * 2 + 2)
            plt.ylim(0, vmax)
            plt.plot(np.abs(mean_freq - mean_freq2), c="g")

        fig.savefig(f"0results/{dir}/{model}/samples_fft.png")
        logger.info(f"Saving samples means")

        fig = plt.figure()
        fig.set_figheight(16)
        fig.set_figwidth(18)
        for lsel in range(nclasses):
            sdatas = sdata[datal == lsel]

            mean_r = np.mean(sdatas, axis=0)
            max_r = np.max(sdatas, axis=0)
            min_r = np.min(sdatas, axis=0)

            samps = samp[lab == lsel].squeeze()

            mean_g = np.mean(samps, axis=0)
            max_g = np.max(samps, axis=0)
            min_g = np.min(samps, axis=0)

            sp1 = fig.add_subplot(nclasses, 2, (lsel * 2) + 1)
            plt.plot(mean_r, c="r")
            plt.plot(mean_g, c="b")

            sp1 = fig.add_subplot(nclasses, 2, (lsel * 2) + 2)
            plt.plot(min_r, c="r")
            plt.plot(min_g, c="b")
            plt.plot(max_r, c="r")
            plt.plot(max_g, c="b")

        fig.savefig(f"0results/{dir}/{model}/samples_means.png")
        plt.close(fig)
    else:
        logger.info(f"Saving samples")
        samp = samples["samples"][:, :, paddingl:paddingl + length]
        for ch in range(channels):
            logger.info(f"Saving samples channel {ch}")

            fig = plt.figure(figsize=(20, 30))
            d = 0
            for i in range(10):
                plt.subplot(10, 1, i + 1)
                plt.plot(samp[i + d][ch], c="r")
                plt.title(lab[i + d])

            fig.savefig(f"0results/{dir}/{model}/samples_{ch:02d}.png")
            plt.close(fig)

        logger.info(f"Saving samples fft")
        nfreq = 100

        fig = plt.figure()
        fig.set_figheight(4 * nclasses)
        fig.set_figwidth(12)
        for lsel in range(nclasses):
            logger.info(f"Saving samples fft class {lsel}")
            sdatas = sdata[datal == lsel]

            mean_freq = np.zeros((sdatas.shape[1], nfreq))

            for n in range(sdatas.shape[0]):
                for i in range(sdatas.shape[1]):
                    mean_freq[i] += np.abs(fft(sdatas[n, i])[:nfreq]) / length

            mean_freq /= sdatas.shape[0]

            samps = samp[lab == lsel]

            mean_freq2 = np.zeros((samps.shape[1], nfreq))
            for n in range(samps.shape[0]):
                for i in range(samps.shape[1]):
                    mean_freq2[i] += np.abs(fft(samps[n, i])[:nfreq]) / length

            mean_freq2 /= samps.shape[0]

            plt.title(f'class {lsel}')
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 1)
            vmax = np.max([np.max(mean_freq), np.max(mean_freq2)])
            sns.heatmap(mean_freq,
                        cmap='seismic',
                        annot=False,
                        center=0,
                        fmt='.2f',
                        vmax=vmax)
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 2)
            sns.heatmap(mean_freq2,
                        cmap='seismic',
                        annot=False,
                        center=0,
                        fmt='.2f',
                        vmax=vmax)
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 3)
            sns.heatmap(mean_freq - mean_freq2,
                        cmap='seismic',
                        annot=False,
                        center=0)  # fmt='.2f', vmax=vmax)
        fig.savefig(f"0results/{dir}/{model}/samples_fft.png")
        plt.close(fig)

        logger.info(f"Saving samples means")
        for lsel in range(nclasses):
            logger.info(f"Saving samples means class {lsel}")
            fig = plt.figure()
            fig.set_figheight(14)
            fig.set_figwidth(12)
            for ch in range(channels):

                sdatas = sdata[datal == lsel]

                mean_r = np.mean(sdatas[:, ch, :], axis=0)

                samps = samp[lab == lsel].squeeze()

                mean_g = np.mean(samps[:, ch, :], axis=0)

                sp1 = fig.add_subplot(channels // 2, 2, ch + 1)
                plt.title(f'Channel {ch} - class {lsel}')
                plt.plot(mean_r, c='r')
                plt.plot(mean_g, c='b')
            fig.savefig(f"0results/{dir}/{model}/samples_means_{lsel:02d}.png")
            plt.close(fig)

        logger.info(f"Saving samples correlations")
        fig = plt.figure()
        fig.set_figheight(4 * nclasses)
        fig.set_figwidth(12)
        for lsel in range(nclasses):
            logger.info(f"Saving samples correlations class {lsel}")
            sdatas = sdata[datal == lsel]

            corr = np.zeros((sdatas.shape[1], sdatas.shape[1]))

            for n in range(sdatas.shape[0]):
                for i in range(sdatas.shape[1]):
                    for j in range(sdatas.shape[1]):
                        if i > j:
                            cr = np.corrcoef(sdatas[n, i], sdatas[n, j])[0, 1]
                            if np.isnan(cr):
                                corr[i, j] += 0
                            else:
                                corr[i, j] += cr
            corr /= sdatas.shape[0]
            samps = samp[lab == lsel]

            corr2 = np.zeros((samps.shape[1], samps.shape[1]))
            for n in range(samps.shape[0]):
                for i in range(samps.shape[1]):
                    for j in range(samps.shape[1]):
                        if i > j:
                            corr2[i, j] += np.corrcoef(samps[n, i],
                                                       samps[n, j])[0, 1]

            corr2 /= samps.shape[0]

            plt.title(f'class {lsel}')
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 1)
            sns.heatmap(corr,
                        cmap='seismic',
                        annot=False,
                        center=0,
                        fmt='.2f',
                        vmin=-1,
                        vmax=1)
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 2)
            sns.heatmap(corr2,
                        cmap='seismic',
                        annot=False,
                        center=0,
                        fmt='.2f',
                        vmin=-1,
                        vmax=1)
            lcorr = []
            for i in range(corr.shape[0]):
                for j in range(corr.shape[0]):
                    if i > j:
                        lcorr.append(corr2[i, j] - corr[i, j])
            sp1 = fig.add_subplot(nclasses, 3, lsel * 3 + 3)
            sns.histplot(np.array(lcorr), bins=100, kde=True, ax=sp1)
        fig.savefig(f"0results/{dir}/{model}/samples_corr.png")
        plt.close(fig)

        # if "tsne" in config and config["tsne"]["save"]:
        #     logger.info(f"Saving t-SNE real and generated data")
        #     if "samples" in config["tsne"]:
        #         datasamp = config["tsne"]["samples"]
        #     else:
        #         datasamp = sdata.shape[0]
        #     all = np.concatenate((samp[:, ch, :], sdata[:datasamp, ch, :]),
        #                          axis=0)
        #     tsne = TSNE(
        #         n_components=2,
        #         perplexity=10,
        #         init="random"  # max_iter=2000,
        #     ).fit_transform(all)
        #     fig = plt.figure()
        #     fig.set_figwidth(12)
        #     for lsel in range(nclasses):
        #         sp1 = fig.add_subplot(1, 2, lsel * 2 + 1)
        #         plt.scatter(
        #             tsne[:samp.shape[0], 0][lab == lsel],
        #             tsne[:samp.shape[0], 1][lab == lsel],
        #             c="r",
        #             s=1,
        #             marker="x",
        #         )
        #         plt.scatter(
        #             tsne[samp.shape[0]:, 0][datal[:datasamp] == lsel],
        #             tsne[samp.shape[0]:, 1][datal[:datasamp] == lsel],
        #             c="g",
        #             s=3,
        #         )

        #         sp1 = fig.add_subplot(nclasses, 2, lsel * 2 + 2)

        #         plt.scatter(
        #             tsne[:samp.shape[0], 0][lab == lsel],
        #             tsne[:samp.shape[0], 1][lab == lsel],
        #             c=lab[lab == lsel],
        #             cmap="viridis",
        #             s=1,
        #             marker="x",
        #         )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def TimeDiffusionResults(cfg: DictConfig) -> None:
    # Convert to dictionary so we can modify it
    cfg = OmegaConf.to_container(cfg, structured_config_mode=SCMode.DICT_CONFIG)
    cfg["local"] = True
    cfg = fix_paths(cfg, cfg["local"])
    cfg["name"] = experiment_name_diffusion(cfg)
    main(cfg)


if __name__ == "__main__":
    TimeDiffusionResults()
