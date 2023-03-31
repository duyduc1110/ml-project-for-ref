import numpy as np
import pandas as pd
import os, math
from matplotlib import pyplot as plt, animation
from matplotlib.animation import FuncAnimation


def read_imp(path='C:/Users/BruceNguyen/Rocsole Oy/Data Scientist Projects - data//imperial/caps/'):
    df = pd.concat(
        [pd.read_csv(path+f, header=None, sep='   ', index_col=0, names=range(768)) for f in os.listdir(path)],
        ignore_index=True,
    ).reset_index(drop=True)
    df.to_csv('imperial_data.csv', sep=',', index=False, header=None)
    t1, t2, t3 = df.values.reshape(-1, 32, 3, 8).transpose(2, 0, 1, 3)
    data = np.concatenate([np.pad(t1, ((0, 0), (0, 32), (0, 0))),
                           np.pad(t2, ((0, 0), (16, 16), (0, 0))),
                           np.pad(t3, ((0, 0), (32, 0), (0, 0)))],
                          axis=-1)
    return data


def read_shell():
    df = pd.read_csv(
        'C:/Users/BruceNguyen/Rocsole Oy/Data Scientist Projects - data/shell/data/shell_brunei_liti_air.txt',
        header=None, sep=' ', index_col=0)
    t1, t2, t3 = df.values.reshape(-1, 32, 3, 16).transpose(2, 0, 1, 3)
    data = np.concatenate([np.pad(t1, ((0, 0), (0, 32), (0, 0))),
                           np.pad(t2, ((0, 0), (16, 16), (0, 0))),
                           np.pad(t3, ((0, 0), (32, 0), (0, 0)))],
                          axis=-1)
    return data


def read_neste():
    df = pd.read_csv('data/neste/data/data_1/data_1.txt', header=None, sep=' ', index_col=0)
    t1, t2, t3 = df.values.reshape(-1, 32, 3, 16).transpose(2, 0, 1, 3)
    data = np.concatenate([np.pad(t1, ((0, 0), (0, 32), (0, 0))),
                           np.pad(t2, ((0, 0), (16, 16), (0, 0))),
                           np.pad(t3, ((0, 0), (32, 0), (0, 0)))],
                          axis=-1)
    return data


def read_cedre():
    df = pd.read_csv(
        'C:/Users/BruceNguyen/Rocsole Oy/Data Scientist Projects - data/cedre/cedre_data_interval_100/caps/datafile_all.txt',
        header=None, sep=' ', index_col=0)
    t1, t2 = df.values.reshape(-1, 2, 32, 16).transpose(1, 0, 2, 3)
    data = np.concatenate([np.pad(t1, ((0, 0), (0, 32), (0, 0))),
                           np.pad(t2, ((0, 0), (32, 0), (0, 0)))],
                          axis=-1)
    return data


def read_teflon(path='C:/Users/BruceNguyen/Documents/Github/ipig-dashboard/src/liquid-detection/data/Teflon/'):
    data, water_h, y = [], [], []
    water_level = {
        26: 33.6,
        27: 37.4,
        28: 36.3,
        29: 37.3,
        30: 37.9,
        31: 38.8,
        32: 39.7,
        33: 40.8,
        34: 42.6,
        35: 43.7,
        36: 44.8,
        37: 46.2,
        38: 48.9,
        39: 52.8,
        40: 54.7,
        41: 58.1,
        42: 59.2,
        43: 61.7,
        44: 63.7,
        45: 65.3,
        46: 69.5,
        47: 72.9,
        48: 75.2,
        49: 78.5,
        50: 84.6,
        51: 83.2,
        52: 80,
        53: 75.6,
        54: 71.1,
        55: 67.8,
        56: 63.9,
        57: 60.1,
        58: 57.4
    }
    for folder in os.listdir(path):
        test_id = int(folder[-2:])
        if 'A' in folder and test_id < 59:
            label = (water_level[test_id] - (1.2 + 2.8 + 0.3)) / 2.25
            label = int(label//1) if label%1 < 0.2 else int(label//1 + 1)
            files = os.listdir(path + folder + '/ect_1/data1/2022/2022-09/2022-09-07')
            if len(files) > 1:
                print(folder)
            for f in files:
                temp = pd.read_csv(
                    path + folder + '/ect_1/data1/2022/2022-09/2022-09-07/' + f, header=None,
                    sep='\t', index_col=0).drop(columns=1793)
                data.extend(temp.values.tolist())
                water_h.extend([water_level[test_id]] * temp.shape[0])
                y.extend([label] * temp.shape[0])
    df = pd.DataFrame(data)
    df['water_h'] = water_h
    df['y'] = y
    return df


class PauseAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Click to pause/resume the animation')
        x = np.linspace(-0.1, 0.1, 1000)

        self.animation = FuncAnimation(
            self.fig, self.update, frames=data_imp.shape[0], interval=50, blit=True)
        self.paused = False

        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, i):
        self.ax.set_title(f"frame {i}")
        self.ax.imshow(data_imp[i])





if __name__ == '__main__':
    data_neste = read_neste()
    data_imp = read_imp()
    data_cedre = read_cedre()
    data_shell = read_shell()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    #
    # ax1.set_title('Neste')
    # ax1.imshow(data_neste[0], aspect="auto")
    #
    # ax2.set_title('Imperial')
    # ax2.imshow(data_imp[0])
    #
    # ax3.set_title('Cedre')
    # ax3.imshow(data_cedre[0])
    #
    # ax4.set_title('Shell')
    # ax4.imshow(data_shell[1])
    #
    # plt.show()

    # NORMAL ANIMATION
    # fig, ax = plt.subplots(figsize=(6,8))
    #
    # def update(i):
    #     ax.imshow(data_imp[i])
    #     ax.set_title(f"frame {i}")
    #
    #
    # anim = FuncAnimation(fig, update, frames=len(data_imp), interval=100, repeat=False)
    #
    # plt.show()


    # # FAST ANIMATION
    # fig, ax = plt.subplots()
    #
    # for i, img in enumerate(data_imp):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(f"frame {i}")
    #     # Note that using time.sleep does *not* work here!
    #     plt.pause(0.1)

    fig, ax = plt.subplots()
    im = plt.imshow(data_imp[0], vmax=2000, vmin=0)
    plt.colorbar()

    def animate_func(i):
        im.set_array(data_imp[i])
        ax.set_title(f'Frame {i}')
        return [im]


    anim = FuncAnimation(
        fig,
        animate_func,
        frames=data_imp.shape[0],
        interval=10,  # in ms
    )

    writervideo = animation.FFMpegWriter(fps=5)
    anim.save('imperial_frame.mp4', writer=writervideo)

    #plt.show()


    # # ANIMATION WITH PAUSE
    # pa = PauseAnimation()
    # plt.show()