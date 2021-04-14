import numpy as np
import matplotlib.pyplot as plt


def curve_plot(path_result, frame_idx, rewards, losses):
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    plt.title('frame: %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title("loss")
    plt.plot(losses)
    plt.savefig(path_result)
    plt.close()