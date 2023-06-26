import numpy as np
import cv2


def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)

def hist_compare(hist_fram1: np.ndarray, hist_fram2: np.ndarray) -> float:
    metric_val: float =cv2.compareHist(hist_fram1, hist_fram2, cv2.HISTCMP_CORREL)
    return metric_val

############################ FUNCTIONS FOR DROWING ################################
class data_linewidth_plot():
    """
    Draws lines that could scale along with figure size
    Source: https://stackoverflow.com/questions/19394505/matplotlib-expand-the-line-with-specified-width-in-data-unit/42972469#42972469
    """
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


def plot_distances_chart(distances: np.ndarray, scene_borders: np.ndarray, ax: matplotlib.axes.Axes) -> None:
    """
    Plot scene borders on top of the pairwise distances matrix

    :param distances: pairwise distances matrix
    :param scene_borders:
    """
    ax.imshow(distances, cmap='gray')
    borders_from_zero = np.concatenate(([0], scene_borders))
    for i in range(1, len(borders_from_zero)):
        data_linewidth_plot(
            x=[borders_from_zero[i-1], borders_from_zero[i-1]],
            y=[borders_from_zero[i-1], borders_from_zero[i]],
            ax=ax, linewidth=1,
            color='red',
            alpha=0.5
        )
        data_linewidth_plot(
            x=[borders_from_zero[i-1], borders_from_zero[i]],
            y=[borders_from_zero[i-1], borders_from_zero[i-1]],
            ax=ax, linewidth=1,
            color='red',
            alpha=0.5
        )
        data_linewidth_plot(
            x=[borders_from_zero[i-1], borders_from_zero[i]],
            y=[borders_from_zero[i], borders_from_zero[i]],
            ax=ax,
            linewidth=1,
            color='red',
            alpha=0.5
        )
        data_linewidth_plot(
            x=[borders_from_zero[i], borders_from_zero[i]],
            y=[borders_from_zero[i-1], borders_from_zero[i]],
            ax=ax,
            linewidth=1,
            color='red',
            alpha=0.5
        )




